# d:/quantly/quanta/quantalytics/ai_agents/tools/logistic_regression_tool.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from statsmodels.formula.api import logit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from django.core.files.storage import default_storage
from typing import Any, Dict, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class LogisticRegressionTool(BaseAnalysisTool):
    """
    A tool to perform logistic regression for binary classification.
    """

    @property
    def name(self) -> str:
        return "logistic_regression"

    @property
    def description(self) -> str:
        return "Models the probability of a binary outcome based on one or more predictor variables."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="dependent_variable", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
            label="Dependent Variable (Y)", description="The binary (two-level) categorical variable you want to predict.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="positive_class", parameter_type=ParameterType.TEXT,
            label="Positive Class", description="The value in the dependent variable that represents the 'success' or 'event' case (e.g., 'Yes', 'True', '1').",
            required=True, help_text="Case-sensitive."
        ))
        params.add_parameter(ToolParameter(
            name="independent_variables", parameter_type=ParameterType.MULTISELECT,
            label="Independent Variables (X)", description="Select one or more variables to use as predictors.",
            required=True, column_source="all"
        ))
        params.add_parameter(ToolParameter(
            name="alpha", parameter_type=ParameterType.SELECT,
            label="Significance Level (α)", description="The threshold for determining if a predictor is statistically significant.",
            required=True, default_value="0.05",
            options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
        ))
        params.add_parameter(ToolParameter(
            name="penalizer", parameter_type=ParameterType.NUMBER,
            label="Penalizer (L1 Regularization)",
            description="Add a small penalty to handle collinearity. Try 0.01 to start if you get a 'Singular matrix' error.",
            required=False, default_value=0.0,
            help_text="A value > 0 enables L2 regularization."
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            dep_var = parameters.get("dependent_variable")
            pos_class = parameters.get("positive_class")
            ind_vars = parameters.get("independent_variables", [])
            penalizer = float(parameters.get("penalizer") or 0.0)

            if isinstance(ind_vars, str):
                ind_vars = [ind_vars]
            
            # Prevent dependent variable from being in independent variables
            if dep_var in ind_vars:
                ind_vars = [v for v in ind_vars if v != dep_var]
                if not ind_vars:
                     return {"status": "error", "summary": "You cannot select the Dependent Variable as the only Independent Variable."}

            if not all([dep_var, pos_class, ind_vars]):
                return {"status": "error", "summary": "Dependent Variable, Positive Class, and at least one Independent Variable are required."}

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            # Prepare data
            df_model = df[[dep_var] + ind_vars].dropna()
            
            if df_model[dep_var].nunique() != 2:
                return {"status": "error", "summary": f"The dependent variable '{dep_var}' must have exactly two unique values."}
            # Handle Type Mismatch for Positive Class (e.g. User types "1" but data is int(1))
            unique_vals = df_model[dep_var].unique()
            
            # 1. Try direct match
            found = pos_class in unique_vals
            
            # 2. If not found, try casting data to string
            if not found:
                 str_vals = [str(x) for x in unique_vals]
                 if pos_class in str_vals:
                     # It's a match, but we need to find the original value to filter correctly
                     # E.g. found "1" in strings, original was 1 (int)
                     # We update pos_class to be the original value from the data
                     original_val = next(x for x in unique_vals if str(x) == pos_class)
                     pos_class = original_val
                     found = True
            
            # 3. If still not found, try stripping decimals from input (e.g. user "1.0", data "1")
            if not found:
                try:
                    f_val = float(pos_class)
                    if f_val in unique_vals:
                        pos_class = f_val
                        found = True
                    # If data is int, float(1.0) == 1 (int) usually works in python lists, but let's be safe
                except:
                    pass

            if not found:
                return {"status": "error", "summary": f"The positive class '{pos_class}' was not found in the dependent variable '{dep_var}'. Available values: {list(unique_vals)}"}

            # Create binary dependent variable
            df_model[dep_var] = (df_model[dep_var] == pos_class).astype(int)

            # Clean column names for the formula
            clean_dep = self.clean_column_names([dep_var])[0]
            clean_ind = self.clean_column_names(ind_vars)
            rename_map = {**{dep_var: clean_dep}, **dict(zip(ind_vars, clean_ind))}
            df_model.rename(columns=rename_map, inplace=True)

            # Build formula
            formula_parts = [f"C(Q('{v}'))" if df_model[v].dtype in ['object', 'category', 'bool'] else f"Q('{v}')" for v in clean_ind]
            formula = f"Q('{clean_dep}') ~ {' + '.join(formula_parts)}"

            # Fit model
            artifacts: List[Dict[str, Any]] = []
            sections: List[Dict[str, Any]] = []
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)

                if penalizer > 0:
                    # For regularized models, summary() can fail. We build it manually.
                    model_instance = logit(formula, data=df_model)
                    model = model_instance.fit_regularized(method='l1', alpha=penalizer, disp=False)
                    
                    # Manually create summary tables
                    sections.append({'type': 'table', 'title': 'Model Summary', 'headers': ['Statistic', 'Value'], 'data': [
                        ['Dep. Variable:', f"Q('{clean_dep}')"],
                        ['Model:', 'Logit (L1 Regularized)'],
                        ['Method:', 'MLE'],
                        ['Pseudo R-squ.:', f"{model.prsquared:.4f}"],
                        ['Log-Likelihood:', f"{model.llf:.4f}"],
                        ['LL-Null:', f"{model.llnull:.4f}"],
                    ]})

                    coeffs_df = pd.DataFrame({
                        'coef': model.params,
                        'std err': 'N/A',
                        'z': 'N/A',
                        'P>|z|': 'N/A',
                        '[0.025': 'N/A',
                        '0.975]': 'N/A'
                    }).reset_index().rename(columns={'index': ''})

                    sections.append({
                        'type': 'table', 'title': 'Model Coefficients', 
                        'headers': coeffs_df.columns.tolist(), 
                        'data': coeffs_df.to_numpy().tolist(),
                        'footer': 'Standard errors and p-values are not calculated for regularized models.'
                    })

                else:
                    model = logit(formula, data=df_model).fit()
                    # --- Model Summary Tables from statsmodels ---
                    summary_tables = model.summary().tables
                    if len(summary_tables) > 0:
                        model_overview_data = [[row[0].strip(), row[1].strip()] for row in summary_tables[0].data]
                        sections.append({'type': 'table', 'title': 'Model Summary', 'headers': ['Statistic', 'Value'], 'data': model_overview_data})

                    if len(summary_tables) > 1:
                        coeffs_table = summary_tables[1]
                        coeffs_data = [list(map(lambda x: x.strip(), row)) for row in coeffs_table.data[1:]]
                        headers = [str(h).strip() for h in coeffs_table.data[0]]
                        sections.append({'type': 'table', 'title': 'Model Coefficients', 'headers': headers, 'data': coeffs_data})

                # If any ConvergenceWarning was caught, add it to the sections
                if w:
                    warning_messages = [str(warn.message) for warn in w]
                    sections.append({
                        'type': 'text',
                        'title': 'Convergence Warnings',
                        'content': "The model fitting process produced the following warnings, which may indicate that the model did not fully converge. This can happen with highly correlated predictors or other numerical issues.\n\n- " + "\n- ".join(warning_messages)
                    })

            summary = f"Logistic regression model fitted for '{dep_var}'. Pseudo R-squared: {model.prsquared:.3f}."

            # --- Performance Metrics & Plots ---
            predictions = (model.predict(df_model) > 0.5).astype(int)
            actuals = df_model[clean_dep]

            # Confusion Matrix
            cm = confusion_matrix(actuals, predictions)
            cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
            sections.append({
                'type': 'table', 'title': 'Confusion Matrix',
                'headers': [''] + cm_df.columns.tolist(),
                'data': [[idx] + list(row) for idx, row in cm_df.iterrows()]
            })

            with PlotUtils.setup_plotting():
                # Confusion Matrix Heatmap
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                ax1.set_xlabel('Predicted Label')
                ax1.set_ylabel('True Label')
                ax1.set_title('Confusion Matrix')
                artifacts.append({"type": "plot", "id": "confusion_matrix_plot", "title": "Confusion Matrix", "content": PlotUtils.fig_to_base64(fig1)})
                plt.close(fig1)

                # ROC Curve
                fpr, tpr, _ = roc_curve(actuals, model.predict(df_model))
                roc_auc = auc(fpr, tpr)
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax2.plot([0, 1], [0, 1], lw=2, linestyle='--')
                ax2.set_xlim([0.0, 1.0])
                ax2.set_ylim([0.0, 1.05])
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax2.legend(loc="lower right")
                artifacts.append({"type": "plot", "id": "roc_curve_plot", "title": "ROC Curve", "content": PlotUtils.fig_to_base64(fig2)})
                plt.close(fig2)

            # ---------- Odds Ratios (robust) ----------
            # Not all regularized results expose conf_int(); guard accordingly
            try:
                conf_int = model.conf_int()
                or_df = pd.DataFrame({
                    "Odds Ratio": model.params.apply(np.exp),
                    "95% CI Lower": conf_int.iloc[:, 0].apply(np.exp),
                    "95% CI Upper": conf_int.iloc[:, 1].apply(np.exp),
                })
            except Exception:
                or_df = pd.DataFrame({
                    "Odds Ratio": model.params.apply(np.exp),
                    "95% CI Lower": np.nan,
                    "95% CI Upper": np.nan,
                })
            odds_ratios = or_df.reset_index().rename(columns={'index': 'Variable'})
            sections.append({
                'type': 'table', 'title': 'Odds Ratios',
                'headers': odds_ratios.columns.tolist(),
                'data': odds_ratios.round(4).to_numpy().tolist(),
                'footer': 'An odds ratio > 1 indicates increased odds of the event; < 1 indicates decreased odds.'
            })

            # ---------- Key Takeaways builder ----------
            def _safe_float(x, default=np.nan):
                try:
                    return float(x)
                except Exception:
                    return default

            # Get/compute McFadden pseudo R^2 robustly
            pseudo_r2 = np.nan
            try:
                pseudo_r2 = _safe_float(getattr(model, "prsquared", np.nan))
                if not np.isfinite(pseudo_r2):
                    llf = _safe_float(getattr(model, "llf", np.nan))
                    llnull = _safe_float(getattr(model, "llnull", np.nan))
                    if np.isfinite(llf) and np.isfinite(llnull) and llnull != 0:
                        pseudo_r2 = max(0.0, 1.0 - (llf / llnull))
            except Exception:
                pass

            # Significant/retained variables (p < alpha for non-regularized; non-zero for regularized)
            alpha_val = float(parameters.get("alpha", 0.05))
            key_vars = []
            try:
                if penalizer > 0:
                    # retained (non-zero) coefficients excluding Intercept
                    non_zero = model.params[model.params != 0]
                    for var, coef in non_zero.items():
                        if str(var).lower() == 'intercept':
                            continue
                        or_val = float(np.exp(coef))
                        if or_val >= 1:
                            desc = f"↑ odds by ×{or_val:.2f} per unit"
                        else:
                            desc = f"↓ odds by ×{1/or_val:.2f} per unit"
                        key_vars.append(f"{var}: {desc}")
                else:
                    # pull p-values from the coefficients table we already added
                    coeff_section = next((s for s in sections if s.get('title') == 'Model Coefficients'), None)
                    pmap = {}
                    if coeff_section:
                        headers = coeff_section.get('headers', [])
                        rows = coeff_section.get('data', [])
                        # header can be 'P>|z|' or HTML-escaped 'P&gt;|z|'
                        try:
                            p_idx = headers.index('P>|z|')
                        except ValueError:
                            p_idx = headers.index('P&gt;|z|')
                        name_idx = 0
                        for r in rows:
                            name = r[name_idx]
                            pval = _safe_float(r[p_idx], default=1.0)
                            pmap[name] = pval

                    # map ORs by name
                    or_map = dict(zip(odds_ratios['Variable'].astype(str), odds_ratios['Odds Ratio'].astype(float)))
                    for name, pval in pmap.items():
                        if name.lower() == 'intercept':
                            continue
                        if pval < alpha_val and name in or_map:
                            or_val = or_map[name]
                            if or_val >= 1:
                                desc = f"↑ odds by ×{or_val:.2f} per unit (p={pval:.3f})"
                            else:
                                desc = f"↓ odds by ×{1/or_val:.2f} per unit (p={pval:.3f})"
                            key_vars.append(f"{name}: {desc}")
            except Exception:
                # swallow — key vars are optional
                pass

            # Any convergence warnings collected earlier?
            conv_section = next((s for s in sections if s.get('title') == 'Convergence Warnings'), None)
            has_conv_warn = conv_section is not None

            # Build bullet list
            key_takeaways: List[str] = []
            if np.isfinite(pseudo_r2):
                key_takeaways.append(f"Model fit: McFadden pseudo R² ≈ {pseudo_r2:.3f}.")
            key_takeaways.append(f"Discrimination: ROC AUC ≈ {roc_auc:.3f}.")
            if key_vars:
                key_takeaways.append("Important predictors: " + "; ".join(key_vars) + ".")
            else:
                key_takeaways.append("No individual predictors stood out at the selected significance/retention threshold.")
            if has_conv_warn:
                key_takeaways.append("Fitting issued convergence warnings — consider removing collinear features or adding a small penalizer (e.g., 0.01).")

            # Present as a readable text section (bullets)
            sections.append({
                'type': 'text',
                'title': 'Key Takeaways',
                'content': "• " + "\n• ".join(key_takeaways)
            })

            # Final summary (keep yours; make it safe)
            if np.isfinite(pseudo_r2):
                summary = f"Logistic regression model fitted for '{dep_var}'. Pseudo R-squared: {pseudo_r2:.3f}."
            else:
                summary = f"Logistic regression model fitted for '{dep_var}'."

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "data": {
                    "key_takeaways": key_takeaways
                },
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                    "statistical_results": {"auc": roc_auc, "pseudo_r2": None if not np.isfinite(pseudo_r2) else float(pseudo_r2)}
                },
            }

        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e):
                return {
                    "status": "error",
                    "summary": "Model fitting failed due to perfect multicollinearity (Singular matrix).\n\n"
                               "This means one or more of your independent variables are perfectly redundant.\n\n"
                               "**Suggestion:** Please review your selected predictors. For example, '6th Stage' might be perfectly predictable from 'T Stage' and 'N Stage'. Try removing one of these redundant variables and run the analysis again."
                }
            return {"status": "error", "summary": f"A linear algebra error occurred: {str(e)}"}
        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}
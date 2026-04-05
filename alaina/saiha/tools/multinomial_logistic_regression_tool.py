
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from django.core.files.storage import default_storage
from typing import Any, Dict, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class MultinomialLogisticRegressionTool(BaseAnalysisTool):
    """
    Multinomial Logistic Regression for multi-category outcomes (K >= 3).
    Includes adaptive collapsing of sparse outcome levels to 'Other'.
    """

    @property
    def name(self) -> str:
        return "multinomial_logistic_regression"

    @property
    def description(self) -> str:
        return "Models the probability of a multi-category outcome based on one or more predictor variables."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="dependent_variable", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
            label="Dependent Variable (Y)", description="The categorical outcome (3+ levels).", required=True
        ))
        params.add_parameter(ToolParameter(
            name="reference_category", parameter_type=ParameterType.TEXT,
            label="Reference (Base) Category",
            description="Category used as the baseline (must exist in the outcome).",
            required=True, help_text="Case-sensitive; spaces/underscores will be normalized like the outcome."
        ))
        params.add_parameter(ToolParameter(
            name="independent_variables", parameter_type=ParameterType.MULTISELECT,
            label="Independent Variables (X)",
            description="One or more predictors. Categorical predictors will be one-hot encoded.",
            required=True, column_source="all"
        ))
        # NEW: rare class handling controls
        params.add_parameter(ToolParameter(
            name="collapse_rare_classes", parameter_type=ParameterType.CHECKBOX,
            label="Collapse Rare Outcome Classes",
            description="If checked, rare outcome classes are collapsed into 'Other' to stabilize the model.",
            required=False, default_value=True
        ))
        params.add_parameter(ToolParameter(
            name="min_class_count", parameter_type=ParameterType.NUMBER,
            label="Minimum Count per Outcome Class",
            description="Attempt to keep only classes with at least this many rows (adaptive: tries this, then 2).",
            required=False, default_value=3, validation_rules={"min": 1, "step": 1}
        ))
        params.add_parameter(ToolParameter(
            name="generate_confusion_matrix", parameter_type=ParameterType.CHECKBOX,
            label="Generate Confusion Matrix Plot", description="Visualize model performance with a heatmap.",
            required=False, default_value=True
        ))
        params.add_parameter(ToolParameter(
            name="generate_coefficient_plot", parameter_type=ParameterType.CHECKBOX,
            label="Generate Coefficient Plot", description="Visualize the model coefficients and their confidence intervals.",
            required=False, default_value=True
        ))
        return params

    # ----------------- helpers -----------------

    def _fmt_num(self, x, nd=4) -> str:
        try:
            xv = float(x)
            if np.isfinite(xv):
                return f"{xv:.{nd}f}"
        except Exception:
            pass
        return str(x)

    def _clean_label(self, s: Any) -> Optional[str]:
        """Normalize labels like TNM; return None for missing-like tokens."""
        if s is None:
            return None
        s = str(s).strip()
        if s == "" or s in {"-", "NA", "N/A"}:
            return None
        if s.lower() in {"nan", "none", "null"}:
            return None
        s = " ".join(s.split())    # collapse internal spaces
        s = s.replace(" ", "")     # TNM often without spaces
        s = s.replace("__", "_").replace("_", "")  # strip underscores for consistency
        return s

    def _collapse_rare_classes(self, y: pd.Series, ref: str, min_class_count: int = 3) -> pd.Series:
        """Collapse classes with count < min_class_count into 'Other', protecting the reference."""
        vc = y.value_counts(dropna=False)
        rare = set(vc[vc < min_class_count].index)
        if ref in rare:
            raise ValueError(
                f"Reference category '{ref}' has only {int(vc[ref])} row(s) (< {min_class_count})."
            )
        if not rare:
            return y
        return y.apply(lambda v: "Other" if v in rare else v)

    def _drop_constant_columns(self, X: pd.DataFrame):
        """Drop predictors with zero variance; return cleaned X and list of dropped names."""
        dropped = []
        for c in list(X.columns):
            if X[c].nunique(dropna=True) <= 1:
                dropped.append(c)
                X = X.drop(columns=[c])
        return X, dropped

    # ----------------- main -----------------

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            p = kwargs
            dep_var = p.get("dependent_variable")
            ref_cat_input = p.get("reference_category")
            ind_vars = p.get("independent_variables", [])
            gen_cm_plot = str(p.get("generate_confusion_matrix", "true")).lower() in ('true', 'on', '1')
            gen_coef_plot = str(p.get("generate_coefficient_plot", "true")).lower() in ('true', 'on', '1')
            collapse_rare = bool(p.get("collapse_rare_classes", True))
            min_class_count = int(p.get("min_class_count", 3) or 3)

            if isinstance(ind_vars, str):
                ind_vars = [ind_vars]

            if not all([dep_var, ref_cat_input, ind_vars]):
                return {"status": "error", "summary": "Dependent Variable, Reference Category, and at least one Independent Variable are required."}

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            sections: List[Dict[str, Any]] = []
            artifacts: List[Dict[str, Any]] = []

            # Column presence
            use_cols = [dep_var] + ind_vars
            missing_cols = [c for c in use_cols if c not in df.columns]
            if missing_cols:
                return {"status": "error", "summary": f"Column(s) not found: {', '.join(missing_cols)}."}

            initial_rows = len(df)

            # Clean target labels; build modeling frame & drop NAs across used columns
            y_raw = df[dep_var].apply(self._clean_label)
            model_df = pd.concat([y_raw.rename(dep_var), df[ind_vars]], axis=1).dropna()
            rows_dropped = initial_rows - len(model_df)
            if len(model_df) == 0:
                return {"status": "error", "summary": "No usable rows after cleaning target and predictors."}

            # Normalize reference category same way
            ref_cat = self._clean_label(ref_cat_input)
            if ref_cat is None:
                return {"status": "error", "summary": f"The provided reference category '{ref_cat_input}' normalizes to empty. Choose a valid base."}

            y_clean = model_df[dep_var].astype(str)

            # -------- Adaptive collapse of rare outcome classes --------
            collapse_notes = []
            if collapse_rare:
                y_collapsed = None
                tried_thresholds = [max(1, min_class_count), 2] if min_class_count != 2 else [2, 1]
                # ensure uniqueness & order
                tried_thresholds = list(dict.fromkeys(tried_thresholds))

                for thr in tried_thresholds:
                    try:
                        tmp = self._collapse_rare_classes(y_clean, ref_cat, min_class_count=thr)
                        if pd.Series(tmp).nunique() >= 3:
                            y_collapsed = tmp
                            collapse_notes.append(f"Collapsed rare outcome classes with threshold ≥{thr}.")
                            break
                        else:
                            collapse_notes.append(f"Threshold ≥{thr}: left <3 classes; trying a lower threshold.")
                    except ValueError as ve:
                        collapse_notes.append(f"Threshold ≥{thr}: skipped ({ve}).")

                if y_collapsed is None:
                    y_collapsed = y_clean
                    collapse_notes.append("Skipped collapsing to preserve ≥3 outcome categories.")
            else:
                y_collapsed = y_clean
                collapse_notes.append("Rare class collapsing disabled by user.")

            # Need at least 3 classes
            n_classes = pd.Series(y_collapsed).nunique()
            if n_classes < 3:
                return {
                    "status": "error",
                    "summary": (
                        "After cleaning, the outcome has fewer than 3 categories. "
                        "Multinomial logistic regression requires ≥3 classes.\n\n"
                        "Suggestions:\n"
                        "1) Use the **binary logistic regression** tool (one-vs-rest) for your current outcome, or\n"
                        "2) Pre-collapse outcome categories (merge very sparse TNM levels) so ≥3 remain."
                    )
                }

            # Order categories with base last (statsmodels uses last as base when passing integer-coded outcome)
            cats = [c for c in pd.Series(y_collapsed).unique() if c != ref_cat] + [ref_cat]
            y_cat = pd.Categorical(y_collapsed, categories=cats, ordered=True)
            y_codes = y_cat.codes  # 0..K-1

            # -------- Encode X and drop constant columns --------
            X = pd.get_dummies(model_df[ind_vars], drop_first=True, dtype=float)
            # coerce to numeric
            for c in X.columns:
                if not pd.api.types.is_numeric_dtype(X[c]):
                    X[c] = pd.to_numeric(X[c], errors="coerce")
            X = X.dropna(axis=1, how="all")
            X, dropped_const = self._drop_constant_columns(X)

            notes = []
            if rows_dropped > 0:
                notes.append(f"Excluded **{rows_dropped} rows** due to missing/invalid values.")
            if dropped_const:
                notes.append("Dropped constant predictors: " + ", ".join(dropped_const))
            if collapse_notes:
                notes.extend(collapse_notes)
            if notes:
                sections.append({'type': 'text', 'title': 'Data Preparation Notes', 'content': "\n".join(notes)})

            if X.shape[1] == 0:
                return {"status": "error", "summary": "No usable predictor columns after encoding and constant-drop. Check your inputs."}

            # add constant
            X = sm.add_constant(X, has_constant="add")

            # -------- Fit MNLogit --------
            model = sm.MNLogit(y_codes, X).fit(disp=0)

            # -------- Summaries --------
            sections.append({
                'type': 'table',
                'title': 'Model Summary',
                'headers': ['Statistic', 'Value'],
                'data': [
                    ['Dep. Variable:', dep_var],
                    ['Reference Category:', ref_cat],
                    ['Model:', 'Multinomial Logit (MNLogit)'],
                    ['No. Observations:', int(model.nobs)],
                    ['Pseudo R-squ.:', self._fmt_num(model.prsquared, 4)],
                    ['Log-Likelihood:', self._fmt_num(model.llf, 4)],
                ]
            })

            nonbase = cats[:-1]  # columns correspond to non-base outcomes
            coef_df = pd.DataFrame(model.params, index=X.columns, columns=nonbase).reset_index().rename(columns={'index': 'Predictor'})
            sections.append({
                'type': 'table',
                'title': 'Model Coefficients (Log-odds relative to base)',
                'headers': coef_df.columns.tolist(),
                'data': coef_df.round(4).to_numpy().tolist(),
                'footer': f"Each column is log-odds vs base '{ref_cat}'."
            })

            or_df = np.exp(model.params)
            or_df = pd.DataFrame(or_df, index=X.columns, columns=nonbase).reset_index().rename(columns={'index': 'Predictor'})
            sections.append({
                'type': 'table',
                'title': 'Relative Risk Ratios (exp(coef))',
                'headers': or_df.columns.tolist(),
                'data': or_df.round(4).to_numpy().tolist(),
                'footer': f"Values > 1 increase odds vs '{ref_cat}' per one-unit increase in predictor."
            })

            # -------- Predictions & metrics --------
            probs = pd.DataFrame(model.predict(X), columns=nonbase + [ref_cat])  # ensure full label set
            pred_labels = probs.idxmax(axis=1).astype(str)

            # Map back the integer-coded actuals to their label strings
            actual_labels = pd.Series(cats, index=range(len(cats))).reindex(y_codes).reset_index(drop=True).astype(str)

            report = classification_report(actual_labels, pred_labels, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'Category'})
            sections.append({
                'type': 'table',
                'title': 'Classification Report',
                'headers': report_df.columns.tolist(),
                'data': report_df.round(4).to_numpy().tolist()
            })

            # -------- Plots --------
            with PlotUtils.setup_plotting():
                if gen_cm_plot:
                    labels = [str(c) for c in cats]
                    cm = confusion_matrix(actual_labels, pred_labels, labels=labels)
                    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                xticklabels=labels, yticklabels=labels)
                    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual'); ax_cm.set_title('Confusion Matrix')
                    artifacts.append({
                        "type": "plot", "id": "mnl_confusion_matrix", "title": "Confusion Matrix",
                        "content": PlotUtils.fig_to_base64(fig_cm)
                    })
                    plt.close(fig_cm)

                if gen_coef_plot:
                    ci = model.conf_int()
                    predictors = list(X.columns)
                    outcomes = nonbase
                    params_long, ci_long = [], []
                    for j, out in enumerate(outcomes):
                        for i, pred in enumerate(predictors):
                            params_long.append({"Outcome": out, "Predictor": pred, "coef": model.params.iloc[i, j]})
                    ci_idx = 0
                    for j, out in enumerate(outcomes):
                        for i, pred in enumerate(predictors):
                            ci_long.append({
                                "Outcome": out, "Predictor": pred,
                                "ci_lower": ci.iloc[ci_idx, 0],
                                "ci_upper": ci.iloc[ci_idx, 1]
                            })
                            ci_idx += 1
                    plot_df = pd.merge(pd.DataFrame(params_long), pd.DataFrame(ci_long), on=["Outcome", "Predictor"])
                    plot_df = plot_df[plot_df["Predictor"].str.lower() != "const"]

                    fig_b, ax_b = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.03)))
                    plot_df = plot_df.sort_values(["Outcome", "coef"])
                    ylabels = [f"{o} | {p}" for o, p in zip(plot_df["Outcome"], plot_df["Predictor"])]
                    y_pos = np.arange(len(plot_df))
                    errors = [plot_df["coef"] - plot_df["ci_lower"], plot_df["ci_upper"] - plot_df["coef"]]
                    ax_b.errorbar(plot_df["coef"].values, y_pos, xerr=np.vstack(errors), fmt='o', capsize=3)
                    ax_b.axvline(0, color='red', linestyle='--', linewidth=1)
                    ax_b.set_yticks(y_pos); ax_b.set_yticklabels(ylabels, fontsize=8)
                    ax_b.set_title("Coefficients with 95% CI (per Outcome vs Base)")
                    ax_b.set_xlabel("Log-odds (coef)"); ax_b.set_ylabel("Outcome | Predictor")
                    plt.tight_layout()
                    artifacts.append({
                        "type": "plot", "id": "mnl_coef_plot", "title": "Coefficient Plot",
                        "content": PlotUtils.fig_to_base64(fig_b)
                    })
                    plt.close(fig_b)

            summary = (
                f"Multinomial logistic regression fitted for '{dep_var}' with base '{ref_cat}'. "
                f"Pseudo R-squared: {self._fmt_num(model.prsquared, 3)}."
            )

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": p},
            }

        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e):
                return {
                    "status": "error",
                    "summary": ("Model failed (Singular matrix). Likely complete separation or severe collinearity.\n"
                                "Try removing redundant predictors, merging very sparse outcome levels, or using fewer predictors.")
                }
            return {"status": "error", "summary": f"A linear algebra error occurred: {str(e)}"}
        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get('status') != 'ok':
            return None
        try:
            params = results.get('meta', {}).get('parameters', {})
            dep_var = params.get('dependent_variable')
            ref_cat = params.get('reference_category')

            prsquared = None
            summary_section = next((s for s in results.get('sections', []) if s.get('title') == 'Model Summary'), None)
            if summary_section:
                for k, v in summary_section.get('data', []):
                    if k == 'Pseudo R-squ.:':
                        try:
                            prsquared = float(v)
                        except Exception:
                            prsquared = None
                        break

            acc = None
            report_section = next((s for s in results.get('sections', []) if s.get('title') == 'Classification Report'), None)
            if report_section:
                for row in report_section.get('data', []):
                    if str(row[0]).lower() == 'accuracy':
                        try:
                            acc = float(row[1])
                        except Exception:
                            acc = None
                        break

            parts = [f"A multinomial logistic regression was built to predict **{dep_var}** (base category: **{ref_cat}**)."]
            if acc is not None:
                parts.append(f"The model accuracy is **{acc:.2%}**.")
            if prsquared is not None:
                parts.append(f"McFadden's pseudo R² is **{prsquared:.3f}** (higher indicates better fit).")
            parts.append("Odds ratios (>1) indicate higher odds of the outcome (vs base) per one-unit increase in the predictor; "
                         "values <1 indicate lower odds.")
            return " ".join(parts)

        except Exception:
            return ("Multinomial model fitted. Review the coefficients / odds ratio tables and confusion matrix "
                    "for performance and effect directions.")
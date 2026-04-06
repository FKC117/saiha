
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from django.core.files.storage import default_storage
from typing import Any, Dict, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class LinearRegressionTool(BaseAnalysisTool):
    """
    A tool to perform multiple linear regression analysis.
    """

    @property
    def name(self) -> str:
        return "linear_regression"

    @property
    def description(self) -> str:
        return "Models the relationship between a dependent variable and one or more independent variables."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="dependent_variable", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Dependent Variable (Y)", description="The numeric variable you want to predict.", required=True
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
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            dep_var = parameters.get("dependent_variable")
            ind_vars = parameters.get("independent_variables", [])

            # Ensure ind_vars is a list, as single selections can come as a string
            if isinstance(ind_vars, str):
                ind_vars = [ind_vars]

            if not dep_var or not ind_vars:
                return {"status": "error", "summary": "Dependent and at least one Independent variable are required."}

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            # Clean column names for the formula
            clean_dep = self.clean_column_names([dep_var])[0]
            clean_ind = self.clean_column_names(ind_vars)
            rename_map = {**{dep_var: clean_dep}, **dict(zip(ind_vars, clean_ind))}
            df.rename(columns=rename_map, inplace=True)

            # Build the formula string, wrapping categorical variables with C()
            formula_parts = []
            for var, clean_var in zip(ind_vars, clean_ind):
                if df[clean_var].dtype in ['object', 'category', 'bool']:
                    formula_parts.append(f"C(Q('{clean_var}'))")
                else:
                    formula_parts.append(f"Q('{clean_var}')")
            
            formula = f"Q('{clean_dep}') ~ {' + '.join(formula_parts)}"

            # Fit the model
            model = ols(formula, data=df).fit()

            summary = f"Linear regression model fitted for '{dep_var}'. Adjusted R-squared: {model.rsquared_adj:.3f}."

            artifacts: List[Dict[str, Any]] = []
            sections: List[Dict[str, Any]] = []

            # --- Model Summary Tables ---
            summary_tables = model.summary().tables
            if len(summary_tables) > 0:
                # First table is key-value, reformat it
                model_overview_data = []
                for row in summary_tables[0].data:
                    model_overview_data.append([row[0].strip(), row[1].strip()])
                    if len(row) > 2:
                        model_overview_data.append([row[2].strip(), row[3].strip()])
                sections.append({
                    'type': 'table', 'title': 'Model Summary',
                    'headers': ['Statistic', 'Value'],
                    'data': model_overview_data
                })
            
            # Coefficients table
            if len(summary_tables) > 1:
                coeffs_table = summary_tables[1]
                sections.append({
                    'type': 'table', 'title': 'Model Coefficients',
                    'headers': [str(h).strip() for h in coeffs_table.data[0]],
                    'data': [list(map(lambda x: x.strip(), row)) for row in coeffs_table.data[1:]]
                })

            # Diagnostics table
            if len(summary_tables) > 2:
                diag_table = summary_tables[2]
                diag_data = []
                for row in diag_table.data:
                    diag_data.append([row[0].strip(), row[1].strip()])
                    if len(row) > 2:
                        diag_data.append([row[2].strip(), row[3].strip()])
                sections.append({
                    'type': 'table', 'title': 'Model Diagnostics',
                    'headers': ['Statistic', 'Value'],
                    'data': diag_data
                })

            # --- Diagnostic Plots ---
            with PlotUtils.setup_plotting():
                # Residuals vs. Fitted
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax1, line_kws={'color': 'red', 'lw': 1})
                ax1.set_title('Residuals vs. Fitted Values')
                ax1.set_xlabel('Fitted Values')
                ax1.set_ylabel('Residuals')
                artifacts.append({"type": "plot", "id": "residuals_plot", "title": "Residuals vs. Fitted", "content": PlotUtils.fig_to_base64(fig1)})
                plt.close(fig1)

                # Normal Q-Q Plot
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sm.qqplot(model.resid, line='s', ax=ax2)
                ax2.set_title('Normal Q-Q Plot of Residuals')
                artifacts.append({"type": "plot", "id": "qq_plot", "title": "Normal Q-Q Plot", "content": PlotUtils.fig_to_base64(fig2)})
                plt.close(fig2)

                # Predicted vs. Actual Plot
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=df[clean_dep], y=model.fittedvalues, ax=ax3, alpha=0.6)
                # Add a 45-degree line for reference
                min_val = min(df[clean_dep].min(), model.fittedvalues.min())
                max_val = max(df[clean_dep].max(), model.fittedvalues.max())
                ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                ax3.set_title('Predicted vs. Actual Values')
                ax3.set_xlabel('Actual Values')
                ax3.set_ylabel('Predicted Values')
                artifacts.append({"type": "plot", "id": "predicted_vs_actual_plot", "title": "Predicted vs. Actual", "content": PlotUtils.fig_to_base64(fig3)})
                plt.close(fig3)

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": parameters},
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Provides a formal interpretation of the linear regression results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            dep_var = params.get('dependent_variable', 'the dependent variable')

            r_squared_adj = "N/A"
            f_prob = 1.0
            significant_vars = []

            # Parse model summary table
            summary_section = next((s for s in results.get('sections', []) if s.get('title') == 'Model Summary'), None)
            if summary_section:
                for stat, val in summary_section.get('data', []):
                    if stat == 'Adj. R-squared:': r_squared_adj = float(val)
                    if stat == 'Prob (F-statistic):': f_prob = float(val)

            # Parse coefficients table
            coeffs_section = next((s for s in results.get('sections', []) if s.get('title') == 'Model Coefficients'), None)
            if coeffs_section:
                headers = coeffs_section.get('headers', [])
                p_val_idx = headers.index('P>|t|')
                coeff_idx = headers.index('coef')
                var_idx = 0 # First column is the variable name

                for row in coeffs_section.get('data', []):
                    var_name = row[var_idx]
                    if var_name == 'Intercept': continue
                    p_value = float(row[p_val_idx])
                    if p_value < alpha:
                        coeff_val = float(row[coeff_idx])
                        direction = "an increase" if coeff_val > 0 else "a decrease"
                        significant_vars.append(f"'{var_name}' (a one-unit increase is associated with {direction} of {abs(coeff_val):.3f} in '{dep_var}')")

            # Build interpretation
            interpretation_parts = []
            
            # Overall model significance
            if f_prob < alpha:
                interpretation_parts.append(f"The overall model is statistically significant (F-test p-value: {f_prob:.4f}), suggesting that the predictors as a group reliably predict '{dep_var}'.")
            else:
                interpretation_parts.append(f"The overall model is not statistically significant (F-test p-value: {f_prob:.4f}), suggesting the predictors do not reliably predict '{dep_var}'.")

            # R-squared
            interpretation_parts.append(f"The Adjusted R-squared is {r_squared_adj:.3f}, which means approximately {r_squared_adj*100:.1f}% of the variance in '{dep_var}' can be explained by the predictors in the model.")

            # Significant predictors
            if significant_vars:
                interpretation_parts.append(f"The following predictors were found to be statistically significant at the α={alpha} level: {'; '.join(significant_vars)}.")
            else:
                interpretation_parts.append(f"No individual predictors were found to be statistically significant at the α={alpha} level.")

            return " ".join(interpretation_parts)

        except (ValueError, TypeError, IndexError, KeyError):
            return "Could not automatically interpret the linear regression results."
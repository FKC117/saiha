# d:/quantly/quanta/quantalytics/ai_agents/tools/log_linear_tool.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
import statsmodels.api as sm
import statsmodels.formula.api as smf
from django.core.files.storage import default_storage
from typing import Any, Dict, Optional, List, Tuple
from itertools import combinations

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class LogLinearTool(BaseAnalysisTool):
    """
    A tool to perform Log-Linear Analysis on two or more categorical variables to model their association structure.
    """

    @property
    def name(self) -> str:
        return "log_linear_analysis"

    @property
    def description(self) -> str:
        return "Models the association structure between 2 or more categorical variables using hierarchical log-linear models."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(
            ToolParameter(
                name="variables", parameter_type=ParameterType.MULTISELECT,
                label="Categorical Variables",
                description="Select two or more categorical columns to analyze.",
                required=True,
                column_source='categorical'
            )
        )
        params.add_parameter(
            ToolParameter(
                name="stabilize_zeros", parameter_type=ParameterType.CHECKBOX,
                label="Stabilize for Zero Counts",
                description="Adds a small constant (0.5) to all cell counts. This is a standard method to prevent errors when some combinations of categories have zero observations.",
                required=False,
                default_value=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_mosaic_plot", parameter_type=ParameterType.CHECKBOX,
                label="Generate Mosaic Plot",
                description="Visualize observed frequencies. Available for 2 or 3 variables.",
                required=False, default_value=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_coefficient_plot", parameter_type=ParameterType.CHECKBOX,
                label="Generate Coefficient Plot",
                description="Plot the coefficients and confidence intervals for the best-fitting model.",
                required=False, default_value=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_residual_heatmap", parameter_type=ParameterType.CHECKBOX,
                label="Generate Residuals Heatmap",
                description="Visualize where the best model's predictions differ from observed counts. Available for 2 or 3 variables.",
                required=False, default_value=False
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_interaction_plot", parameter_type=ParameterType.CHECKBOX,
                label="Generate Interaction Plot",
                description="Visualize a three-way interaction. Available only for 3 variables.",
                required=False,
                default_value=True
            )
        )
        return params

    def _clean_var_name(self, var_name: str) -> str:
        """
        Cleans and quotes a variable name for use in a Patsy formula.
        It quotes if the name contains special characters and escapes existing backticks.
        """
        # Escape any existing backticks
        cleaned_name = var_name.replace('`', '\\`')
        # Quote if it contains spaces or operators that would confuse patsy
        if any(c in cleaned_name for c in ' +-/*()[]~'):
            return f"`{cleaned_name}`"
        return cleaned_name

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            variables = parameters.get("variables")
            stabilize = str(parameters.get("stabilize_zeros", "true")).lower() in ('true', 'on', '1')
            gen_mosaic = str(parameters.get("generate_mosaic_plot", "true")).lower() in ('true', 'on', '1')
            gen_coef_plot = str(parameters.get("generate_coefficient_plot", "true")).lower() in ('true', 'on', '1')
            gen_resid_heatmap = str(parameters.get("generate_residual_heatmap", "false")).lower() in ('true', 'on', '1')
            gen_interaction_plot = str(parameters.get("generate_interaction_plot", "true")).lower() in ('true', 'on', '1')

            if not variables or len(variables) < 2:
                return {"status": "error", "summary": "Please select at least two categorical variables."}
            
            # Add a hard limit to prevent performance issues from combinatorial explosion
            MAX_LOGLINEAR_VARS = 6
            if len(variables) > MAX_LOGLINEAR_VARS:
                return {"status": "error", "summary": (
                    f"Log-Linear Analysis with more than {MAX_LOGLINEAR_VARS} variables is computationally infeasible for an interactive session. "
                    "Please select fewer variables (4-5 is recommended) and try again.")}

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            # --- Prepare data with stabilization for sparse tables ---
            # 1. Create the full contingency table structure to identify zero-cells
            all_levels = [df[v].astype('category').cat.categories.tolist() for v in variables]
            full_index = pd.MultiIndex.from_product(all_levels, names=variables)
            full_contingency = pd.DataFrame(index=full_index).reset_index()

            # 2. Get the observed counts
            observed_counts_df = df.groupby(variables).size().reset_index(name='count')

            # 3. Merge to get the full table, filling missing combinations with 0
            freq_df = pd.merge(full_contingency, observed_counts_df, on=variables, how='left').fillna(0)
            
            # 4. Optionally, add a small constant (0.5) to all cells to avoid issues with zeros (stabilization)
            if stabilize:
                freq_df['count'] += 0.5

            # Clean column names for formula compatibility
            clean_vars = [self._clean_var_name(v) for v in variables]
            freq_df.columns = clean_vars + ['count']

            models = []
            # --- Fit Hierarchical Models ---
            # Model 0: Independence model (main effects only)
            formula_main = f"count ~ {' + '.join(clean_vars)}"
            fit_main = smf.poisson(formula_main, data=freq_df).fit(disp=0)
            models.append({'name': 'Independence (Main Effects)', 'fit': fit_main, 'terms': 'Main Effects'})

            # Model 1 to N-1: Add k-way interactions
            for k in range(2, len(clean_vars) + 1):
                formula_k_way = f"count ~ ({' + '.join(clean_vars)})**{k}"
                fit_k_way = smf.poisson(formula_k_way, data=freq_df).fit(disp=0)
                
                term_names = [f"{k}-way interactions"]
                if k > 2:
                    term_names.insert(0, f"up to {k-1}-way")

                models.append({'name': f'All {k}-way Interactions', 'fit': fit_k_way, 'terms': ' + '.join(term_names)})

            # The last model is the saturated model
            models[-1]['name'] = 'Saturated Model'

            # --- Prepare Results ---
            sections = []
            model_comparison_data = []
            for model in models:
                fit = model['fit']
                model_comparison_data.append([
                    model['name'],
                    model['terms'],
                    f"{fit.llf:.4f}",
                    f"{fit.llr:.4f}", # G-squared (Likelihood Ratio)
                    fit.df_resid,
                    f"{fit.aic:.4f}",
                    f"{fit.bic:.4f}"
                ])

            sections.append({
                'type': 'table', 'title': 'Log-Linear Model Comparison',
                'headers': ['Model', 'Highest-Order Terms', 'Log-Likelihood', 'G-squared', 'df', 'AIC', 'BIC'],
                'data': model_comparison_data,
                'footer': ("The best model typically has the lowest AIC/BIC. G-squared tests the goodness of fit against the saturated model. "
                           f"Note: Zero-cell stabilization {'was applied' if stabilize else 'was NOT applied'}.")

            })

            # Add parameter estimates for the best model (lowest AIC)
            best_model = min(models, key=lambda m: m['fit'].aic)
            best_fit_summary = best_model['fit'].summary2().tables[1]
            best_fit_summary.reset_index(inplace=True)
            best_fit_summary.rename(columns={'index': 'Parameter'}, inplace=True)

            sections.append({
                'type': 'table', 'title': f"Parameter Estimates for Best Model: {best_model['name']}",
                'headers': best_fit_summary.columns.tolist(),
                'data': best_fit_summary.round(4).values.tolist(),
                'footer': "These are the coefficients of the log-linear model. Significant terms indicate important associations."
            })

            summary_text = (
                f"Log-Linear analysis performed on variables: {', '.join(variables)}. "
                f"The best-fitting model based on AIC is the '{best_model['name']}' model. "
                "This suggests the presence of the associations indicated by the significant parameters in its estimates table."
            )

            # --- Visualizations ---
            artifacts = []
            with PlotUtils.setup_plotting():
                # Mosaic Plot (for 2 or 3 variables)
                if gen_mosaic:
                    if len(variables) in [2, 3]:
                        try:
                            # Use observed counts (before stabilization) for mosaic plot
                            mosaic_data = observed_counts_df.set_index(variables)['count']
                            fig, _ = mosaic(mosaic_data, title=f'Mosaic Plot for {", ".join(variables)}', gap=0.02)
                            fig.set_size_inches(10, 7)
                            artifacts.append({"type": "plot", "id": "loglinear_mosaic", "title": "Mosaic Plot of Observed Frequencies", "content": PlotUtils.fig_to_base64(fig)})
                        except Exception as e:
                            sections.append({'type': 'text', 'title': 'Mosaic Plot Skipped', 'content': f"Could not generate mosaic plot due to an error: {str(e)}"})
                    else:
                        sections.append({'type': 'text', 'title': 'Mosaic Plot Skipped', 'content': "Mosaic plots are only generated for analyses with 2 or 3 variables to ensure readability."})

                # Coefficient Plot
                if gen_coef_plot:
                    try:
                        # Exclude intercept for clarity
                        plot_df = best_fit_summary[~best_fit_summary['Parameter'].str.contains('Intercept', case=False)].copy()
                        plot_df['ci_lower'] = plot_df['[0.025']
                        plot_df['ci_upper'] = plot_df['0.975]']
                        plot_df['error'] = (plot_df['ci_upper'] - plot_df['ci_lower']) / 2

                        fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.5)))
                        ax.errorbar(x=plot_df['Coef.'], y=plot_df['Parameter'], xerr=plot_df['error'], fmt='o', capsize=5)
                        ax.axvline(0, color='red', linestyle='--')
                        ax.set_title(f'Coefficient Plot for {best_model["name"]}')
                        ax.set_xlabel('Coefficient (Log Scale)')
                        ax.set_ylabel('Model Parameter')
                        plt.tight_layout()
                        artifacts.append({"type": "plot", "id": "loglinear_coef_plot", "title": "Coefficient Plot with 95% CI", "content": PlotUtils.fig_to_base64(fig)})
                    except Exception as e:
                        sections.append({'type': 'text', 'title': 'Coefficient Plot Skipped', 'content': f"Could not generate coefficient plot due to an error: {str(e)}"})

                # Pearson Residuals Heatmap (for 2 or 3 variables)
                if gen_resid_heatmap:
                    if len(variables) in [2, 3]:
                        try:
                            # Use observed counts (before stabilization)
                            observed = freq_df.set_index(variables)['count']
                            if stabilize:
                                observed = (observed - 0.5).clip(lower=0)
                            
                            expected = best_model['fit'].predict(freq_df[clean_vars])
                            expected.index = observed.index

                            # Pearson residuals: (Observed - Expected) / sqrt(Expected)
                            with np.errstate(divide='ignore', invalid='ignore'):
                                pearson_residuals = (observed - expected) / np.sqrt(expected)
                            pearson_residuals = pearson_residuals.fillna(0)

                            if len(variables) == 2:
                                resid_table = pearson_residuals.unstack()
                                fig, ax = plt.subplots(figsize=(10, 8))
                                sns.heatmap(resid_table, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
                                ax.set_title(f'Heatmap of Pearson Residuals for {best_model["name"]}')
                                artifacts.append({"type": "plot", "id": "loglinear_resid_heatmap", "title": "Heatmap of Pearson Residuals", "content": PlotUtils.fig_to_base64(fig)})
                            elif len(variables) == 3:
                                resid_table = pearson_residuals.unstack(level=-1)
                                g = sns.FacetGrid(resid_table.stack().reset_index(), col=variables[2], col_wrap=min(3, len(all_levels[2])), sharex=False, sharey=False)
                                g.map_dataframe(lambda data, color: sns.heatmap(data.pivot(index=variables[0], columns=variables[1], values=0), annot=True, fmt=".2f", cmap="vlag", center=0))
                                g.fig.suptitle(f'Heatmap of Pearson Residuals by {variables[2]}', y=1.03)
                                artifacts.append({"type": "plot", "id": "loglinear_resid_heatmap_faceted", "title": f"Faceted Heatmap of Pearson Residuals by {variables[2]}", "content": PlotUtils.fig_to_base64(g.fig)})

                        except Exception as e:
                            sections.append({'type': 'text', 'title': 'Residuals Heatmap Skipped', 'content': f"Could not generate residuals heatmap due to an error: {str(e)}"})
                    else:
                        sections.append({'type': 'text', 'title': 'Residuals Heatmap Skipped', 'content': "Residuals heatmaps are only generated for analyses with 2 or 3 variables to ensure readability."})

                # Three-way Interaction Plot
                if gen_interaction_plot:
                    if len(variables) == 3:
                        try:
                            # Use the model's predictions on the full grid of possibilities
                            plot_df = freq_df.copy()
                            plot_df['predicted_count'] = best_model['fit'].predict(freq_df[clean_vars])
                            
                            g = sns.catplot(data=plot_df, x=variables[0], y='predicted_count', hue=variables[1], col=variables[2], kind='point', col_wrap=min(4, len(all_levels[2])))
                            g.fig.suptitle(f'Three-Way Interaction Plot for {best_model["name"]}', y=1.03)
                            g.set_axis_labels(variables[0], "Predicted Count")
                            artifacts.append({"type": "plot", "id": "loglinear_interaction_plot", "title": "Three-Way Interaction Plot", "content": PlotUtils.fig_to_base64(g.fig)})
                        except Exception as e:
                            sections.append({'type': 'text', 'title': 'Interaction Plot Skipped', 'content': f"Could not generate interaction plot due to an error: {str(e)}"})
                    elif gen_interaction_plot: # Only show message if user checked the box but didn't provide 3 vars
                        sections.append({'type': 'text', 'title': 'Interaction Plot Skipped', 'content': "Interaction plots are only generated for analyses with exactly 3 variables."})
            return {
                "status": "ok",
                "summary": summary_text,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": parameters},
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            variables = params.get('variables', [])
            
            model_comparison_section = next((s for s in results['sections'] if s['title'] == 'Log-Linear Model Comparison'), None)
            if not model_comparison_section:
                return "Could not find model comparison results for interpretation."

            # Find the model with the lowest AIC
            models_data = model_comparison_section['data']
            best_model_row = min(models_data, key=lambda row: float(row[5])) # AIC is at index 5
            best_model_name = best_model_row[0]
            best_model_terms = best_model_row[1]

            interpretation = (
                f"The analysis fitted several hierarchical models to understand the relationships between {', '.join(variables)}. "
                f"Based on the Akaike Information Criterion (AIC), the best-fitting model is the **'{best_model_name}'**. "
            )

            if 'Saturated' in best_model_name:
                interpretation += (
                    "This indicates a complex relationship where the highest-order interaction "
                    f"({len(variables)}-way interaction) is significant. In simpler terms, the relationship between any two variables "
                    "depends on the levels of all other variables in the model."
                )
            elif 'Main Effects' in best_model_name:
                interpretation += (
                    "This suggests that while there are no significant **interaction effects** between the variables, "
                    "one or more variables may still have a significant main effect on the cell counts. "
                    "You should examine the 'Parameter Estimates' table to see which individual variables are statistically significant (P>|z| < 0.05)."
                )
            else:
                interpretation += (
                    f"This model includes **{best_model_terms}**. This means there are significant associations "
                    "at this level. You should examine the parameter estimates for this model to see which specific "
                    "interactions are statistically significant (typically p-value < 0.05)."
                )

            return interpretation

        except Exception as e:
            return f"Could not automatically interpret the results due to an error: {e}"
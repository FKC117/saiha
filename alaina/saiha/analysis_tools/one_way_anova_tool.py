# d:/quantly/quanta/quantalytics/ai_agents/tools/one_way_anova_tool.py

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import base64
import io
from django.core.files.storage import default_storage
from typing import Dict, Any, List

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from saiha.ai_agents.tools.plot_utils import PlotUtils

class OneWayAnovaTool(BaseAnalysisTool):
    """
    A tool to perform a One-Way Analysis of Variance (ANOVA) to compare the means
    of a numeric variable across two or more groups of a categorical variable.
    """

    @property
    def name(self) -> str:
        """The unique name of the tool."""
        return "one_way_anova"

    @property
    def description(self) -> str:
        """
        Returns a human-friendly description of what the tool does.
        """
        return "Performs a One-Way ANOVA to compare means across different groups."

    def get_parameters_schema(self) -> ToolParameterSet:
        """
        Defines the parameters the tool accepts for UI generation and validation.
        """
        params = ToolParameterSet(tool_name="one_way_anova")
        params.add_parameter(
            ToolParameter(
                name="dependent_variable",
                label="Dependent Variable (Numeric)",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                required=True,
                help_text="Select the numeric column whose means you want to compare."
            )
        )
        params.add_parameter(
            ToolParameter(
                name="group_variable",
                label="Grouping Variable (Categorical)",
                parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                required=True,
                help_text="Select the categorical column that defines the groups."
            )
        )
        params.add_parameter(
            ToolParameter(
                name="alpha",
                label="Significance Level (α)",
                parameter_type=ParameterType.SELECT,
                required=True,
                default_value="0.05",
                options=[
                    {"value": "0.01", "label": "0.01 (99% Confidence)"},
                    {"value": "0.05", "label": "0.05 (95% Confidence)"},
                    {"value": "0.10", "label": "0.10 (90% Confidence)"}
                ],
                help_text="The threshold for statistical significance."
            )
        )
        params.add_parameter(
            ToolParameter(
                name="post_hoc_test",
                label="Perform Post-Hoc Test (Tukey's HSD)",
                parameter_type=ParameterType.CHECKBOX,
                description="If ANOVA is significant, perform pairwise comparisons to see which groups differ.",
                required=False,
                default_value=False
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        """
        Executes the One-Way ANOVA, returning results in the standard Result Envelope.
        """
        try:
            # Force matplotlib to use a non-GUI backend to prevent tkinter errors on the server
            matplotlib.use('Agg')

            dataset_id = kwargs.get("dataset_id") # noqa
            parameters = kwargs
            
            # Correctly load the dataset using the storage manager
            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()
            
            dependent_var = parameters.get("dependent_variable")
            group_var = parameters.get("group_variable")
            alpha = float(parameters.get("alpha", 0.05))
            perform_posthoc = str(parameters.get("post_hoc_test", "false")).lower() in ('true', 'on', '1')

            # Clean column names for statsmodels formula
            clean_dependent, clean_group = self.clean_column_names([dependent_var, group_var])
            df.rename(columns={dependent_var: clean_dependent, group_var: clean_group}, inplace=True)

            # Validate Data Requirements
            if df[group_var].nunique() < 2:
                return {
                    "status": "error",
                    "summary": f"The grouping variable '{group_var}' must have at least two unique groups (levels) to perform ANOVA. Found: {df[group_var].unique().tolist()}"
                }
            
            if df[dependent_var].nunique() <= 1:
                 return {
                    "status": "error",
                    "summary": f"The dependent variable '{dependent_var}' must have some variance (more than 1 unique value). All values are identical."
                }

            # Perform ANOVA
            try:
                model = ols(f"Q('{clean_dependent}') ~ C(Q('{clean_group}'))", data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                # Safely extract stats with fallbacks
                if 'F' in anova_table and len(anova_table['F']) > 0:
                     f_statistic = anova_table['F'][0]
                else:
                     f_statistic = np.nan
                     
                if 'PR(>F)' in anova_table and len(anova_table['PR(>F)']) > 0:
                    p_value = anova_table['PR(>F)'][0]
                else:
                    p_value = np.nan
                    
                is_significant = bool(p_value < alpha) if not pd.isna(p_value) else False
                
            except Exception as stat_err:
                 return {
                    "status": "error",
                    "summary": f"Statistical calculation failed: {str(stat_err)}. This often happens if groups have too few samples or zero variance."
                }

            sections: List[Dict[str, Any]] = []
            artifacts: List[Dict[str, Any]] = []

            # Perform post-hoc test (Tukey's HSD) if significant
            if is_significant and perform_posthoc:
                try:
                    tukey = pairwise_tukeyhsd(endog=df[clean_dependent], groups=df[clean_group], alpha=alpha)
                    post_hoc_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                    
                    sections.append({
                        'type': 'table', 'title': "Tukey's HSD Post-Hoc Test",
                        'headers': post_hoc_df.columns.tolist(),
                        'data': post_hoc_df.to_numpy().tolist(),
                        "footer": "Shows pairwise comparisons between groups."
                    })

                    # Generate heatmap visualization from the tukey results
                    heatmap_df = post_hoc_df.pivot(index='group1', columns='group2', values='p-adj')
                    fig_heatmap, ax_heatmap = plt.subplots(figsize=(max(8, len(heatmap_df.columns)), max(6, len(heatmap_df.index))))
                    sns.heatmap(heatmap_df, annot=True, cmap='coolwarm_r', fmt=".4f", ax=ax_heatmap, cbar_kws={'label': 'Adjusted P-value'})
                    ax_heatmap.set_title("Post-Hoc P-Value Heatmap (Tukey's HSD)")
                    plt.tight_layout()
                    artifacts.append({
                        "type": "plot", "id": "anova_posthoc_heatmap", "title": "Post-Hoc Heatmap",
                        "content": PlotUtils.fig_to_base64(fig_heatmap)
                    })
                except Exception as posthoc_ex:
                    sections.append({"type": "text", "title": "Post-Hoc Test Failed", "content": str(posthoc_ex)})

            # Generate Visualizations
            with PlotUtils.setup_plotting():
                # Box Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=clean_group, y=clean_dependent, data=df, ax=ax)
                ax.set_title(f'Distribution of {dependent_var} by {group_var}')
                ax.set_xlabel(group_var)
                ax.set_ylabel(dependent_var)
                plt.xticks(rotation=45, ha='right')
                artifacts.append({
                    "type": "plot", "id": "anova_boxplot", "title": "Box Plot",
                    "content": PlotUtils.fig_to_base64(fig)
                })

                # Violin Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.violinplot(x=clean_group, y=clean_dependent, data=df, ax=ax, inner="quartile")
                ax.set_title(f'Distribution of {dependent_var} by {group_var}')
                ax.set_xlabel(group_var)
                ax.set_ylabel(dependent_var)
                plt.xticks(rotation=45, ha='right')
                artifacts.append({
                    "type": "plot", "id": "anova_violinplot", "title": "Violin Plot",
                    "content": PlotUtils.fig_to_base64(fig)
                })

                # Means Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.pointplot(x=clean_group, y=clean_dependent, data=df, ax=ax, capsize=.1, errorbar='sd')
                ax.set_title(f'Mean of {dependent_var} by {group_var} (with SD)')
                ax.set_xlabel(group_var)
                ax.set_ylabel(f'Mean of {dependent_var}')
                plt.xticks(rotation=45, ha='right')
                artifacts.append({
                    "type": "plot", "id": "anova_meansplot", "title": "Means Plot",
                    "content": PlotUtils.fig_to_base64(fig)
                })

            # Add main ANOVA table to sections
            anova_table_df = anova_table.reset_index().rename(columns={'index': 'Source'})
            sections.insert(0, {
                'type': 'table', 'title': 'One-Way ANOVA Results',
                'headers': anova_table_df.columns.tolist(),
                'data': anova_table_df.round(4).to_numpy().tolist()
            })

            # Construct Result Envelope
            return {
                "status": "ok",
                "summary": f"One-Way ANOVA for '{dependent_var}' by '{group_var}' completed. The result is {'significant' if is_significant else 'not significant'} (p={p_value:.4f}).",
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": parameters}
            }
        except Exception as e:
            self.log_error(e)
            return {
                "status": "error",
                "summary": f"An unexpected error occurred during One-Way ANOVA: {str(e)}",
            }

    def interpret(self, results: Dict[str, Any]) -> str:
        """
        Provides a formal interpretation of the One-Way ANOVA results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            dependent_var = params.get('dependent_variable', 'the dependent variable')
            group_var = params.get('group_variable', 'the grouping variable')

            p_value = None
            anova_section = next((s for s in results.get('sections', []) if s.get('title') == 'One-Way ANOVA Results'), None)
            
            if anova_section:
                # Find the 'PR(>F)' header index
                headers = anova_section.get('headers', [])
                try:
                    p_value_index = headers.index('PR(>F)')
                    # The p-value is in the first data row at that index
                    p_value = float(anova_section['data'][0][p_value_index])
                except (ValueError, IndexError):
                    pass

            if p_value is None:
                return "Could not automatically determine the p-value for interpretation."

            if p_value < alpha:
                post_hoc_ran = any("Post-Hoc Test" in s.get('title', '') for s in results.get('sections', []))
                post_hoc_advice = ""

                tukey_results = next((s for s in results.get('sections', []) if "Tukey's HSD Post-Hoc Test (for all group combinations)" in s.get('title', '')), None)
                significant_pairs = []
                if tukey_results:
                    # Look up reject results in Tukey's HSD table and construct the output string
                    reject_col_index = tukey_results['headers'].index('reject')
                    group1_col_index = tukey_results['headers'].index('group1')
                    group2_col_index = tukey_results['headers'].index('group2')

                    for row in tukey_results.get('data', []):
                        if row[reject_col_index] == True:  # If 'reject' is True
                            significant_pairs.append(f"'{row[0]}' and '{row[1]}'")

                if significant_pairs and post_hoc_ran:
                    pairings_string = ", ".join(significant_pairs)  # Joining those significant
                    post_hoc_advice = f" The post-hoc test (Tukey's HSD) shows significant differences between the following group pairings: {pairings_string}."
                elif post_hoc_ran:
                     post_hoc_advice = " However, the post-hoc test did not identify any significant differences between specific group pairings at the selected alpha level."


                return f"Since the p-value ({p_value:.4f}) is less than the significance level (α = {alpha}), we reject the null hypothesis. This indicates there is a statistically significant difference in the mean of '{dependent_var}' across the different groups of '{group_var}'.{post_hoc_advice}"
            else:
                return f"Since the p-value ({p_value:.4f}) is greater than or equal to the significance level (α = {alpha}), we fail to reject the null hypothesis. There is not enough evidence to conclude that there are any significant differences in the mean of '{dependent_var}' across the groups of '{group_var}'."

        except (ValueError, TypeError, IndexError):
            return "Could not automatically interpret the ANOVA results."

import pandas as pd
import numpy as np
from statsmodels.stats.anova import AnovaRM
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import scikit_posthocs as sp
    SCIKIT_POSTHOCS_AVAILABLE = True
except ImportError:
    SCIKIT_POSTHOCS_AVAILABLE = False
from django.core.files.storage import default_storage
from typing import Dict, Any, List

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from quantalytics.ai_agents.tools.plot_utils import PlotUtils

class RepeatedMeasuresAnovaTool(BaseAnalysisTool):
    """
    A tool to perform a Repeated Measures Analysis of Variance (ANOVA) to test for
    differences between related group means.
    """

    @property
    def name(self) -> str:
        """The unique name of the tool."""
        return "repeated_measures_anova"

    @property
    def description(self) -> str:
        """A human-friendly description of what the tool does."""
        return "Performs a Repeated Measures ANOVA for related group comparisons."

    def get_parameters_schema(self) -> ToolParameterSet:
        """Defines the parameters the tool accepts for UI generation."""
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(
            ToolParameter(
                name="subject_id",
                label="Subject ID Column",
                parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                required=True,
                help_text="Column that uniquely identifies each subject or participant."
            )
        )
        params.add_parameter(
            ToolParameter(
                name="within_factor",
                label="Within-Subject Factor (Time/Condition)",
                parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                required=True,
                help_text="Categorical variable representing the repeated conditions (e.g., Time 1, Time 2)."
            )
        )
        params.add_parameter(
            ToolParameter(
                name="dependent_variable",
                label="Dependent Variable (Numeric)",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                required=True,
                help_text="The numeric outcome variable measured at each condition."
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
                    {"value": "0.01", "label": "0.01"},
                    {"value": "0.05", "label": "0.05"},
                    {"value": "0.10", "label": "0.10"}
                ],
                help_text="The threshold for statistical significance."
            )
        )
        params.add_parameter(
            ToolParameter(
                name="perform_posthoc", parameter_type=ParameterType.CHECKBOX,
                label="Perform Post-Hoc Test (Paired T-Tests)",
                description="If the main test is significant, perform pairwise comparisons between conditions.",
                required=False, default_value=False
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Executes the Repeated Measures ANOVA."""
        try:
            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            subject = kwargs.get("subject_id")
            within = kwargs.get("within_factor")
            dv = kwargs.get("dependent_variable")
            alpha = float(kwargs.get("alpha", 0.05))
            perform_posthoc = str(kwargs.get("perform_posthoc", "false")).lower() in ('true', 'on', '1')

            # Clean column names for statsmodels
            clean_subject, clean_within, clean_dv = self.clean_column_names([subject, within, dv])
            df.rename(columns={subject: clean_subject, within: clean_within, dv: clean_dv}, inplace=True)

            # Perform Repeated Measures ANOVA
            # Use aggregate_func='mean' to handle cases where there are multiple observations
            # per subject and cell. This is a common scenario in real-world data.
            aov = AnovaRM(data=df, 
                        depvar=clean_dv, 
                        subject=clean_subject, 
                        within=[clean_within], 
                        aggregate_func='mean')
            res = aov.fit()
            anova_table = res.anova_table

            f_statistic = anova_table.loc[clean_within, 'F Value']
            p_value = anova_table.loc[clean_within, 'Pr > F']
            is_significant = bool(p_value < alpha)

            sections: List[Dict[str, Any]] = []
            artifacts: List[Dict[str, Any]] = []

            # Add main ANOVA table to sections
            anova_table_df = anova_table.reset_index().rename(columns={'index': 'Source'})
            sections.append({
                'type': 'table', 'title': 'Repeated Measures ANOVA Results',
                'headers': anova_table_df.columns.tolist(),
                'data': anova_table_df.round(4).to_numpy().tolist()
            })

            # Perform post-hoc test if significant
            if is_significant and perform_posthoc:
                if not SCIKIT_POSTHOCS_AVAILABLE:
                    sections.append({"type": "text", "title": "Post-Hoc Test Skipped", "content": "The 'scikit-posthocs' library is not installed. Please install it to run post-hoc tests."})
                else:
                    try:
                        # Pivot data to wide format for paired t-tests
                        wide_df = df.pivot(index=clean_subject, columns=clean_within, values=clean_dv)
                        posthoc_df = sp.posthoc_ttest(wide_df, paired=True, p_adjust='bonferroni')
                        
                        posthoc_table_df = posthoc_df.round(4)
                        posthoc_table_df.reset_index(inplace=True)
                        posthoc_table_df.rename(columns={'index': ''}, inplace=True)
                        
                        sections.append({
                            'type': 'table', 'title': "Post-Hoc Paired T-Tests (Bonferroni Correction)",
                            'headers': posthoc_table_df.columns.tolist(),
                            'data': posthoc_table_df.values.tolist(),
                            "footer": "Shows p-values for pairwise condition comparisons."
                        })

                        # Generate heatmap visualization
                        fig_heatmap, ax_heatmap = plt.subplots(figsize=(max(8, len(posthoc_df.columns)), max(6, len(posthoc_df.index))))
                        sns.heatmap(posthoc_df, annot=True, cmap='coolwarm_r', fmt=".4f", ax=ax_heatmap, cbar_kws={'label': 'P-value'})
                        ax_heatmap.set_title("Post-Hoc P-Value Heatmap (Paired T-Tests)")
                        plt.tight_layout()
                        artifacts.append({
                            "type": "plot", "id": "rm_anova_posthoc_heatmap", "title": "Post-Hoc Heatmap",
                            "content": PlotUtils.fig_to_base64(fig_heatmap)
                        })
                    except Exception as posthoc_ex:
                        sections.append({"type": "text", "title": "Post-Hoc Test Failed", "content": str(posthoc_ex)})

            # Generate Visualizations
            with PlotUtils.setup_plotting():
                # Profile Plot (Interaction Plot)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.pointplot(x=clean_within, y=clean_dv, data=df, ax=ax, markers='o', linestyles='-', capsize=.1, errorbar='ci')
                ax.set_title(f'Profile Plot of {dv} across {within}')
                ax.set_xlabel(within)
                ax.set_ylabel(dv)
                plt.xticks(rotation=45, ha='right')
                artifacts.append({
                    "type": "plot", "id": "rm_anova_profile_plot", "title": "Profile Plot",
                    "content": PlotUtils.fig_to_base64(fig)
                })

                # Violin Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.violinplot(x=clean_within, y=clean_dv, data=df, ax=ax, inner="quartile")
                ax.set_title(f'Distribution of {dv} across {within}')
                ax.set_xlabel(within)
                ax.set_ylabel(dv)
                plt.xticks(rotation=45, ha='right')
                artifacts.append({
                    "type": "plot", "id": "rm_anova_violin_plot", "title": "Violin Plot",
                    "content": PlotUtils.fig_to_base64(fig)
                })

            # Construct Result Envelope
            return {
                "status": "ok",
                "summary": f"Repeated Measures ANOVA for '{dv}' by '{within}' completed. The result is {'significant' if is_significant else 'not significant'} (p={p_value:.4f}).",
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": kwargs}
            }
        except Exception as e:
            self.log_error(e)
            return {
                "status": "error",
                "summary": f"An error occurred during Repeated Measures ANOVA: {str(e)}",
            }

    def interpret(self, results: Dict[str, Any]) -> str:
        """
        Provides a formal interpretation of the Repeated Measures ANOVA results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            dependent_var = params.get('dependent_variable', 'the dependent variable')
            within_factor = params.get('within_factor', 'the within-subject factor')

            p_value = None
            anova_section = next((s for s in results.get('sections', []) if s.get('title') == 'Repeated Measures ANOVA Results'), None)
            
            if anova_section:
                # The p-value is in the first row, last column for AnovaRM
                p_value = float(anova_section['data'][0][-1])

            if p_value is None:
                return "Could not automatically determine the p-value for interpretation."

            if p_value < alpha:
                main_conclusion = f"Since the p-value ({p_value:.4f}) is less than α ({alpha}), we reject the null hypothesis. This indicates a statistically significant difference in the mean of '{dependent_var}' across the levels of '{within_factor}'."
                
                post_hoc_ran = any("Post-Hoc" in s.get('title', '') for s in results.get('sections', []))
                post_hoc_advice = ""
                if post_hoc_ran:
                    posthoc_section = next((s for s in results.get('sections', []) if "Post-Hoc" in s.get('title', '')), None)
                    posthoc_df = pd.DataFrame(posthoc_section['data'], columns=posthoc_section['headers'])
                    
                    significant_pairs = []
                    # Melt the DataFrame to easily iterate through pairs
                    melted_df = posthoc_df.melt(id_vars=[''], var_name='group2', value_name='p-adj')
                    melted_df.rename(columns={'': 'group1'}, inplace=True)
                    
                    for _, row in melted_df.iterrows():
                        if row['p-adj'] < alpha:
                            significant_pairs.append(f"'{row['group1']}' and '{row['group2']}'")

                    if significant_pairs:
                        post_hoc_advice = f" The post-hoc test shows significant differences between the following pairings: {', '.join(significant_pairs)}."
                
                return main_conclusion + post_hoc_advice
            else:
                return f"Since the p-value ({p_value:.4f}) is greater than or equal to α ({alpha}), we fail to reject the null hypothesis. There is not enough evidence to conclude a significant difference in the mean of '{dependent_var}' across the levels of '{within_factor}'."

        except (ValueError, TypeError, IndexError, KeyError):
            return "Could not automatically interpret the Repeated Measures ANOVA results."
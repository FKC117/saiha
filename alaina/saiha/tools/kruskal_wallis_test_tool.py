
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
try:
    import scikit_posthocs as sp
    SCIKIT_POSTHOCS_AVAILABLE = True
except ImportError:
    SCIKIT_POSTHOCS_AVAILABLE = False
from django.core.files.storage import default_storage
from typing import Any, Dict, List

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class KruskalWallisTestTool(BaseAnalysisTool):
    """
    A tool to perform the Kruskal-Wallis H test for independent samples.
    """

    @property
    def name(self) -> str:
        return "kruskal_wallis_test"

    @property
    def description(self) -> str:
        return "Compares distributions for two or more independent groups. Non-parametric equivalent of One-Way ANOVA."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="variable", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Numeric Variable", description="Select the numeric variable whose distributions you want to compare.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="group_column", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
            label="Grouping Variable", description="Select the categorical variable with two or more groups.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="alpha", parameter_type=ParameterType.SELECT,
            label="Significance Level (α)", description="The threshold for statistical significance.",
            required=True, default_value="0.05",
            options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
        ))
        params.add_parameter(ToolParameter(name="generate_boxplots", parameter_type=ParameterType.CHECKBOX, label="Generate Side-by-side Box Plots", required=False, default_value=True))
        params.add_parameter(ToolParameter(name="generate_violinplots", parameter_type=ParameterType.CHECKBOX, label="Generate Violin Plots", required=False, default_value=True))
        params.add_parameter(ToolParameter(
            name="perform_posthoc", parameter_type=ParameterType.CHECKBOX,
            label="Perform Post-Hoc Test (Dunn's)",
            description="If the main test is significant, perform pairwise comparisons to see which groups differ.",
            required=False, default_value=False
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            var = parameters.get("variable")
            group_col = parameters.get("group_column")
            alpha = float(parameters.get("alpha", 0.05))
            gen_box = str(parameters.get("generate_boxplots", "true")).lower() in ('true', 'on', '1')
            gen_violin = str(parameters.get("generate_violinplots", "true")).lower() in ('true', 'on', '1')
            perform_posthoc = str(parameters.get("perform_posthoc", "false")).lower() in ('true', 'on', '1')

            if not var or not group_col:
                return {"status": "error", "summary": "Both a numeric variable and a grouping variable are required."}

            # Use efficient column projection loading from BaseAnalysisTool
            df = self.load_dataset(columns=[var, group_col])

            df_clean = df[[var, group_col]].dropna()
            groups = df_clean[group_col].unique()
            if len(groups) < 2:
                return {"status": "error", "summary": f"The grouping variable '{group_col}' must have at least two unique groups, but it has {len(groups)}."}

            samples = [df_clean[df_clean[group_col] == g][var] for g in groups]

            h_stat, p_val = stats.kruskal(*samples)
            is_significant = p_val < alpha

            summary = (
                f"The Kruskal-Wallis test indicates that there is a {'statistically significant' if is_significant else 'not statistically significant'} "
                f"difference in the distributions of '{var}' across the groups in '{group_col}' (p={p_val:.4f})."
            )

            artifacts: List[Dict[str, Any]] = []
            sections = [{
                'type': 'table', 'title': 'Kruskal-Wallis Test Results',
                'headers': ['Statistic', 'Value'],
                'data': [
                    ['H-Statistic', f"{h_stat:.4f}"],
                    ['P-Value', f"{p_val:.6f}"],
                    ['Number of Groups', len(groups)],
                    ['Is Significant at ' + str(alpha), 'Yes' if is_significant else 'No']
                ]
            }]

            if is_significant and perform_posthoc:
                if not SCIKIT_POSTHOCS_AVAILABLE:
                    sections.append({"type": "text", "title": "Post-Hoc Test Skipped", "content": "The 'scikit-posthocs' library is not installed. Please install it to run post-hoc tests."})
                else:
                    try:
                        posthoc_df = sp.posthoc_dunn(df_clean, val_col=var, group_col=group_col, p_adjust='bonferroni')
                        
                        # Create a rounded version for the table
                        posthoc_table_df = posthoc_df.round(4)
                        posthoc_table_df.reset_index(inplace=True)
                        posthoc_table_df.rename(columns={'index': ''}, inplace=True)
                        
                        sections.append({
                            'type': 'table', 'title': "Dunn's Post-Hoc Test (Bonferroni Correction)",
                            'headers': posthoc_table_df.columns.tolist(),
                            'data': posthoc_table_df.values.tolist(),
                            "footer": "Shows p-values for pairwise group comparisons."
                        })

                        # Generate heatmap visualization
                        fig_heatmap, ax_heatmap = plt.subplots(figsize=(max(8, len(posthoc_df.columns)), max(6, len(posthoc_df.index))))
                        sns.heatmap(posthoc_df, annot=True, cmap='coolwarm_r', fmt=".4f", ax=ax_heatmap, cbar_kws={'label': 'P-value'})
                        ax_heatmap.set_title("Post-Hoc P-Value Heatmap (Dunn's Test)")
                        plt.tight_layout()
                        artifacts.append({
                            "type": "plot",
                            "id": "kw_posthoc_heatmap",
                            "title": "Post-Hoc Heatmap",
                            "content": PlotUtils.fig_to_base64(fig_heatmap)
                        })
                    except Exception as posthoc_ex:
                        sections.append({"type": "text", "title": "Post-Hoc Test Failed", "content": str(posthoc_ex)})


            # --- Optional Visualizations ---
            with PlotUtils.setup_plotting():
                if gen_box:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.boxplot(x=group_col, y=var, data=df_clean, ax=ax)
                        ax.set_title(f'Box Plots of {var} by {group_col}')
                        plt.xticks(rotation=45, ha='right')
                        artifacts.append({"type": "plot", "id": "kw_boxplot", "title": "Side-by-side Box Plots", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_violin:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(10, 7))
                        sns.violinplot(x=group_col, y=var, data=df_clean, ax=ax, inner='quartile', cut=0)
                        ax.set_title(f'Violin Plots of {var} by {group_col}')
                        plt.xticks(rotation=45, ha='right')
                        artifacts.append({"type": "plot", "id": "kw_violinplot", "title": "Violin Plots", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

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

    def interpret(self, results: Dict[str, Any]) -> str:
        """
        Provides a formal interpretation of the Kruskal-Wallis test results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            var = params.get('variable', 'the variable')
            group_col = params.get('group_column', 'the grouping variable')

            p_value = None
            kruskal_section = next((s for s in results.get('sections', []) if s.get('title') == 'Kruskal-Wallis Test Results'), None)
            
            if kruskal_section:
                for row in kruskal_section.get('data', []):
                    if row[0] == 'P-Value':
                        p_value = float(row[1])
                        break

            if p_value is None:
                return "Could not automatically determine the p-value for interpretation."

            if p_value < alpha:
                post_hoc_ran = any("Post-Hoc Test" in s.get('title', '') for s in results.get('sections', []))
                post_hoc_advice = " The post-hoc test results can identify which specific groups differ." if post_hoc_ran else ""

                return f"Since the p-value ({p_value:.4f}) is less than the significance level (α = {alpha}), we reject the null hypothesis. This indicates there is a statistically significant difference in the distribution of '{var}' across the different groups of '{group_col}'.{post_hoc_advice}"
            else:
                return f"Since the p-value ({p_value:.4f}) is greater than or equal to the significance level (α = {alpha}), we fail to reject the null hypothesis. There is not enough evidence to conclude that there are any significant differences in the distribution of '{var}' across the groups of '{group_col}'."

        except (ValueError, TypeError, IndexError):
            return "Could not automatically interpret the Kruskal-Wallis test results."
# d:/quantly/quanta/quantalytics/ai_agents/tools/friedman_test_tool.py

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


class FriedmanTestTool(BaseAnalysisTool):
    """
    A tool to perform the Friedman test for repeated measures.
    """

    @property
    def name(self) -> str:
        return "friedman_test"

    @property
    def description(self) -> str:
        return "Compares distributions for three or more related groups. Non-parametric equivalent of Repeated Measures ANOVA."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="subject_id", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
            label="Subject ID Column", description="Column that uniquely identifies each subject.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="within_factor", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
            label="Within-Subject Factor (Time/Condition)", description="Categorical variable representing the repeated conditions.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="dependent_variable", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Dependent Variable (Numeric)", description="The numeric outcome variable measured at each condition.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="alpha", parameter_type=ParameterType.SELECT,
            label="Significance Level (α)", description="The threshold for statistical significance.",
            required=True, default_value="0.05",
            options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
        ))
        params.add_parameter(ToolParameter(name="generate_boxplots", parameter_type=ParameterType.CHECKBOX, label="Generate Box Plots by Condition", required=False, default_value=True))
        params.add_parameter(ToolParameter(name="generate_profile_plot", parameter_type=ParameterType.CHECKBOX, label="Generate Profile Plot", required=False, default_value=True))
        params.add_parameter(ToolParameter(
            name="perform_posthoc", parameter_type=ParameterType.CHECKBOX,
            label="Perform Post-Hoc Test (Conover's)",
            description="If the main test is significant, perform pairwise comparisons to see which conditions differ.",
            required=False, default_value=False
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            subject_col = parameters.get("subject_id")
            within_col = parameters.get("within_factor")
            dv_col = parameters.get("dependent_variable")
            alpha = float(parameters.get("alpha", 0.05))
            gen_box = str(parameters.get("generate_boxplots", "true")).lower() in ('true', 'on', '1')
            gen_profile = str(parameters.get("generate_profile_plot", "true")).lower() in ('true', 'on', '1')
            perform_posthoc = str(parameters.get("perform_posthoc", "false")).lower() in ('true', 'on', '1')

            if not all([subject_col, within_col, dv_col]):
                return {"status": "error", "summary": "Subject ID, Within-Subject Factor, and Dependent Variable are all required."}

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            df_clean = df[[subject_col, within_col, dv_col]].dropna()
            
            # Pivot data from long to wide format for the test
            wide_df = df_clean.pivot(index=subject_col, columns=within_col, values=dv_col).dropna()
            
            if len(wide_df) < 2:
                return {"status": "error", "summary": "Not enough complete subject data across all conditions to perform the test."}
            if len(wide_df.columns) < 3:
                return {"status": "error", "summary": f"The Friedman test requires at least 3 conditions, but '{within_col}' has only {len(wide_df.columns)}."}

            samples = [wide_df[col] for col in wide_df.columns]
            chi2_stat, p_val = stats.friedmanchisquare(*samples)
            is_significant = p_val < alpha

            summary = (
                f"The Friedman test indicates that there is a {'statistically significant' if is_significant else 'not statistically significant'} "
                f"difference in the distributions of '{dv_col}' across the conditions in '{within_col}' (p={p_val:.4f})."
            )

            artifacts: List[Dict[str, Any]] = []
            sections = [{
                'type': 'table', 'title': 'Friedman Test Results',
                'headers': ['Statistic', 'Value'],
                'data': [
                    ['Chi-squared Statistic', f"{chi2_stat:.4f}"],
                    ['P-Value', f"{p_val:.6f}"],
                    ['Number of Conditions', len(wide_df.columns)],
                    ['Is Significant at ' + str(alpha), 'Yes' if is_significant else 'No']
                ]
            }]

            if is_significant and perform_posthoc:
                if not SCIKIT_POSTHOCS_AVAILABLE:
                    sections.append({"type": "text", "title": "Post-Hoc Test Skipped", "content": "The 'scikit-posthocs' library is not installed. Please install it to run post-hoc tests."})
                else:
                    try:
                        # scikit-posthocs needs long format data
                        posthoc_df = sp.posthoc_conover_friedman(df_clean, dv=dv_col, within=within_col, subject=subject_col, p_adjust='bonferroni')
                        
                        # Create a rounded version for the table
                        posthoc_table_df = posthoc_df.round(4)
                        posthoc_table_df.reset_index(inplace=True)
                        posthoc_table_df.rename(columns={'index': ''}, inplace=True)
                        
                        sections.append({
                            'type': 'table', 'title': "Conover's Post-Hoc Test (Bonferroni Correction)",
                            'headers': posthoc_table_df.columns.tolist(),
                            'data': posthoc_table_df.values.tolist(),
                            "footer": "Shows p-values for pairwise condition comparisons."
                        })

                        # Generate heatmap visualization
                        fig_heatmap, ax_heatmap = plt.subplots(figsize=(max(8, len(posthoc_df.columns)), max(6, len(posthoc_df.index))))
                        sns.heatmap(posthoc_df, annot=True, cmap='coolwarm_r', fmt=".4f", ax=ax_heatmap, cbar_kws={'label': 'P-value'})
                        ax_heatmap.set_title("Post-Hoc P-Value Heatmap (Conover's Test)")
                        plt.tight_layout()
                        artifacts.append({
                            "type": "plot",
                            "id": "friedman_posthoc_heatmap",
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
                        sns.boxplot(x=within_col, y=dv_col, data=df_clean, ax=ax)
                        ax.set_title(f'Box Plots of {dv_col} by {within_col}')
                        plt.xticks(rotation=45, ha='right')
                        artifacts.append({"type": "plot", "id": "friedman_boxplot", "title": "Box Plots by Condition", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_profile:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.pointplot(x=within_col, y=dv_col, data=df_clean, ax=ax, markers='o', linestyles='-', capsize=.1, errorbar='ci')
                        ax.set_title(f'Profile Plot of {dv_col} across {within_col}')
                        plt.xticks(rotation=45, ha='right')
                        artifacts.append({"type": "plot", "id": "friedman_profile_plot", "title": "Profile Plot", "content": PlotUtils.fig_to_base64(fig)})
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
        Provides a formal interpretation of the Friedman test results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            dv_col = params.get('dependent_variable', 'the dependent variable')
            within_col = params.get('within_factor', 'the within-subject factor')

            p_value = None
            friedman_section = next((s for s in results.get('sections', []) if s.get('title') == 'Friedman Test Results'), None)
            
            if friedman_section:
                for row in friedman_section.get('data', []):
                    if row[0] == 'P-Value':
                        p_value = float(row[1])
                        break

            if p_value is None:
                return "Could not automatically determine the p-value for interpretation."

            if p_value < alpha:
                post_hoc_ran = any("Post-Hoc Test" in s.get('title', '') for s in results.get('sections', []))
                post_hoc_advice = " The post-hoc test results can identify which specific conditions differ." if post_hoc_ran else ""

                return f"Since the p-value ({p_value:.4f}) is less than α ({alpha}), we reject the null hypothesis. This indicates there is a statistically significant difference in the distributions of '{dv_col}' across the conditions in '{within_col}'.{post_hoc_advice}"
            else:
                return f"Since the p-value ({p_value:.4f}) is greater than or equal to α ({alpha}), we fail to reject the null hypothesis. There is not enough evidence to conclude that there are any significant differences in the distribution of '{dv_col}' across the conditions in '{within_col}'."

        except (ValueError, TypeError, IndexError):
            return "Could not automatically interpret the Friedman test results."
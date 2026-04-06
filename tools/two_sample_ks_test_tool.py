
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from django.core.files.storage import default_storage
from typing import Any, Dict, List

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class TwoSampleKSTestTool(BaseAnalysisTool):
    """
    A tool to perform the two-sample Kolmogorov-Smirnov test.
    """

    @property
    def name(self) -> str:
        return "two_sample_ks_test"

    @property
    def description(self) -> str:
        return "Tests if two independent samples are drawn from the same distribution."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="variable", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Numeric Variable", description="Select the numeric variable whose distributions you want to compare.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="group_column", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
            label="Grouping Variable", description="Select the categorical variable with exactly two groups.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="alternative", parameter_type=ParameterType.SELECT,
            label="Alternative Hypothesis", description="Defines the alternative hypothesis for the test.",
            required=True, default_value="two-sided",
            options=[
                {"value": "two-sided", "label": "Two-sided (Distributions are not identical)"},
                {"value": "less", "label": "Less (Dist. of Group 1 < Dist. of Group 2)"},
                {"value": "greater", "label": "Greater (Dist. of Group 1 > Dist. of Group 2)"},
            ]
        ))
        params.add_parameter(ToolParameter(
            name="alpha", parameter_type=ParameterType.SELECT,
            label="Significance Level (α)", description="The threshold for statistical significance.",
            required=True, default_value="0.05",
            options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
        ))
        params.add_parameter(ToolParameter(
            name="generate_boxplots", parameter_type=ParameterType.CHECKBOX,
            label="Generate Side-by-side Box Plots", required=False, default_value=True
        ))
        params.add_parameter(ToolParameter(
            name="generate_violinplots", parameter_type=ParameterType.CHECKBOX,
            label="Generate Violin Plots", required=False, default_value=True
        ))
        params.add_parameter(ToolParameter(
            name="generate_histograms", parameter_type=ParameterType.CHECKBOX,
            label="Generate Back-to-back Histograms", required=False, default_value=False
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            var = parameters.get("variable")
            group_col = parameters.get("group_column")
            alternative = parameters.get("alternative", "two-sided")
            alpha = float(parameters.get("alpha", 0.05))
            gen_box = str(parameters.get("generate_boxplots", "true")).lower() in ('true', 'on', '1')
            gen_violin = str(parameters.get("generate_violinplots", "true")).lower() in ('true', 'on', '1')
            gen_hist = str(parameters.get("generate_histograms", "false")).lower() in ('true', 'on', '1')

            if not var or not group_col:
                return {"status": "error", "summary": "Both a numeric variable and a grouping variable are required."}

            # Use efficient column projection loading from BaseAnalysisTool
            df = self.load_dataset(columns=group_column)

            df_clean = df[[var, group_col]].dropna()
            
            # Sort groups alphabetically for deterministic order
            groups = sorted(df_clean[group_col].unique())
            
            if len(groups) != 2:
                return {"status": "error", "summary": f"The grouping variable '{group_col}' must have exactly two unique groups, but it has {len(groups)}."}

            group1_data = df_clean[df_clean[group_col] == groups[0]][var].rename(f"Group 1: {groups[0]}")
            group2_data = df_clean[df_clean[group_col] == groups[1]][var].rename(f"Group 2: {groups[1]}")

            if group1_data.empty or group2_data.empty:
                return {"status": "error", "summary": "One or both groups have no data to perform the test."}

            ks_stat, p_val = stats.ks_2samp(group1_data, group2_data, alternative=alternative)
            is_significant = p_val < alpha

            if alternative == 'two-sided':
                summary = (
                    f"The two-sample KS test indicates that the distributions of '{var}' for group '{groups[0]}' and group '{groups[1]}' "
                    f"are {'statistically significantly' if is_significant else 'not statistically significantly'} different (p={p_val:.4f})."
                )
            else:
                direction = "less than" if alternative == "less" else "greater than"
                summary = (
                    f"The one-sided KS test indicates that the distribution of '{var}' for Group 1 ('{groups[0]}') is "
                    f"{'statistically significantly' if is_significant else 'not statistically significantly'} stochastically {direction} the distribution for Group 2 ('{groups[1]}') "
                    f"(p={p_val:.4f})."
                )

            artifacts: List[Dict[str, Any]] = []
            sections = [{
                'type': 'table', 'title': 'Two-Sample Kolmogorov-Smirnov Test Results',
                'headers': ['Statistic', 'Value'],
                'data': [
                    ['KS Statistic', f"{ks_stat:.4f}"],
                    ['P-Value', f"{p_val:.6f}"],
                    ['Is Significant at ' + str(alpha), 'Yes' if is_significant else 'No']
                ]
            }, {
                'type': 'table', 'title': 'Descriptive Statistics by Group',
                'headers': ['Group', 'Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
                'data': [
                    [f"Group 1: {groups[0]}"] + group1_data.describe().round(4).tolist(),
                    [f"Group 2: {groups[1]}"] + group2_data.describe().round(4).tolist()
                ]
            }]

            # --- Visualization ---
            with PlotUtils.setup_plotting():
                fig = None
                try:
                    # Always generate the primary ECDF plot
                    fig, ax = plt.subplots(figsize=(10, 7))
                    sns.ecdfplot(data=df_clean, x=var, hue=group_col, ax=ax)
                    ax.set_title(f'ECDF of {var} by {group_col}')
                    ax.set_xlabel(var)
                    ax.set_ylabel('Cumulative Probability')
                    artifacts.append({"type": "plot", "id": "two_sample_ks_ecdf", "title": "ECDF Comparison Plot", "content": PlotUtils.fig_to_base64(fig)})
                finally:
                    if fig: plt.close(fig)
                
                if gen_box:
                    fig_box = None
                    try:
                        fig_box, ax_box = plt.subplots(figsize=(8, 6))
                        sns.boxplot(x=group_col, y=var, data=df_clean, ax=ax_box)
                        ax_box.set_title(f'Box Plots of {var} by {group_col}')
                        artifacts.append({"type": "plot", "id": "two_sample_ks_boxplot", "title": "Side-by-side Box Plots", "content": PlotUtils.fig_to_base64(fig_box)})
                    finally:
                        if fig_box: plt.close(fig_box)

                if gen_violin:
                    fig_violin = None
                    try:
                        fig_violin, ax_violin = plt.subplots(figsize=(10, 7))
                        sns.violinplot(x=group_col, y=var, data=df_clean, ax=ax_violin, inner='quartile', cut=0)
                        ax_violin.set_title(f'Violin Plots of {var} by {group_col}')
                        artifacts.append({"type": "plot", "id": "two_sample_ks_violinplot", "title": "Violin Plots", "content": PlotUtils.fig_to_base64(fig_violin)})
                    finally:
                        if fig_violin: plt.close(fig_violin)

                if gen_hist:
                    fig_hist = None
                    try:
                        fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
                        sns.histplot(group1_data, ax=ax1, kde=True)
                        ax1.set_title(f"Distribution for Group 1: '{groups[0]}'")
                        sns.histplot(group2_data, ax=ax2, kde=True)
                        ax2.set_title(f"Distribution for Group 2: '{groups[1]}'")
                        fig_hist.suptitle(f'Back-to-back Histograms for {var}')
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        artifacts.append({"type": "plot", "id": "two_sample_ks_histograms", "title": "Back-to-back Histograms", "content": PlotUtils.fig_to_base64(fig_hist)})
                    finally:
                        if fig_hist: plt.close(fig_hist)

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
        Provides a formal interpretation of the Two-Sample KS Test results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            alternative = params.get('alternative', 'two-sided')
            var = params.get('variable', 'the variable')
            group_col = params.get('group_column', 'the grouping variable')

            p_value = None
            for section in results.get('sections', []):
                if section.get('title') == 'Two-Sample Kolmogorov-Smirnov Test Results':
                    for row in section.get('data', []):
                        if row[0] == 'P-Value':
                            p_value = float(row[1])
                            break
            
            if p_value is None:
                return "Could not automatically determine the p-value for interpretation."

            conclusion = "we reject the null hypothesis" if p_value < alpha else "we fail to reject the null hypothesis"
            evidence = "There is significant evidence to suggest that" if p_value < alpha else "There is not enough evidence to suggest that"
            
            alt_text = {"two-sided": "are different", "greater": "is stochastically greater than the other", "less": "is stochastically less than the other"}.get(alternative, "are different")

            return f"The Two-Sample KS test was performed to compare the distributions of '{var}' between the groups in '{group_col}'. Since the p-value ({p_value:.4f}) is {'less than' if p_value < alpha else 'not less than'} α ({alpha}), {conclusion}. {evidence} the two sample distributions {alt_text}."

        except (ValueError, TypeError, IndexError):
            return "Could not automatically interpret the Two-Sample KS test results."
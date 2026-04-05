
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


class WilcoxonSignedRankTestTool(BaseAnalysisTool):
    """
    A tool to perform the Wilcoxon signed-rank test for paired samples.
    """

    @property
    def name(self) -> str:
        return "wilcoxon_signed_rank_test"

    @property
    def description(self) -> str:
        return "Compares two related paired samples. Non-parametric equivalent of the Paired T-Test."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="variable1", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Variable 1 (e.g., Before)", description="Select the first numeric variable for the paired test.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="variable2", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Variable 2 (e.g., After)", description="Select the second numeric variable for the paired test.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="alternative", parameter_type=ParameterType.SELECT,
            label="Alternative Hypothesis", description="Specifies the alternative hypothesis.",
            required=True, default_value="two-sided",
            options=[
                {"value": "two-sided", "label": "Two-sided (Distributions are not equal)"},
                {"value": "less", "label": "Less (Distribution 1 < Distribution 2)"},
                {"value": "greater", "label": "Greater (Distribution 1 > Distribution 2)"},
            ]
        ))
        params.add_parameter(ToolParameter(
            name="alpha", parameter_type=ParameterType.SELECT,
            label="Significance Level (α)", description="The threshold for statistical significance.",
            required=True, default_value="0.05",
            options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
        ))
        params.add_parameter(ToolParameter(name="generate_slope_plot", parameter_type=ParameterType.CHECKBOX, label="Generate Paired Lines Plot (Slope Graph)", required=False, default_value=True))
        params.add_parameter(ToolParameter(name="generate_diff_plot", parameter_type=ParameterType.CHECKBOX, label="Generate Difference Plot", required=False, default_value=True))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            var1 = parameters.get("variable1")
            var2 = parameters.get("variable2")
            alpha = float(parameters.get("alpha", 0.05))
            alternative = parameters.get("alternative", "two-sided")
            gen_slope = str(parameters.get("generate_slope_plot", "true")).lower() in ('true', 'on', '1')
            gen_diff = str(parameters.get("generate_diff_plot", "true")).lower() in ('true', 'on', '1')

            if not var1 or not var2:
                return {"status": "error", "summary": "Both variables are required for a paired test."}
            if var1 == var2:
                return {"status": "error", "summary": "Please select two different variables for the paired test."}

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            paired_df = df[[var1, var2]].dropna()
            if len(paired_df) < 2:
                return {"status": "error", "summary": "Not enough complete pairs of data to perform the test."}

            sample1 = paired_df[var1]
            sample2 = paired_df[var2]

            w_stat, p_val = stats.wilcoxon(sample1, sample2, alternative=alternative)
            is_significant = p_val < alpha

            summary = (
                f"The Wilcoxon signed-rank test indicates that the distributions of '{var1}' and '{var2}' "
                f"are {'statistically significantly' if is_significant else 'not statistically significantly'} different (p={p_val:.4f})."
            )

            artifacts: List[Dict[str, Any]] = []
            sections = [{
                'type': 'table', 'title': 'Wilcoxon Signed-Rank Test Results',
                'headers': ['Statistic', 'Value'],
                'data': [
                    ['W-Statistic', f"{w_stat:.4f}"],
                    ['P-Value', f"{p_val:.6f}"],
                    ['Number of Pairs', len(paired_df)],
                    ['Is Significant at ' + str(alpha), 'Yes' if is_significant else 'No']
                ]
            }]

            # --- Optional Visualizations ---
            with PlotUtils.setup_plotting():
                if gen_slope:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plot_df = paired_df.head(50) # Limit to 50 for readability
                        ax.plot([plot_df[var1], plot_df[var2]], color='gray', alpha=0.5, marker='o')
                        ax.set_xticks([0, 1])
                        ax.set_xticklabels([var1, var2])
                        ax.set_title(f'Paired Lines Plot (Slope Graph) for first {len(plot_df)} pairs')
                        ax.set_ylabel('Value')
                        artifacts.append({"type": "plot", "id": "wilcoxon_slope_plot", "title": "Paired Lines Plot", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_diff:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        differences = sample2 - sample1
                        sns.histplot(differences, kde=True, ax=ax)
                        ax.axvline(differences.median(), color='r', linestyle='--', label=f'Median Difference ({differences.median():.2f})')
                        ax.set_title('Distribution of Differences (Variable 2 - Variable 1)')
                        ax.legend()
                        artifacts.append({"type": "plot", "id": "wilcoxon_diff_plot", "title": "Difference Plot", "content": PlotUtils.fig_to_base64(fig)})
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
        Provides a formal interpretation of the Wilcoxon Signed-Rank Test results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            alternative = params.get('alternative', 'two-sided')
            var1 = params.get('variable1', 'Variable 1')
            var2 = params.get('variable2', 'Variable 2')

            p_value = None
            for section in results.get('sections', []):
                if section.get('title') == 'Wilcoxon Signed-Rank Test Results':
                    for row in section.get('data', []):
                        if row[0] == 'P-Value': p_value = float(row[1])

            if p_value is None:
                return "Could not automatically determine the p-value for interpretation."

            conclusion = "we reject the null hypothesis" if p_value < alpha else "we fail to reject the null hypothesis"
            evidence = "There is significant evidence to suggest that" if p_value < alpha else "There is not enough evidence to suggest that"
            alt_text = {"two-sided": "are different", "greater": f"for '{var1}' is stochastically greater than for '{var2}'", "less": f"for '{var1}' is stochastically less than for '{var2}'"}.get(alternative, "are different")

            return f"Since the p-value ({p_value:.4f}) is less than α ({alpha}), {conclusion}. {evidence} the distributions for the paired samples '{var1}' and '{var2}' {alt_text}."

        except (ValueError, TypeError, IndexError):
            return "Could not automatically interpret the Wilcoxon signed-rank test results."
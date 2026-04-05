# d:/quantly/quanta/quantalytics/ai_agents/tools/paired_t_test_tool.py

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


class PairedTTestTool(BaseAnalysisTool):
    """
    A tool to perform a paired samples T-test.
    """

    @property
    def name(self) -> str:
        return "paired_t_test"

    @property
    def description(self) -> str:
        return "Compares the means of two related measurements from the same subjects (e.g., before and after)."

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
                {"value": "two-sided", "label": "Two-sided (Means are not equal)"},
                {"value": "greater", "label": "Greater Than (Mean 1 > Mean 2)"},
                {"value": "less", "label": "Less Than (Mean 1 < Mean 2)"},
            ]
        ))
        params.add_parameter(ToolParameter(
            name="alpha", parameter_type=ParameterType.SELECT,
            label="Significance Level (α)", description="The threshold for statistical significance.",
            required=True, default_value="0.05",
            options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
        ))
        # Plotting options
        params.add_parameter(ToolParameter(name="generate_slope_plot", parameter_type=ParameterType.CHECKBOX, label="Generate Paired Lines Plot (Slope Graph)", required=False, default_value=True))
        params.add_parameter(ToolParameter(name="generate_diff_plot", parameter_type=ParameterType.CHECKBOX, label="Generate Difference Plot", required=False, default_value=True))
        params.add_parameter(ToolParameter(name="generate_bland_altman", parameter_type=ParameterType.CHECKBOX, label="Generate Bland-Altman Plot", required=False, default_value=False))
        params.add_parameter(ToolParameter(name="generate_bar_chart", parameter_type=ParameterType.CHECKBOX, label="Generate Paired Bar Chart", required=False, default_value=False))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            var1 = parameters.get("variable1")
            var2 = parameters.get("variable2")
            alpha = float(parameters.get("alpha", 0.05))
            alternative = parameters.get("alternative", "two-sided")

            # Plotting flags
            gen_slope = str(parameters.get("generate_slope_plot", "true")).lower() in ('true', 'on', '1')
            gen_diff = str(parameters.get("generate_diff_plot", "true")).lower() in ('true', 'on', '1')
            gen_bland = str(parameters.get("generate_bland_altman", "false")).lower() in ('true', 'on', '1')
            gen_bar = str(parameters.get("generate_bar_chart", "false")).lower() in ('true', 'on', '1')

            if not var1 or not var2:
                return {"status": "error", "summary": "Both variables are required for a paired T-test."}
            if var1 == var2:
                return {"status": "error", "summary": "Please select two different variables for the paired test."}

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            paired_df = df[[var1, var2]].dropna()
            if len(paired_df) < 2:
                return {"status": "error", "summary": "Not enough complete pairs of data to perform the test."}

            sample1 = paired_df[var1]
            sample2 = paired_df[var2]

            t_stat, p_val = stats.ttest_rel(sample1, sample2, alternative=alternative)
            dof = len(paired_df) - 1
            is_significant = p_val < alpha

            summary = (
                f"The mean difference between '{var1}' ({sample1.mean():.2f}) and '{var2}' ({sample2.mean():.2f}) "
                f"is {'statistically significant' if is_significant else 'not statistically significant'} (p={p_val:.4f})."
            )

            artifacts: List[Dict[str, Any]] = []
            sections = [{
                'type': 'table', 'title': 'Paired T-Test Results',
                'headers': ['Statistic', 'Value'],
                'data': [
                    ['T-Statistic', f"{t_stat:.4f}"],
                    ['P-Value', f"{p_val:.6f}"],
                    ['Degrees of Freedom', dof],
                    ['Mean Difference', f"{(sample2.mean() - sample1.mean()):.4f}"],
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
                        artifacts.append({"type": "plot", "id": "paired_slope_plot", "title": "Paired Lines Plot", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_diff:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        differences = sample2 - sample1
                        sns.histplot(differences, kde=True, ax=ax)
                        ax.axvline(differences.mean(), color='r', linestyle='--', label=f'Mean Difference ({differences.mean():.2f})')
                        ax.set_title('Distribution of Differences (Variable 2 - Variable 1)')
                        ax.legend()
                        artifacts.append({"type": "plot", "id": "paired_diff_plot", "title": "Difference Plot", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_bland:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(10, 7))
                        means = (sample1 + sample2) / 2
                        diffs = sample2 - sample1
                        mean_diff = diffs.mean()
                        std_diff = diffs.std()
                        ax.scatter(means, diffs, alpha=0.5)
                        ax.axhline(mean_diff, color='r', linestyle='-', label='Mean Difference')
                        ax.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--', label='±1.96 SD')
                        ax.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')
                        ax.set_title('Bland-Altman Plot')
                        ax.set_xlabel('Average of Measurements')
                        ax.set_ylabel('Difference between Measurements')
                        ax.legend()
                        artifacts.append({"type": "plot", "id": "paired_bland_altman", "title": "Bland-Altman Plot", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_bar:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        means = [sample1.mean(), sample2.mean()]
                        errors = [stats.sem(sample1), stats.sem(sample2)]
                        ax.bar([var1, var2], means, yerr=errors, capsize=5, color=['#007bff', '#28a745'])
                        ax.set_title('Mean of Paired Variables with Standard Error')
                        ax.set_ylabel('Mean Value')
                        artifacts.append({"type": "plot", "id": "paired_bar_chart", "title": "Paired Bar Chart", "content": PlotUtils.fig_to_base64(fig)})
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
        Provides a formal interpretation of the Paired T-Test results.
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
            mean_diff = None

            for section in results.get('sections', []):
                if section.get('title') == 'Paired T-Test Results':
                    for row in section.get('data', []):
                        if row[0] == 'P-Value': p_value = float(row[1])
                        elif row[0] == 'Mean Difference': mean_diff = float(row[1])

            if p_value is None or mean_diff is None:
                return "Could not automatically determine test results for interpretation."

            if p_value < alpha:
                conclusion = "we reject the null hypothesis"
                evidence = "There is significant evidence to suggest that"
            else:
                conclusion = "we fail to reject the null hypothesis"
                evidence = "There is not enough evidence to suggest that"

            if alternative == 'two-sided':
                return f"Since the p-value ({p_value:.4f}) is less than α ({alpha}), {conclusion}. {evidence} there is a difference in the means between '{var1}' and '{var2}'."
            elif alternative == 'greater': # This tests if mean(var1) > mean(var2)
                return f"Since the p-value ({p_value:.4f}) is less than α ({alpha}), {conclusion}. {evidence} the mean of '{var1}' is greater than the mean of '{var2}'."
            else:  # 'less', tests if mean(var1) < mean(var2)
                return f"Since the p-value ({p_value:.4f}) is less than α ({alpha}), {conclusion}. {evidence} the mean of '{var1}' is less than the mean of '{var2}'."

        except (ValueError, TypeError, IndexError):
            return "Could not automatically interpret the paired t-test results."
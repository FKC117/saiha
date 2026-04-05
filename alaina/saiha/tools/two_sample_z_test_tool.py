
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from django.core.files.storage import default_storage
from typing import Any, Dict, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class TwoSampleZTestTool(BaseAnalysisTool):
    """
    A tool to perform an independent two-sample Z-test.
    """

    @property
    def name(self) -> str:
        return "two_sample_z_test"

    @property
    def description(self) -> str:
        return "Compares the means of two independent groups when population standard deviations are known."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="variable", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Numeric Variable", description="Select the numeric variable to compare.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="group_column", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
            label="Grouping Variable", description="Select the categorical variable with exactly two groups.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="stddev1", parameter_type=ParameterType.NUMBER,
            label="Population SD for Group 1 (σ₁)", description="Known population standard deviation for the first group. Must be > 0.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="stddev2", parameter_type=ParameterType.NUMBER,
            label="Population SD for Group 2 (σ₂)", description="Known population standard deviation for the second group. Must be > 0.", required=True
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
        params.add_parameter(ToolParameter(name="generate_boxplots", parameter_type=ParameterType.CHECKBOX, label="Generate Side-by-side Box Plots", required=False, default_value=True))
        params.add_parameter(ToolParameter(name="generate_dist_plots", parameter_type=ParameterType.CHECKBOX, label="Generate Distribution Plots", required=False, default_value=True))
        params.add_parameter(ToolParameter(
            name="generate_mean_ci_plot", parameter_type=ParameterType.CHECKBOX,
            label="Generate Group Means with Confidence Intervals", required=False, default_value=False
        ))
        params.add_parameter(ToolParameter(
            name="generate_overlapping_normal_plot", parameter_type=ParameterType.CHECKBOX,
            label="Generate Overlapping Normal Distributions", required=False, default_value=False
        ))
        params.add_parameter(ToolParameter(
            name="generate_diff_mean_ci_plot", parameter_type=ParameterType.CHECKBOX,
            label="Generate Difference in Means with CI Plot", required=False, default_value=False
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            def _safe_float(val, default=0.0):
                return float(val) if val else default

            var = parameters.get("variable")
            group_col = parameters.get("group_column")
            stddev1 = _safe_float(parameters.get("stddev1"))
            stddev2 = _safe_float(parameters.get("stddev2"))
            alpha = float(parameters.get("alpha", 0.05))
            alternative = parameters.get("alternative", "two-sided")
            gen_box = str(parameters.get("generate_boxplots", "true")).lower() in ('true', 'on', '1')
            gen_dist = str(parameters.get("generate_dist_plots", "true")).lower() in ('true', 'on', '1')
            gen_mean_ci = str(parameters.get("generate_mean_ci_plot", "false")).lower() in ('true', 'on', '1')
            gen_overlap = str(parameters.get("generate_overlapping_normal_plot", "false")).lower() in ('true', 'on', '1')
            gen_diff_ci = str(parameters.get("generate_diff_mean_ci_plot", "false")).lower() in ('true', 'on', '1')

            if not var or not group_col:
                return {"status": "error", "summary": "Both a numeric variable and a grouping variable are required."}
            if stddev1 <= 0 or stddev2 <= 0:
                return {"status": "error", "summary": "Population standard deviations must be greater than zero."}

            # Use efficient column projection loading from BaseAnalysisTool
            df = self.load_dataset(columns=[var, group_col])

            df_clean = df[[var, group_col]].dropna()
            groups = df_clean[group_col].unique()
            if len(groups) != 2:
                return {"status": "error", "summary": f"The grouping variable '{group_col}' must have exactly two unique groups, but it has {len(groups)}."}

            group1_data = df_clean[df_clean[group_col] == groups[0]][var]
            group2_data = df_clean[df_clean[group_col] == groups[1]][var]
            n1, n2 = len(group1_data), len(group2_data)

            if n1 < 2 or n2 < 2:
                return {"status": "error", "summary": "One or both groups have insufficient data to perform the test."}

            mean1, mean2 = group1_data.mean(), group2_data.mean()
            
            # Manual Z-test calculation
            z_stat = (mean1 - mean2) / np.sqrt((stddev1**2 / n1) + (stddev2**2 / n2))

            if alternative == 'two-sided':
                p_val = 2 * stats.norm.sf(np.abs(z_stat))
            elif alternative == 'greater':
                p_val = stats.norm.sf(z_stat)
            else:  # 'less'
                p_val = stats.norm.cdf(z_stat)

            is_significant = p_val < alpha

            summary = (
                f"The mean of '{var}' for group '{groups[0]}' ({mean1:.2f}) and group '{groups[1]}' ({mean2:.2f}) "
                f"are {'statistically significantly' if is_significant else 'not statistically significantly'} different (p={p_val:.4f})."
            )

            artifacts: List[Dict[str, Any]] = []
            sections = [{
                'type': 'table', 'title': 'Two-Sample Z-Test Results',
                'headers': ['Statistic', 'Value'],
                'data': [
                    ['Z-Statistic', f"{z_stat:.4f}"],
                    ['P-Value', f"{p_val:.6f}"],
                    [f"Mean of '{groups[0]}'", f"{mean1:.4f}"],
                    [f"Mean of '{groups[1]}'", f"{mean2:.4f}"],
                    ['Is Significant at ' + str(alpha), 'Yes' if is_significant else 'No']
                ]
            }]

            # --- Optional Visualizations ---
            with PlotUtils.setup_plotting():
                if gen_box:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.boxplot(x=group_col, y=var, data=df_clean, ax=ax)
                        ax.set_title(f'Box Plots of {var} by {group_col}')
                        artifacts.append({"type": "plot", "id": "two_sample_z_boxplot", "title": "Side-by-side Box Plots", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_dist:
                    fig = None
                    try:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
                        sns.histplot(group1_data, ax=ax1, kde=True)
                        ax1.set_title(f'Distribution for {groups[0]}')
                        sns.histplot(group2_data, ax=ax2, kde=True)
                        ax2.set_title(f'Distribution for {groups[1]}')
                        fig.suptitle(f'Distribution Plots for {var}')
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        artifacts.append({"type": "plot", "id": "two_sample_z_distplots", "title": "Distribution Plots", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_mean_ci:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        z_critical = stats.norm.ppf(1 - alpha / 2)
                        ci_error1 = z_critical * (stddev1 / np.sqrt(n1))
                        ci_error2 = z_critical * (stddev2 / np.sqrt(n2))
                        
                        means = [mean1, mean2]
                        errors = [ci_error1, ci_error2]
                        
                        ax.bar(groups, means, yerr=errors, capsize=5, color=['#007bff', '#28a745'])
                        ax.set_title(f'Mean of {var} by {group_col} with {1-alpha:.0%} CI')
                        ax.set_ylabel('Mean Value')
                        artifacts.append({"type": "plot", "id": "two_sample_z_mean_ci", "title": "Group Means with CI", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_overlap:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # Define range for plotting
                        min_x = min(mean1 - 4 * stddev1, mean2 - 4 * stddev2)
                        max_x = max(mean1 + 4 * stddev1, mean2 + 4 * stddev2)
                        x = np.linspace(min_x, max_x, 1000)
                        
                        # Plot distributions
                        y1 = stats.norm.pdf(x, mean1, stddev1)
                        ax.plot(x, y1, label=f'Group {groups[0]} (Assumed Pop.)')
                        ax.fill_between(x, y1, alpha=0.3)

                        y2 = stats.norm.pdf(x, mean2, stddev2)
                        ax.plot(x, y2, label=f'Group {groups[1]} (Assumed Pop.)')
                        ax.fill_between(x, y2, alpha=0.3)

                        ax.set_title('Overlapping Normal Distributions for Each Group')
                        ax.legend()
                        artifacts.append({"type": "plot", "id": "two_sample_z_overlap", "title": "Overlapping Normal Distributions", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_diff_ci:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        mean_diff = mean1 - mean2
                        se_diff = np.sqrt((stddev1**2 / n1) + (stddev2**2 / n2))
                        z_critical = stats.norm.ppf(1 - alpha / 2)
                        ci_error = z_critical * se_diff

                        ax.bar('Difference', mean_diff, yerr=ci_error, capsize=10, color='#6f42c1')
                        ax.axhline(0, color='black', linestyle='--')
                        
                        ax.set_title(f'Difference in Means with {1-alpha:.0%} CI')
                        ax.set_ylabel('Mean Difference')
                        # Add text label for the mean difference
                        ax.text(0, mean_diff, f'{mean_diff:.2f}', ha='center', va='bottom' if mean_diff > 0 else 'top',
                                bbox=dict(facecolor='white', alpha=0.5, pad=2))

                        artifacts.append({"type": "plot", "id": "two_sample_z_diff_ci", "title": "Difference in Means with CI", "content": PlotUtils.fig_to_base64(fig)})
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
        Provides a formal interpretation of the Two-Sample Z-Test results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            alternative = params.get('alternative', 'two-sided')
            stddev1 = params.get('stddev1', 'σ1')
            stddev2 = params.get('stddev2', 'σ2')

            p_value = None
            mean1, mean2 = None, None
            group1_name, group2_name = 'Group 1', 'Group 2'

            for section in results.get('sections', []):
                if section.get('title') == 'Two-Sample Z-Test Results':
                    for row in section.get('data', []):
                        if row[0] == 'P-Value': p_value = float(row[1])
                        elif 'Mean of' in row[0] and group1_name == 'Group 1':
                            mean1 = float(row[1])
                            group1_name = row[0].split("'")[1]
                        elif 'Mean of' in row[0]:
                            mean2 = float(row[1])
                            group2_name = row[0].split("'")[1]

            if p_value is None or mean1 is None or mean2 is None:
                return "Could not automatically determine test results for interpretation."

            conclusion = "we reject the null hypothesis" if p_value < alpha else "we fail to reject the null hypothesis"
            evidence = "There is significant evidence to conclude that" if p_value < alpha else "There is not enough evidence to conclude that"
            alt_text = {"two-sided": "is different from", "greater": "is greater than", "less": "is less than"}.get(alternative, "is different from")

            return f"Using known population standard deviations (σ₁={stddev1}, σ₂={stddev2}), the Z-test was performed. Since the p-value ({p_value:.4f}) is less than α ({alpha}), {conclusion}. {evidence} the mean for group '{group1_name}' ({mean1:.2f}) {alt_text} the mean for group '{group2_name}' ({mean2:.2f})."

        except (ValueError, TypeError, IndexError):
            return "Could not automatically interpret the two-sample z-test results."
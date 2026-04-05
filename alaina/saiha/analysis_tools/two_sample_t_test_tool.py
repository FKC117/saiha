# d:/quantly/quanta/quantalytics/ai_agents/tools/two_sample_t_test_tool.py

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


class TwoSampleTTestTool(BaseAnalysisTool):
    """
    A tool to perform an independent two-sample T-test.
    """

    @property
    def name(self) -> str:
        return "two_sample_t_test"

    @property
    def description(self) -> str:
        return "Compares the means of a numeric variable between two independent groups."

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
            name="alternative", parameter_type=ParameterType.SELECT,
            label="Alternative Hypothesis", description="Specifies the alternative hypothesis.",
            required=True, default_value="two-sided",
            options=[
                {"value": "two-sided", "label": "Two-sided (Means are not equal)"},
                {"value": "greater", "label": "Greater Than (Group 1 > Group 2)"},
                {"value": "less", "label": "Less Than (Group 1 < Group 2)"},
            ]
        ))
        params.add_parameter(ToolParameter(
            name="alpha", parameter_type=ParameterType.SELECT,
            label="Significance Level (α)", description="The threshold for statistical significance.",
            required=True, default_value="0.05",
            options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
        ))
        # Plotting options
        params.add_parameter(ToolParameter(name="generate_boxplots", parameter_type=ParameterType.CHECKBOX, label="Generate Side-by-side Box Plots", required=False, default_value=True))
        params.add_parameter(ToolParameter(name="generate_violinplots", parameter_type=ParameterType.CHECKBOX, label="Generate Violin Plots with Swarm Plots", required=False, default_value=True))
        params.add_parameter(ToolParameter(name="generate_mean_plot", parameter_type=ParameterType.CHECKBOX, label="Generate Group Means with CI Plot", required=False, default_value=False))
        params.add_parameter(ToolParameter(name="generate_histograms", parameter_type=ParameterType.CHECKBOX, label="Generate Back-to-back Histograms", required=False, default_value=False))
        params.add_parameter(ToolParameter(name="generate_beeswarm", parameter_type=ParameterType.CHECKBOX, label="Generate Beeswarm Plots", required=False, default_value=False))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            var = parameters.get("variable")
            group_col = parameters.get("group_column")
            alpha = float(parameters.get("alpha", 0.05))
            alternative = parameters.get("alternative", "two-sided")

            # Plotting flags
            gen_box = str(parameters.get("generate_boxplots", "true")).lower() in ('true', 'on', '1')
            gen_violin = str(parameters.get("generate_violinplots", "true")).lower() in ('true', 'on', '1')
            gen_mean = str(parameters.get("generate_mean_plot", "false")).lower() in ('true', 'on', '1')
            gen_hist = str(parameters.get("generate_histograms", "false")).lower() in ('true', 'on', '1')
            gen_swarm = str(parameters.get("generate_beeswarm", "false")).lower() in ('true', 'on', '1')

            if not var or not group_col:
                return {"status": "error", "summary": "Both a numeric variable and a grouping variable are required."}

            # Use efficient column projection loading from BaseAnalysisTool
            df = self.load_dataset(columns=[var, group_col])

            df_clean = df[[var, group_col]].dropna()
            groups = df_clean[group_col].unique()
            if len(groups) != 2:
                return {"status": "error", "summary": f"The grouping variable '{group_col}' must have exactly two unique groups, but it has {len(groups)}."}

            group1_data = df_clean[df_clean[group_col] == groups[0]][var]
            group2_data = df_clean[df_clean[group_col] == groups[1]][var]

            if len(group1_data) < 2 or len(group2_data) < 2:
                return {"status": "error", "summary": "One or both groups have insufficient data to perform the test."}

            # Welch's T-test is default as it's more robust
            t_stat, p_val = stats.ttest_ind(group1_data, group2_data, alternative=alternative, equal_var=False)
            is_significant = p_val < alpha

            summary = (
                f"The mean of '{var}' for group '{groups[0]}' ({group1_data.mean():.2f}) and group '{groups[1]}' ({group2_data.mean():.2f}) "
                f"are {'statistically significantly' if is_significant else 'not statistically significantly'} different (p={p_val:.4f})."
            )

            artifacts: List[Dict[str, Any]] = []
            sections = [{
                'type': 'table', 'title': 'Two-Sample T-Test Results',
                'headers': ['Statistic', 'Value'],
                'data': [
                    ['T-Statistic', f"{t_stat:.4f}"],
                    ['P-Value', f"{p_val:.6f}"],
                    [f"Mean of '{groups[0]}'", f"{group1_data.mean():.4f}"],
                    [f"Mean of '{groups[1]}'", f"{group2_data.mean():.4f}"],
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
                        artifacts.append({"type": "plot", "id": "two_sample_boxplot", "title": "Side-by-side Box Plots", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_violin:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(10, 7))
                        sns.violinplot(x=group_col, y=var, data=df_clean, ax=ax, inner='quartile', cut=0)
                        # Use stripplot with jitter for performance instead of swarmplot
                        sns.stripplot(x=group_col, y=var, data=df_clean, color='k', alpha=0.3, ax=ax, jitter=True)
                        ax.set_title(f'Violin Plots of {var} by {group_col}')
                        artifacts.append({"type": "plot", "id": "two_sample_violinplot", "title": "Violin Plots with Data Points", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_mean:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.pointplot(x=group_col, y=var, data=df_clean, ax=ax, capsize=.1, errorbar='ci', n_boot=1000)
                        ax.set_title(f'Mean of {var} by {group_col} with 95% CI')
                        artifacts.append({"type": "plot", "id": "two_sample_meanplot", "title": "Group Means with Confidence Intervals", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_hist:
                    fig = None
                    try:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
                        sns.histplot(group1_data, ax=ax1, kde=True)
                        ax1.set_title(f'Distribution for {groups[0]}')
                        sns.histplot(group2_data, ax=ax2, kde=True)
                        ax2.set_title(f'Distribution for {groups[1]}')
                        fig.suptitle(f'Back-to-back Histograms for {var}')
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        artifacts.append({"type": "plot", "id": "two_sample_histograms", "title": "Back-to-back Histograms", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_swarm:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        # Use stripplot with jitter for performance instead of swarmplot
                        sns.stripplot(x=group_col, y=var, data=df_clean, ax=ax, jitter=True, alpha=0.7)
                        ax.set_title(f'Beeswarm-style Plot of {var} by {group_col}')
                        artifacts.append({"type": "plot", "id": "two_sample_beeswarm", "title": "Beeswarm-style Plots", "content": PlotUtils.fig_to_base64(fig)})
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
        Provides a formal interpretation of the Two-Sample T-Test results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            alternative = params.get('alternative', 'two-sided')
            group_col = params.get('group_column')

            p_value = None
            mean1, mean2 = None, None
            group1_name, group2_name = 'Group 1', 'Group 2'

            for section in results.get('sections', []):
                if section.get('title') == 'Two-Sample T-Test Results':
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

            if p_value < alpha:
                conclusion = "we reject the null hypothesis"
                evidence = "There is significant evidence to conclude that"
            else:
                conclusion = "we fail to reject the null hypothesis"
                evidence = "There is not enough evidence to conclude that"

            if alternative == 'two-sided':
                return f"Since the p-value ({p_value:.4f}) is less than α ({alpha}), {conclusion}. {evidence} the mean for group '{group1_name}' ({mean1:.2f}) is different from the mean for group '{group2_name}' ({mean2:.2f})."
            elif alternative == 'greater':
                return f"Since the p-value ({p_value:.4f}) is less than α ({alpha}), {conclusion}. {evidence} the mean for group '{group1_name}' ({mean1:.2f}) is greater than the mean for group '{group2_name}' ({mean2:.2f})."
            else:  # 'less'
                return f"Since the p-value ({p_value:.4f}) is less than α ({alpha}), {conclusion}. {evidence} the mean for group '{group1_name}' ({mean1:.2f}) is less than the mean for group '{group2_name}' ({mean2:.2f})."

        except (ValueError, TypeError, IndexError):
            return "Could not automatically interpret the two-sample t-test results."
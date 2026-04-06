
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


class MannWhitneyUTestTool(BaseAnalysisTool):
    """
    A tool to perform the Mann-Whitney U test.
    """

    @property
    def name(self) -> str:
        return "mann_whitney_u_test"

    @property
    def description(self) -> str:
        return "Compares distributions between two independent groups. Non-parametric equivalent of the two-sample t-test."

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
            label="Alternative Hypothesis", description="Specifies the alternative hypothesis.",
            required=True, default_value="two-sided",
            options=[
                {"value": "two-sided", "label": "Two-sided (Distributions are not equal)"},
                {"value": "less", "label": "Less (Distribution 1 is stochastically less than 2)"},
                {"value": "greater", "label": "Greater (Distribution 1 is stochastically greater than 2)"},
            ]
        ))
        params.add_parameter(ToolParameter(
            name="alpha", parameter_type=ParameterType.SELECT,
            label="Significance Level (α)", description="The threshold for statistical significance.",
            required=True, default_value="0.05",
            options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
        ))
        params.add_parameter(ToolParameter(name="generate_boxplots", parameter_type=ParameterType.CHECKBOX, label="Generate Side-by-side Box Plots", required=False, default_value=True))
        params.add_parameter(ToolParameter(name="generate_violinplots", parameter_type=ParameterType.CHECKBOX, label="Generate Violin Plots", required=False, default_value=True))
        params.add_parameter(ToolParameter(name="generate_ecdf_plot", parameter_type=ParameterType.CHECKBOX, label="Generate ECDF Plots", required=False, default_value=False))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            var = parameters.get("variable")
            group_col = parameters.get("group_column")
            alpha = float(parameters.get("alpha", 0.05))
            alternative = parameters.get("alternative", "two-sided")
            gen_box = str(parameters.get("generate_boxplots", "true")).lower() in ('true', 'on', '1')
            gen_violin = str(parameters.get("generate_violinplots", "true")).lower() in ('true', 'on', '1')
            gen_ecdf = str(parameters.get("generate_ecdf_plot", "false")).lower() in ('true', 'on', '1')

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

            if len(group1_data) < 1 or len(group2_data) < 1:
                return {"status": "error", "summary": "One or both groups have no data to perform the test."}

            u_stat, p_val = stats.mannwhitneyu(group1_data, group2_data, alternative=alternative)
            is_significant = p_val < alpha

            summary = (
                f"The Mann-Whitney U test indicates that the distributions of '{var}' for group '{groups[0]}' and group '{groups[1]}' "
                f"are {'statistically significantly' if is_significant else 'not statistically significantly'} different (p={p_val:.4f})."
            )

            artifacts: List[Dict[str, Any]] = []
            sections = [{
                'type': 'table', 'title': 'Mann-Whitney U Test Results',
                'headers': ['Statistic', 'Value'],
                'data': [
                    ['U-Statistic', f"{u_stat:.4f}"],
                    ['P-Value', f"{p_val:.6f}"],
                    [f"Median of '{groups[0]}'", f"{group1_data.median():.4f}"],
                    [f"Median of '{groups[1]}'", f"{group2_data.median():.4f}"],
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
                        artifacts.append({"type": "plot", "id": "mwu_boxplot", "title": "Side-by-side Box Plots", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_violin:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(10, 7))
                        sns.violinplot(x=group_col, y=var, data=df_clean, ax=ax, inner='quartile', cut=0)
                        sns.stripplot(x=group_col, y=var, data=df_clean, color='k', alpha=0.3, ax=ax, jitter=True)
                        ax.set_title(f'Violin Plots of {var} by {group_col}')
                        artifacts.append({"type": "plot", "id": "mwu_violinplot", "title": "Violin Plots with Data Points", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_ecdf:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(10, 7))
                        sns.ecdfplot(data=df_clean, x=var, hue=group_col, ax=ax)
                        ax.set_title(f'ECDF of {var} by {group_col}')
                        ax.set_xlabel(var)
                        ax.set_ylabel('Cumulative Probability')
                        artifacts.append({"type": "plot", "id": "mwu_ecdf_plot", "title": "ECDF Plots", "content": PlotUtils.fig_to_base64(fig)})
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
        Provides a formal interpretation of the Mann-Whitney U Test results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            alternative = params.get('alternative', 'two-sided')
            
            p_value = None
            median1, median2 = None, None
            group1_name, group2_name = 'Group 1', 'Group 2'

            for section in results.get('sections', []):
                if section.get('title') == 'Mann-Whitney U Test Results':
                    for row in section.get('data', []):
                        if row[0] == 'P-Value': p_value = float(row[1])
                        elif 'Median of' in row[0] and group1_name == 'Group 1':
                            median1 = float(row[1])
                            group1_name = row[0].split("'")[1]
                        elif 'Median of' in row[0]:
                            median2 = float(row[1])
                            group2_name = row[0].split("'")[1]

            if p_value is None:
                return "Could not automatically determine the p-value for interpretation."

            conclusion = "we reject the null hypothesis" if p_value < alpha else "we fail to reject the null hypothesis"
            evidence = "There is significant evidence to suggest that" if p_value < alpha else "There is not enough evidence to suggest that"
            alt_text = {"two-sided": "are different", "greater": f"for '{group1_name}' is stochastically greater than for '{group2_name}'", "less": f"for '{group1_name}' is stochastically less than for '{group2_name}'"}.get(alternative, "are different")

            return f"Since the p-value ({p_value:.4f}) is less than α ({alpha}), {conclusion}. {evidence} the distributions for group '{group1_name}' and group '{group2_name}' {alt_text}."

        except (ValueError, TypeError, IndexError):
            return "Could not automatically interpret the Mann-Whitney U test results."
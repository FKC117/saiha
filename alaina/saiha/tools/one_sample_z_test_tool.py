# d:/quantly/quanta/quantalytics/ai_agents/tools/one_sample_z_test_tool.py

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


class OneSampleZTestTool(BaseAnalysisTool):
    """
    A tool to perform a one-sample Z-test.
    """

    @property
    def name(self) -> str:
        return "one_sample_z_test"

    @property
    def description(self) -> str:
        return "Tests if a sample mean is equal to a known population mean, when the population standard deviation is known."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="variable", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Variable to Test", description="Select the numeric variable whose mean you want to test.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="population_mean", parameter_type=ParameterType.NUMBER,
            label="Hypothesized Population Mean (μ₀)", description="The known population mean to test against.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="population_stddev", parameter_type=ParameterType.NUMBER,
            label="Population Standard Deviation (σ)", description="The known population standard deviation. Must be greater than 0.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="alternative", parameter_type=ParameterType.SELECT,
            label="Alternative Hypothesis", description="Specifies the alternative hypothesis for the test.",
            required=True, default_value="two-sided",
            options=[
                {"value": "two-sided", "label": "Two-sided (Mean ≠ μ₀)"},
                {"value": "greater", "label": "Greater Than (Mean > μ₀)"},
                {"value": "less", "label": "Less Than (Mean < μ₀)"},
            ]
        ))
        params.add_parameter(ToolParameter(
            name="alpha", parameter_type=ParameterType.SELECT,
            label="Significance Level (α)", description="The threshold for determining statistical significance.",
            required=True, default_value="0.05",
            options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
        ))
        params.add_parameter(ToolParameter(name="generate_boxplot", parameter_type=ParameterType.CHECKBOX, label="Generate Box Plot with Reference Line", required=False, default_value=True))
        params.add_parameter(ToolParameter(name="generate_dist_plot", parameter_type=ParameterType.CHECKBOX, label="Generate Distribution Plot with Mean Lines", required=False, default_value=True))
        params.add_parameter(ToolParameter(
            name="generate_mean_ci_plot", parameter_type=ParameterType.CHECKBOX,
            label="Generate Sample Mean with Confidence Interval Plot", required=False, default_value=False
        ))
        params.add_parameter(ToolParameter(
            name="generate_critical_region_plot", parameter_type=ParameterType.CHECKBOX,
            label="Generate Normal Curve with Critical Regions", required=False, default_value=False
        ))
        params.add_parameter(ToolParameter(
            name="generate_zscore_marker_plot", parameter_type=ParameterType.CHECKBOX,
            label="Generate Standard Normal with Z-Score Marker", required=False, default_value=False
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            def _safe_float(val, default=0.0):
                return float(val) if val else default

            var = parameters.get("variable")
            pop_mean = _safe_float(parameters.get("population_mean"))
            pop_stddev = _safe_float(parameters.get("population_stddev"))
            alpha = float(parameters.get("alpha", 0.05))
            alternative = parameters.get("alternative", "two-sided")
            gen_boxplot = str(parameters.get("generate_boxplot", "true")).lower() in ('true', 'on', '1')
            gen_dist_plot = str(parameters.get("generate_dist_plot", "true")).lower() in ('true', 'on', '1')
            gen_mean_ci = str(parameters.get("generate_mean_ci_plot", "false")).lower() in ('true', 'on', '1')
            gen_critical = str(parameters.get("generate_critical_region_plot", "false")).lower() in ('true', 'on', '1')
            gen_z_marker = str(parameters.get("generate_zscore_marker_plot", "false")).lower() in ('true', 'on', '1')

            if not var:
                return {"status": "error", "summary": "A variable to test is required."}
            if pop_stddev <= 0:
                return {"status": "error", "summary": "Population standard deviation must be greater than zero."}

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            sample = df[var].dropna()
            n = len(sample)
            if n < 2:
                return {"status": "error", "summary": f"Not enough data in '{var}' to perform a Z-test."}

            sample_mean = sample.mean()
            
            # Manual Z-test calculation
            z_stat = (sample_mean - pop_mean) / (pop_stddev / np.sqrt(n))

            if alternative == 'two-sided':
                p_val = 2 * stats.norm.sf(np.abs(z_stat))
            elif alternative == 'greater':
                p_val = stats.norm.sf(z_stat)
            else:  # 'less'
                p_val = stats.norm.cdf(z_stat)

            is_significant = p_val < alpha

            summary = (
                f"The mean of '{var}' ({sample_mean:.2f}) is {'statistically significantly' if is_significant else 'not statistically significantly'} "
                f"different from the hypothesized population mean of {pop_mean} (p={p_val:.4f})."
            )

            artifacts: List[Dict[str, Any]] = []
            sections = [{
                'type': 'table', 'title': 'One-Sample Z-Test Results',
                'headers': ['Statistic', 'Value'],
                'data': [
                    ['Z-Statistic', f"{z_stat:.4f}"],
                    ['P-Value', f"{p_val:.6f}"],
                    ['Sample Size', n],
                    ['Sample Mean', f"{sample_mean:.4f}"],
                    ['Is Significant at ' + str(alpha), 'Yes' if is_significant else 'No']
                ]
            }]

            # --- Optional Visualizations ---
            with PlotUtils.setup_plotting():
                if gen_boxplot:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.boxplot(y=sample, ax=ax)
                        ax.axhline(pop_mean, color='r', linestyle='--', label=f'Hypothesized Mean ({pop_mean})')
                        ax.set_title(f'Box Plot of {var}')
                        ax.set_ylabel(var)
                        ax.legend()
                        artifacts.append({"type": "plot", "id": "one_sample_z_boxplot", "title": "Box Plot", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_dist_plot:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.histplot(sample, kde=True, ax=ax, label='Sample Distribution')
                        ax.axvline(sample.mean(), color='g', linestyle='-', linewidth=2, label=f'Sample Mean ({sample.mean():.2f})')
                        ax.axvline(pop_mean, color='r', linestyle='--', linewidth=2, label=f'Hypothesized Mean ({pop_mean})')
                        ax.set_title(f'Distribution of {var}')
                        ax.legend()
                        artifacts.append({"type": "plot", "id": "one_sample_z_dist_plot", "title": "Distribution Plot", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_mean_ci:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        z_critical = stats.norm.ppf(1 - alpha / 2)
                        ci_error = z_critical * (pop_stddev / np.sqrt(n))
                        
                        ax.bar('Sample Mean', sample_mean, yerr=ci_error, capsize=5, color='#007bff', label=f'Sample Mean ({1-alpha:.0%} CI)')
                        ax.axhline(pop_mean, color='#dc3545', linestyle='--', label=f'Hypothesized Mean ({pop_mean})')
                        
                        ax.set_title(f'Sample Mean of {var} vs. Hypothesized Mean')
                        ax.set_ylabel('Mean Value')
                        ax.legend()
                        artifacts.append({"type": "plot", "id": "one_sample_z_mean_ci", "title": "Sample Mean with CI", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_critical:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        se = pop_stddev / np.sqrt(n)
                        x = np.linspace(pop_mean - 4 * se, pop_mean + 4 * se, 1000)
                        y = stats.norm.pdf(x, pop_mean, se)
                        ax.plot(x, y, label='Normal Distribution under H₀')
                        
                        if alternative == 'two-sided':
                            crit_upper = stats.norm.ppf(1 - alpha / 2, pop_mean, se)
                            crit_lower = stats.norm.ppf(alpha / 2, pop_mean, se)
                            ax.fill_between(x, y, where=(x >= crit_upper) | (x <= crit_lower), color='red', alpha=0.3, label='Critical Region')
                        elif alternative == 'greater':
                            crit_upper = stats.norm.ppf(1 - alpha, pop_mean, se)
                            ax.fill_between(x, y, where=(x >= crit_upper), color='red', alpha=0.3, label='Critical Region')
                        else: # less
                            crit_lower = stats.norm.ppf(alpha, pop_mean, se)
                            ax.fill_between(x, y, where=(x <= crit_lower), color='red', alpha=0.3, label='Critical Region')
                        
                        ax.axvline(sample_mean, color='green', linestyle='--', label=f'Sample Mean ({sample_mean:.2f})')
                        ax.set_title('Normal Curve with Critical Regions')
                        ax.legend()
                        artifacts.append({"type": "plot", "id": "one_sample_z_critical", "title": "Critical Region Plot", "content": PlotUtils.fig_to_base64(fig)})
                    finally:
                        if fig: plt.close(fig)

                if gen_z_marker:
                    fig = None
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        x = np.linspace(-4, 4, 1000)
                        y = stats.norm.pdf(x, 0, 1)
                        ax.plot(x, y, label='Standard Normal Distribution')

                        if alternative == 'two-sided':
                            z_crit = stats.norm.ppf(1 - alpha / 2)
                            ax.fill_between(x, y, where=(x >= z_crit) | (x <= -z_crit), color='red', alpha=0.3, label='Critical Region')
                        elif alternative == 'greater':
                            z_crit = stats.norm.ppf(1 - alpha)
                            ax.fill_between(x, y, where=(x >= z_crit), color='red', alpha=0.3, label='Critical Region')
                        else: # less
                            z_crit = stats.norm.ppf(alpha)
                            ax.fill_between(x, y, where=(x <= z_crit), color='red', alpha=0.3, label='Critical Region')

                        ax.axvline(z_stat, color='green', linestyle='--', label=f'Z-Statistic ({z_stat:.2f})')
                        ax.set_title('Standard Normal Distribution with Z-Score Marker')
                        ax.legend()
                        artifacts.append({"type": "plot", "id": "one_sample_z_marker", "title": "Z-Score Marker Plot", "content": PlotUtils.fig_to_base64(fig)})
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
        Provides a formal interpretation of the One-Sample Z-Test results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            alternative = params.get('alternative', 'two-sided')
            pop_mean = params.get('population_mean', 'the hypothesized mean')
            pop_stddev = params.get('population_stddev', 'a known value')

            p_value = None
            sample_mean = None

            for section in results.get('sections', []):
                if section.get('title') == 'One-Sample Z-Test Results':
                    for row in section.get('data', []):
                        if row[0] == 'P-Value': p_value = float(row[1])
                        elif row[0] == 'Sample Mean': sample_mean = float(row[1])

            if p_value is None or sample_mean is None:
                return "Could not automatically determine test results for interpretation."

            if p_value < alpha:
                conclusion = "we reject the null hypothesis"
                evidence = "There is significant evidence to conclude that"
            else:
                conclusion = "we fail to reject the null hypothesis"
                evidence = "There is not enough evidence to conclude that"

            alt_text = {"two-sided": "is different from", "greater": "is greater than", "less": "is less than"}.get(alternative, "is different from")

            return f"Using a known population standard deviation of {pop_stddev}, the Z-test was performed. Since the p-value ({p_value:.4f}) is less than α ({alpha}), {conclusion}. {evidence} the sample mean ({sample_mean:.2f}) {alt_text} the hypothesized population mean of {pop_mean}."

        except (ValueError, TypeError, IndexError):
            return "Could not automatically interpret the one-sample z-test results."
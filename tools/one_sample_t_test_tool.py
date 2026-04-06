
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from django.core.files.storage import default_storage
from typing import Any, Dict, List
from typing import Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class OneSampleTTestTool(BaseAnalysisTool):
    """
    A tool to perform a one-sample T-test.
    """

    @property
    def name(self) -> str:
        return "one_sample_t_test"

    @property
    def description(self) -> str:
        return "Tests if the mean of a single sample is equal to a known or hypothesized population mean."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(
            ToolParameter(
                name="variable", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Variable to Test", description="Select the numeric variable whose mean you want to test.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="population_mean", parameter_type=ParameterType.NUMBER,
                label="Hypothesized Population Mean (μ₀)", description="The known or hypothesized mean to test against.",
                required=True, default_value=0.0
            )
        )
        params.add_parameter(
            ToolParameter(
                name="alternative", parameter_type=ParameterType.SELECT,
                label="Alternative Hypothesis", description="Specifies the alternative hypothesis for the test.",
                required=True, default_value="two-sided",
                options=[
                    {"value": "two-sided", "label": "Two-sided (Mean ≠ μ₀)"},
                    {"value": "greater", "label": "Greater Than (Mean > μ₀)"},
                    {"value": "less", "label": "Less Than (Mean < μ₀)"},
                ]
            )
        )
        params.add_parameter(
            ToolParameter(
                name="alpha", parameter_type=ParameterType.SELECT,
                label="Significance Level (α)", description="The threshold for determining statistical significance.",
                required=True, default_value="0.05",
                options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_boxplot", parameter_type=ParameterType.CHECKBOX,
                label="Generate Box Plot with Reference Line", required=False, default_value=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_dist_plot", parameter_type=ParameterType.CHECKBOX,
                label="Generate Distribution Plot with Mean Lines", required=False, default_value=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_qq_plot", parameter_type=ParameterType.CHECKBOX,
                label="Generate Q-Q Plot for Normality Check", required=False, default_value=False
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_mean_plot", parameter_type=ParameterType.CHECKBOX,
                label="Generate Mean Comparison Plot with CI", required=False, default_value=False
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            def _safe_float(val, default=0.0):
                return float(val) if val else default
            var = parameters.get("variable")
            pop_mean = _safe_float(parameters.get("population_mean"), 0.0)
            alpha = float(parameters.get("alpha", 0.05))
            alternative = parameters.get("alternative", "two-sided")
            gen_boxplot = str(parameters.get("generate_boxplot", "true")).lower() in ('true', 'on', '1')
            gen_dist_plot = str(parameters.get("generate_dist_plot", "true")).lower() in ('true', 'on', '1')
            gen_qq_plot = str(parameters.get("generate_qq_plot", "false")).lower() in ('true', 'on', '1')
            gen_mean_plot = str(parameters.get("generate_mean_plot", "false")).lower() in ('true', 'on', '1')

            if not var:
                return {"status": "error", "summary": "A variable to test is required."}

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            sample = df[var].dropna()
            if len(sample) < 2:
                return {"status": "error", "summary": f"Not enough data in '{var}' to perform a one-sample T-test."}

            t_stat, p_val = stats.ttest_1samp(sample, pop_mean, alternative=alternative)
            dof = len(sample) - 1
            is_significant = p_val < alpha

            summary = (
                f"The mean of '{var}' ({sample.mean():.2f}) is {'statistically significantly' if is_significant else 'not statistically significantly'} "
                f"different from the hypothesized mean of {pop_mean} (p={p_val:.4f})."
            )

            artifacts: List[Dict[str, Any]] = []
            sections = [{
                'type': 'table', 'title': 'One-Sample T-Test Results',
                'headers': ['Statistic', 'Value'],
                'data': [
                    ['T-Statistic', f"{t_stat:.4f}"],
                    ['P-Value', f"{p_val:.6f}"],
                    ['Degrees of Freedom', dof],
                    ['Sample Mean', f"{sample.mean():.4f}"],
                    ['Is Significant at ' + str(alpha), 'Yes' if is_significant else 'No']
                ]
            }]

            # --- Optional Visualizations ---
            with PlotUtils.setup_plotting():
                if gen_boxplot:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(y=sample, ax=ax)
                    ax.axhline(pop_mean, color='r', linestyle='--', label=f'Hypothesized Mean ({pop_mean})')
                    ax.set_title(f'Box Plot of {var}')
                    ax.set_ylabel(var)
                    ax.legend()
                    artifacts.append({"type": "plot", "id": "one_sample_boxplot", "title": "Box Plot", "content": PlotUtils.fig_to_base64(fig)})

                if gen_dist_plot:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(sample, kde=True, ax=ax, label='Sample Distribution')
                    ax.axvline(sample.mean(), color='g', linestyle='-', linewidth=2, label=f'Sample Mean ({sample.mean():.2f})')
                    ax.axvline(pop_mean, color='r', linestyle='--', linewidth=2, label=f'Hypothesized Mean ({pop_mean})')
                    ax.set_title(f'Distribution of {var}')
                    ax.legend()
                    artifacts.append({"type": "plot", "id": "one_sample_dist_plot", "title": "Distribution Plot", "content": PlotUtils.fig_to_base64(fig)})

                if gen_qq_plot:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    stats.probplot(sample, dist="norm", plot=ax)
                    ax.set_title('Q-Q Plot for Normality Check')
                    artifacts.append({"type": "plot", "id": "one_sample_qq_plot", "title": "Q-Q Plot", "content": PlotUtils.fig_to_base64(fig)})

                if gen_mean_plot:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sample_mean = sample.mean()
                    # Calculate confidence interval
                    ci = stats.t.interval(1 - alpha, dof, loc=sample_mean, scale=stats.sem(sample))
                    ci_error = sample_mean - ci[0]

                    bar_labels = ['Sample Mean', 'Hypothesized Mean']
                    means = [sample_mean, pop_mean]
                    colors = ['#007bff', '#dc3545']
                    
                    ax.bar(bar_labels[0], means[0], yerr=ci_error, capsize=5, color=colors[0], label=f'Sample Mean (95% CI)')
                    ax.axhline(pop_mean, color=colors[1], linestyle='--', label=f'Hypothesized Mean ({pop_mean})')

                    # Add text labels for clarity
                    ax.text(0, sample_mean + ci_error * 1.1, f'{sample_mean:.2f}', ha='center', va='bottom', color='white')
                    
                    ax.set_title(f'Mean of {var} vs. Hypothesized Mean')
                    ax.set_ylabel('Mean Value')
                    ax.legend()
                    # Remove x-axis ticks for the second bar as it's a line
                    ax.set_xticks([0]) 
                    ax.set_xticklabels([bar_labels[0]])
                    
                    artifacts.append({"type": "plot", "id": "one_sample_mean_plot", "title": "Mean Comparison Plot", "content": PlotUtils.fig_to_base64(fig)})


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
        

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Provides a simple, rule-based interpretation of the One-Sample T-Test results
        by parsing the 'sections' data.
        """
        if results.get('status') != 'ok':
            return None

        p_value = None
        params = results.get('meta', {}).get('parameters', {})
        test_value = params.get('population_mean', 'the hypothesized mean')
        alternative = params.get('alternative', 'two-sided')
        alpha = float(params.get('alpha', 0.05))
        sample_mean = None

        # Find the p-value within the sections data
        try:
            for section in results.get('sections', []):
                if section.get('type') == 'table':
                    for row in section.get('data', []):
                        if row[0] == 'P-Value':
                            p_value = float(row[1])
                        elif row[0] == 'Sample Mean':
                            sample_mean = float(row[1])
                            break
                if p_value is not None:
                    break
        except (ValueError, TypeError, IndexError):
            return "Could not automatically determine the p-value from the results table."

        if p_value is None:
            return "P-value not found in the analysis results."

        # Generate a more specific interpretation based on the alternative hypothesis
        if p_value < alpha:
            # Significant result
            if alternative == 'two-sided':
                return f"Since the p-value ({p_value:.4f}) is less than the significance level (α = {alpha}), we reject the null hypothesis. There is significant evidence to conclude that the sample mean ({sample_mean:.2f}) is different from the hypothesized mean of {test_value}."
            elif alternative == 'greater':
                return f"Since the p-value ({p_value:.4f}) is less than the significance level (α = {alpha}), we reject the null hypothesis. There is significant evidence to conclude that the sample mean ({sample_mean:.2f}) is greater than the hypothesized mean of {test_value}."
            else:  # 'less'
                return f"Since the p-value ({p_value:.4f}) is less than the significance level (α = {alpha}), we reject the null hypothesis. There is significant evidence to conclude that the sample mean ({sample_mean:.2f}) is less than the hypothesized mean of {test_value}."
        else:
            # Not a significant result
            if alternative == 'two-sided':
                return f"Since the p-value ({p_value:.4f}) is greater than or equal to the significance level (α = {alpha}), we fail to reject the null hypothesis. There is not enough evidence to conclude that the sample mean is different from the hypothesized mean of {test_value}."
            elif alternative == 'greater':
                return f"Since the p-value ({p_value:.4f}) is greater than or equal to the significance level (α = {alpha}), we fail to reject the null hypothesis. There is not enough evidence to conclude that the sample mean is greater than the hypothesized mean of {test_value}."
            else:  # 'less'
                return f"Since the p-value ({p_value:.4f}) is greater than or equal to the significance level (α = {alpha}), we fail to reject the null hypothesis. There is not enough evidence to conclude that the sample mean is less than the hypothesized mean of {test_value}."
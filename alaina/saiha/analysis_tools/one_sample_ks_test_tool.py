# d:/quantly/quanta/quantalytics/ai_agents/tools/one_sample_ks_test_tool.py

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


class OneSampleKSTestTool(BaseAnalysisTool):
    """
    A tool to perform the one-sample Kolmogorov-Smirnov test.
    """

    @property
    def name(self) -> str:
        return "one_sample_ks_test"

    @property
    def description(self) -> str:
        return "Tests if a sample distribution differs from a specified theoretical distribution (e.g., normal)."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="variable", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Variable to Test", description="Select the numeric variable to test against a distribution.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="test_method", parameter_type=ParameterType.SELECT,
            label="Test Method for Normality", description="Shapiro-Wilk is often more powerful for testing normality.",
            required=True, default_value="ks",
            options=[{"value": "ks", "label": "Kolmogorov-Smirnov"}, {"value": "shapiro", "label": "Shapiro-Wilk"}]
        ))
        params.add_parameter(ToolParameter(
            name="distribution", parameter_type=ParameterType.SELECT,
            label="Theoretical Distribution", description="The theoretical distribution to compare against.",
            required=True, default_value="norm",
            options=[
                {"value": "norm", "label": "Normal"},
                {"value": "uniform", "label": "Uniform"},
                {"value": "expon", "label": "Exponential"},
            ]
        ))
        params.add_parameter(ToolParameter(
            name="alpha", parameter_type=ParameterType.SELECT,
            label="Significance Level (α)", description="The threshold for statistical significance.",
            required=True, default_value="0.05",
            options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            var = parameters.get("variable")
            method = parameters.get("test_method", "ks")
            dist_name = parameters.get("distribution", "norm")
            alpha = float(parameters.get("alpha", 0.05))

            if not var:
                return {"status": "error", "summary": "A variable to test is required."}

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            sample = df[var].dropna()
            if len(sample) < 2:
                return {"status": "error", "summary": f"Not enough data in '{var}' to perform the test."}

            # --- Perform the statistical test ---
            if method == 'shapiro' and dist_name == 'norm':
                if len(sample) < 3:
                    return {"status": "error", "summary": "Shapiro-Wilk test requires at least 3 data points."}
                test_stat, p_val = stats.shapiro(sample)
                test_name = "Shapiro-Wilk Test for Normality"
                stat_name = "W Statistic"
                summary = (
                    f"The Shapiro-Wilk test indicates that the distribution of '{var}' is "
                    f"{'statistically significantly different' if p_val < alpha else 'not statistically significantly different'} "
                    f"from a normal distribution (p={p_val:.4f})."
                )
            else: # Default to KS test
                test_name = "One-Sample Kolmogorov-Smirnov Test"
                stat_name = "KS Statistic"
                # For KS test against normal, we standardize. For others, we use the raw data.
                if dist_name == 'norm':
                    test_sample = (sample - sample.mean()) / sample.std()
                else:
                    test_sample = sample
                
                test_stat, p_val = stats.kstest(test_sample, dist_name)
                summary = (
                    f"The one-sample KS test indicates that the distribution of '{var}' is "
                    f"{'statistically significantly different' if p_val < alpha else 'not statistically significantly different'} "
                    f"from a standard {dist_name} distribution (p={p_val:.4f})."
                )

            is_significant = p_val < alpha

            artifacts: List[Dict[str, Any]] = []
            sections = [{
                'type': 'table', 'title': test_name,
                'headers': ['Statistic', 'Value'],
                'data': [
                    [stat_name, f"{test_stat:.4f}"],
                    ['P-Value', f"{p_val:.6f}"],
                    ['Is Significant at ' + str(alpha), 'Yes' if is_significant else 'No']
                ]
            }]

            # --- Visualization ---
            with PlotUtils.setup_plotting():
                fig = None
                try:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    test_sample_for_plot = (sample - sample.mean()) / sample.std() if dist_name == 'norm' else sample
                    sns.ecdfplot(data=test_sample_for_plot, ax=ax, label='Empirical CDF')
                    
                    # Plot theoretical CDF
                    x_vals = np.linspace(test_sample_for_plot.min(), test_sample_for_plot.max(), 100)
                    theoretical_cdf = getattr(stats, dist_name).cdf(x_vals)
                    ax.plot(x_vals, theoretical_cdf, 'r--', label=f'Theoretical {dist_name.capitalize()} CDF')
                    
                    ax.set_title(f'ECDF of {var} vs. Theoretical {dist_name.capitalize()} CDF')
                    ax.legend()
                    artifacts.append({"type": "plot", "id": "one_sample_ks_ecdf", "title": "ECDF Comparison Plot", "content": PlotUtils.fig_to_base64(fig)})
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
        Provides a formal interpretation of the one-sample KS or Shapiro-Wilk test results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            var = params.get('variable', 'the variable')
            method = params.get('test_method', 'ks')
            dist_name = params.get('distribution', 'norm')

            # Map internal distribution names to user-friendly labels
            dist_map = {
                'norm': 'Normal',
                'expon': 'Exponential',
                'uniform': 'Uniform'
            }
            friendly_dist_name = dist_map.get(dist_name, dist_name)

            p_value = None
            test_name = "Shapiro-Wilk Test for Normality" if method == 'shapiro' else "One-Sample Kolmogorov-Smirnov Test"
            
            test_section = next((s for s in results.get('sections', []) if s.get('title') == test_name), None)
            if test_section:
                for row in test_section.get('data', []):
                    if row[0] == 'P-Value':
                        p_value = float(row[1])
                        break

            if p_value is None:
                return "Could not automatically determine the p-value for interpretation."

            if p_value < alpha:
                conclusion = "we reject the null hypothesis"
                evidence = (
                    f"There is significant evidence to suggest that the distribution of '{var}' is not normal."
                    if method == 'shapiro' else
                    f"There is significant evidence to suggest that the distribution of '{var}' does not follow the specified theoretical '{friendly_dist_name}' distribution."
                )
            else:
                conclusion = "we fail to reject the null hypothesis"
                evidence = (
                    f"There is not enough evidence to conclude that the distribution of '{var}' is different from a normal distribution."
                    if method == 'shapiro' else
                    f"There is not enough evidence to conclude that the distribution of '{var}' is different from the specified theoretical '{friendly_dist_name}' distribution (when standardized)."
                )

            return f"The {test_name} was performed to test if the data follows the specified distribution. Since the p-value ({p_value:.4f}) is {'less than' if p_value < alpha else 'not less than'} α ({alpha}), {conclusion}. {evidence}"

        except (ValueError, TypeError, IndexError):
            return "Could not automatically interpret the test results."
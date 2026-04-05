
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import chi2_contingency
from django.core.files.storage import default_storage
from typing import Any, Dict, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from saiha.ai_agents.tools.plot_utils import PlotUtils


class ChiSquareTool(BaseAnalysisTool):
    """
    A tool to perform a Chi-Square Test of Independence between two categorical variables.
    """

    @property
    def name(self) -> str:
        return "chi_square_test"

    @property
    def description(self) -> str:
        return "Tests for a statistically significant association between two categorical variables."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(
            ToolParameter(
                name="variable1", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                label="Categorical Variable 1", description="Select the first categorical column.", required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="variable2", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                label="Categorical Variable 2", description="Select the second categorical column.", required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="alpha", parameter_type=ParameterType.SELECT,
                label="Significance Level (α)", description="The threshold for determining statistical significance.",
                required=True, default_value="0.05",
                options=[
                    {"value": "0.05", "label": "0.05"},
                    {"value": "0.01", "label": "0.01"},
                    {"value": "0.10", "label": "0.10"},
                ]
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_stacked_bar", parameter_type=ParameterType.CHECKBOX,
                label="Generate Stacked Bar Chart", description="Visualize the relationship using a stacked bar chart.",
                required=False, default_value=False
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_heatmap", parameter_type=ParameterType.CHECKBOX,
                label="Generate Heatmap", description="Visualize the contingency table as a heatmap.",
                required=False, default_value=False
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_mosaic_plot", parameter_type=ParameterType.CHECKBOX,
                label="Generate Mosaic Plot", description="Visualize the association using a mosaic plot.",
                required=False, default_value=False
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            var1 = parameters.get("variable1")
            var2 = parameters.get("variable2")
            alpha = float(parameters.get("alpha", 0.05))
            gen_stacked = parameters.get("generate_stacked_bar", False)
            gen_heatmap = parameters.get("generate_heatmap", False)
            gen_mosaic = parameters.get("generate_mosaic_plot", False)

            if not var1 or not var2:
                return {"status": "error", "summary": "Both categorical variables are required."}
            if var1 == var2:
                return {"status": "error", "summary": "Please select two different variables."}

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            # Create contingency table
            contingency_table = pd.crosstab(df[var1], df[var2])

            # Perform Chi-Square test
            chi2, p, dof, expected = chi2_contingency(contingency_table)

            is_significant = p < alpha
            summary = (
                f"The Chi-Square test for independence between '{var1}' and '{var2}' was performed. "
                f"The result is {'statistically significant' if is_significant else 'not statistically significant'} "
                f"at the {alpha} level (p-value = {p:.4f})."
            )

            # Prepare results tables
            sections = []
            artifacts = []
            
            # Main results
            sections.append({
                'type': 'table', 'title': 'Chi-Square Test Results',
                'headers': ['Statistic', 'Value'],
                'data': [
                    ['Chi-Square Statistic', f"{chi2:.4f}"],
                    ['P-Value', f"{p:.6f}"],
                    ['Degrees of Freedom', dof],
                    ['Is Significant', 'Yes' if is_significant else 'No']
                ]
            })

            # Observed Frequencies
            observed_df = contingency_table.reset_index()
            sections.append({
                'type': 'table', 'title': 'Observed Frequencies (Contingency Table)',
                'headers': observed_df.columns.tolist(),
                'data': observed_df.values.tolist()
            })

            # Expected Frequencies
            expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns).round(2).reset_index()
            sections.append({
                'type': 'table', 'title': 'Expected Frequencies',
                'headers': expected_df.columns.tolist(),
                'data': expected_df.values.tolist(),
                'footer': "Expected frequencies are calculated under the assumption that the two variables are independent."
            })

            # --- Optional Visualizations ---
            with PlotUtils.setup_plotting():
                # Stacked Bar Chart
                if gen_stacked:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    contingency_table.plot(kind='bar', stacked=True, ax=ax)
                    ax.set_title(f'Stacked Bar Chart of {var2} by {var1}')
                    ax.set_xlabel(var1)
                    ax.set_ylabel('Count')
                    plt.tight_layout()
                    artifacts.append({
                        "type": "plot", "id": "chi_square_stacked_bar",
                        "title": "Stacked Bar Chart", "content": PlotUtils.fig_to_base64(fig)
                    })

                # Heatmap
                if gen_heatmap:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='viridis', ax=ax)
                    ax.set_title(f'Heatmap of Observed Frequencies')
                    plt.tight_layout()
                    artifacts.append({
                        "type": "plot", "id": "chi_square_heatmap",
                        "title": "Heatmap of Frequencies", "content": PlotUtils.fig_to_base64(fig)
                    })

                # Mosaic Plot
                if gen_mosaic:
                    try:
                        fig, _ = mosaic(contingency_table.stack(), title=f'Mosaic Plot for {var1} and {var2}', gap=0.02)
                        fig.set_size_inches(10, 7)
                        plt.tight_layout()
                        artifacts.append({
                            "type": "plot", "id": "chi_square_mosaic",
                            "title": "Mosaic Plot", "content": PlotUtils.fig_to_base64(fig)
                        })
                    except Exception as e:
                        sections.append({
                            'type': 'text',
                            'title': 'Mosaic Plot Failed',
                            'content': f"Could not generate mosaic plot. This can happen with very sparse data. Error: {str(e)}"
                        })


            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": parameters},
            }

        except ValueError as ve:
            # Catch errors from chi2_contingency, e.g., low expected frequencies
            return {"status": "error", "summary": f"Statistical validation failed: {str(ve)}"}
        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Provides a simple, rule-based interpretation of the Chi-Square test results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            var1 = params.get('variable1', 'Variable 1')
            var2 = params.get('variable2', 'Variable 2')

            p_value = None
            for section in results.get('sections', []):
                if section.get('title') == 'Chi-Square Test Results':
                    for row in section.get('data', []):
                        if row[0] == 'P-Value':
                            p_value = float(row[1])
                            break
            
            if p_value is None:
                return "Could not automatically determine the p-value for interpretation."

            return f"Since the p-value ({p_value:.4f}) is less than the significance level (α = {alpha}), we reject the null hypothesis. There is a statistically significant association between '{var1}' and '{var2}'." if p_value < alpha else f"Since the p-value ({p_value:.4f}) is greater than or equal to the significance level (α = {alpha}), we fail to reject the null hypothesis. There is not enough evidence to conclude that there is a significant association between '{var1}' and '{var2}'."
        except Exception as e:
            return f"Could not automatically interpret the Chi-Square test results due to an error: {e}"
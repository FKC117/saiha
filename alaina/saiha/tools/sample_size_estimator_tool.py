# d:/quantly/quanta/quantalytics/ai_agents/tools/sample_size_estimator_tool.py

from statsmodels.stats.power import TTestIndPower, TTestPower, FTestAnovaPower, GofChisquarePower
from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class SampleSizeEstimatorTool(BaseAnalysisTool):
    """
    A tool to estimate the required sample size for a study.
    """

    @property
    def name(self) -> str:
        return "sample_size_estimator"

    @property
    def description(self) -> str:
        return "Estimates the required sample size for a study based on power, effect size, and significance level."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="effect_size", parameter_type=ParameterType.NUMBER,
            label="Effect Size (e.g., Cohen's d, f, or w)",
            description="Standardized effect size. T-Test (d): 0.2, 0.5, 0.8. ANOVA (f): 0.1, 0.25, 0.4. Chi-Square (w): 0.1, 0.3, 0.5.",
            required=True, default_value=0.5,
            validation_rules={"min": 0.01, "step": 0.01}
        ))
        params.add_parameter(ToolParameter(
            name="power", parameter_type=ParameterType.NUMBER,
            label="Statistical Power (1 - β)",
            description="The probability of finding an effect if it exists. Typically 0.80 or higher.",
            required=True, default_value=0.80,
            validation_rules={"min": 0.01, "max": 0.99, "step": 0.01}
        ))
        params.add_parameter(ToolParameter(
            name="alpha", parameter_type=ParameterType.SELECT,
            label="Significance Level (α)",
            description="The probability of a Type I error (false positive).",
            required=True, default_value="0.05",
            options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
        ))
        params.add_parameter(ToolParameter(
            name="test_type", parameter_type=ParameterType.SELECT,
            label="Test Type",
            description="The type of t-test you plan to use.",
            required=True, default_value="two_sample",
            options=[
                {"value": "two_sample", "label": "Two-Sample Independent T-Test"},
                {"value": "one_sample", "label": "One-Sample T-Test"},
                {"value": "paired", "label": "Paired T-Test"},
                {"value": "anova", "label": "One-Way ANOVA"},
                {"value": "chi_square", "label": "Chi-Square Test of Independence"},
            ]
        ))
        params.add_parameter(ToolParameter(
            name="k_groups", parameter_type=ParameterType.NUMBER,
            label="Number of Groups (for ANOVA)",
            description="The total number of groups in your ANOVA design (e.g., 3).",
            required=False, help_text="Only used when Test Type is One-Way ANOVA."
        ))
        params.add_parameter(ToolParameter(
            name="num_rows", parameter_type=ParameterType.NUMBER,
            label="Number of Rows (for Chi-Square)",
            description="The number of categories in the first variable for the Chi-Square test.",
            required=False, help_text="Only used for Chi-Square Test."
        ))
        params.add_parameter(ToolParameter(
            name="num_cols", parameter_type=ParameterType.NUMBER,
            label="Number of Columns (for Chi-Square)",
            description="The number of categories in the second variable for the Chi-Square test.",
            required=False, help_text="Only used for Chi-Square Test."
        ))
        params.add_parameter(ToolParameter(
            name="show_multiple_effects", parameter_type=ParameterType.CHECKBOX,
            label="Plot Multiple Effect Sizes",
            description="Show power curves for small, medium, and large effect sizes for comparison.",
            required=False, default_value=False
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            effect_size = float(parameters.get("effect_size", 0.5))
            power = float(parameters.get("power", 0.8))
            alpha = float(parameters.get("alpha", 0.05))
            test_type = parameters.get("test_type", "two_sample")

            # Handle optional integer fields that may be empty strings
            k_groups_str = parameters.get("k_groups")
            num_rows_str = parameters.get("num_rows")
            num_cols_str = parameters.get("num_cols")
            k_groups = int(k_groups_str) if k_groups_str else 3
            num_rows = int(num_rows_str) if num_rows_str else 2
            num_cols = int(num_cols_str) if num_cols_str else 2

            if test_type == "two_sample":
                power_analysis = TTestIndPower()
                required_n = power_analysis.solve_power(
                    effect_size=effect_size, power=power, alpha=alpha, alternative='two-sided'
                )
                sample_size = int(np.ceil(required_n))
                summary = f"For a two-sample t-test, to detect an effect size (d) of {effect_size} with {power*100:.0f}% power at α={alpha}, you need **{sample_size} participants per group**."
                result_label = "Required Sample Size (per group)"
            elif test_type == "anova":
                power_analysis = FTestAnovaPower()
                required_n = power_analysis.solve_power(
                    effect_size=effect_size, power=power, alpha=alpha, k_groups=k_groups
                )
                sample_size = int(np.ceil(required_n))
                total_n = sample_size * k_groups
                summary = f"For a One-Way ANOVA with {k_groups} groups, to detect an effect size (f) of {effect_size} with {power*100:.0f}% power at α={alpha}, you need **{sample_size} participants per group** (total of {total_n})."
                result_label = "Required Sample Size (per group)"
            elif test_type == "chi_square":
                power_analysis = GofChisquarePower()
                df = (num_rows - 1) * (num_cols - 1)
                if df == 0:
                    return {"status": "error", "summary": "For a Chi-Square test, both rows and columns must be 2 or greater."}
                required_n = power_analysis.solve_power(
                    effect_size=effect_size, power=power, alpha=alpha, n_bins=df + 1
                )
                sample_size = int(np.ceil(required_n))
                summary = f"For a Chi-Square test on a {num_rows}x{num_cols} table, to detect an effect size (w) of {effect_size} with {power*100:.0f}% power at α={alpha}, you need a total sample size of **{sample_size}**."
                result_label = "Required Sample Size (total)"
            else: # one_sample or paired
                power_analysis = TTestPower()
                required_n = power_analysis.solve_power(
                    effect_size=effect_size, power=power, alpha=alpha, alternative='two-sided'
                )
                sample_size = int(np.ceil(required_n))
                summary = f"For a {test_type.replace('_', ' ')} t-test, to detect an effect size (d) of {effect_size} with {power*100:.0f}% power at α={alpha}, you need a total of **{sample_size} observations**."
                result_label = "Required Sample Size (total)"

            artifacts: List[Dict[str, Any]] = []
            sections = [{
                'type': 'table', 'title': 'Sample Size Estimation Results',
                'headers': ['Parameter', 'Value'],
                'data': [
                    ['Effect Size', f"{effect_size:.2f}"],
                    ['Statistical Power', f"{power:.2f}"],
                    ['Significance Level (α)', f"{alpha:.2f}"],
                    ['Test Type', test_type.replace('_', ' ').title()],
                    [result_label, sample_size],
                ]
            }]

            # --- Generate Power Curve Plot ---
            with PlotUtils.setup_plotting():
                fig = None
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Determine plot range and labels
                    is_per_group = test_type in ['two_sample', 'anova']
                    plot_label = "Sample Size (per group)" if is_per_group else "Total Sample Size"
                    max_n = max(50, sample_size * 2)
                    n_range = np.linspace(10, max_n, 100).astype(int)
                    
                    # Define effect sizes for plotting
                    show_multiple = str(parameters.get("show_multiple_effects")).lower() in ('true', 'on', '1')

                    if show_multiple:
                        effect_size_map = {
                            't_test': {'Small': 0.2, 'Medium': 0.5, 'Large': 0.8},
                            'anova': {'Small': 0.1, 'Medium': 0.25, 'Large': 0.4},
                            'chi_square': {'Small': 0.1, 'Medium': 0.3, 'Large': 0.5}
                        }
                        plot_effects = effect_size_map.get('t_test' if 'sample' in test_type else test_type, {})

                        # Plot a curve for each effect size
                        for label, es_val in plot_effects.items():
                            if test_type == "two_sample":
                                powers = power_analysis.power(effect_size=es_val, nobs1=n_range, alpha=alpha, ratio=1.0)
                            elif test_type == "anova":
                                powers = power_analysis.power(effect_size=es_val, nobs=n_range * k_groups, alpha=alpha, k_groups=k_groups)
                            elif test_type == "chi_square":
                                powers = power_analysis.power(effect_size=es_val, nobs=n_range, alpha=alpha, n_bins=(num_rows - 1) * (num_cols - 1) + 1)
                            else: # one_sample or paired
                                powers = power_analysis.power(effect_size=es_val, nobs=n_range, alpha=alpha)
                            
                            is_user_curve = np.isclose(es_val, effect_size)
                            ax.plot(n_range, powers, label=f'{label} Effect (es={es_val})', linewidth=2.5 if is_user_curve else 1.5, alpha=1.0 if is_user_curve else 0.7)
                        title = 'Power vs. Sample Size for Different Effect Sizes'
                    else:
                        # Plot only the user-specified effect size
                        if test_type == "two_sample":
                            powers = power_analysis.power(effect_size=effect_size, nobs1=n_range, alpha=alpha, ratio=1.0)
                        elif test_type == "anova":
                            powers = power_analysis.power(effect_size=effect_size, nobs=n_range * k_groups, alpha=alpha, k_groups=k_groups)
                        elif test_type == "chi_square":
                            powers = power_analysis.power(effect_size=effect_size, nobs=n_range, alpha=alpha, n_bins=(num_rows - 1) * (num_cols - 1) + 1)
                        else: # one_sample or paired
                            powers = power_analysis.power(effect_size=effect_size, nobs=n_range, alpha=alpha)
                        ax.plot(n_range, powers, label=f'Power Curve (es={effect_size})', linewidth=2)
                        title = 'Power vs. Sample Size'

                    # Add reference lines for the user's specific calculation
                    ax.axhline(power, color='r', linestyle='--', label=f'Target Power ({power:.2f})')
                    ax.axvline(sample_size, color='g', linestyle='--', label=f'Required N ({sample_size})')
                    
                    if test_type == 'anova': title += f' (k={k_groups})'
                    elif test_type == 'chi_square': title += f' ({num_rows}x{num_cols} table)'

                    ax.set_title(title)
                    ax.set_xlabel(plot_label)
                    ax.set_ylabel('Statistical Power')
                    ax.set_ylim(0, 1.05)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    artifacts.append({"type": "plot", "id": "power_curve_plot", "title": "Power Curve", "content": PlotUtils.fig_to_base64(fig)})
                except Exception as plot_ex:
                    sections.append({"type": "text", "title": "Power Curve Failed", "content": str(plot_ex)})
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
        Provides a formal interpretation of the sample size estimation results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            effect_size = float(params.get('effect_size', 0.5))
            power = float(params.get('power', 0.8))
            alpha = float(params.get('alpha', 0.05))
            test_type = params.get('test_type', 'two_sample')
            show_multiple = str(params.get("show_multiple_effects")).lower() in ('true', 'on', '1')

            sample_size_str = "N/A"
            result_label = "Required Sample Size"
            for section in results.get('sections', []):
                if section.get('title') == 'Sample Size Estimation Results':
                    for row in section.get('data', []):
                        if 'Required Sample Size' in row[0]:
                            sample_size_str = str(row[1])
                            result_label = row[0]
                            break
            
            if sample_size_str == "N/A":
                return "Could not automatically determine the calculated sample size for interpretation."

            # Describe effect size
            if test_type == 'anova': # Cohen's f for ANOVA
                if effect_size <= 0.15: effect_desc = "small"
                elif effect_size <= 0.35: effect_desc = "medium"
                else: effect_desc = "large"
            elif test_type == 'chi_square': # Cohen's w for Chi-Square
                if effect_size <= 0.2: effect_desc = "small"
                elif effect_size <= 0.4: effect_desc = "medium"
                else: effect_desc = "large"
            else: # Cohen's d for t-tests
                if effect_size <= 0.3: effect_desc = "small"
                elif effect_size <= 0.7: effect_desc = "medium"
                else: effect_desc = "large"

            interpretation_parts = [
                f"This calculation helps you design a statistically robust study. You specified that you want to detect a '{effect_desc}' effect size ({effect_size:.2f}) with a statistical power of {power*100:.0f}%.",
                f"This means you want an {power*100:.0f}% chance of finding a statistically significant result if the effect truly exists.",
                f"Given a significance level (α) of {alpha}, the analysis determined that the {result_label.lower()} is **{sample_size_str}**.",
                "Conducting your study with this sample size ensures that your findings are more likely to be reliable and not just due to random chance.",
            ]

            if show_multiple:
                interpretation_parts.append("The Power Curve plot visualizes this relationship, showing how the required sample size changes for small, medium, and large effects.")
            else:
                interpretation_parts.append("The Power Curve plot visualizes this relationship, showing how power increases as sample size grows.")

            return " ".join(interpretation_parts)

        except (ValueError, TypeError, IndexError, KeyError):
            return "Could not automatically interpret the sample size estimation results."
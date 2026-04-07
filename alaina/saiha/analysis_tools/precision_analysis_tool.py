from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower, TTestPower, FTestAnovaPower, GofChisquarePower

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class PrecisionAnalysisTool(BaseAnalysisTool):
    """
    A tool to calculate the Minimum Detectable Effect (Precision/Sensitivity Analysis).
    """

    @property
    def name(self) -> str:
        return "precision_analysis"

    @property
    def description(self) -> str:
        return "Calculate the Minimum Detectable Effect (Sensitivity Analysis) for a fixed sample size."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        
        params.add_parameter(ToolParameter(
            name="sample_size", parameter_type=ParameterType.NUMBER,
            label="Sample Size (per group/total)",
            description="The number of participants you have (per group for T-Test/ANOVA, total for others).",
            required=True, default_value=50,
            validation_rules={"min": 2, "step": 1}
        ))
        
        params.add_parameter(ToolParameter(
            name="power", parameter_type=ParameterType.NUMBER,
            label="Target Power (1 - β)",
            description="The desired probability of detecting an effect.",
            required=True, default_value=0.80,
            validation_rules={"min": 0.01, "max": 0.99, "step": 0.01}
        ))
        
        params.add_parameter(ToolParameter(
            name="alpha", parameter_type=ParameterType.SELECT,
            label="Significance Level (α)",
            description="The probability of a Type I error.",
            required=True, default_value="0.05",
            options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
        ))
        
        params.add_parameter(ToolParameter(
            name="test_type", parameter_type=ParameterType.SELECT,
            label="Test Type",
            description="The statistical test you plan to use.",
            required=True, default_value="two_sample",
            options=[
                {"value": "two_sample", "label": "Two-Sample Independent T-Test"},
                {"value": "one_sample", "label": "One-Sample T-Test"},
                {"value": "paired", "label": "Paired T-Test"},
                {"value": "anova", "label": "One-Way ANOVA"},
                {"value": "chi_square", "label": "Chi-Square Test"},
            ]
        ))
        
        params.add_parameter(ToolParameter(
            name="k_groups", parameter_type=ParameterType.NUMBER,
            label="Number of Groups (ANOVA)",
            required=False, default_value=3,
            help_text="Only used for ANOVA."
        ))
        
        params.add_parameter(ToolParameter(
            name="degrees_of_freedom", parameter_type=ParameterType.NUMBER,
            label="Degrees of Freedom (Chi-Square)",
            required=False, default_value=1,
            help_text="df = (rows-1)*(cols-1). Only used for Chi-Square."
        ))

        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            params = kwargs
            sample_size = int(params.get("sample_size", 50))
            power = float(params.get("power", 0.8))
            alpha = float(params.get("alpha", 0.05))
            test_type = params.get("test_type", "two_sample")
            k_groups = int(params.get("k_groups", 3))
            df = int(params.get("degrees_of_freedom", 1))

            if test_type == "two_sample":
                power_analysis = TTestIndPower()
                effect_size = float(power_analysis.solve_power(
                    nobs1=sample_size, power=power, alpha=alpha, ratio=1.0, alternative='two-sided'
                ))
                metric = "Cohen's d"
            elif test_type == "anova":
                power_analysis = FTestAnovaPower()
                effect_size = float(power_analysis.solve_power(
                    nobs=sample_size, power=power, alpha=alpha, k_groups=k_groups
                ))
                metric = "Cohen's f"
            elif test_type == "chi_square":
                power_analysis = GofChisquarePower()
                effect_size = float(power_analysis.solve_power(
                    nobs=sample_size, power=power, alpha=alpha, n_bins=df+1
                ))
                metric = "Cohen's w"
            else: # one_sample or paired
                power_analysis = TTestPower()
                effect_size = float(power_analysis.solve_power(
                    nobs=sample_size, power=power, alpha=alpha, alternative='two-sided'
                ))
                metric = "Cohen's d"

            summary = f"With {sample_size} participants (per group/total) and {power*100:.0f}% power, the smallest effect you can calculate is **{metric} = {effect_size:.3f}**."
            
            # Interpretation
            if test_type == 'anova':
                if effect_size < 0.1: size_label = "Negligible"
                elif effect_size < 0.25: size_label = "Small"
                elif effect_size < 0.4: size_label = "Medium"
                else: size_label = "Large"
            elif test_type == 'chi_square':
                 if effect_size < 0.1: size_label = "Negligible"
                 elif effect_size < 0.3: size_label = "Small"
                 elif effect_size < 0.5: size_label = "Medium"
                 else: size_label = "Large"
            else:
                if effect_size < 0.2: size_label = "Negligible"
                elif effect_size < 0.5: size_label = "Small"
                elif effect_size < 0.8: size_label = "Medium"
                else: size_label = "Large"
            
            summary += f" This relates to a **{size_label}** effect size."

            sections = [{
                'type': 'table', 'title': 'Precision Analysis Results',
                'headers': ['Parameter', 'Value'],
                'data': [
                    ['Sample Size (N)', sample_size],
                    ['Target Power', power],
                    ['Alpha', alpha],
                    ['Min. Detectable Effect', f"**{effect_size:.3f}** ({metric})"],
                    ['Sensitivity', size_label]
                ]
            }]
            
            # Plot
            artifacts = []
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(8, 5))
                # Plot range of N around user's N
                n_start = max(5, int(sample_size * 0.2))
                n_end = int(sample_size * 2)
                n_range = np.linspace(n_start, n_end, 50).astype(int)
                
                effects = []
                for n_val in n_range:
                    if test_type == "two_sample":
                        es = float(power_analysis.solve_power(nobs1=n_val, power=power, alpha=alpha, ratio=1.0, alternative='two-sided'))
                    elif test_type == "anova":
                        es = float(power_analysis.solve_power(nobs=n_val, power=power, alpha=alpha, k_groups=k_groups))
                    elif test_type == "chi_square":
                        es = float(power_analysis.solve_power(nobs=n_val, power=power, alpha=alpha, n_bins=df+1))
                    else: # one_sample or paired
                        es = float(power_analysis.solve_power(nobs=n_val, power=power, alpha=alpha, alternative='two-sided'))
                    effects.append(es)
                
                ax.plot(n_range, effects, label=f'MDE at {power*100:.0f}% Power', color='purple')
                ax.axvline(sample_size, color='r', linestyle='--', label=f'Your N={sample_size}')
                ax.axhline(effect_size, color='g', linestyle='--', label=f'MDE={effect_size:.2f}')
                
                ax.set_title(f'Minimum Detectable Effect vs. Sample Size ({metric})')
                ax.set_xlabel('Sample Size')
                ax.set_ylabel(f'Effect Size ({metric})')
                ax.legend()
                ax.grid(alpha=0.3)
                
                artifacts.append({"type": "plot", "id": "mde_plot", "title": "Precision Plot", "content": PlotUtils.fig_to_base64(fig)})
                plt.close(fig)

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": params}
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"Error: {str(e)}"}

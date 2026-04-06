from typing import Any, Dict, List
import numpy as np
from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType

class EffectSizeCalculatorTool(BaseAnalysisTool):
    """
    A standalone calculator for Effect Sizes (Cohen's d, Eta Squared).
    """

    @property
    def name(self) -> str:
        return "effect_size_calculator"

    @property
    def description(self) -> str:
        return "Calculate Cohen's d (T-Tests) or Eta-Squared (ANOVA) from summary statistics."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        
        params.add_parameter(ToolParameter(
            name="calculator_type", parameter_type=ParameterType.SELECT,
            label="Calculation Type",
            description="Choose the type of effect size to calculate.",
            required=True, default_value="cohens_d",
            options=[
                {"value": "cohens_d", "label": "Cohen's d (Two Means being compared)"},
                {"value": "eta_squared", "label": "Eta Squared (ANOVA F-statistic conversion)"},
            ]
        ))
        
        # --- Cohen's d Parameters ---
        params.add_parameter(ToolParameter(
            name="mean1", parameter_type=ParameterType.NUMBER,
            label="Mean 1 (Group 1)",
            required=False, help_text="Required for Cohen's d"
        ))
        params.add_parameter(ToolParameter(
            name="sd1", parameter_type=ParameterType.NUMBER,
            label="Standard Deviation 1",
            required=False, help_text="Required for Cohen's d"
        ))
        params.add_parameter(ToolParameter(
            name="n1", parameter_type=ParameterType.NUMBER,
            label="Sample Size 1",
            required=False, help_text="Required for Cohen's d (for pooled SD)"
        ))
        
        params.add_parameter(ToolParameter(
            name="mean2", parameter_type=ParameterType.NUMBER,
            label="Mean 2 (Group 2)",
            required=False, help_text="Required for Cohen's d"
        ))
        params.add_parameter(ToolParameter(
            name="sd2", parameter_type=ParameterType.NUMBER,
            label="Standard Deviation 2",
            required=False, help_text="Required for Cohen's d"
        ))
        params.add_parameter(ToolParameter(
            name="n2", parameter_type=ParameterType.NUMBER,
            label="Sample Size 2",
            required=False, help_text="Required for Cohen's d (for pooled SD)"
        ))

        # --- Eta Squared Parameters ---
        params.add_parameter(ToolParameter(
            name="f_statistic", parameter_type=ParameterType.NUMBER,
            label="F-Statistic",
            required=False, help_text="Required for Eta Squared"
        ))
        params.add_parameter(ToolParameter(
            name="df1", parameter_type=ParameterType.NUMBER,
            label="Degrees of Freedom 1 (Between)",
            required=False, help_text="Required for Eta Squared"
        ))
        params.add_parameter(ToolParameter(
            name="df2", parameter_type=ParameterType.NUMBER,
            label="Degrees of Freedom 2 (Error)",
            required=False, help_text="Required for Eta Squared"
        ))

        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            params = kwargs
            calc_type = params.get("calculator_type", "cohens_d")
            
            sections = []
            summary = ""
            artifacts = []

            if calc_type == "cohens_d":
                m1 = float(params.get("mean1") or 0)
                m2 = float(params.get("mean2") or 0)
                sd1 = float(params.get("sd1") or 1)
                sd2 = float(params.get("sd2") or 1)
                n1 = float(params.get("n1") or 10)
                n2 = float(params.get("n2") or 10)

                # Calculate Pooled Standard Deviation
                # SD_pooled = sqrt( ((n1-1)SD1^2 + (n2-1)SD2^2) / (n1+n2-2) )
                numerator = ((n1 - 1) * sd1**2) + ((n2 - 1) * sd2**2)
                denominator = n1 + n2 - 2
                
                if denominator <= 0:
                    return {"status": "error", "summary": "Sample sizes must be greater than 1."}
                    
                pooled_sd = np.sqrt(numerator / denominator)
                
                # Cohen's d = (Mean1 - Mean2) / Pooled_SD
                cohens_d = (m1 - m2) / pooled_sd
                
                # Interpretation
                d_abs = abs(cohens_d)
                if d_abs < 0.2: interpretation = "Negligible"
                elif d_abs < 0.5: interpretation = "Small"
                elif d_abs < 0.8: interpretation = "Medium"
                else: interpretation = "Large"

                summary = f"Based on the provided statistics, **Cohen's d is {cohens_d:.2f}**, which is considered a **{interpretation}** effect size."
                
                sections.append({
                    'type': 'table', 'title': "Cohen's d Results",
                    'headers': ['Metric', 'Value'],
                    'data': [
                        ['Mean 1', m1], ['SD 1', sd1],
                        ['Mean 2', m2], ['SD 2', sd2],
                        ['Pooled SD', f"{pooled_sd:.3f}"],
                        ["Cohen's d", f"**{cohens_d:.3f}**"],
                        ["Interpretation", interpretation]
                    ]
                })

            elif calc_type == "eta_squared":
                f_val = float(params.get("f_statistic") or 1)
                df1 = float(params.get("df1") or 1)
                df2 = float(params.get("df2") or 10)
                
                # Eta Squared = (F * df1) / (F * df1 + df2)
                # Cohen's f = sqrt(Eta / (1 - Eta))
                
                eta_sq = (f_val * df1) / ((f_val * df1) + df2)
                cohens_f = np.sqrt(eta_sq / (1 - eta_sq)) if eta_sq < 1 else 999.0

                if eta_sq < 0.01: interpretation = "Negligible"
                elif eta_sq < 0.06: interpretation = "Small"
                elif eta_sq < 0.14: interpretation = "Medium"
                else: interpretation = "Large"

                summary = f"Based on F={f_val}, **Eta Squared is {eta_sq:.3f}** (Cohen's f ≈ {cohens_f:.2f}), which is a **{interpretation}** effect size."
                
                sections.append({
                    'type': 'table', 'title': "Eta Squared / Cohen's f Results",
                    'headers': ['Metric', 'Value'],
                    'data': [
                        ['F-Statistic', f_val],
                        ['Eta Squared (η²)', f"**{eta_sq:.3f}**"],
                        ["Cohen's f", f"{cohens_f:.3f}"],
                        ["Interpretation", interpretation]
                    ]
                })

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": params}
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"Calculation error: {str(e)}"}

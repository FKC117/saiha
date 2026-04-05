"""
Mediation Analysis Tool
Tests whether the effect of X on Y is mediated by a third variable M.
Implements the Baron & Kenny causal steps approach.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class MediationAnalysisTool(BaseAnalysisTool):
    """Tool for Mediation Analysis."""

    @property
    def name(self) -> str:
        return "mediation_analysis"

    @property
    def description(self) -> str:
        return "Test if M mediates the relationship between X and Y (Baron & Kenny)."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="mediation_analysis")
        params.add_parameter(
            ToolParameter(
                name="independent_variable",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Independent Variable (X)",
                description="The predictor variable.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="mediator_variable",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Mediator Variable (M)",
                description="The potential mediator.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="dependent_variable",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Dependent Variable (Y)",
                description="The outcome variable.",
                required=True
            )
        )
        return params

    def _run_regression(self, df: pd.DataFrame, x_col: str, y_col: str, m_col: Optional[str] = None):
        """Run OLS regression: Y ~ X + [M]"""
        X = df[[x_col]]
        if m_col:
            X = df[[x_col, m_col]]
        
        X = sm.add_constant(X, has_constant='add')
        if 'const' not in X.columns:
            X['const'] = 1.0
        y = df[y_col].astype(float)
        X = X.astype(float)
        
        model = sm.OLS(y, X).fit()
        return model

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters
            parameters = kwargs
            iv_col = parameters.get("independent_variable")
            mv_col = parameters.get("mediator_variable")
            dv_col = parameters.get("dependent_variable")

            if not all([iv_col, mv_col, dv_col]):
                return {"status": "error", "summary": "Missing required variables (X, M, Y)."}
            
            if iv_col == mv_col or iv_col == dv_col or mv_col == dv_col:
                 return {"status": "error", "summary": "X, M, and Y must be different variables."}

            df = self.load_dataset(columns=[iv_col, mv_col, dv_col])
            df_clean = df.dropna()

            if df_clean.empty:
                return {"status": "error", "summary": "No valid data after removing missing values."}

            # 2. Step 1: Total Effect (X -> Y)
            model_c = self._run_regression(df_clean, iv_col, dv_col)
            c_path = model_c.params[iv_col]
            c_pvald = model_c.pvalues[iv_col]
            
            # 3. Step 2: X -> M (Path a)
            model_a = self._run_regression(df_clean, iv_col, mv_col)
            a_path = model_a.params[iv_col]
            a_pval = model_a.pvalues[iv_col]

            # 4. Step 3: X + M -> Y (Path b and c')
            model_b = self._run_regression(df_clean, iv_col, dv_col, m_col=mv_col)
            b_path = model_b.params[mv_col]
            b_pval = model_b.pvalues[mv_col]
            c_prime_path = model_b.params[iv_col]
            c_prime_pval = model_b.pvalues[iv_col]

            # 5. Interpretation
            # Mediation exists if:
            # - c is significant (sometimes relaxed in modern theory)
            # - a is significant
            # - b is significant
            
            significance_level = 0.05
            is_c_sig = c_pvald < significance_level
            is_a_sig = a_pval < significance_level
            is_b_sig = b_pval < significance_level
            is_c_prime_sig = c_prime_pval < significance_level
            
            conclusion = ""
            if is_a_sig and is_b_sig:
                if not is_c_prime_sig:
                    conclusion = "Full Mediation suggested (a and b significant, c' not significant)."
                else:
                    conclusion = "Partial Mediation suggested (a, b, and c' significant)."
            else:
                 conclusion = "No Mediation detected (one of the paths 'a' or 'b' is not significant)."


            if not is_c_sig:
                conclusion += " Note: Total effect (X->Y) was not significant initially."

            # Enhanced Narrative
            indirect_effect = a_path * b_path
            proportion_mediated = (indirect_effect / c_path) * 100 if c_path != 0 else 0
            
            narrative = f"\n\n### Detailed Interpretation:\n"
            narrative += f"1. **Total Effect**: The overall effect of {iv_col} on {dv_col} is {c_path:.3f} (p={c_pvald:.3f}).\n"
            narrative += f"2. **Indirect Effect**: {iv_col} influences {dv_col} through {mv_col} by an estimated {indirect_effect:.3f}.\n"
            narrative += f"3. **Direct Effect**: After controlling for {mv_col}, the remaining effect of {iv_col} on {dv_col} is {c_prime_path:.3f} (p={c_prime_pval:.3f}).\n"
            
            if conclusion.startswith("Full Mediation") or conclusion.startswith("Partial Mediation"):
                 narrative += f"4. **Mediation Portion**: Approximately {proportion_mediated:.1f}% of the total effect is mediated by {mv_col}."
            
            conclusion += narrative


            artifacts = []
            sections = []

            # Results Table
            sections.append({
                'type': 'table',
                'title': 'Mediation Steps (Baron & Kenny)',
                'headers': ['Path', 'Coeff', 'P-Value', 'Significant?'],
                'data': [
                    ['Total Effect (c): X -> Y', f"{c_path:.4f}", f"{c_pvald:.4f}", "Yes" if is_c_sig else "No"],
                    ['Path a: X -> M', f"{a_path:.4f}", f"{a_pval:.4f}", "Yes" if is_a_sig else "No"],
                    ['Path b: M -> Y (controlling X)', f"{b_path:.4f}", f"{b_pval:.4f}", "Yes" if is_b_sig else "No"],
                    ['Direct Effect (c\'): X -> Y (controlling M)', f"{c_prime_path:.4f}", f"{c_prime_pval:.4f}", "Yes" if is_c_prime_sig else "No"]
                ]
            })
            
            sections.append({
                'type': 'text',
                'title': 'Conclusion',
                'content': conclusion
            })

            # Sobel Test (Manual calculation approx)
            sa = model_a.bse[iv_col]
            sb = model_b.bse[mv_col]
            
            # Sobel statistic = a*b / sqrt(b^2*sa^2 + a^2*sb^2)
            sobel_top = a_path * b_path
            sobel_bottom = np.sqrt(b_path**2 * sa**2 + a_path**2 * sb**2)
            sobel_z = sobel_top / sobel_bottom
            from scipy import stats
            sobel_p = (1 - stats.norm.cdf(abs(sobel_z))) * 2 
            
            sections.append({
                'type': 'text',
                'title': 'Sobel Test (Significance of Mediation Effect)',
                'content': f"Z-score: {sobel_z:.4f}, P-value: {sobel_p:.4f}. {'Significant' if sobel_p < 0.05 else 'Not Significant'} at p<0.05."
            })

            # Visualizations: Path Diagram
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Coordinates
                x_coord, y_coord = 0.1, 0.1
                m_coord, m_height = 0.5, 0.8
                y_coord_x, y_coord_y = 0.9, 0.1
                
                # Nodes
                ax.text(x_coord, y_coord, iv_col, ha='center', va='center', bbox=dict(boxstyle="round", fc="w"))
                ax.text(m_coord, m_height, mv_col, ha='center', va='center', bbox=dict(boxstyle="round", fc="w"))
                ax.text(y_coord_x, y_coord_y, dv_col, ha='center', va='center', bbox=dict(boxstyle="round", fc="w"))
                
                # Arrows
                # X -> M (a)
                ax.annotate("", xy=(m_coord, m_height-0.05), xytext=(x_coord+0.05, y_coord+0.05),
                            arrowprops=dict(arrowstyle="->", lw=1.5))
                ax.text((x_coord+m_coord)/2 - 0.05, (y_coord+m_height)/2, f"a = {a_path:.3f}{'*' if is_a_sig else ''}", fontsize=10)
                
                # M -> Y (b)
                ax.annotate("", xy=(y_coord_x-0.05, y_coord_y+0.05), xytext=(m_coord, m_height-0.05),
                            arrowprops=dict(arrowstyle="->", lw=1.5))
                ax.text((m_coord+y_coord_x)/2 + 0.05, (m_height+y_coord_y)/2, f"b = {b_path:.3f}{'*' if is_b_sig else ''}", fontsize=10)
                
                # X -> Y (c')
                ax.annotate("", xy=(y_coord_x-0.05, y_coord_y), xytext=(x_coord+0.05, y_coord),
                            arrowprops=dict(arrowstyle="->", lw=1.5))
                ax.text(0.5, 0.15, f"c' = {c_prime_path:.3f}{'*' if is_c_prime_sig else ''}", ha='center', fontsize=10)
                
                # Total Effect note
                ax.text(0.5, 0.02, f"Total Effect (c) = {c_path:.3f}{'*' if is_c_sig else ''}", ha='center', fontsize=9, color='gray')
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title("Mediation Path Diagram")
                
                artifacts.append({
                    "type": "plot",
                    "id": "mediation_path_diagram",
                    "title": "Path Diagram",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)

            return {
                "status": "ok",
                "summary": f"Mediation Analysis comparing {iv_col} -> {mv_col} -> {dv_col}. Result: {conclusion}",
                "sections": sections,
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                }
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get('status') != 'ok':
            return None
        return results.get('summary', "Mediation Analysis Completed.")
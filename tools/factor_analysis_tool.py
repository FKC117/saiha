"""
Factor Analysis Tool (EFA)
Performs Exploratory Factor Analysis to identify latent relationships between variables.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class FactorAnalysisTool(BaseAnalysisTool):
    """Tool for Exploratory Factor Analysis."""

    @property
    def name(self) -> str:
        return "factor_analysis"

    @property
    def description(self) -> str:
        return "Identify latent factors explaining the correlation between variables."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="factor_analysis")
        params.add_parameter(
            ToolParameter(
                name="target_columns",
                parameter_type=ParameterType.MULTISELECT,
                label="Variables",
                description="Select numeric variables to analyze.",
                required=True,
                column_source="numeric",
                validation_rules={"minItems": 2},
                help_text="Select at least 2 variables for factor analysis."
            )
        )
        params.add_parameter(
            ToolParameter(
                name="n_factors",
                parameter_type=ParameterType.NUMBER,
                label="Number of Factors",
                description="Number of factors to extract.",
                default_value=2,
                required=True,
                validation_rules={"min": 1}
            )
        )
        params.add_parameter(
            ToolParameter(
                name="rotation",
                parameter_type=ParameterType.SELECT,
                label="Rotation",
                options=[
                    {"value": "varimax", "label": "Varimax (Orthogonal)"},
                    {"value": "quartimax", "label": "Quartimax (Orthogonal)"},
                    {"value": None, "label": "None"},
                ],
                default_value="varimax",
                required=False
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters and load data
            parameters = kwargs
            target_cols = parameters.get("target_columns", [])
            # Handle potential single string input
            if isinstance(target_cols, str):
                target_cols = [target_cols]

            n_factors = int(parameters.get("n_factors", 2))
            rotation = parameters.get("rotation")
            # Handle string 'None' from frontend
            if rotation == "None" or rotation == "":
                rotation = None

            if not target_cols or len(target_cols) < 2:
                return {"status": "error", "summary": "Please select at least 2 numeric variables."}

            df = self.load_dataset(columns=target_cols)
            df_clean = df.dropna()
            
            if df_clean.empty:
                return {"status": "error", "summary": "No data remaining after removing missing values."}

            if len(target_cols) < n_factors:
                 return {"status": "error", "summary": f"Number of factors ({n_factors}) cannot exceed number of variables ({len(target_cols)})."}

            # 2. Preprocess
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clean)

            # 3. Fit Factor Analysis
            fa = FactorAnalysis(n_components=n_factors, rotation=rotation, random_state=42)
            fa.fit(X_scaled)
            
            loadings = fa.components_.T
            
            # 4. Process Results
            artifacts = []
            sections = []

            # Loadings Table
            loadings_df = pd.DataFrame(
                loadings, 
                index=target_cols, 
                columns=[f'Factor {i+1}' for i in range(n_factors)]
            )
            
            # Highlight high loadings
            # We'll just present the table, formatting handled by frontend or string conversion
            
            sections.append({
                'type': 'table',
                'title': 'Factor Loadings',
                'headers': ['Variable'] + loadings_df.columns.tolist(),
                'data': [[idx] + [f"{x:.4f}" for x in row] for idx, row in zip(loadings_df.index, loadings_df.values)]
            })

            # 5. Visualizations
            with PlotUtils.setup_plotting():
                # Heatmap of Loadings
                fig, ax = plt.subplots(figsize=(8, len(target_cols) * 0.5 + 2))
                sns.heatmap(loadings_df, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
                ax.set_title("Factor Loadings Heatmap")
                plt.tight_layout()
                artifacts.append({
                    "type": "plot",
                    "id": "fa_loadings",
                    "title": "Factor Loadings Heatmap",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)

                # Scree Plot (Eigenvalues of correlation matrix)
                # Helps user decide true number of factors
                corr_matrix = np.corrcoef(X_scaled.T)
                eigenvalues = np.linalg.eigvals(corr_matrix)
                eigenvalues = sorted(eigenvalues, reverse=True)
                
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', color='blue')
                ax2.axhline(y=1, color='r', linestyle='--')
                ax2.set_title("Scree Plot")
                ax2.set_xlabel("Factor Number")
                ax2.set_ylabel("Eigenvalue")
                plt.tight_layout()
                artifacts.append({
                    "type": "plot",
                    "id": "scree_plot",
                    "title": "Scree Plot",
                    "content": PlotUtils.fig_to_base64(fig2)
                })
                plt.close(fig2)

            # 6. Dynamic Interpretation
            interpretation_lines = []
            for i in range(n_factors):
                factor_name = f"Factor {i+1}"
                # Get loadings for this factor
                factor_loadings = loadings_df[factor_name]
                # Filter significant loadings (abs > 0.4 is a common rule of thumb)
                sig_loadings = factor_loadings[factor_loadings.abs() > 0.4].sort_values(key=abs, ascending=False)
                
                if not sig_loadings.empty:
                    concepts = []
                    for var, loading in sig_loadings.items():
                        direction = "positive" if loading > 0 else "negative"
                        concepts.append(f"{var} ({direction}, {loading:.2f})")
                    
                    interpretation_lines.append(f"- **{factor_name}**: Driven by {', '.join(concepts)}.")
                else:
                    interpretation_lines.append(f"- **{factor_name}**: No strong specific variable associations found (>0.4).")
            
            interpretation_text = "\n".join(interpretation_lines)
            
            summary_text = f"Factor Analysis extracted {n_factors} factors.\n\n### Factor Interpretation:\n{interpretation_text}"

            return {
                "status": "ok",
                "summary": summary_text,
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
        return results.get('summary', "Factor Analysis Completed.")
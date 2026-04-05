"""
Principal Component Analysis (PCA) Tool
Performs dimensionality reduction using PCA.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional
from io import BytesIO
import base64

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class PCATool(BaseAnalysisTool):
    """Tool for performing Principal Component Analysis (PCA)."""

    @property
    def name(self) -> str:
        return "pca"

    @property
    def description(self) -> str:
        return "Perform Principal Component Analysis (PCA) to reduce dimensionality and visualize data structure."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="pca")
        params.add_parameter(
            ToolParameter(
                name="numeric_columns",
                parameter_type=ParameterType.MULTISELECT,
                label="Select Numeric Columns",
                description="Choose numeric columns for PCA.",
                required=True,
                column_source="numeric"
            )
        )
        params.add_parameter(
            ToolParameter(
                name="n_components",
                parameter_type=ParameterType.NUMBER,
                label="Number of Components (Optional)",
                description="Number of components to keep. Default is 2.",
                required=False,
                default_value=2
            )
        )
        params.add_parameter(
            ToolParameter(
                name="scale_data",
                parameter_type=ParameterType.CHECKBOX,
                label="Scale Data",
                description="Standardize features by removing the mean and scaling to unit variance.",
                required=False,
                default_value=True
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters and load data
            parameters = kwargs
            numeric_columns = parameters.get("numeric_columns", [])
            if isinstance(numeric_columns, str):
                numeric_columns = [numeric_columns]
            
            n_components = parameters.get("n_components")
            try:
                n_components = int(n_components) if n_components else 2
            except (ValueError, TypeError):
                n_components = 2

            scale_data = parameters.get("scale_data", True)

            if not numeric_columns or len(numeric_columns) < 2:
                return {"status": "error", "summary": "Please select at least 2 numeric columns for PCA."}

            df = self.load_dataset(columns=numeric_columns)
            
            # Handle missing values (simple drop for now)
            df_clean = df.dropna()
            if df_clean.empty:
                return {"status": "error", "summary": "Dataset is empty after removing missing values."}
            
            X = df_clean[numeric_columns]

            # 2. Scaling
            if scale_data:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X

            # 3. Perform PCA
            # Limit components to min(n_samples, n_features)
            max_components = min(X_scaled.shape)
            final_n_components = min(n_components, max_components)
            
            pca = PCA(n_components=final_n_components)
            principal_components = pca.fit_transform(X_scaled)
            
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)

            # 4. Generate Visualizations
            artifacts = []

            with PlotUtils.setup_plotting():
                # Scree Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_indices = range(1, len(explained_variance_ratio) + 1)
                ax.plot(feature_indices, explained_variance_ratio, 'bo-', linewidth=2, label='Individual')
                ax.plot(feature_indices, cumulative_variance, 'rs--', linewidth=2, label='Cumulative')
                ax.set_title('Scree Plot: Explained Variance by Components')
                ax.set_xlabel('Principal Component')
                ax.set_ylabel('Explained Variance Ratio')
                ax.legend(loc='best')
                ax.grid(True)
                
                artifacts.append({
                    "type": "plot",
                    "id": "scree_plot",
                    "title": "Scree Plot",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)

                # 2D Scatter Plot (if at least 2 components)
                if final_n_components >= 2:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    pc_df = pd.DataFrame(data=principal_components[:, :2], columns=['PC1', 'PC2'])
                    sns.scatterplot(x='PC1', y='PC2', data=pc_df, ax=ax, alpha=0.7)
                    ax.set_title(f'PCA Scatter Plot (Explains {cumulative_variance[1]:.2%} of Variance)')
                    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.2%} Variance)')
                    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.2%} Variance)')
                    ax.grid(True, alpha=0.3)

                    artifacts.append({
                        "type": "plot",
                        "id": "pca_scatter",
                        "title": "PCA 2D Scatter Plot",
                        "content": PlotUtils.fig_to_base64(fig)
                    })
                    plt.close(fig)

            # 5. Result Construction
            summary = f"PCA performed on {len(numeric_columns)} columns. " \
                      f"First {final_n_components} components explain {cumulative_variance[-1]:.2%} of the variance."
            
            # Dynamic Interpretation
            interpretation_lines = []
            
            # Helper to interpret first few components (up to 3)
            comps_to_interpret = min(final_n_components, 3)
            loading_matrix = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(final_n_components)], index=numeric_columns)
            
            for i in range(comps_to_interpret):
                pc_name = f"PC{i+1}"
                pc_var = explained_variance_ratio[i]
                
                # Get top loadings
                # Threshold > 0.3 usually significant enough for PCA
                top_contributors = loading_matrix[pc_name][loading_matrix[pc_name].abs() > 0.3].sort_values(key=abs, ascending=False)
                
                if not top_contributors.empty:
                    concepts = []
                    for var, loading in top_contributors.items():
                         direction = "(+)" if loading > 0 else "(-)"
                         concepts.append(f"{var} {direction}")
                    
                    interpretation_lines.append(f"- **{pc_name}** ({pc_var:.1%} var): Driven by {', '.join(concepts)}.")
                else:
                    interpretation_lines.append(f"- **{pc_name}** ({pc_var:.1%} var): Distributed contribution (no single dominant variable > 0.3).")

            summary += "\n\n### Key Components:\n" + "\n".join(interpretation_lines)
            
            # Create explained variance table
            variance_data = []
            for i, var in enumerate(explained_variance_ratio):
                variance_data.append([f"PC{i+1}", f"{var:.4f}", f"{cumulative_variance[i]:.4f}"])

            sections = [
                {
                    'type': 'table',
                    'title': 'Explained Variance',
                    'icon': 'bi bi-table',
                    'headers': ['Component', 'Explained Variance', 'Cumulative Variance'],
                    'data': variance_data
                }
            ]
            
            # Component Loadings (Component-Feature correlation)
            loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(final_n_components)], index=numeric_columns)
            loadings_data = loadings.reset_index().values.tolist()
            sections.append({
                'type': 'table',
                'title': 'Component Loadings',
                'icon': 'bi bi-list-columns',
                'headers': ['Feature'] + [f"PC{i+1}" for i in range(final_n_components)],
                'data': loadings_data
            })

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                    "columns_analyzed": numeric_columns
                }
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get('status') != 'ok':
            return None
        return results.get('summary', "PCA Analysis Completed.")
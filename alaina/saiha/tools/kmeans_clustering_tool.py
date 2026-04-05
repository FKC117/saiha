"""
K-Means Clustering Tool
Performs K-Means clustering algorithm.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional
from io import BytesIO
import base64

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class KMeansClusteringTool(BaseAnalysisTool):
    """Tool for performing K-Means Clustering."""

    @property
    def name(self) -> str:
        return "kmeans_clustering"

    @property
    def description(self) -> str:
        return "Perform K-Means Clustering to group similar data points."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="kmeans_clustering")
        params.add_parameter(
            ToolParameter(
                name="feature_columns",
                parameter_type=ParameterType.MULTISELECT,
                label="Select Features",
                description="Choose numeric and/or categorical columns for clustering.",
                required=True,
                column_source="numeric,categorical"
            )
        )
        params.add_parameter(
            ToolParameter(
                name="n_clusters",
                parameter_type=ParameterType.NUMBER,
                label="Number of Clusters (k)",
                description="The number of clusters to form. Default is 3.",
                required=False,
                default_value=3
            )
        )
        params.add_parameter(
            ToolParameter(
                name="auto_find_k",
                parameter_type=ParameterType.CHECKBOX,
                label="Auto-find k (Elbow Method)",
                description="Automatically suggest optimal k using the Elbow Method.",
                required=False,
                default_value=False
            )
        )
        params.add_parameter(
            ToolParameter(
                name="encoding_method",
                parameter_type=ParameterType.SELECT,
                label="Categorical Encoding",
                description="How to transform text variables for the model. One-Hot is usually safer for this model.",
                required=True,
                default_value="one_hot",
                options=[
                    {"value": "one_hot", "label": "One-Hot Encoding (Dummies)"},
                    {"value": "label", "label": "Label Encoding (Ordinal)"}
                ]
            )
        )
        params.add_parameter(
            ToolParameter(
                name="save_as_new_dataset",
                parameter_type=ParameterType.CHECKBOX,
                label="Save as New Dataset",
                description="Create a new dataset with the assigned Cluster labels.",
                required=False,
                default_value=False
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters and load data
            parameters = kwargs
            feature_columns = parameters.get("feature_columns", [])
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]
            
            n_clusters_param = parameters.get("n_clusters")
            try:
                k = int(n_clusters_param) if n_clusters_param else 3
            except (ValueError, TypeError):
                k = 3

            auto_find_k = parameters.get("auto_find_k", False)

            if not feature_columns or len(feature_columns) < 2:
                return {"status": "error", "summary": "Please select at least 2 columns for clustering."}

            df = self.load_dataset(columns=feature_columns)
            
            # Handle missing values
            df_clean = df.dropna()
            if df_clean.empty:
                return {"status": "error", "summary": "Dataset is empty after removing missing values."}
            
            # Identify categorical columns for encoding
            categorical_features = [c for c in feature_columns if df_clean[c].dtype in ['object', 'category', 'bool']]
            encoding_method = parameters.get("encoding_method", "one_hot")

            if encoding_method == "label":
                from sklearn.preprocessing import LabelEncoder
                X = df_clean[feature_columns].copy()
                for col in categorical_features:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            else:
                # Default One-Hot
                X = pd.get_dummies(df_clean[feature_columns], columns=categorical_features, drop_first=True)

            # 2. Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            artifacts = []
            
            # 3. Auto-find k (Elbow Method)
            optimal_k_msg = ""
            if auto_find_k:
                inertias = []
                K_range = range(1, 11)
                
                # Optimize: Use sampling for Elbow Method if dataset is large and not using weighted kmeans
                elbow_X = X_scaled
                sampling_msg = ""
                if X_scaled.shape[0] > 10000:
                    # Random sample of 10k rows
                    np.random.seed(42)
                    indices = np.random.choice(X_scaled.shape[0], 10000, replace=False)
                    elbow_X = X_scaled[indices]
                    sampling_msg = " (calculated on 10k sample)"

                for i in K_range:
                    km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
                    km.fit(elbow_X)
                    inertias.append(km.inertia_)
                
                with PlotUtils.setup_plotting():
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(K_range, inertias, 'bo-', linewidth=2)
                    ax.set_xlabel('Number of Clusters (k)')
                    ax.set_ylabel(f'Inertia (Within-cluster Sum of Squares){sampling_msg}')
                    ax.set_title(f'Elbow Method for Optimal k{sampling_msg}')
                    ax.grid(True)
                    
                    artifacts.append({
                        "type": "plot",
                        "id": "elbow_plot",
                        "title": "Elbow Plot",
                        "content": PlotUtils.fig_to_base64(fig)
                    })
                    plt.close(fig)
                
                # Simple heuristic for "optimal" k might be complex to automate perfectly without heavier libs,
                # but we present the plot. For the actual clustering below, we stick to user provided k 
                # unless we want to override it. Let's stick to user k but add a note.
                optimal_k_msg = " (Review Elbow Plot to adjust k)"

            # 4. Perform K-Means
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to a display dataframe (limited rows)
            df_display = df_clean.copy()
            df_display['Cluster'] = cluster_labels
            
            # 5. Visualize Clusters (using PCA for 2D projection)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = cluster_labels

            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100, ax=ax)
                ax.set_title(f'K-Means Clustering (k={k}) Visualized with PCA')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.legend(title='Cluster')
                ax.grid(True, alpha=0.3)
                
                artifacts.append({
                    "type": "plot",
                    "id": "cluster_plot",
                    "title": f"Cluster Plot (k={k})",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)

            # 6. Cluster Centers
            centers_df = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)
            centers_df.insert(0, 'Cluster', range(k))
            
            # Round for display
            for col in X.columns:
                centers_df[col] = centers_df[col].round(4)
                
            centers_data = centers_df.values.tolist()
            sections = [
                {
                    'type': 'table',
                    'title': 'Cluster Centers (Centroids)',
                    'icon': 'bi bi-geo-alt',
                    'headers': ['Cluster'] + list(X.columns),
                    'data': centers_data
                }
            ]
            
            # Cluster Counts
            counts = pd.Series(cluster_labels).value_counts().sort_index()
            counts_data = [[i, count, f"{count/len(cluster_labels):.1%}"] for i, count in counts.items()]
            sections.append({
                'type': 'table',
                'title': 'Cluster Size',
                'icon': 'bi bi-pie-chart',
                'headers': ['Cluster', 'Count', 'Percentage'],
                'data': counts_data
            })

            summary = f"K-Means clustering performed with k={k}.{optimal_k_msg} Data partitioned into {k} clusters."
            
            # Dynamic Cluster Characterization
            # Compare cluster centers to global means
            global_means = X.mean()
            global_stds = X.std()
            
            interpretation_lines = []
            for i in range(k):
                cluster_center = centers_df[centers_df['Cluster'] == i].iloc[0]
                cluster_desc = []
                
                for col in X.columns:
                    val = cluster_center[col]
                    mean_val = global_means[col]
                    std_val = global_stds[col]
                    z_score = (val - mean_val) / std_val if std_val != 0 else 0
                    
                    # Highlight if > 0.5 std dev away
                    if z_score > 0.5:
                        cluster_desc.append(f"High {col}")
                    elif z_score < -0.5:
                        cluster_desc.append(f"Low {col}")
                
                size_pct = counts_data[i][2]
                if cluster_desc:
                    interpretation_lines.append(f"- **Cluster {i} ({size_pct})**: Characterized by {', '.join(cluster_desc)}.")
                else:
                    interpretation_lines.append(f"- **Cluster {i} ({size_pct})**: Average values across features.")

            summary += "\n\n### Cluster Profiles:\n" + "\n".join(interpretation_lines)

            # Save as New Dataset Logic
            save_as_new = parameters.get("save_as_new_dataset", False)
            new_dataset_info = None
            if save_as_new:
                # Add cluster labels to the FULL original dataframe
                # Note: df_clean might have fewer rows if we dropped NaNs. 
                # We need to align carefully or likely just save the cleaned version + clusters.
                
                # Option 1: Save the CLEANED version with clusters (simplest consistency)
                df_to_save = df_clean.copy()
                df_to_save['Cluster'] = cluster_labels
                
                from ...dataset_utils import save_dataframe_as_dataset
                suffix = f"KMeans (k={k})"
                new_dataset = save_dataframe_as_dataset(df_to_save, self.dataset, suffix)
                summary += f"\n\n**Dataset Saved**: New dataset '{new_dataset.name}' created with 'Cluster' column."
                new_dataset_info = {"id": str(new_dataset.id), "name": new_dataset.name}

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                    "columns_analyzed": list(X.columns),
                    "new_dataset": new_dataset_info
                }
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get('status') != 'ok':
            return None
        return results.get('summary', "K-Means Analysis Completed.")
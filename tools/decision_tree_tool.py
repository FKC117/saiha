"""
Decision Tree Tool
Fits a Decision Tree model to visualize decision rules and feature importance.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class DecisionTreeTool(BaseAnalysisTool):
    """Tool for Decision Tree Analysis."""

    @property
    def name(self) -> str:
        return "decision_tree"

    @property
    def description(self) -> str:
        return "Visualize decision rules and identify important features using a Decision Tree."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="decision_tree")
        params.add_parameter(
            ToolParameter(
                name="target_variable",
                parameter_type=ParameterType.COLUMN_SELECT,
                label="Target Variable (Y)",
                description="The outcome you want to explain (can be categorical or numeric).",
                required=True,
                 column_source="numeric,categorical" # Both allowed
            )
        )
        params.add_parameter(
            ToolParameter(
                name="feature_columns",
                parameter_type=ParameterType.MULTISELECT,
                label="Feature Columns (X)",
                description="Variables used to predict the target.",
                required=True,
                column_source="numeric,categorical" 
            )
        )
        params.add_parameter(
            ToolParameter(
                name="encoding_method",
                parameter_type=ParameterType.SELECT,
                label="Categorical Encoding",
                description="How to handle non-numeric variables.",
                required=False,
                default_value="one_hot",
                options=[
                    {"value": "one_hot", "label": "One-Hot Encoding (Best for non-ordinal)"},
                    {"value": "label", "label": "Label Encoding (Best for ordinal/high-cardinality)"}
                ]
            )
        )
        params.add_parameter(
            ToolParameter(
                name="max_depth",
                parameter_type=ParameterType.NUMBER,
                label="Max Depth",
                description="Maximum depth of the tree (limits complexity).",
                required=True,
                default_value=3,
                validation_rules={"minimum": 1, "maximum": 10}
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters
            parameters = kwargs
            target_col = parameters.get("target_variable")
            features = parameters.get("feature_columns", [])
            
            # Clamp max_depth to safety limits
            max_depth = min(max(int(parameters.get("max_depth", 3)), 1), 10)

            if isinstance(features, str):
                features = [features]
            
            if not target_col or not features:
                return {"status": "error", "summary": "Target and features are required."}

            cols = [target_col] + features
            df = self.load_dataset(columns=cols)
            df_clean = df.dropna()

            if df_clean.empty:
                return {"status": "error", "summary": "No data remaining after removing missing values."}

            # 2. Determine Mode (Classification vs Regression)
            y = df_clean[target_col]
            X = df_clean[features]
            
            encoding_method = parameters.get("encoding_method", "one_hot")
            
            is_categorical_y = False
            if y.dtype == 'object' or str(y.dtype) == 'category' or len(y.unique()) < 10:
                # Heuristic: strings or few unique integers treated as class
                is_categorical_y = True
            
            # Feature Encoding (X)
            if encoding_method == "one_hot":
                X_encoded = pd.get_dummies(X, drop_first=True)
            else:
                # Label Encoding for categorical columns in X
                X_encoded = X.copy()
                for col_name in X_encoded.columns:
                    if X_encoded[col_name].dtype == 'object' or str(X_encoded[col_name].dtype) == 'category':
                        le = LabelEncoder()
                        X_encoded[col_name] = le.fit_transform(X_encoded[col_name].astype(str))
            
            if is_categorical_y:
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                type_name = "Classification"
                # If y is string, sklearn handles it for fit but we might want label encoding
                # actually sklearn needs numeric Y usually or it might warn
                # Let's force str for consistent class names
                y = y.astype(str)
                model.fit(X_encoded, y)
                score = accuracy_score(y, model.predict(X_encoded))
                metric_name = "Accuracy"
            else:
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                type_name = "Regression"
                model.fit(X_encoded, y)
                score = r2_score(y, model.predict(X_encoded))
                metric_name = "R-Squared"

            # 3. Outputs
            artifacts = []
            sections = []
            
            summary_text = f"Decision Tree ({type_name}) fitted.\n"
            summary_text += f"Target: {target_col}\n"
            summary_text += f"Model {metric_name}: {score:.4f}\n"
            
            # Feature Importance
            importance_df = pd.DataFrame({
                'Feature': X_encoded.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            sections.append({
                'type': 'table',
                'title': 'Feature Importance',
                'headers': ['Feature', 'Importance'],
                'data': [[r['Feature'], f"{r['Importance']:.4f}"] for i, r in importance_df.iterrows()]
            })

            # Plot Tree
            with PlotUtils.setup_plotting():
                # Dynamic Figure Sizing: Expand exponentially with depth
                # Potential leaves = 2^depth. Each leaf needs ~0.8 inches of width.
                width = max(24, (2**max_depth) * 0.8)
                height = max(14, max_depth * 2.5)
                fig, ax = plt.subplots(figsize=(width, height))
                
                # Adaptive Font Size: Smaller for deeper trees
                adaptive_font = max(6, 12 - max_depth)
                
                plot_tree(
                    model, 
                    feature_names=X_encoded.columns, 
                    class_names=model.classes_ if (is_categorical_y and hasattr(model, 'classes_')) else None, 
                    filled=True, 
                    rounded=True, 
                    ax=ax, 
                    fontsize=adaptive_font,
                    precision=2,
                    proportion=True
                )
                
                ax.set_title(f"Decision Tree Visualization (Max Depth {max_depth})", fontsize=max(16, adaptive_font * 1.5))
                plt.tight_layout()
                
                artifacts.append({
                    "type": "plot",
                    "id": "decision_tree_plot",
                    "title": "Tree Diagram",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)
            
            # Plot Importance
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis', ax=ax)
                ax.set_title("Top 10 Feature Importances")
                artifacts.append({
                    "type": "plot",
                    "id": "feature_importance_plot",
                    "title": "Feature Importance",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)

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
         return results.get('summary', "Decision Tree Analysis Completed.")
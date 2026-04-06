"""
Regularized Regression Tool
Performs Lasso (L1) and Ridge (L2) regression.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class LassoRidgeRegressionTool(BaseAnalysisTool):
    """Tool for performing Regularized Regression (Lasso and Ridge)."""

    @property
    def name(self) -> str:
        return "lasso_ridge_regression"

    @property
    def description(self) -> str:
        return "Perform Regularized Regression (Lasso or Ridge) to penalize model complexity."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="lasso_ridge_regression")
        params.add_parameter(
            ToolParameter(
                name="target_variable",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Target Variable (Dependent)",
                description="The numeric variable you want to predict.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="feature_columns",
                parameter_type=ParameterType.MULTISELECT,
                label="Feature Columns (Independent)",
                description="The predictor variables.",
                required=True,
                column_source="numeric,categorical"
            )
        )
        params.add_parameter(
            ToolParameter(
                name="regularization_type",
                parameter_type=ParameterType.SELECT,
                label="Regularization Type",
                options=[
                    {"value": "Lasso", "label": "Lasso (L1) - Feature Selection"},
                    {"value": "Ridge", "label": "Ridge (L2) - Shrinkage"}
                ],
                default_value="Lasso",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="alpha",
                parameter_type=ParameterType.NUMBER,
                label="Alpha (Regularization Strength)",
                description="Constant that multiplies the penalty terms. Higher alpha = more regularization. set 0 for OLS.",
                default_value=1.0,
                required=False
            )
        )
        params.add_parameter(
            ToolParameter(
                name="use_cv",
                parameter_type=ParameterType.CHECKBOX,
                label="Use Cross-Validation to find best Alpha",
                description="If checked, automatically finds the best alpha from a range of values.",
                default_value=False,
                required=False
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
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters and load data
            parameters = kwargs
            target_col = parameters.get("target_variable")
            feature_cols = parameters.get("feature_columns", [])
            if isinstance(feature_cols, str):
                feature_cols = [feature_cols]
            
            reg_type = parameters.get("regularization_type", "Lasso")
            use_cv = parameters.get("use_cv", False)
            
            try:
                alpha_val = float(parameters.get("alpha", 1.0))
            except (ValueError, TypeError):
                alpha_val = 1.0

            if not target_col or not feature_cols:
                return {"status": "error", "summary": "Please select both a target variable and at least one feature."}
            
            if target_col in feature_cols:
                feature_cols = [c for c in feature_cols if c != target_col]
                if not feature_cols:
                     return {"status": "error", "summary": "Target variable cannot be the only feature."}

            cols_to_load = feature_cols + [target_col]
            df = self.load_dataset(columns=cols_to_load)
            
            # Handle missing values
            df_clean = df.dropna()
            if df_clean.empty:
                return {"status": "error", "summary": "Dataset is empty after removing missing values."}

            # Identify categorical columns for encoding
            categorical_features = [c for c in feature_cols if df_clean[c].dtype in ['object', 'category', 'bool']]
            encoding_method = parameters.get("encoding_method", "one_hot")

            if encoding_method == "label":
                from sklearn.preprocessing import LabelEncoder
                X = df_clean[feature_cols].copy()
                for col in categorical_features:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            else:
                # Default One-Hot
                X = pd.get_dummies(df_clean[feature_cols], columns=categorical_features, drop_first=True)
            
            y = df_clean[target_col].astype(float)
            X = X.astype(float)

            # 2. Scaling (Important for Regularization)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 3. Fit Model
            best_alpha_msg = ""
            
            if use_cv:
                if reg_type == "Lasso":
                    model = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
                    used_alpha = model.alpha_
                else:
                    model = RidgeCV(cv=None, scoring='neg_mean_squared_error').fit(X_scaled, y)
                    used_alpha = model.alpha_
                best_alpha_msg = f" (Best CV Alpha: {used_alpha:.4f})"
            else:
                used_alpha = alpha_val
                if reg_type == "Lasso":
                    model = Lasso(alpha=used_alpha, random_state=42).fit(X_scaled, y)
                else:
                    model = Ridge(alpha=used_alpha, random_state=42).fit(X_scaled, y)

            y_pred = model.predict(X_scaled)
            
            # 4. Metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)

            artifacts = []
            sections = []

            # 5. Metrics Table
            sections.append({
                'type': 'table',
                'title': 'Model Performance',
                'icon': 'bi bi-speedometer2',
                'headers': ['Metric', 'Value'],
                'data': [
                    ['R-Squared', f"{r2:.4f}"],
                    ['RMSE', f"{rmse:.4f}"],
                    ['MSE', f"{mse:.4f}"],
                    ['Alpha Used', f"{used_alpha:.4f}"]
                ]
            })

            # 6. Coefficients Table
            coeffs_df = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': model.coef_
            })
            # Sort by absolute coefficient size
            coeffs_df['Abs_Coeff'] = coeffs_df['Coefficient'].abs()
            coeffs_df = coeffs_df.sort_values('Abs_Coeff', ascending=False).drop(columns=['Abs_Coeff'])
            
            sections.append({
                'type': 'table',
                'title': 'Feature Coefficients',
                'icon': 'bi bi-list-ol',
                'headers': ['Feature', 'Coefficient'],
                'data': [[r['Feature'], f"{r['Coefficient']:.4f}"] for _, r in coeffs_df.iterrows()]
            })

            # 7. Coefficient Plot
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Color code: Zero vs Non-Zero (especially for Lasso)
                coeffs_df['Color'] = np.where(coeffs_df['Coefficient'] == 0, 'lightgray', 
                                            np.where(coeffs_df['Coefficient'] > 0, 'green', 'red'))
                
                sns.barplot(x='Coefficient', y='Feature', data=coeffs_df, ax=ax, palette=coeffs_df['Color'].tolist())
                ax.set_title(f'{reg_type} Regression Coefficients (Alpha={used_alpha:.4f})')
                ax.axvline(x=0, color='black', linestyle='--')
                plt.tight_layout()
                
                artifacts.append({
                    "type": "plot",
                    "id": "coeff_plot",
                    "title": "Coefficient Plot",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)

            # 8. Predicted vs Actual Plot
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(y, y_pred, alpha=0.5)
                
                # Perfect prediction line
                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),
                    np.max([ax.get_xlim(), ax.get_ylim()]),
                ]
                ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title('Actual vs Predicted')
                plt.tight_layout()

                artifacts.append({
                    "type": "plot",
                    "id": "pred_vs_actual",
                    "title": "Predicted vs Actual",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)

            summary = f"{reg_type} Regression performed using alpha={used_alpha:.4f}{best_alpha_msg}. R-Squared = {r2:.4f}."
            if reg_type == "Lasso":
                zero_coeffs = (model.coef_ == 0).sum()
                summary += f" {zero_coeffs} features eliminated (coefficient reduced to zero)."

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                    "target": target_col
                }
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get('status') != 'ok':
            return None
        return results.get('summary', "Regression Analysis Completed.")
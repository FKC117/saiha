"""
Sensitivity Analysis ("What-If") Tool
Simulates how changes in an input variable affect a target outcome using a linear regression model.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class SensitivityAnalysisTool(BaseAnalysisTool):
    """Tool for performing Sensitivity (What-If) Analysis."""

    @property
    def name(self) -> str:
        return "sensitivity_analysis"

    @property
    def description(self) -> str:
        return "Simulate how changes in one key variable affect the target outcome (What-If Analysis)."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="sensitivity_analysis")
        params.add_parameter(
            ToolParameter(
                name="target_variable",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Target Outcome (Y)",
                description="The variable you want to predict/analyze (e.g., Sales, Profit).",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="varying_variable",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Variable to Change (X)",
                description="The input variable you want to vary (e.g., Price, Ad Spend).",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="covariates",
                parameter_type=ParameterType.MULTISELECT,
                label="Other Fixed Factors",
                description="Other variables to include in the model (held constant at average).",
                required=False,
                column_source="numeric,categorical"
            )
        )
        params.add_parameter(
            ToolParameter(
                name="variation_range",
                parameter_type=ParameterType.SELECT,
                label="Variation Range",
                description="How much to vary the input variable.",
                required=True,
                default_value="percent_20",
                options=[
                    {"value": "percent_10", "label": "+/- 10%"},
                    {"value": "percent_20", "label": "+/- 20%"},
                    {"value": "percent_50", "label": "+/- 50%"},
                    {"value": "min_max", "label": "Full Range (Min to Max)"}
                ]
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
            # 1. Get parameters
            parameters = kwargs
            target_col = parameters.get("target_variable")
            vary_col = parameters.get("varying_variable")
            covariates = parameters.get("covariates", [])
            variation_range = parameters.get("variation_range", "percent_20")

            if isinstance(covariates, str):
                covariates = [covariates]
            
            # Ensure vary_col is not in covariates to avoid dupes
            if vary_col in covariates:
                covariates.remove(vary_col)

            all_cols = [target_col, vary_col] + covariates
            if not all([target_col, vary_col]):
                 return {"status": "error", "summary": "Target and Varying variables are required."}

            df = self.load_dataset(columns=all_cols)
            df_clean = df.dropna()

            if df_clean.empty:
                return {"status": "error", "summary": "No data remaining after removing missing values."}

            # 2. Fit Model (OLS)
            # Y = b0 + b1*Vary + b2*Cov1 + ...
            X = df_clean[[vary_col] + covariates]
            cat_cols = [c for c in covariates if df_clean[c].dtype in ['object', 'category', 'bool']]
            encoding_method = parameters.get("encoding_method", "one_hot")

            if encoding_method == "label":
                from sklearn.preprocessing import LabelEncoder
                X_model = df_clean[[vary_col] + covariates].copy()
                for col in cat_cols:
                    le = LabelEncoder()
                    X_model[col] = le.fit_transform(X_model[col].astype(str))
                X_encoded = sm.add_constant(X_model, has_constant='add')
                if 'const' not in X_encoded.columns:
                    X_encoded['const'] = 1.0
            else:
                # Default One-Hot
                X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
                X_encoded = sm.add_constant(X_encoded, has_constant='add')
                if 'const' not in X_encoded.columns:
                    X_encoded['const'] = 1.0
            
            y = df_clean[target_col].astype(float)
            X_encoded = X_encoded.astype(float)
            
            model = sm.OLS(y, X_encoded).fit()
            
            # 3. Create Simulation Data
            # Base Case: averages of numeric inputs, mode of categorical inputs
            base_row = {}
            for col in [vary_col] + covariates:
                if df_clean[col].dtype in ['object', 'category', 'bool']:
                    base_row[col] = df_clean[col].mode()[0]
                else:
                    base_row[col] = df_clean[col].mean()
            
            # Define range for varying variable (always numeric in this tool's current design for varying)
            vary_mean = base_row[vary_col]
            vary_min = df_clean[vary_col].min()
            vary_max = df_clean[vary_col].max()
            
            if variation_range == "min_max":
                sim_values = np.linspace(vary_min, vary_max, 20)
            elif variation_range == "percent_10":
                sim_values = np.linspace(vary_mean * 0.9, vary_mean * 1.1, 20)
            elif variation_range == "percent_20":
                 sim_values = np.linspace(vary_mean * 0.8, vary_mean * 1.2, 20)
            elif variation_range == "percent_50":
                 sim_values = np.linspace(vary_mean * 0.5, vary_mean * 1.5, 20)
            else:
                 sim_values = np.linspace(vary_mean * 0.8, vary_mean * 1.2, 20)
            
            # Create simulation dataframe
            sim_df = pd.DataFrame([base_row] * len(sim_values))
            sim_df[vary_col] = sim_values
            
            # Encode simulation data same as X
            if encoding_method == "label":
                from sklearn.preprocessing import LabelEncoder
                sim_X_model = sim_df.copy()
                for col in cat_cols:
                    le = LabelEncoder()
                    # We must use the same mapping as the model
                    # For simplicity in this tool context, we re-fit on combined to ensure levels match
                    full_vals = pd.concat([df_clean[col], sim_df[col]]).astype(str)
                    le.fit(full_vals)
                    sim_X_model[col] = le.transform(sim_df[col].astype(str))
                sim_X_final = sm.add_constant(sim_X_model, has_constant='add')
                if 'const' not in sim_X_final.columns:
                    sim_X_final['const'] = 1.0
            else:
                # One-Hot
                sim_X_raw = pd.get_dummies(sim_df, columns=cat_cols, drop_first=False)
                # Reindex to match model features (handling missing levels if any)
                sim_X_final = pd.DataFrame(index=sim_X_raw.index)
                sim_X_final['const'] = 1.0
                for col in X_encoded.columns:
                    if col == 'const': continue
                    if col in sim_X_raw.columns:
                        sim_X_final[col] = sim_X_raw[col]
                    else:
                        sim_X_final[col] = 0
            
            # Ensure order matches and is float
            sim_X_final = sim_X_final[X_encoded.columns].astype(float)
            sim_df['Predicted_Outcome'] = model.predict(sim_X_final)
            
            # 4. Impact metrics
            impact_min = sim_df['Predicted_Outcome'].min()
            impact_max = sim_df['Predicted_Outcome'].max()
            change = impact_max - impact_min
            
            slope = model.params[vary_col]
            
            # 5. Output
            artifacts = []
            sections = []
            
            summary_text = f"Sensitivity Analysis for '{target_col}' varying '{vary_col}'.\n"
            summary_text += f"The model suggests that for every 1 unit increase in {vary_col}, {target_col} changes by {slope:.4f}.\n"
            summary_text += f"Across the simulated range, the outcome varies from {impact_min:.2f} to {impact_max:.2f} (Delta: {change:.2f})."
            
            # Simulation Data Table
            display_sim = sim_df[[vary_col, 'Predicted_Outcome']].copy()
            display_sim.columns = [f"{vary_col} (Input)", f"Predicted {target_col}"]
            # Sample rows for display
            sections.append({
                'type': 'table',
                'title': 'Simulation Results (Sample)',
                'headers': display_sim.columns.tolist(),
                'data': [[f"{row[0]:.2f}", f"{row[1]:.2f}"] for i, row in display_sim.iloc[::2].iterrows()] # Show every other row
            })
            
            # Coefficients Table
            coef_df = pd.DataFrame({"Feature": model.params.index, "Coefficient": model.params.values, "P-Value": model.pvalues.values})
            coef_df = coef_df[coef_df["Feature"] != "const"]
            sections.append({
                 'type': 'table',
                'title': 'underlying Model Coefficients',
                'headers': ['Feature', 'Coefficient', 'P-Value'],
                'data': [[r['Feature'], f"{r['Coefficient']:.4f}", f"{r['P-Value']:.4f}"] for i, r in coef_df.iterrows()]
            })

            # Visualization
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot Simulation Line
                ax.plot(sim_df[vary_col], sim_df['Predicted_Outcome'], color='blue', linewidth=2, label='Predicted Trend')
                
                # Calculate base prediction
                base_case_df = pd.DataFrame([base_row])
                if encoding_method == "label":
                    base_case_X_model = base_case_df.copy()
                    for col in cat_cols:
                        le = LabelEncoder()
                        full_vals = pd.concat([df_clean[col], base_case_df[col]]).astype(str)
                        le.fit(full_vals)
                        base_case_X_model[col] = le.transform(base_case_df[col].astype(str))
                    base_case_final = sm.add_constant(base_case_X_model, has_constant='add')
                    if 'const' not in base_case_final.columns:
                        base_case_final['const'] = 1.0
                else:
                    base_case_X_raw = pd.get_dummies(base_case_df, columns=cat_cols, drop_first=False)
                    base_case_final = pd.DataFrame(index=base_case_X_raw.index)
                    base_case_final['const'] = 1.0
                    for col in X_encoded.columns:
                        if col == 'const': continue
                        base_case_final[col] = base_case_X_raw[col].iloc[0] if col in base_case_X_raw.columns else 0
                
                base_pred = model.predict(base_case_final[X_encoded.columns].astype(float))[0]

                # Plot Base Case point
                ax.plot(vary_mean, base_pred, 'ro', markersize=8, label='Base Case')
                
                ax.set_xlabel(f"{vary_col} (Varying Input)")
                ax.set_ylabel(f"Predicted {target_col}")
                ax.set_title(f"Sensitivity: {target_col} vs {vary_col}")
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                artifacts.append({
                    "type": "plot",
                    "id": "sensitivity_plot",
                    "title": "Sensitivity Plot (What-If)",
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
         return results.get('summary', "Sensitivity Analysis Completed.")
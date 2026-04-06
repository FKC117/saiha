"""
Interrupted Time Series (ITS) Tool
Analyzes the effect of an intervention on a time series outcome using segmented regression.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class InterruptedTimeSeriesTool(BaseAnalysisTool):
    """Tool for Interrupted Time Series (ITS) Analysis."""

    @property
    def name(self) -> str:
        return "interrupted_time_series"

    @property
    def description(self) -> str:
        return "Analyze the impact of an intervention on a time series using segmented regression."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="interrupted_time_series")
        params.add_parameter(
            ToolParameter(
                name="outcome_variable",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Outcome Variable (Y)",
                description="The dependent variable to analyze over time.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="time_variable",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Time Variable (X)",
                description="Continuous time variable or sequence index.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="intervention_point",
                parameter_type=ParameterType.NUMBER,
                label="Intervention Cutoff (Time Value)",
                description="The value of the time variable where the intervention occurred.",
                required=True
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters
            parameters = kwargs
            outcome_col = parameters.get("outcome_variable")
            time_col = parameters.get("time_variable")
            cutoff = parameters.get("intervention_point")
            
            try:
                cutoff = float(cutoff)
            except (ValueError, TypeError):
                return {"status": "error", "summary": "Intervention point must be a number."}

            if not all([outcome_col, time_col]):
                return {"status": "error", "summary": "Missing required variables."}

            df = self.load_dataset(columns=[outcome_col, time_col])
            df_clean = df.dropna()

            if df_clean.empty:
                return {"status": "error", "summary": "No data remaining after removing missing values."}

            # 2. Prepare Data for Segmented Regression
            # Model: Y = beta0 + beta1*Time + beta2*Intervention + beta3*Time_After_Intervention + e
            
            df_clean = df_clean.sort_values(by=time_col)
            
            # Simple numeric conversion if needed
            try:
                df_clean['Time'] = pd.to_numeric(df_clean[time_col])
                df_clean['Outcome'] = pd.to_numeric(df_clean[outcome_col])
            except:
                return {"status": "error", "summary": "Time and Outcome variables must be numeric."}
            
            # Create dummy variables
            df_clean['Intervention'] = (df_clean['Time'] >= cutoff).astype(int)
            df_clean['Time_After'] = (df_clean['Time'] - cutoff) * df_clean['Intervention']
            
            # 3. Fit Model
            model = smf.ols("Outcome ~ Time + Intervention + Time_After", data=df_clean).fit()
            
            # 4. Extract Results
            # beta2 (Intervention) = Immediate level change
            # beta3 (Time_After) = Slope change
            
            params = model.params
            pvalues = model.pvalues
            
            level_change = params.get('Intervention', 0)
            level_p = pvalues.get('Intervention', 1)
            
            slope_change = params.get('Time_After', 0)
            slope_p = pvalues.get('Time_After', 1)
            
            baseline_slope = params.get('Time', 0)
            
            # 5. Output Construction
            artifacts = []
            sections = []
            
            summary_text = f"Interrupted Time Series Analysis for '{outcome_col}' at cutoff {cutoff}.\n"
            summary_text += f"Immediate Level Change: {level_change:.4f} (p={level_p:.4f}).\n"
            summary_text += f"Slope Change (Trend): {slope_change:.4f} (p={slope_p:.4f}).\n"
            
            # Interpretation
            interp = []
            if level_p < 0.05:
                interp.append("Significant immediate shift in the outcome level at the intervention point.")
            else:
                interp.append("No significant immediate shift in level.")
                
            if slope_p < 0.05:
                direction = "increased" if slope_change > 0 else "decreased"
                interp.append(f"Significant change in trend. The slope {direction} by {abs(slope_change):.4f} after intervention.")
            else:
                interp.append("No significant change in the trend/slope.")
                
            summary_text += "\n" + " ".join(interp)
            
            sections.append({
                'type': 'table',
                'title': 'Regression Results',
                'headers': ['Term', 'Coefficient', 'P-Value'],
                'data': [
                    ['Intercept', f"{params['Intercept']:.4f}", f"{pvalues['Intercept']:.4f}"],
                    ['Baseline Trend (Time)', f"{baseline_slope:.4f}", f"{pvalues['Time']:.4f}"],
                    ['Level Change (Intervention)', f"{level_change:.4f}", f"{level_p:.4f}"],
                    ['Slope Change (Time_After)', f"{slope_change:.4f}", f"{slope_p:.4f}"]
                ]
            })
            
            # Visualizations
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot Raw Data
                ax.scatter(df_clean['Time'], df_clean['Outcome'], color='gray', alpha=0.5, label='Observed Data')
                
                # Plot Fitted Lines
                # Pre-intervention
                pre_df = df_clean[df_clean['Time'] < cutoff]
                if not pre_df.empty:
                    # Model prediction: Y = b0 + b1*Time (Intervention=0, Time_After=0)
                    pre_pred = model.predict(pre_df)
                    ax.plot(pre_df['Time'], pre_pred, 'b-', linewidth=2, label='Pre-Intervention Trend')
                    
                # Post-intervention
                post_df = df_clean[df_clean['Time'] >= cutoff]
                if not post_df.empty:
                    post_pred = model.predict(post_df)
                    ax.plot(post_df['Time'], post_pred, 'r-', linewidth=2, label='Post-Intervention Trend')
                
                # Counterfactual (if trend continued)
                if not post_df.empty:
                    # Counterfactual: Y = b0 + b1*Time (Intervention=0, Time_After=0)
                    # We manually calc using just intercept and time coeff
                    counterfactual = params['Intercept'] + params['Time'] * post_df['Time']
                    ax.plot(post_df['Time'], counterfactual, 'b--', alpha=0.5, label='Counterfactual')

                ax.axvline(cutoff, color='k', linestyle=':', label='Intervention')
                ax.set_xlabel(time_col)
                ax.set_ylabel(outcome_col)
                ax.set_title("Interrupted Time Series Plot")
                ax.legend()
                
                artifacts.append({
                    "type": "plot",
                    "id": "its_plot",
                    "title": "Segmented Regression Plot",
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
        return results.get('summary', "ITS Analysis Completed.")
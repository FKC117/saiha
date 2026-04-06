"""
Difference-in-Differences (DiD) Tool
Estimates the causal effect of a treatment by comparing the change in outcome over time between a treatment group and a control group.
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

class DifferenceInDifferencesTool(BaseAnalysisTool):
    """Tool for Difference-in-Differences (DiD) estimation."""

    @property
    def name(self) -> str:
        return "difference_in_differences"

    @property
    def description(self) -> str:
        return "Estimate causal effects by comparing changes over time between treatment and control groups."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="difference_in_differences")
        params.add_parameter(
            ToolParameter(
                name="outcome_variable",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Outcome Variable (Y)",
                description="The dependent variable to analyze.",
                required=True,
                help_text="Select the continuous outcome variable."
            )
        )
        params.add_parameter(
            ToolParameter(
                name="group_variable",
                parameter_type=ParameterType.COLUMN_SELECT,
                label="Group Variable (Treatment/Control)",
                description="Binary variable indicating Treatment (1) vs Control (0) group.",
                required=True,
                column_source="numeric,categorical",
                help_text="Select the binary variable distinguishing Treatment vs Control groups."
            )
        )
        params.add_parameter(
            ToolParameter(
                name="time_variable",
                parameter_type=ParameterType.COLUMN_SELECT,
                label="Time Variable (Pre/Post)",
                description="Binary variable indicating Post-Intervention (1) vs Pre-Intervention (0).",
                required=True,
                column_source="numeric,categorical",
                help_text="Select the binary variable distinguishing Pre vs Post time periods."
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters
            parameters = kwargs
            outcome_col = parameters.get("outcome_variable")
            group_col = parameters.get("group_variable")
            time_col = parameters.get("time_variable")

            if not all([outcome_col, group_col, time_col]):
                return {"status": "error", "summary": "Missing required variables."}

            df = self.load_dataset(columns=[outcome_col, group_col, time_col])
            df_clean = df.dropna()

            if df_clean.empty:
                return {"status": "error", "summary": "No data remaining after removing missing values."}

            # 2. Fit DiD Model (Interaction Model)
            # Y = beta0 + beta1*Time + beta2*Group + beta3*(Time*Group) + epsilon
            # beta3 is the DiD estimator.
            
            # Ensure group/time are treated as categories if needed, but statsmodels formula handles this well if 0/1.
            # We rename to safe names to avoid formula syntax errors
            df_safe = df_clean.rename(columns={
                outcome_col: 'Outcome',
                group_col: 'Group',
                time_col: 'Time'
            })
            
            # Use 'C()' to force categorical interpretation if they are string/object
            # But simpler to just let statsmodels handle it and find the term
            model = smf.ols("Outcome ~ Time * Group", data=df_safe).fit()
            
            # 3. Extract Results
            # Robustly identify the interaction term
            # It should contain both 'Time' and 'Group' and a ':' separator
            interaction_term = None
            for term in model.params.index:
                if "Time" in term and "Group" in term and ":" in term:
                    interaction_term = term
                    break
            
            if not interaction_term:
                # Fallback: Check strict names just in case
                 if "Time:Group" in model.params:
                     interaction_term = "Time:Group"
                 elif "Group:Time" in model.params:
                     interaction_term = "Group:Time"

            if not interaction_term:
                return {"status": "error", "summary": f"Could not identify interaction term in model. Available terms: {list(model.params.index)}"}

            did_estimate = model.params[interaction_term]
            p_value = model.pvalues[interaction_term]
            conf_int = model.conf_int().loc[interaction_term]
            
            # Group Means for Table
            means = df_safe.groupby(['Group', 'Time'])['Outcome'].mean().unstack()
            # Expected structure: 
            # Time      0         1
            # Group
            # 0      Mean_C_Pre  Mean_C_Post
            # 1      Mean_T_Pre  Mean_T_Post
            
            # Handle if 0/1 or actual names are used. We assume 0/1 or similar sortable order for Pre/Post
            # Generally assume: Min val = Pre, Max val = Post; Min group = Control, Max group = Treatment.
            # But the regression handles the logic based on values. 
            
            # Validation: Ensure we have data for all 4 cells (2 groups x 2 time periods)
            if means.size != 4 or means.isnull().values.any():
                missing_info = []
                try:
                    # Check safely which are missing
                    if pd.isna(means.iloc[0, 0]): missing_info.append("Control Group (First Level) at Pre-Intervention (First Time)")
                    if pd.isna(means.iloc[0, 1]): missing_info.append("Control Group (First Level) at Post-Intervention (Second Time)")
                    if pd.isna(means.iloc[1, 0]): missing_info.append("Treatment Group (Second Level) at Pre-Intervention (First Time)")
                    if pd.isna(means.iloc[1, 1]): missing_info.append("Treatment Group (Second Level) at Post-Intervention (Second Time)")
                except IndexError:
                     missing_info.append("One or more entire groups or time periods are missing from the dataset.")

                return {
                    "status": "error", 
                    "summary": f"Data Missing for DiD Analysis. The following combinations satisfy 0 observations:\n- " + "\n- ".join(missing_info) + "\n\nEnsure your dataset contains observations for both groups in both time periods."
                }

            control_pre = means.iloc[0, 0]
            control_post = means.iloc[0, 1]
            treat_pre = means.iloc[1, 0]
            treat_post = means.iloc[1, 1]
            
            # Manual DiD calc to verify
            did_manual = (treat_post - treat_pre) - (control_post - control_pre)
            
            # 4. Construct Output
            artifacts = []
            sections = []
            
            summary_text = f"Difference-in-Differences Analysis for '{outcome_col}'.\n"
            summary_text += f"DiD Estimator: {did_estimate:.4f} (p={p_value:.4f}).\n"
            
            if p_value < 0.05:
                summary_text += "Result is Statistically Significant at p < 0.05."
            else:
                summary_text += "Result is Not Statistically Significant."
            
            # Expanded Interpretation
            interp_lines = []
            interp_lines.append(f"- **Treatment Group Baseline**: {treat_pre:.2f} -> Post: {treat_post:.2f} (Change: {treat_post-treat_pre:.2f})")
            interp_lines.append(f"- **Control Group Baseline**: {control_pre:.2f} -> Post: {control_post:.2f} (Change: {control_post-control_pre:.2f})")
            interp_lines.append(f"- **Net Impact (DiD)**: {did_estimate:.4f}")
            interp_lines.append(f"- **What this means**: The treatment group changed by {did_estimate:.2f} units *more* (or less) than would be expected based on the control group's trend.")
            
            summary_text += "\n\n### Detailed Breakdown:\n" + "\n".join(interp_lines)

            # Table
            sections.append({
                'type': 'table',
                'title': 'Group Means Table',
                'headers': ['Group', 'Pre-Intervention', 'Post-Intervention', 'Change'],
                'data': [
                    ['Control', f"{control_pre:.4f}", f"{control_post:.4f}", f"{control_post-control_pre:.4f}"],
                    ['Treatment', f"{treat_pre:.4f}", f"{treat_post:.4f}", f"{treat_post-treat_pre:.4f}"],
                    ['Difference', f"{treat_pre-control_pre:.4f}", f"{treat_post-control_post:.4f}", f"{did_manual:.4f} (DiD)"]
                ]
            })

            # Visualizations
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Interaction Plot
                # X-axis: Time (0, 1), Y-axis: Outcome, Lines: Group
                groups = [0, 1] # Assumes mapped/coerced to 0/1 or sorted
                
                # Plot Control Trend
                ax.plot([0, 1], [control_pre, control_post], 'o-', label='Control', color='blue', linewidth=2)
                
                # Plot Treatment Trend
                ax.plot([0, 1], [treat_pre, treat_post], 'o-', label='Treatment', color='orange', linewidth=2)
                
                # Plot Counterfactual (Treatment if it followed Control trend)
                # Counterfactual Post = Treat Pre + (Control Post - Control Pre)
                counterfactual_post = treat_pre + (control_post - control_pre)
                ax.plot([0, 1], [treat_pre, counterfactual_post], 'o--', label='Parallel Trend (Counterfactual)', color='gray', alpha=0.7)
                
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Pre-Intervention', 'Post-Intervention'])
                ax.set_ylabel(outcome_col)
                ax.set_title("Difference-in-Differences Interaction Plot")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Annotate DiD
                ax.annotate(f"DiD Effect: {did_estimate:.2f}", 
                            xy=(1, (treat_post + counterfactual_post)/2),
                            xytext=(1.1, (treat_post + counterfactual_post)/2),
                            arrowprops=dict(arrowstyle="-[, widthB=1.0, lengthB=0.2", lw=1.5),
                            ha='left', va='center', fontsize=10)

                plt.tight_layout()
                artifacts.append({
                    "type": "plot",
                    "id": "did_interaction_plot",
                    "title": "Interaction Plot (Parallel Trends)",
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
        return results.get('summary', "DiD Analysis Completed.")
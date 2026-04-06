"""
Monte Carlo Simulation Tool
Performs stochastic simulation based on historical data to forecast future outcomes with confidence intervals.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class MonteCarloSimulationTool(BaseAnalysisTool):
    """Tool for Monte Carlo Simulation."""

    @property
    def name(self) -> str:
        return "monte_carlo_simulation"

    @property
    def description(self) -> str:
        return "Forecast future outcomes using probabilistic Monte Carlo Simulation."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="monte_carlo_simulation")
        params.add_parameter(
            ToolParameter(
                name="data_column",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Historical Data Variable",
                description="The numeric variable (e.g., Sales, Returns) to base the simulation on.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="num_simulations",
                parameter_type=ParameterType.NUMBER,
                label="Number of Simulations",
                description="How many scenarios to run (Max 1000).",
                required=True,
                default_value=100,
                validation_rules={"minimum": 1, "maximum": 1000}
            )
        )
        params.add_parameter(
            ToolParameter(
                name="periods",
                parameter_type=ParameterType.NUMBER,
                label="Forecast Periods",
                description="How many steps into the future to simulate (Max 100).",
                required=True,
                default_value=10,
                validation_rules={"minimum": 1, "maximum": 100}
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters and apply strict limits (Resource Guard)
            parameters = kwargs
            col = parameters.get("data_column")
            
            # Clamp values to prevent resource exhaustion even if validation is bypassed
            num_sims = min(int(parameters.get("num_simulations", 100)), 1000)
            periods = min(int(parameters.get("periods", 10)), 100)

            if not col:
                return {"status": "error", "summary": "Variable selection is required."}

            df = self.load_dataset(columns=[col])
            series = df[col].dropna()

            if series.empty or len(series) < 2:
                return {"status": "error", "summary": "Insufficient data for simulation."}

            # 2. Check Data & Select Model (Additive vs Multiplicative)
            is_positive = (series > 0).all()
            
            mu, sigma, drift = 0.0, 0.0, 0.0
            mu_diff, sigma_diff = 0.0, 0.0

            if is_positive:
                pct_change = series.pct_change()
                pct_change = pct_change.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(pct_change) < 2:
                    is_positive = False 
                else:
                    mu = pct_change.mean()
                    sigma = pct_change.std()
                    drift = mu - 0.5 * sigma**2
            
            if not is_positive:
                diffs = series.diff().dropna()
                if len(diffs) < 2:
                     return {"status": "error", "summary": "Insufficient variation in data for simulation."}
                
                mu_diff = diffs.mean()
                sigma_diff = diffs.std()
            
            last_value = series.iloc[-1]
            
            # 3. Simulate (Vectorized - NO PYTHON LOOPS)
            # Generate all random shocks at once: (periods, num_sims)
            Z = np.random.normal(0, 1, (periods, num_sims))
            
            # Base array with starting values
            # shape: (periods+1, num_sims)
            price_paths = np.zeros((periods + 1, num_sims))
            price_paths[0] = last_value

            if is_positive:
                # Geometric: P_t = P_0 * exp(cumsum(drift + sigma*Z))
                daily_log_returns = drift + sigma * Z
                cumulative_log_returns = np.cumsum(daily_log_returns, axis=0)
                price_paths[1:] = last_value * np.exp(cumulative_log_returns)
            else:
                # Additive: P_t = P_0 + cumsum(mu + sigma*Z)
                daily_shocks = mu_diff + sigma_diff * Z
                cumulative_shocks = np.cumsum(daily_shocks, axis=0)
                price_paths[1:] = last_value + cumulative_shocks

            # Numerical Stability: Cap values to prevent plotting crashes with extreme divergence
            # We cap at +/- 5x the historical range (50x was too much and caused plotting memory errors)
            hist_max = series.max()
            hist_min = series.min()
            margin = (hist_max - hist_min) * 5.0 if hist_max != hist_min else abs(last_value) * 5.0
            
            upper_cap = hist_max + margin
            lower_cap = hist_min - margin if not is_positive else 0
            
            price_paths = np.clip(price_paths, lower_cap, upper_cap)
            
            # Ensure no NaNs or Infs reached the analysis step
            price_paths = np.nan_to_num(price_paths, nan=last_value, posinf=upper_cap, neginf=lower_cap)

            # 4. Analyze Results
            final_values = price_paths[-1]
            mean_final = np.mean(final_values)
            median_final = np.median(final_values)
            percentile_5 = np.percentile(final_values, 5)
            percentile_95 = np.percentile(final_values, 95)
            
            # 5. Output
            artifacts = []
            sections = []
            
            summary_text = f"Monte Carlo Simulation ({num_sims} runs, {periods} periods).\n"
            summary_text += f"Starting Value: {last_value:.2f}\n"
            summary_text += f"Projected Median: {median_final:.2f}\n"
            summary_text += f"90% Confidence Interval: [{percentile_5:.2f} - {percentile_95:.2f}]"
            
            sections.append({
                'type': 'table',
                'title': 'Simulation Statistics',
                'headers': ['Metric', 'Value'],
                'data': [
                    ['Historical Mean Return', f"{mu:.4%} per period" if is_positive else f"{mu_diff:.4f} (Avg Change)"],
                    ['Historical Volatility (Std)', f"{sigma:.4%} per period" if is_positive else f"{sigma_diff:.4f} (Std Dev)"],
                    ['Projected Median', f"{median_final:.2f}"],
                    ['Likely Best Case (95%)', f"{percentile_95:.2f}"],
                    ['Likely Worst Case (5%)', f"{percentile_5:.2f}"]
                ]
            })
            
            # Visualization
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot first 50 paths
                ax.plot(price_paths[:, :50], color='blue', alpha=0.1)
                
                # Plot Median path
                median_path = np.median(price_paths, axis=1)
                ax.plot(median_path, color='red', linewidth=2, label='Median Forecast')
                
                ax.set_title(f"Monte Carlo Simulation: {col} Forecast")
                ax.set_xlabel("Periods Future")
                ax.set_ylabel("Value")
                ax.legend()
                
                artifacts.append({
                    "type": "plot",
                    "id": "mc_fan_chart",
                    "title": "Simulation Paths",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)
                
            # Histogram of Final Values
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(8, 5))
                # Explicitly set bins and range to avoid memory explosion in automatic bin estimation
                sns.histplot(final_values, kde=True, ax=ax, color='purple', bins=30)
                ax.axvline(percentile_5, color='r', linestyle='--', label='5th Percentile')
                ax.axvline(percentile_95, color='r', linestyle='--', label='95th Percentile')
                ax.set_title(f"Distribution of Outcomes at Period {periods}")
                ax.legend()
                
                artifacts.append({
                    "type": "plot",
                    "id": "mc_final_dist",
                    "title": "Final Outcome Distribution",
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
         return results.get('summary', "Simulation Completed.")
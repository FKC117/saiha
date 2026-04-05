# d:/quantly/quanta/quantalytics/ai_agents/tools/auto_arima_tool.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
from django.core.files.storage import default_storage
from typing import Any, Dict, List

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class AutoArimaTool(BaseAnalysisTool):
    """
    A tool to automatically find the best ARIMA model and generate forecasts.
    """

    @property
    def name(self) -> str:
        return "auto_arima"

    @property
    def description(self) -> str:
        return "Automatically finds the best ARIMA model for a time series and generates future forecasts."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="date_column", parameter_type=ParameterType.COLUMN_SELECT,
            label="Date/Time Column", description="Select the column containing the date or time information.",
            required=True, column_source="date"
        ))
        params.add_parameter(ToolParameter(
            name="value_column", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Value Column", description="Select the numeric column to forecast.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="forecast_periods", parameter_type=ParameterType.NUMBER,
            label="Forecast Periods", description="The number of periods to forecast into the future.",
            required=True, default_value=12
        ))
        params.add_parameter(ToolParameter(
            name="seasonal", parameter_type=ParameterType.CHECKBOX,
            label="Consider Seasonality", description="Allow the model to search for seasonal patterns.",
            required=False, default_value=True
        ))
        params.add_parameter(ToolParameter(
            name="seasonal_period", parameter_type=ParameterType.NUMBER,
            label="Seasonal Period (m)", description="The number of periods in a seasonal cycle (e.g., 12 for monthly, 7 for daily).",
            required=False, default_value=12, help_text="Required if 'Consider Seasonality' is checked."
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        if not PMDARIMA_AVAILABLE:
            return {"status": "error", "summary": "The 'pmdarima' library is not installed. Please install it to use Auto ARIMA."}

        try:
            parameters = kwargs
            date_col = parameters.get("date_column")
            value_col = parameters.get("value_column")
            forecast_periods = int(parameters.get("forecast_periods", 12))
            is_seasonal = str(parameters.get("seasonal", "true")).lower() in ('true', 'on', '1')
            m = int(parameters.get("seasonal_period", 12)) if is_seasonal else 1

            if not all([date_col, value_col]):
                return {"status": "error", "summary": "Date Column and Value Column are required."}

            # Use efficient column projection loading from BaseAnalysisTool
            df = self.load_dataset(columns=date_column)

            # Prepare the time series
            ts_df = df[[date_col, value_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col])
            ts_df = ts_df.sort_values(by=date_col).set_index(date_col)
            
            # Resample to a consistent daily frequency, summing values for each day.
            # This is crucial for models that require a regular time index for forecasting.
            series = ts_df[value_col].resample('D').sum()

            # Find the best ARIMA model
            model = pm.auto_arima(series, seasonal=is_seasonal, m=m,
                                  stepwise=True,  # Use stepwise algorithm to speed up the search
                                  suppress_warnings=True,  # Don't print warnings
                                  error_action='ignore',  # Don't stop on models that fail to fit
                                  trace=True,  # Print status updates to the console
                                  max_p=5, max_q=5,  # Limit the search space for p and q
                                  max_P=2, max_Q=2   # Limit the search space for seasonal P and Q
                                  )

            # Generate forecast
            forecast_values, conf_int = model.predict(n_periods=forecast_periods, return_conf_int=True)
            
            # Create forecast DataFrame
            forecast_index = pd.date_range(start=series.index[-1], periods=forecast_periods + 1, freq=series.index.freq)[1:]
            forecast_df = pd.DataFrame({
                'mean': forecast_values,
                'mean_ci_lower': conf_int[:, 0],
                'mean_ci_upper': conf_int[:, 1]
            }, index=forecast_index)
            forecast_df.index.name = 'Forecast Date'
            forecast_df.reset_index(inplace=True)

            # Add model order to parameters for interpretation
            meta_params = parameters.copy()
            meta_params['best_order'] = model.order
            if is_seasonal and hasattr(model, 'seasonal_order'):
                meta_params['best_seasonal_order'] = model.seasonal_order


            summary = f"Auto ARIMA found best model: {model.order} {'with seasonal order ' + str(model.seasonal_order) if is_seasonal else ''}. Forecasted for {forecast_periods} periods."

            artifacts: List[Dict[str, Any]] = []
            sections: List[Dict[str, Any]] = []

            # Generate plot
            with PlotUtils.setup_plotting():
                fig = None
                try:
                    fig, ax = plt.subplots(figsize=(14, 7))
                    ax.plot(series.index, series, label='Observed')
                    ax.plot(forecast_df['Forecast Date'], forecast_df['mean'], color='r', label='Forecast')
                    ax.fill_between(forecast_df['Forecast Date'],
                                    forecast_df['mean_ci_lower'],
                                    forecast_df['mean_ci_upper'], color='k', alpha=.15)
                    ax.set_title(f'Auto ARIMA Forecast for {value_col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel(value_col)
                    ax.legend()
                    artifacts.append({"type": "plot", "id": "auto_arima_forecast_plot", "title": "Auto ARIMA Forecast", "content": PlotUtils.fig_to_base64(fig)})
                finally:
                    if fig: plt.close(fig)

            # --- Prepare sections after plotting ---

            # Convert timestamp to string for JSON serialization in the table
            forecast_df['Forecast Date'] = forecast_df['Forecast Date'].dt.strftime('%Y-%m-%d')

            # Add forecast table
            sections.append({
                'type': 'table', 'title': 'Forecasted Values',
                'headers': forecast_df.columns.tolist(),
                'data': forecast_df.round(4).to_numpy().tolist()
            })

            # Add model summary tables
            summary_tables = model.summary().tables
            if len(summary_tables) > 0:
                # First table is key-value, reformat it
                model_overview_data = []
                for row in summary_tables[0].data:
                    model_overview_data.append([row[0], row[1]])
                    model_overview_data.append([row[2], row[3]])
                sections.append({
                    'type': 'table', 'title': 'Best ARIMA Model Summary',
                    'headers': ['Statistic', 'Value'],
                    'data': model_overview_data
                })

            # Add other tables (coefficients, diagnostics)
            for i, table in enumerate(summary_tables[1:]):
                sections.append({
                    'type': 'table', 'title': f'Best Model Details (Table {i+1})',
                    'headers': [str(h) for h in table.data[0]],
                    'data': table.data[1:]
                })

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": meta_params},
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> str:
        """
        Provides a formal interpretation of the Auto ARIMA model results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            best_order = params.get('best_order')
            best_seasonal_order = params.get('best_seasonal_order')
            forecast_periods = params.get('forecast_periods', 'N')

            aic_val = "N/A"
            significant_coeffs = []

            summary_section = next((s for s in results.get('sections', []) if s.get('title') == 'Best ARIMA Model Summary'), None)
            if summary_section:
                for row in summary_section.get('data', []):
                    if row[0].strip() == 'AIC':
                        aic_val = row[1]
                        break

            coeffs_section = next((s for s in results.get('sections', []) if 'Best Model Details' in s.get('title', '')), None)
            if coeffs_section:
                headers = [str(h).strip() for h in coeffs_section.get('headers', [])]
                try:
                    p_val_idx = headers.index('P>|z|')
                    coeff_name_idx = 0 # First column
                    for row in coeffs_section.get('data', []):
                        if float(row[p_val_idx]) < 0.05:
                            significant_coeffs.append(row[coeff_name_idx].strip())
                except (ValueError, IndexError):
                    pass # Could not parse coefficients

            order_str = f"ARIMA{best_order}" if best_order else "an ARIMA model"
            if best_seasonal_order:
                order_str += f" with seasonal order {best_seasonal_order}"

            interpretation_parts = [f"Auto ARIMA selected the best model based on the AIC. The chosen model is {order_str}."]
            interpretation_parts.append(f"The model's AIC (Akaike Information Criterion) is {aic_val}. Lower AIC values suggest a better model fit.")

            if significant_coeffs:
                interpretation_parts.append(f"Statistically significant coefficients (p < 0.05) in this model include: {', '.join(significant_coeffs)}.")
            
            interpretation_parts.append(f"This model was used to generate a forecast for the next {forecast_periods} periods.")

            return " ".join(interpretation_parts)

        except (ValueError, TypeError, IndexError, KeyError):
            return "Could not automatically interpret the Auto ARIMA model results."
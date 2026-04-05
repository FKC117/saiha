# d:/quantly/quanta/quantalytics/ai_agents/tools/arima_forecasting_tool.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from django.core.files.storage import default_storage
from typing import Any, Dict, List

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class ArimaForecastingTool(BaseAnalysisTool):
    """
    A tool to fit an ARIMA model and generate forecasts.
    """

    @property
    def name(self) -> str:
        return "arima_forecasting"

    @property
    def description(self) -> str:
        return "Fits an ARIMA model to a time series and generates future forecasts."

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
            name="p_order", parameter_type=ParameterType.NUMBER,
            label="AR Order (p)", description="The order of the autoregressive part of the model.",
            required=True, default_value=1
        ))
        params.add_parameter(ToolParameter(
            name="d_order", parameter_type=ParameterType.NUMBER,
            label="Differencing Order (d)", description="The degree of differencing needed to make the series stationary.",
            required=True, default_value=1
        ))
        params.add_parameter(ToolParameter(
            name="q_order", parameter_type=ParameterType.NUMBER,
            label="MA Order (q)", description="The order of the moving-average part of the model.",
            required=True, default_value=1
        ))
        params.add_parameter(ToolParameter(
            name="forecast_periods", parameter_type=ParameterType.NUMBER,
            label="Forecast Periods", description="The number of periods to forecast into the future.",
            required=True, default_value=12
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            date_col = parameters.get("date_column")
            value_col = parameters.get("value_column")
            p = int(parameters.get("p_order", 1))
            d = int(parameters.get("d_order", 1))
            q = int(parameters.get("q_order", 1))
            forecast_periods = int(parameters.get("forecast_periods", 12))

            if not all([date_col, value_col]):
                return {"status": "error", "summary": "Date Column and Value Column are required."}

            # Use efficient column projection loading from BaseAnalysisTool
            df = self.load_dataset(columns=date_column)

            # Prepare the time series
            ts_df = df[[date_col, value_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col])
            ts_df = ts_df.sort_values(by=date_col).set_index(date_col)
            series = ts_df[value_col].dropna()

            # Fit ARIMA model
            model = ARIMA(series, order=(p, d, q))
            model_fit = model.fit()

            # Generate forecast
            forecast = model_fit.get_forecast(steps=forecast_periods)
            forecast_df = forecast.summary_frame()
            forecast_df.index.name = 'Forecast Date'
            forecast_df.reset_index(inplace=True)

            summary = f"ARIMA({p},{d},{q}) model fitted and forecasted for {forecast_periods} periods."

            artifacts: List[Dict[str, Any]] = []
            sections: List[Dict[str, Any]] = []

            # Add forecast table
            sections.append({
                'type': 'table', 'title': 'Forecasted Values',
                'headers': forecast_df.columns.tolist(),
                'data': forecast_df.round(4).to_numpy().tolist()
            })

            # Add model summary tables
            summary_tables = model_fit.summary().tables
            if len(summary_tables) > 0:
                # First table is key-value, reformat it
                model_overview_data = []
                for row in summary_tables[0].data:
                    model_overview_data.append([row[0], row[1]])
                    model_overview_data.append([row[2], row[3]])
                sections.append({
                    'type': 'table', 'title': 'ARIMA Model Summary',
                    'headers': ['Statistic', 'Value'],
                    'data': model_overview_data
                })
            
            # Add other tables (coefficients, diagnostics)
            for i, table in enumerate(summary_tables[1:]):
                sections.append({
                    'type': 'table', 'title': f'ARIMA Model Details (Table {i+1})',
                    'headers': [str(h) for h in table.data[0]],
                    'data': table.data[1:]
                })

            # Generate plot
            with PlotUtils.setup_plotting():
                fig = None
                try:
                    fig, ax = plt.subplots(figsize=(14, 7))
                    series.plot(ax=ax, label='Observed')
                    model_fit.get_prediction(dynamic=False).predicted_mean.plot(ax=ax, label='In-sample fit', alpha=.7)
                    forecast.predicted_mean.plot(ax=ax, label='Forecast')
                    ax.fill_between(forecast.conf_int().index,
                                    forecast.conf_int().iloc[:, 0],
                                    forecast.conf_int().iloc[:, 1], color='k', alpha=.2)
                    ax.set_xlabel('Date')
                    ax.set_ylabel(value_col)
                    ax.set_title(f'ARIMA({p},{d},{q}) Forecast')
                    ax.legend()
                    artifacts.append({"type": "plot", "id": "arima_forecast_plot", "title": "ARIMA Forecast", "content": PlotUtils.fig_to_base64(fig)})
                finally:
                    if fig: plt.close(fig)

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": parameters},
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> str:
        """
        Provides a formal interpretation of the ARIMA model results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            p = params.get('p_order', '?')
            d = params.get('d_order', '?')
            q = params.get('q_order', '?')
            forecast_periods = params.get('forecast_periods', 'N')

            aic_val = "N/A"
            significant_coeffs = []

            summary_section = next((s for s in results.get('sections', []) if s.get('title') == 'ARIMA Model Summary'), None)
            if summary_section:
                for row in summary_section.get('data', []):
                    if row[0].strip() == 'AIC':
                        aic_val = row[1]
                        break

            coeffs_section = next((s for s in results.get('sections', []) if 'ARIMA Model Details' in s.get('title', '')), None)
            if coeffs_section:
                headers = [h.strip() for h in coeffs_section.get('headers', [])]
                try:
                    p_val_idx = headers.index('P>|z|')
                    coeff_name_idx = 0 # First column
                    for row in coeffs_section.get('data', []):
                        if float(row[p_val_idx]) < 0.05:
                            significant_coeffs.append(row[coeff_name_idx].strip())
                except (ValueError, IndexError):
                    pass # Could not parse coefficients

            interpretation_parts = [f"An ARIMA({p},{d},{q}) model was fitted to the time series."]
            interpretation_parts.append(f"The model's AIC (Akaike Information Criterion) is {aic_val}. Lower AIC values suggest a better model fit when comparing different models.")

            if significant_coeffs:
                interpretation_parts.append(f"Statistically significant coefficients (p < 0.05) were found for: {', '.join(significant_coeffs)}.")
            else:
                interpretation_parts.append("No statistically significant coefficients were found at the 0.05 level.")

            interpretation_parts.append(f"The model was used to generate a forecast for the next {forecast_periods} periods, as shown in the plot and table.")

            return " ".join(interpretation_parts)

        except (ValueError, TypeError, IndexError, KeyError):
            return "Could not automatically interpret the ARIMA model results."
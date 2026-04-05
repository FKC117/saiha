# d:/quantly/quanta/quantalytics/ai_agents/tools/time_series_decomposition_tool.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from django.core.files.storage import default_storage
from typing import Any, Dict, List

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class TimeSeriesDecompositionTool(BaseAnalysisTool):
    """
    A tool to perform time series decomposition.
    """

    @property
    def name(self) -> str:
        return "time_series_decomposition"

    @property
    def description(self) -> str:
        return "Decomposes a time series into trend, seasonal, and residual components."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="date_column", parameter_type=ParameterType.COLUMN_SELECT,
            label="Date/Time Column", description="Select the column containing the date or time information.",
            required=True, column_source="date"
        ))
        params.add_parameter(ToolParameter(
            name="value_column", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Value Column", description="Select the numeric column to analyze.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="model", parameter_type=ParameterType.SELECT,
            label="Decomposition Model", description="Choose the model type for decomposition.",
            required=True, default_value="additive",
            options=[
                {"value": "additive", "label": "Additive"},
                {"value": "multiplicative", "label": "Multiplicative"},
            ]
        ))
        params.add_parameter(ToolParameter(
            name="period", parameter_type=ParameterType.NUMBER,
            label="Seasonal Period (integer)", description="The number of observations per seasonal cycle (e.g., 12 for monthly data, 7 for daily).",
            required=True, help_text="Must be an integer greater than 1."
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            date_col = parameters.get("date_column")
            value_col = parameters.get("value_column")
            model = parameters.get("model", "additive")
            period_str = parameters.get("period")

            if not all([date_col, value_col, period_str]):
                return {"status": "error", "summary": "Date Column, Value Column, and Seasonal Period are all required."}

            try:
                period = int(period_str)
                if period <= 1:
                    raise ValueError()
            except (ValueError, TypeError):
                return {"status": "error", "summary": "Seasonal Period must be an integer greater than 1."}

            # Use efficient column projection loading from BaseAnalysisTool
            df = self.load_dataset(columns=date_column)

            if date_col not in df.columns or value_col not in df.columns:
                return {"status": "error", "summary": "Selected columns not found in the dataset."}

            # Prepare the time series
            ts_df = df[[date_col, value_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col])
            ts_df = ts_df.sort_values(by=date_col).set_index(date_col)
            series = ts_df[value_col].dropna()

            if len(series) < 2 * period:
                return {"status": "error", "summary": f"Not enough data for the specified period. You need at least {2 * period} data points, but have {len(series)}."}

            decomposition = seasonal_decompose(series, model=model, period=period)

            # Calculate strength of trend and seasonality for interpretation
            trend = decomposition.trend.dropna()
            seasonal = decomposition.seasonal.dropna()
            resid = decomposition.resid.dropna()
            
            # Strength of trend: 1 - Var(Resid) / Var(Observed - Seasonal)
            detrended = series - seasonal
            strength_of_trend = max(0, 1 - np.var(resid) / np.var(detrended.dropna()))

            # Strength of seasonality: 1 - Var(Resid) / Var(Observed - Trend)
            deseasonalized = series - trend
            strength_of_seasonality = max(0, 1 - np.var(resid) / np.var(deseasonalized.dropna()))


            summary = f"Time series decomposition of '{value_col}' completed using a {model} model with a period of {period}."

            artifacts: List[Dict[str, Any]] = []
            sections: List[Dict[str, Any]] = []

            # Create a table with a sample of the decomposed values
            try:
                result_df = pd.DataFrame({
                    'Observed': decomposition.observed,
                    'Trend': decomposition.trend,
                    'Seasonal': decomposition.seasonal,
                    'Residual': decomposition.resid
                }).dropna().head(20).round(4)
                
                result_df.reset_index(inplace=True)
                result_df[date_col] = result_df[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')

                sections.append({
                    'type': 'table', 'title': 'Decomposition Values (Sample)',
                    'headers': result_df.columns.tolist(),
                    'data': result_df.values.tolist(),
                    'footer': 'Showing a sample of the first 20 decomposed time series components.'
                })
            except Exception as table_ex:
                sections.append({"type": "text", "title": "Decomposition Table Failed", "content": str(table_ex)})

            with PlotUtils.setup_plotting():
                fig = None
                try:
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                    decomposition.observed.plot(ax=ax1, legend=False)
                    ax1.set_ylabel('Observed')
                    decomposition.trend.plot(ax=ax2, legend=False)
                    ax2.set_ylabel('Trend')
                    decomposition.seasonal.plot(ax=ax3, legend=False)
                    ax3.set_ylabel('Seasonal')
                    decomposition.resid.plot(ax=ax4, legend=False)
                    ax4.set_ylabel('Residual')
                    fig.suptitle(f"Time Series Decomposition of {value_col}", fontsize=16)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    artifacts.append({"type": "plot", "id": "ts_decomposition_plot", "title": "Decomposition Plot", "content": PlotUtils.fig_to_base64(fig)})
                finally:
                    if fig: plt.close(fig)

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name, 
                    "parameters": parameters,
                    "statistical_results": {
                        "strength_of_trend": strength_of_trend,
                        "strength_of_seasonality": strength_of_seasonality,
                        "trend_start": trend.iloc[0] if not trend.empty else None,
                        "trend_end": trend.iloc[-1] if not trend.empty else None,
                    }
                },
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> str:
        """
        Provides a formal interpretation of the time series decomposition results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            stats = results.get('meta', {}).get('statistical_results', {})
            
            model = params.get('model', 'additive')
            value_col = params.get('value_column', 'the series')

            sot = stats.get('strength_of_seasonality')
            sotrend = stats.get('strength_of_trend')
            trend_start = stats.get('trend_start')
            trend_end = stats.get('trend_end')

            interpretation_parts = [f"The time series for '{value_col}' was decomposed using a {model} model."]

            # Interpret Trend
            if sotrend is not None and trend_start is not None and trend_end is not None:
                trend_direction = "increasing" if trend_end > trend_start else "decreasing" if trend_end < trend_start else "stable"
                trend_strength_desc = "strong" if sotrend > 0.8 else "moderate" if sotrend > 0.5 else "weak"
                interpretation_parts.append(f"The analysis reveals a {trend_strength_desc}, generally {trend_direction} trend (Strength: {sotrend:.2f}).")

            # Interpret Seasonality
            if sot is not None:
                season_strength_desc = "strong" if sot > 0.8 else "moderate" if sot > 0.5 else "weak"
                interpretation_parts.append(f"The data exhibits {season_strength_desc} seasonality (Strength: {sot:.2f}).")

            interpretation_parts.append("The decomposition plot visually separates these components from the random residual noise.")

            return " ".join(interpretation_parts)

        except (ValueError, TypeError, IndexError, KeyError):
            return "Could not automatically interpret the time series decomposition results."
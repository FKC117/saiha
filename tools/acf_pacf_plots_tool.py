
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf, adfuller
from django.core.files.storage import default_storage
from typing import Any, Dict, List

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class AcfPacfPlotsTool(BaseAnalysisTool):
    """
    A tool to generate ACF and PACF plots for a time series.
    """

    @property
    def name(self) -> str:
        return "acf_pacf_plots"

    @property
    def description(self) -> str:
        return "Generates Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots to analyze time series data."

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
            name="lags", parameter_type=ParameterType.NUMBER,
            label="Number of Lags", description="The number of time lags to include in the plots (e.g., 40).",
            required=False, default_value=40
        ))
        params.add_parameter(ToolParameter(
            name="alpha", parameter_type=ParameterType.SELECT,
            label="Significance Level (α)", description="Significance level for confidence intervals.",
            required=True, default_value="0.05",
            options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            date_col = parameters.get("date_column")
            value_col = parameters.get("value_column")
            lags_str = parameters.get("lags", "40")
            alpha = float(parameters.get("alpha", 0.05))

            if not all([date_col, value_col]):
                return {"status": "error", "summary": "Date Column and Value Column are required."}

            try:
                lags = int(lags_str) if lags_str else 40
            except (ValueError, TypeError):
                return {"status": "error", "summary": "Number of Lags must be an integer."}

            # Use efficient column projection loading from BaseAnalysisTool
            df = self.load_dataset(columns=date_column)

            if date_col not in df.columns or value_col not in df.columns:
                return {"status": "error", "summary": "Selected columns not found in the dataset."}

            # Prepare the time series
            ts_df = df[[date_col, value_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col])
            ts_df = ts_df.sort_values(by=date_col).set_index(date_col)
            series = ts_df[value_col].dropna()

            summary = f"ACF and PACF plots generated for '{value_col}' with {lags} lags."

            artifacts: List[Dict[str, Any]] = []
            sections: List[Dict[str, Any]] = []

            # Perform Augmented Dickey-Fuller test for stationarity
            try:
                adf_result = adfuller(series)
                adf_p_value = adf_result[1]
                is_stationary = adf_p_value < 0.05

                adf_data = [
                    ['ADF Test Statistic', f"{adf_result[0]:.4f}"],
                    ['p-value', f"{adf_p_value:.4f}"],
                    ['Lags Used', adf_result[2]],
                    ['Number of Observations', adf_result[3]],
                    ['Critical Value (1%)', f"{adf_result[4]['1%']:.4f}"],
                    ['Critical Value (5%)', f"{adf_result[4]['5%']:.4f}"],
                    ['Critical Value (10%)', f"{adf_result[4]['10%']:.4f}"],
                ]
                sections.append({
                    'type': 'table', 'title': 'Augmented Dickey-Fuller Test for Stationarity',
                    'headers': ['Statistic', 'Value'],
                    'data': adf_data
                })

                interpretation = (
                    f"The p-value is **{adf_p_value:.4f}**. A p-value below 0.05 suggests the time series is **stationary**. "
                    f"Since the p-value is {'less' if is_stationary else 'not less'} than 0.05, the data is likely **{'stationary' if is_stationary else 'not stationary'}**. "
                    "If the data is not stationary, differencing (d > 0 in ARIMA) is recommended."
                )
                sections.append({"type": "text", "title": "Stationarity Interpretation", "content": interpretation})
            except Exception as adf_ex:
                sections.append({"type": "text", "title": "Stationarity Test Failed", "content": str(adf_ex)})

            # Calculate ACF and PACF values for the table
            try:
                acf_values, acf_confint = acf(series, nlags=lags, alpha=alpha)
                pacf_values, pacf_confint = pacf(series, nlags=lags, alpha=alpha, method='ywm')

                # Create a DataFrame for the results table
                lags_range = range(lags + 1)
                results_df = pd.DataFrame({
                    'Lag': lags_range,
                    'ACF': acf_values,
                    'ACF_Lower_CI': [ci[0] - acf_val for acf_val, ci in zip(acf_values, acf_confint)],
                    'ACF_Upper_CI': [ci[1] - acf_val for acf_val, ci in zip(acf_values, acf_confint)],
                    'PACF': np.nan,
                    'PACF_Lower_CI': np.nan,
                    'PACF_Upper_CI': np.nan,
                })
                results_df.loc[1:, 'PACF'] = pacf_values[1:]
                results_df.loc[1:, 'PACF_Lower_CI'] = [ci[0] - pacf_val for pacf_val, ci in zip(pacf_values[1:], pacf_confint[1:])]
                results_df.loc[1:, 'PACF_Upper_CI'] = [ci[1] - pacf_val for pacf_val, ci in zip(pacf_values[1:], pacf_confint[1:])]

                sections.append({
                    'type': 'table', 'title': 'ACF and PACF Values',
                    'headers': results_df.columns.tolist(),
                    'data': results_df.round(4).to_numpy().tolist()
                })
            except Exception as table_ex:
                sections.append({"type": "text", "title": "ACF/PACF Table Failed", "content": str(table_ex)})
            
            # Suggest p and q values
            try:
                # Find last significant lag for ACF (suggests q)
                # A value is significant if it's outside the confidence interval.
                # confint is [lower, upper], so we check if value is outside this.
                significant_acf_lags = np.where((acf_values < acf_confint[:, 0]) | (acf_values > acf_confint[:, 1]))[0]
                suggested_q = max(significant_acf_lags) if significant_acf_lags.any() else 0

                # Find last significant lag for PACF (suggests p)
                significant_pacf_lags = np.where((pacf_values < pacf_confint[:, 0]) | (pacf_values > pacf_confint[:, 1]))[0]
                suggested_p = max(significant_pacf_lags) if significant_pacf_lags.any() else 0

                suggestion_text = (
                    f"Based on the plots:\n"
                    f"- The **PACF** plot appears to cut off after lag **{suggested_p}**. This suggests an **AR({suggested_p})** model might be appropriate (p={suggested_p}).\n"
                    f"- The **ACF** plot appears to cut off after lag **{suggested_q}**. This suggests an **MA({suggested_q})** model might be appropriate (q={suggested_q}).\n\n"
                    "Review the plots to confirm this interpretation. If both plots tail off, an ARMA model may be needed."
                )
                sections.append({"type": "text", "title": "Suggested ARIMA Parameters (p, q)", "content": suggestion_text})
            except Exception as suggestion_ex:
                sections.append({"type": "text", "title": "Parameter Suggestion Failed", "content": str(suggestion_ex)})

            with PlotUtils.setup_plotting():
                fig = None
                try:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                    plot_acf(series, ax=ax1, lags=lags, alpha=alpha)
                    ax1.set_title('Autocorrelation Function (ACF)')
                    plot_pacf(series, ax=ax2, lags=lags, alpha=alpha, method='ywm')
                    ax2.set_title('Partial Autocorrelation Function (PACF)')
                    plt.tight_layout()
                    artifacts.append({"type": "plot", "id": "acf_pacf_plot", "title": "ACF and PACF Plots", "content": PlotUtils.fig_to_base64(fig)})
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
        Provides a formal interpretation of the ACF/PACF plots and stationarity test.
        """
        if results.get('status') != 'ok':
            return None

        try:
            interpretation_parts = []

            # Find the stationarity interpretation
            stationarity_section = next((s for s in results.get('sections', []) if s.get('title') == 'Stationarity Interpretation'), None)
            if stationarity_section:
                interpretation_parts.append(stationarity_section.get('content', ''))

            # Find the parameter suggestions
            suggestion_section = next((s for s in results.get('sections', []) if s.get('title') == 'Suggested ARIMA Parameters (p, q)'), None)
            if suggestion_section:
                interpretation_parts.append(suggestion_section.get('content', ''))

            if not interpretation_parts:
                return "Could not automatically interpret the ACF/PACF results. Please review the plots and tables."

            return "\n\n".join(filter(None, interpretation_parts))

        except (ValueError, TypeError, IndexError, KeyError):
            return "Could not automatically interpret the ACF/PACF results."
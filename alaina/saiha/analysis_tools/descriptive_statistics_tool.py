
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


class DescriptiveStatisticsTool(BaseAnalysisTool):
    """
    Computes comprehensive descriptive statistics for one or more numeric columns,
    including mean, median, std, skewness, kurtosis, percentiles, and visualizations.
    """

    @property
    def name(self) -> str:
        return "descriptive_statistics"

    @property
    def description(self) -> str:
        return (
            "Use for numeric deep-dive statistics: mean, median, std, variance, skewness, kurtosis, "
            "percentiles (P10/P25/P75/P90), and histograms with mean/median overlays. "
            "Triggered by: 'descriptive statistics', 'describe', 'summarise numeric', 'skewness', 'kurtosis', 'distribution summary'."
        )

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="columns",
            parameter_type=ParameterType.MULTISELECT,
            label="Numeric Columns to Analyse",
            description="Select one or more numeric columns. Leave blank to analyse ALL numeric columns.",
            required=False,
            column_source="numeric"
        ))
        params.add_parameter(ToolParameter(
            name="generate_plots",
            parameter_type=ParameterType.CHECKBOX,
            label="Generate Visualizations",
            description="Generate histograms and box plots for each selected column.",
            required=False,
            default_value=True
        ))
        params.add_parameter(ToolParameter(
            name="percentiles",
            parameter_type=ParameterType.TEXT,
            label="Custom Percentiles (comma-separated, e.g. 10,25,75,90)",
            description="Additional percentiles to include in the report.",
            required=False,
            default_value="10,25,50,75,90"
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            df = self.load_dataset()

            columns = kwargs.get("columns", [])
            if isinstance(columns, str):
                columns = [c.strip() for c in columns.split(",") if c.strip()]
            if not columns:
                columns = df.select_dtypes(include="number").columns.tolist()

            gen_plots = str(kwargs.get("generate_plots", "true")).lower() in ("true", "on", "1")

            # Parse custom percentiles
            pct_raw = kwargs.get("percentiles", "10,25,50,75,90")
            try:
                pcts = [float(p.strip()) / 100 for p in str(pct_raw).split(",") if p.strip()]
            except ValueError:
                pcts = [0.10, 0.25, 0.50, 0.75, 0.90]

            # Validate columns exist
            valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            if not valid_cols:
                return {
                    "status": "error",
                    "summary": f"None of the requested columns are numeric. Available numeric columns: {df.select_dtypes(include='number').columns.tolist()}"
                }

            artifacts: List[Dict[str, Any]] = []
            sections: List[Dict[str, Any]] = []

            for col in valid_cols:
                series = df[col].dropna()
                if series.empty:
                    continue

                # --- Core Stats Table ---
                pct_labels = [f"P{int(p*100)}" for p in pcts]
                pct_values = series.quantile(pcts).tolist()

                stats_rows = [
                    ["Count", int(series.count())],
                    ["Missing", int(df[col].isnull().sum())],
                    ["Mean", round(float(series.mean()), 4)],
                    ["Median", round(float(series.median()), 4)],
                    ["Std Dev", round(float(series.std()), 4)],
                    ["Variance", round(float(series.var()), 4)],
                    ["Min", round(float(series.min()), 4)],
                    ["Max", round(float(series.max()), 4)],
                    ["Range", round(float(series.max() - series.min()), 4)],
                    ["Skewness", round(float(series.skew()), 4)],
                    ["Kurtosis", round(float(series.kurtosis()), 4)],
                ] + list(zip(pct_labels, [round(v, 4) for v in pct_values]))

                sections.append({
                    "type": "table",
                    "title": f"Descriptive Statistics — {col}",
                    "headers": ["Statistic", "Value"],
                    "data": stats_rows,
                    "footer": f"Based on {int(series.count())} non-null observations."
                })

                if gen_plots:
                    try:
                        with PlotUtils.setup_plotting():
                            # Histogram + KDE
                            fig, ax = plt.subplots(figsize=(10, 5))
                            sns.histplot(series, kde=True, ax=ax, color="#8B5CF6", alpha=0.7)
                            ax.axvline(series.mean(), color="#f59e0b", linestyle="--", label=f"Mean={series.mean():.2f}")
                            ax.axvline(series.median(), color="#34d399", linestyle="--", label=f"Median={series.median():.2f}")
                            ax.set_title(f"Distribution of {col}")
                            ax.set_xlabel(col)
                            ax.set_ylabel("Frequency")
                            ax.legend(fontsize=9)
                            plt.tight_layout()
                            artifacts.append(PlotUtils.to_artifact(fig, f"desc_hist_{col}", f"Histogram — {col}"))
                            plt.close(fig)

                            # Box Plot (with 5-number summary for ECharts)
                            fig2, ax2 = plt.subplots(figsize=(10, 4))
                            sns.boxplot(x=series, ax=ax2, color="#8B5CF6")
                            ax2.set_title(f"Box Plot — {col}")
                            ax2.set_xlabel(col)
                            plt.tight_layout()
                            box_data = {
                                "type": "boxplot",
                                "title": f"Box Plot of {col}",
                                "categories": [col],
                                "values": [[
                                    float(series.min()),
                                    float(series.quantile(0.25)),
                                    float(series.median()),
                                    float(series.quantile(0.75)),
                                    float(series.max())
                                ]],
                                "metadata": {"xAxisLabel": col}
                            }
                            artifacts.append(PlotUtils.to_artifact(fig2, f"desc_box_{col}", f"Box Plot — {col}", data_override=box_data))
                            plt.close(fig2)
                    except Exception as plot_ex:
                        sections.append({"type": "text", "title": f"Plot failed for {col}", "content": str(plot_ex)})

            result_cols = ", ".join(valid_cols)
            return {
                "status": "ok",
                "summary": f"Descriptive statistics computed for {len(valid_cols)} column(s): {result_cols}.",
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": kwargs}
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get("status") != "ok":
            return None
        try:
            findings = []
            for section in results.get("sections", []):
                if section.get("type") != "table":
                    continue
                title = section.get("title", "")
                rows = {row[0]: row[1] for row in section.get("data", []) if len(row) == 2}
                col_name = title.replace("Descriptive Statistics — ", "")
                skew = rows.get("Skewness")
                kurt = rows.get("Kurtosis")
                mean = rows.get("Mean")
                median = rows.get("Median")

                desc = f"'{col_name}': Mean={mean}, Median={median}"
                if skew is not None:
                    skew = float(skew)
                    if abs(skew) < 0.5:
                        desc += ", approx. symmetric distribution"
                    elif skew > 0:
                        desc += f", positively skewed ({skew:.2f})"
                    else:
                        desc += f", negatively skewed ({skew:.2f})"
                findings.append(desc)

            return " | ".join(findings) if findings else "No numeric sections found."
        except Exception as e:
            return f"Could not interpret results: {e}"

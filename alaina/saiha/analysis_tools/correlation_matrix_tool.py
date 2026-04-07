
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


class CorrelationMatrixTool(BaseAnalysisTool):
    """
    Computes pairwise Pearson/Spearman/Kendall correlations between numeric columns
    and renders a heatmap and ranked correlation list.
    """

    @property
    def name(self) -> str:
        return "correlation_matrix"

    @property
    def description(self) -> str:
        return (
            "Computes a pairwise correlation matrix for numeric columns and visualises "
            "it as a heatmap. Supports Pearson, Spearman, and Kendall methods. "
            "Use this for 'correlation', 'relationship between variables', or 'collinearity' queries."
        )

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="columns",
            parameter_type=ParameterType.MULTISELECT,
            label="Columns to Include",
            description="Select specific numeric columns. Leave blank to use ALL numeric columns.",
            required=False,
            column_source="numeric"
        ))
        params.add_parameter(ToolParameter(
            name="method",
            parameter_type=ParameterType.SELECT,
            label="Correlation Method",
            description="The statistical method to use for computing correlations.",
            required=False,
            default_value="pearson",
            options=[
                {"value": "pearson", "label": "Pearson (linear relationships)"},
                {"value": "spearman", "label": "Spearman (monotonic, rank-based)"},
                {"value": "kendall", "label": "Kendall (robust for small samples)"},
            ]
        ))
        params.add_parameter(ToolParameter(
            name="threshold",
            parameter_type=ParameterType.NUMBER,
            label="Strong Correlation Threshold (0–1)",
            description="Pairs with |r| above this value will be highlighted as strongly correlated.",
            required=False,
            default_value=0.7,
            validation_rules={"min": 0.1, "max": 1.0, "step": "0.05"}
        ))
        params.add_parameter(ToolParameter(
            name="min_columns",
            parameter_type=ParameterType.NUMBER,
            label="Minimum columns required",
            description="Minimum number of numeric columns needed. Default is 2.",
            required=False,
            default_value=2
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

            method = str(kwargs.get("method", "pearson")).lower()
            if method not in {"pearson", "spearman", "kendall"}:
                method = "pearson"

            try:
                threshold = float(kwargs.get("threshold", 0.7))
            except (ValueError, TypeError):
                threshold = 0.7

            # Validate columns
            valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            if len(valid_cols) < 2:
                return {
                    "status": "error",
                    "summary": f"At least 2 numeric columns are required. Got: {valid_cols}. Available: {df.select_dtypes(include='number').columns.tolist()}"
                }

            work_df = df[valid_cols].dropna()
            corr_matrix = work_df.corr(method=method).round(4)

            artifacts: List[Dict[str, Any]] = []
            sections: List[Dict[str, Any]] = []

            # --- Full correlation table (flat) ---
            table_data = []
            headers = ["Column A", "Column B", f"{method.capitalize()} r", "Strength"]
            seen = set()
            for i, c1 in enumerate(corr_matrix.columns):
                for j, c2 in enumerate(corr_matrix.columns):
                    if i >= j:
                        continue
                    key = tuple(sorted((c1, c2)))
                    if key in seen:
                        continue
                    seen.add(key)
                    r = corr_matrix.loc[c1, c2]
                    abs_r = abs(r)
                    if abs_r >= 0.9:
                        strength = "Very Strong"
                    elif abs_r >= 0.7:
                        strength = "Strong"
                    elif abs_r >= 0.5:
                        strength = "Moderate"
                    elif abs_r >= 0.3:
                        strength = "Weak"
                    else:
                        strength = "Negligible"
                    table_data.append([c1, c2, round(r, 4), strength])

            # Sort by |r| descending
            table_data.sort(key=lambda row: abs(row[2]), reverse=True)
            sections.append({
                "type": "table",
                "title": f"Pairwise Correlations ({method.capitalize()})",
                "headers": headers,
                "data": table_data,
                "footer": f"Threshold for 'strong': |r| >= {threshold}"
            })

            # --- Strong correlations highlight ---
            strong_pairs = [row for row in table_data if abs(row[2]) >= threshold]
            if strong_pairs:
                sections.append({
                    "type": "table",
                    "title": f"Strongly Correlated Pairs (|r| >= {threshold})",
                    "headers": headers,
                    "data": strong_pairs,
                    "footer": "These pairs may indicate multicollinearity or genuine relationships worth investigating."
                })

            # --- Heatmap ---
            try:
                with PlotUtils.setup_plotting():
                    n = len(valid_cols)
                    fig_size = max(8, n * 0.9)
                    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    sns.heatmap(
                        corr_matrix, mask=mask, annot=True, fmt=".2f",
                        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                        ax=ax, square=True, linewidths=0.5,
                        cbar_kws={"label": f"{method.capitalize()} r", "shrink": 0.8}
                    )
                    ax.set_title(f"Correlation Matrix ({method.capitalize()})", fontsize=14, pad=12)
                    plt.xticks(rotation=45, ha="right", fontsize=9)
                    plt.yticks(rotation=0, fontsize=9)
                    plt.tight_layout()
                    artifacts.append(PlotUtils.to_artifact(fig, "correlation_heatmap", f"Correlation Heatmap ({method.capitalize()})"))
                    plt.close(fig)
            except Exception as plot_ex:
                sections.append({"type": "text", "title": "Heatmap failed", "content": str(plot_ex)})

            # --- Bar chart of top correlations (top 15 pairs by |r|) ---
            try:
                with PlotUtils.setup_plotting():
                    top_pairs = table_data[:15]
                    labels = [f"{r[0]} × {r[1]}" for r in top_pairs]
                    values = [r[2] for r in top_pairs]
                    colors = ["#34d399" if v >= 0 else "#f87171" for v in values]

                    fig2, ax2 = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)))
                    bars = ax2.barh(labels[::-1], values[::-1], color=colors[::-1], alpha=0.85)
                    ax2.axvline(0, color="white", linewidth=0.8, alpha=0.5)
                    ax2.axvline(threshold, color="#f59e0b", linewidth=1.2, linestyle="--", label=f"Threshold ({threshold})")
                    ax2.axvline(-threshold, color="#f59e0b", linewidth=1.2, linestyle="--")
                    ax2.set_xlim(-1.05, 1.05)
                    ax2.set_xlabel(f"{method.capitalize()} Correlation Coefficient")
                    ax2.set_title("Top Correlated Pairs")
                    ax2.legend(fontsize=9)
                    plt.tight_layout()
                    
                    bar_chart_data = {
                        "type": "bar",
                        "title": "Top Correlated Pairs",
                        "labels": labels[::-1],
                        "series": [{"name": "Correlation r", "data": [round(v, 4) for v in values[::-1]]}],
                        "metadata": {"xAxisLabel": "Correlation r", "yAxisLabel": "Variable Pair"}
                    }
                    artifacts.append(PlotUtils.to_artifact(fig2, "correlation_bar", "Top Correlated Pairs", data_override=bar_chart_data))
                    plt.close(fig2)
            except Exception as bar_ex:
                pass  # Non-critical; heatmap is the primary visual

            strong_count = len(strong_pairs)
            return {
                "status": "ok",
                "summary": (
                    f"Correlation matrix ({method}) computed for {len(valid_cols)} columns. "
                    f"Found {strong_count} pair(s) with |r| >= {threshold} (strong correlation)."
                ),
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
            meta = results.get("meta", {}).get("parameters", {})
            method = meta.get("method", "pearson").capitalize()
            threshold = float(meta.get("threshold", 0.7))

            strong_section = next((s for s in results.get("sections", []) if "Strongly Correlated" in s.get("title", "")), None)
            all_section = next((s for s in results.get("sections", []) if "Pairwise" in s.get("title", "")), None)

            if not all_section:
                return "Could not extract correlation data for interpretation."

            total_pairs = len(all_section.get("data", []))
            strong_pairs = strong_section["data"] if strong_section else []

            if not strong_pairs:
                return (
                    f"Using the {method} method across {total_pairs} variable pairs, no pairs exceeded "
                    f"the strong correlation threshold of |r| >= {threshold}. "
                    "The variables appear to be largely independent of each other."
                )

            top = strong_pairs[0]
            highlights = [f"'{r[0]}' & '{r[1]}' (r={r[2]:.3f})" for r in strong_pairs[:5]]
            return (
                f"Using the {method} method, {len(strong_pairs)} strongly correlated pair(s) were found "
                f"(|r| ≥ {threshold}) out of {total_pairs} total pairs. "
                f"The strongest correlation is between {top[0]} and {top[1]} (r={top[2]:.3f}). "
                f"Notable pairs: {', '.join(highlights)}. "
                "Consider investigating these relationships for multicollinearity before regression modelling."
            )
        except Exception as e:
            return f"Could not interpret correlation results: {e}"

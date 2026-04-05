# d:/quantly/quanta/quantalytics/ai_agents/tools/correlation_matrix_tool.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from django.core.files.storage import default_storage
from typing import Any, Dict, List, Optional
from itertools import combinations
from pandas.api.types import is_numeric_dtype

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class CorrelationMatrixTool(BaseAnalysisTool):
    """
    A tool to calculate and visualize correlations/associations.
    - Pearson for numeric-numeric (signed, -1..1)
    - Cramér's V for categorical-categorical (0..1)
    - Correlation ratio (eta) for numeric-categorical (0..1)
    """

    @property
    def name(self) -> str:
        return "correlation_matrix"

    @property
    def description(self) -> str:
        return "Calculate and visualize the correlation between numeric and/or categorical variables with appropriate association measures."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="variables", parameter_type=ParameterType.MULTISELECT,
            label="Variables to Correlate",
            description="Select two or more numeric or categorical columns.",
            required=True, column_source="all"
        ))
        return params

    def _cramers_v(self, x, y):
        """Calculate Cramér's V for two categorical variables with bias correction."""
        confusion_matrix = pd.crosstab(x, y)
        if confusion_matrix.size == 0:
            return np.nan
        try:
            chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
        except Exception:
            return np.nan
        n = confusion_matrix.to_numpy().sum()
        if n == 0:
            return np.nan
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        # Bias correction per Bergsma & Wicher
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        denom = min((kcorr-1), (rcorr-1))
        if denom <= 0:
            return 0.0
        return np.sqrt(phi2corr / denom)

    def _correlation_ratio(self, categories, measurements):
        """
        Compute correlation ratio (eta) between categorical (nominal) variable `categories`
        and numeric `measurements`. Returns value in [0,1].
        """
        # drop na pairs
        df = pd.DataFrame({"cat": categories, "num": measurements}).dropna()
        if df.empty:
            return np.nan
        groups = df.groupby("cat")["num"]
        # overall mean
        grand_mean = df["num"].mean()
        # between-group sum of squares
        n_k = groups.count()
        means_k = groups.mean()
        ss_between = (n_k * (means_k - grand_mean) ** 2).sum()
        ss_total = ((df["num"] - grand_mean) ** 2).sum()
        if ss_total == 0:
            return 0.0
        eta2 = ss_between / ss_total
        return np.sqrt(max(0.0, eta2))

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            variables = parameters.get("variables", [])
            if isinstance(variables, str):
                variables = [variables]

            if len(variables) < 2:
                return {"status": "error", "summary": "Please select at least two variables."}

            # Load dataset (projected to requested columns)
            df = self.load_dataset(columns=variables)
            df_selected = df.loc[:, variables].copy()

            # Determine column types robustly
            numeric_cols = [c for c in variables if is_numeric_dtype(df_selected[c])]
            categorical_cols = [c for c in variables if c not in numeric_cols]

            # Initialize matrices with NaN so uncomputed entries remain NaN
            assoc_matrix = pd.DataFrame(np.nan, index=variables, columns=variables, dtype=float)
            pearson_matrix = pd.DataFrame(np.nan, index=numeric_cols, columns=numeric_cols, dtype=float)

            # 1) Pearson for numeric-numeric
            if len(numeric_cols) > 0:
                pearson = df_selected[numeric_cols].corr(method='pearson')
                pearson_matrix.loc[:, :] = pearson

                # Put pearson values into assoc_matrix (symmetric)
                for i in numeric_cols:
                    for j in numeric_cols:
                        assoc_matrix.loc[i, j] = pearson.loc[i, j]

            # 2) Cramér's V for categorical-categorical
            if len(categorical_cols) > 0:
                for a, b in combinations(categorical_cols, 2):
                    v = self._cramers_v(df_selected[a], df_selected[b])
                    assoc_matrix.loc[a, b] = v
                    assoc_matrix.loc[b, a] = v
                # self correlation for categorical = 1
                for c in categorical_cols:
                    assoc_matrix.loc[c, c] = 1.0

            # 3) Correlation ratio (eta) for numeric-categorical (and symmetric)
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    eta = self._correlation_ratio(df_selected[cat_col], df_selected[num_col])
                    assoc_matrix.loc[num_col, cat_col] = eta
                    assoc_matrix.loc[cat_col, num_col] = eta

            # Ensure numeric diagonal is 1
            for ncol in numeric_cols:
                pearson_matrix.loc[ncol, ncol] = 1.0
                assoc_matrix.loc[ncol, ncol] = 1.0

            # --- Key Takeaways ---
            sections = []
            takeaways = []

            # Analyze pearson matrix for numeric multicollinearity / strong signed correlations
            if not pearson_matrix.empty:
                pe = pearson_matrix.copy()
                # Use pandas loc to set diagonal to NaN safely (avoids read-only array issues with pe.values)
                for col in pe.columns:
                    pe.loc[col, col] = np.nan
                stacked_pe = pe.stack()
                stacked_pe = stacked_pe[~stacked_pe.index.duplicated(keep='first')]
                perfect_corr = stacked_pe[stacked_pe.abs() >= 0.999]
                if not perfect_corr.empty:
                    takeaways.append("**High numeric multicollinearity detected:**")
                    for (v1, v2), val in perfect_corr.items():
                        takeaways.append(f"- `{v1}` and `{v2}` (Pearson r = {val:.3f})")
                top_numeric = stacked_pe.abs().sort_values(ascending=False).head(5)
                if not top_numeric.empty:
                    takeaways.append("\n**Top numeric correlations (Pearson):**")
                    for (v1, v2), val_abs in top_numeric.items():
                        val = pearson_matrix.loc[v1, v2]
                        direction = "positive" if val > 0 else "negative"
                        takeaways.append(f"- `{v1}` ↔ `{v2}`: {direction}, r = {val:.3f}")

            # Analyze assoc_matrix for strong associations (Cramér's V / eta)
            am = assoc_matrix.copy()
            # Use pandas loc to set diagonal to NaN safely
            for col in am.columns:
                am.loc[col, col] = np.nan
            stacked_am = am.stack()
            # dedupe by sorted pair
            stacked_am.index = pd.MultiIndex.from_tuples([tuple(sorted(i)) for i in stacked_am.index])
            stacked_am = stacked_am[~stacked_am.index.duplicated(keep='first')]
            top_assoc = stacked_am.dropna().abs().sort_values(ascending=False).head(10)
            if not top_assoc.empty:
                takeaways.append("\n**Top associations (Cramér's V / eta):**")
                for (v1, v2), val in top_assoc.items():
                    takeaways.append(f"- `{v1}` ↔ `{v2}` : association = {val:.3f} (unsigned)")

            if takeaways:
                sections.append({'type': 'text', 'title': 'Key Takeaways', 'content': "\n".join(takeaways)})

            # --- Visualization: produce separate heatmaps ---
            artifacts = []
            with PlotUtils.setup_plotting():
                # 1) Numeric Pearson heatmap (if numeric cols exist)
                if not pearson_matrix.empty:
                    fig1, ax1 = plt.subplots(figsize=(max(6, len(numeric_cols)*0.6), max(4, len(numeric_cols)*0.6)))
                    sns.heatmap(pearson_matrix, annot=True, fmt=".2f", cmap='vlag', center=0, ax=ax1,
                                cbar_kws={'label': "Pearson r"})
                    ax1.set_title("Numeric correlation (Pearson)")
                    plt.tight_layout()
                    artifacts.append({"type": "plot", "id": "pearson_heatmap", "title": "Pearson Correlation (numeric)", "content": PlotUtils.fig_to_base64(fig1)})
                    plt.close(fig1)

                # 2) Association matrix heatmap (Cramér's V and eta combined) — unsigned 0..1
                if len(variables) > 0:
                    fig2, ax2 = plt.subplots(figsize=(max(6, len(variables)*0.5), max(4, len(variables)*0.5)))
                    # Use a sequential colormap since values are 0..1 (unsigned)
                    sns.heatmap(assoc_matrix, annot=True, fmt=".2f", vmin=0, vmax=1, center=None,
                                cmap='YlGnBu', ax=ax2, cbar_kws={'label': "Association (0..1)"})
                    ax2.set_title("Association matrix (Cramér's V for categorical-categorical, eta for numeric-categorical)")
                    plt.tight_layout()
                    artifacts.append({"type": "plot", "id": "association_heatmap", "title": "Association Matrix", "content": PlotUtils.fig_to_base64(fig2)})
                    plt.close(fig2)

            return {
                "status": "ok",
                "summary": "Calculated Pearson for numeric pairs, Cramér's V for categorical pairs, and correlation ratio (eta) for mixed pairs. Separate heatmaps provided for numeric correlations and overall associations.",
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": parameters},
            }
        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}
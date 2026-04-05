# d:/quantly/quanta/quantalytics/ai_agents/tools/stratified_chi_square_tool.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import StratifiedTable
from django.core.files.storage import default_storage
from typing import Any, Dict, Optional, List, Tuple

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class StratifiedChiSquareTool(BaseAnalysisTool):
    """
    A tool to perform a Chi-Square Test of Independence, with an option for stratification
    (Cochran-Mantel-Haenszel test).
    """

    @property
    def name(self) -> str:
        return "stratified_chi_square_test"

    @property
    def description(self) -> str:
        return "Tests association between two categorical variables; if a stratum is provided and both variables are binary, runs CMH."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(
            ToolParameter(
                name="variable1", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                label="Categorical Variable 1 (Rows)", description="Select the first categorical column.", required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="variable2", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                label="Categorical Variable 2 (Columns)", description="Select the second categorical column.", required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="stratum_variable", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                label="Stratum Variable (Optional)",
                description="If provided, and both variables are binary, performs Cochran–Mantel–Haenszel (CMH) test.",
                required=False
            )
        )
        params.add_parameter(
            ToolParameter(
                name="alpha", parameter_type=ParameterType.SELECT,
                label="Significance Level (α)", description="Threshold for statistical significance.",
                required=True, default_value="0.05",
                options=[{"value": "0.05", "label": "0.05"}, {"value": "0.01", "label": "0.01"}]
            )
        )
        # Stratified plot options
        params.add_parameter(
            ToolParameter(
                name="generate_stratified_plots", parameter_type=ParameterType.CHECKBOX,
                label="Generate Stratified Bar Charts",
                description="Visualize the association within each stratum using bar charts. (Used only when a stratum variable is provided).",
                required=False, default_value=False
            )
        )
        # Pooled plot options (for when no stratum is selected)
        params.add_parameter(
            ToolParameter(
                name="generate_stacked_bar", parameter_type=ParameterType.CHECKBOX,
                label="Generate Stacked Bar Chart (Pooled)",
                description="Visualize the overall relationship using a stacked bar chart. (Used only when no stratum is selected).",
                required=False, default_value=False
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_heatmap", parameter_type=ParameterType.CHECKBOX,
                label="Generate Heatmap (Pooled)",
                description="Visualize the overall contingency table as a heatmap. (Used only when no stratum is selected).",
                required=False, default_value=False
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_mosaic_plot", parameter_type=ParameterType.CHECKBOX,
                label="Generate Mosaic Plot (Pooled)",
                description="Visualize the overall association using a mosaic plot. (Used only when no stratum is selected).",
                required=False, default_value=False
            )
        )
        return params

    # ---------------- helpers ----------------

    def _two_levels(self, s: pd.Series) -> List[Any]:
        """Return exactly two levels (order stable) or empty list if not 2-level."""
        lev = pd.Categorical(s).categories.tolist()
        # categories excludes NaN; if user has only 1 or >2 levels → not 2×2
        return lev if len(lev) == 2 else []

    def _aligned_2x2_tables(
        self, df: pd.DataFrame, var1: str, var2: str, stratum_var: str
    ) -> Tuple[List[pd.DataFrame], List[Any], List[Any], List[Any]]:
        """
        Build a list of aligned 2×2 tables across strata.
        Returns: (tables, strata_values, row_levels, col_levels)
        Raises ValueError if variables are not binary.
        """
        # Determine global 2 levels for each variable (across whole df)
        rows2 = self._two_levels(df[var1].dropna())
        cols2 = self._two_levels(df[var2].dropna())

        if not rows2 or not cols2:
            raise ValueError(
                f"CMH requires both '{var1}' and '{var2}' to be binary (exactly two levels). "
                f"Found {len(pd.Categorical(df[var1].dropna()).categories)} level(s) for '{var1}' "
                f"and {len(pd.Categorical(df[var2].dropna()).categories)} level(s) for '{var2}'."
            )

        tables = []
        strata_values = pd.Categorical(df[stratum_var]).categories.tolist()
        for s in strata_values:
            dfs = df[df[stratum_var] == s]
            ct = pd.crosstab(dfs[var1], dfs[var2])
            # align to global 2×2 shape and fill zeros
            ct = ct.reindex(index=rows2, columns=cols2, fill_value=0)
            # If an entire row/col is zero in a stratum, continuity-correct a bit
            if (ct.values == 0).all():
                # completely empty stratum carries no info; skip it
                continue
            if (ct == 0).any().any():
                ct = ct.astype(float)
                ct += 0.5  # Haldane-Anscombe correction avoids singularities
            tables.append(ct)

        if not tables:
            raise ValueError("No informative strata remained after alignment (all-zero tables).")

        return tables, strata_values, rows2, cols2

    # ---------------- main ----------------

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            var1 = parameters.get("variable1")
            var2 = parameters.get("variable2")
            stratum_var = parameters.get("stratum_variable")
            alpha = float(parameters.get("alpha", 0.05))
            # Plotting flags
            gen_strat_plots = str(parameters.get("generate_stratified_plots", "false")).lower() in ('true', 'on', '1')
            gen_stacked = str(parameters.get("generate_stacked_bar", "false")).lower() in ('true', 'on', '1')
            gen_heatmap = str(parameters.get("generate_heatmap", "false")).lower() in ('true', 'on', '1')
            gen_mosaic = str(parameters.get("generate_mosaic_plot", "false")).lower() in ('true', 'on', '1')

            if not var1 or not var2:
                return {"status": "error", "summary": "Both categorical variables are required."}
            if var1 == var2 or (stratum_var and (var1 == stratum_var or var2 == stratum_var)):
                return {"status": "error", "summary": "Please select different columns for rows, columns, and stratum."}

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            # Pre-check: Ensure both variables have at least 2 levels
            n_levels1 = df[var1].nunique(dropna=True)
            n_levels2 = df[var2].nunique(dropna=True)
            if n_levels1 < 2 or n_levels2 < 2:
                return {"status": "error", "summary": (
                    f"The test cannot be performed because at least one variable lacks variation. "
                    f"'{var1}' has {n_levels1} unique level(s) and '{var2}' has {n_levels2} unique level(s). Both need at least 2."
                )}

            sections = []
            artifacts = []

            if stratum_var:
                # ---------- Stratified (CMH) path: only for 2×2 per stratum ----------
                try:
                    tables, strata_vals, row_lvls, col_lvls = self._aligned_2x2_tables(df, var1, var2, stratum_var)

                    # Convert each to numpy array (2×2), feed to StratifiedTable
                    np_tables = [t.values for t in tables]
                    strat_table = StratifiedTable(np_tables)

                    # CMH statistic & p-value (2×2 only)
                    cmh = strat_table.test_null_odds()
                    p_value = float(cmh.pvalue)
                    stat = float(cmh.statistic)
                    is_significant = p_value < alpha

                    # pooled OR
                    or_pooled = getattr(strat_table, "oddsratio_pooled", None)
                    logor_pooled = getattr(strat_table, "logodds_pooled", None)

                    sections.append({
                        'type': 'table',
                        'title': 'Cochran–Mantel–Haenszel (CMH) Results',
                        'headers': ['Statistic', 'Value'],
                        'data': [
                            ['CMH Chi-Square', f"{stat:.4f}"],
                            ['P-Value', f"{p_value:.6f}"],
                            ['Degrees of Freedom', 1],
                            ['Is Significant', 'Yes' if is_significant else 'No'],
                            ['Pooled Odds Ratio', f"{or_pooled:.4f}" if or_pooled is not None else "N/A"],
                            ['Pooled Log(OR)', f"{logor_pooled:.4f}" if logor_pooled is not None else "N/A"],
                        ],
                        'footer': f"Association between '{var1}' and '{var2}' controlling for '{stratum_var}'. "
                                  f"Levels: rows={row_lvls}, cols={col_lvls}."
                    })

                    summary_text = (
                        f"CMH test (2×2 per stratum) for '{var1}' vs '{var2}' controlling for '{stratum_var}': "
                        f"{'significant' if is_significant else 'not significant'} at α={alpha} (p={p_value:.4f})."
                    )

                    # --- Stratified Plots ---
                    if gen_strat_plots:
                        with PlotUtils.setup_plotting():
                            num_strata = len(strata_vals)
                            # Simple grid layout
                            ncols = min(3, num_strata)
                            nrows = int(np.ceil(num_strata / ncols))
                            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
                            axes = axes.flatten()

                            for i, s_val in enumerate(strata_vals):
                                stratum_df = df[df[stratum_var] == s_val]
                                ct = pd.crosstab(stratum_df[var1], stratum_df[var2])
                                ct.plot(kind='bar', stacked=True, ax=axes[i], legend= (i==0) ) # Legend on first plot only
                                axes[i].set_title(f'Stratum: {s_val}')
                                axes[i].set_xlabel(var1)
                                axes[i].set_ylabel('Count')

                            # Hide unused subplots
                            for j in range(i + 1, len(axes)):
                                axes[j].set_visible(False)

                            plt.tight_layout()
                            artifacts.append({
                                "type": "plot", "id": "stratified_bar_charts",
                                "title": "Stacked Bar Charts by Stratum", "content": PlotUtils.fig_to_base64(fig)
                            })

                except ValueError as ve:
                    # Variables are not binary → explain clearly
                    return {
                        "status": "error",
                        "summary": (
                            f"{ve}\n\n"
                            "How to proceed:\n"
                            "• Recode each variable to 2 levels (e.g., combine categories or bin numeric to High/Low), then re-run CMH; OR\n"
                            "• Omit the stratum to run a standard Chi-Square test on the pooled table; OR\n"
                            "• Run separate Chi-Square tests within each stratum (not pooled) using your reporting layer."
                        )
                    }

            else:
                # ---------- Standard (pooled) Chi-Square path ----------
                contingency = pd.crosstab(df[var1], df[var2])
                chi2, p_value, dof, _ = chi2_contingency(contingency)
                is_significant = p_value < alpha

                sections.append({
                    'type': 'table', 'title': 'Chi-Square Test Results',
                    'headers': ['Statistic', 'Value'],
                    'data': [
                        ['Chi-Square', f"{chi2:.4f}"],
                        ['P-Value', f"{p_value:.6f}"],
                        ['Degrees of Freedom', dof],
                        ['Is Significant', 'Yes' if is_significant else 'No']
                    ],
                    'footer': f"Test of independence between '{var1}' and '{var2}'."
                })

                summary_text = (
                    f"Chi-Square test for '{var1}' vs '{var2}': "
                    f"{'significant' if is_significant else 'not significant'} at α={alpha} (p={p_value:.4f})."
                )

            # --- Optional Pooled Visualizations (Run for both stratified and non-stratified cases) ---
            # These plots show the overall relationship, ignoring any stratum.
            with PlotUtils.setup_plotting():
                pooled_contingency = pd.crosstab(df[var1], df[var2])
                if gen_stacked:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    pooled_contingency.plot(kind='bar', stacked=True, ax=ax)
                    ax.set_title(f'Overall Stacked Bar Chart of {var2} by {var1}')
                    artifacts.append({"type": "plot", "id": "pooled_stacked_bar", "title": "Overall Stacked Bar Chart", "content": PlotUtils.fig_to_base64(fig)})

                if gen_heatmap:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    sns.heatmap(pooled_contingency, annot=True, fmt='d', cmap='viridis', ax=ax)
                    ax.set_title(f'Overall Heatmap of Observed Frequencies')
                    artifacts.append({"type": "plot", "id": "pooled_heatmap", "title": "Overall Heatmap", "content": PlotUtils.fig_to_base64(fig)})

                if gen_mosaic:
                    try:
                        fig, _ = mosaic(pooled_contingency.stack(), title=f'Overall Mosaic Plot for {var1} and {var2}', gap=0.02)
                        fig.set_size_inches(10, 7)
                        artifacts.append({"type": "plot", "id": "pooled_mosaic", "title": "Overall Mosaic Plot", "content": PlotUtils.fig_to_base64(fig)})
                    except Exception as e:
                        sections.append({'type': 'text', 'title': 'Mosaic Plot Failed', 'content': str(e)})
                        
            return {
                "status": "ok",
                "summary": summary_text,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": parameters},
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        return results.get("summary")
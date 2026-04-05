
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from django.core.files.storage import default_storage
from typing import Any, Dict, List, Optional

from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import proportional_hazard_test

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from quantalytics.ai_agents.tools.plot_utils import PlotUtils


class CoxPHTool(BaseAnalysisTool):
    """
    A tool to perform Cox Proportional Hazards regression analysis.
    """

    @property
    def name(self) -> str:
        return "cox_ph_regression"

    @property
    def description(self) -> str:
        return "Models the effect of covariates on survival time using Cox Proportional Hazards regression."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(
            ToolParameter(
                name="time_column", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Time to Event Column", description="Select the column with duration data.", required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="event_column", parameter_type=ParameterType.COLUMN_SELECT,
                label="Event Observed Column", description="Select the column indicating if the event occurred.",
                required=True, column_source="all"
            )
        )
        params.add_parameter(
            ToolParameter(
                name="event_value", parameter_type=ParameterType.TEXT,
                label="Event Value", description="Value indicating an event if the event column is text (e.g., 'Dead').",
                help_text="Case-sensitive. Required if the event column contains text.", required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="covariates", parameter_type=ParameterType.MULTISELECT,
                label="Covariates", description="Select columns to include as predictors in the model.",
                required=True, column_source="all"
            )
        )
        params.add_parameter(
            ToolParameter(
                name="strata_column", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                label="Strata Column (Optional)", description="Stratify by this column to allow non-proportional hazards between groups.",
                required=False
            )
        )
        params.add_parameter(
            ToolParameter(
                name="encoding_method", parameter_type=ParameterType.SELECT,
                label="Categorical Encoding", description="Method for encoding categorical covariates.",
                required=True, default_value="dummy",
                options=[
                    {"value": "dummy", "label": "Dummy (n-1, for interpretation)"},
                    {"value": "one_hot", "label": "One-Hot (n, for prediction)"},
                ]
            )
        )
        params.add_parameter(
            ToolParameter(
                name="alpha", parameter_type=ParameterType.SELECT,
                label="Significance Level (α)", description="Controls the confidence interval width (1-α).",
                required=True, default_value="0.05",
                options=[
                    {"value": "0.05", "label": "0.05 (95% Confidence)"},
                    {"value": "0.01", "label": "0.01 (99% Confidence)"},
                ]
            )
        )
        params.add_parameter(
            ToolParameter(
                name="penalizer", parameter_type=ParameterType.NUMBER,
                label="Penalizer (L2 Regularization)",
                description="Add a small penalty to handle collinearity, especially with one-hot encoding. Try 0.01 to start.",
                required=False, default_value=0.0,
                help_text="A value > 0 enables L2 regularization."
            )
        )
        params.add_parameter(
            ToolParameter(
                name="generate_assumption_plots", parameter_type=ParameterType.CHECKBOX,
                label="Generate Assumption Plots",
                description="Generate plots to visually check the proportional hazard assumption. This can be slow.",
                required=False, default_value=False
            )
        )
        return params

    # ---------- helpers ----------

    def _fmt_num(self, x, nd: int = 3) -> str:
        """Safe number formatting: try float fixed-point; else str(x)."""
        try:
            xv = float(x)
            if np.isfinite(xv):
                return f"{xv:.{nd}f}"
        except Exception:
            pass
        return str(x)

    def _plot_ph_schoenfeld(self, cph, df_final, covariate, ax=None):
        """
        Version-agnostic PH diagnostic:
        Scaled Schoenfeld residuals vs. ranked time with a smooth trend line.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        resid_df = cph.compute_residuals(df_final, kind="scaled_schoenfeld")
        if covariate not in resid_df.columns:
            raise ValueError(f"Schoenfeld residuals not available for '{covariate}'")

        r = resid_df[covariate].dropna()
        if r.empty:
            raise ValueError(f"No residuals to plot for '{covariate}'")

        # event times are the index
        t = r.index.values.astype(float)
        order = np.argsort(t)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.linspace(0.0, 1.0, num=r.size, endpoint=True)

        y = r.values.astype(float)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(ranks, y, s=12, alpha=0.6)
        ax.axhline(0.0, linestyle="--", linewidth=1)

        # lightweight smoothing
        try:
            deg = 3 if y.size >= 6 else 1
            coeffs = np.polyfit(ranks, y, deg=deg)
            x_fit = np.linspace(0, 1, 200)
            y_fit = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, y_fit, linewidth=2)
        except Exception:
            pass

        ax.set_title(f"Schoenfeld Residuals vs Ranked Time — {covariate}")
        ax.set_xlabel("Ranked Time (0 → 1)")
        ax.set_ylabel("Scaled Schoenfeld Residuals")
        return ax

    def _prepare_survival_frame(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        event_value: Optional[str],
        covariates: List[str],
        strata_col: Optional[str],
        encoding: str,
    ) -> pd.DataFrame:
        """
        Build a clean modeling dataframe with:
        - duration (>=0, finite)
        - event (binary 0/1)
        - encoded covariates
        - optional strata
        Drops rows with any NA after assembly.
        """
        # Base frame with needed columns
        model_cols = [time_col, event_col] + list(covariates)
        if strata_col:
            model_cols.append(strata_col)
        model_cols = list(dict.fromkeys(model_cols))  # keep order, drop dups

        dfm = df[model_cols].copy()

        # Duration
        dfm["duration"] = pd.to_numeric(dfm[time_col], errors="coerce")

        # Event
        if pd.api.types.is_numeric_dtype(dfm[event_col]):
            e_raw = pd.to_numeric(dfm[event_col], errors="coerce").fillna(0)
            dfm["event"] = (e_raw != 0).astype(int)
        else:
            if not event_value:
                raise ValueError("Event column is text. Please specify the 'Event Value'.")
            dfm["event"] = (dfm[event_col] == event_value).astype(int)

        # Encode covariates
        cov_df = dfm[covariates].copy()
        cat_cols = cov_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if cat_cols:
            drop_first = (encoding == "dummy")
            cov_df = pd.get_dummies(cov_df, columns=cat_cols, drop_first=drop_first, dummy_na=False)

        # Re-assemble
        cols = ["duration", "event"]
        if strata_col:
            cols.append(strata_col)
        df_final = pd.concat([dfm[cols], cov_df], axis=1)

        # Clean: duration finite & >=0
        valid = df_final["duration"].notna() & np.isfinite(df_final["duration"]) & (df_final["duration"] >= 0)
        df_final = df_final.loc[valid].copy()

        # Drop any remaining NA
        df_final.dropna(axis=0, inplace=True)

        # Ensure we still have covariates
        cov_only = [c for c in df_final.columns if c not in ("duration", "event") and c != (strata_col or "")]
        if len(cov_only) == 0:
            raise ValueError("No usable covariates after encoding/cleaning. Check your inputs.")

        return df_final

    # ---------- main ----------

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            time_col = parameters.get("time_column")
            event_col = parameters.get("event_column")
            event_value = parameters.get("event_value")
            covariates_param = parameters.get("covariates", [])
            if isinstance(covariates_param, str):
                covariates = [covariates_param]
            else:
                covariates = covariates_param or []
            strata_col = parameters.get("strata_column") or None
            encoding = parameters.get("encoding_method", "dummy")
            alpha_ui = float(parameters.get("alpha", 0.05))  # significance level from UI
            ci_level = max(0.0, min(1.0, 1.0 - alpha_ui))   # lifelines expects confidence level
            penalizer_str = parameters.get("penalizer")
            penalizer = float(penalizer_str) if penalizer_str else 0.0
            gen_plots = bool(parameters.get("generate_assumption_plots", False))

            if not time_col or not event_col or not covariates:
                return {"status": "error", "summary": "Time, Event, and at least one Covariate are required."}

            # Use efficient column projection loading from BaseAnalysisTool
            cols_needed = [time_col, event_col] + covariates
            if strata_col:
                cols_needed.append(strata_col)
            # Remove potential duplicates
            cols_needed = list(set(cols_needed))
            df = self.load_dataset(columns=cols_needed)

            # Existence checks
            missing = [c for c in [time_col, event_col] + covariates + ([strata_col] if strata_col else []) if c not in df.columns]
            if missing:
                return {"status": "error", "summary": f"Column(s) not found in dataset: {', '.join(missing)}."}

            # Prepare modeling dataframe
            try:
                df_final = self._prepare_survival_frame(
                    df, time_col, event_col, event_value, covariates, strata_col, encoding
                )
            except ValueError as ve:
                return {"status": "error", "summary": str(ve)}

            if len(df_final) < 20:
                return {"status": "error", "summary": f"Not enough valid data ({len(df_final)} rows) to fit the model."}

            # Fit model (pass CI level)
            cph = CoxPHFitter(alpha=ci_level, penalizer=penalizer)
            cph.fit(df_final, duration_col="duration", event_col="event", strata=strata_col if strata_col else None)

            sections: List[Dict[str, Any]] = []
            artifacts: List[Dict[str, Any]] = []

            # === Main results table ===
            summary_df = cph.summary.reset_index()
            # Pretty column names
            summary_df.rename(columns={
                "index": "variable",
                "exp(coef)": "Hazard Ratio",
                "exp(coef) lower 95%": f"Hazard Ratio lower {int(ci_level*100)}%",
                "exp(coef) upper 95%": f"Hazard Ratio upper {int(ci_level*100)}%",
                "p": "p-value",
            }, inplace=True)

            # Round numeric columns
            for col in summary_df.columns:
                if pd.api.types.is_numeric_dtype(summary_df[col]):
                    summary_df[col] = summary_df[col].round(4)

            sections.append({
                "type": "table",
                "title": "Cox Proportional Hazards Model Summary",
                "headers": summary_df.columns.tolist(),
                "data": summary_df.values.tolist()
            })

            # === Forest plot of hazard ratios ===
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(10, max(4, len(cph.params_) * 0.5)))
                cph.plot(ax=ax)  # shows coef & CI; labels reflect CI level
                ax.set_title("Hazard Ratios with Confidence Intervals")
                plt.tight_layout()
                artifacts.append({
                    "type": "plot",
                    "id": "cox_ph_forest_plot",
                    "title": "Forest Plot of Hazard Ratios",
                    "content": PlotUtils.fig_to_base64(fig),
                })
                plt.close(fig) # Explicitly close the figure

            # === Predicted survival curves for one covariate (robust to encoding & strata) ===
            plot_covariate = None
            # Prefer the first *original* categorical covariate (pre-encoding); else fall back to first numeric
            for cov in covariates:
                if pd.api.types.is_categorical_dtype(df[cov]) or pd.api.types.is_object_dtype(df[cov]) or pd.api.types.is_bool_dtype(df[cov]):
                    plot_covariate = cov
                    break
            if plot_covariate is None and covariates:
                plot_covariate = covariates[0]  # fallback: continuous/numeric

            def _model_cols(cph_):
                return list(cph_.params_.index)

            def _sanitize_level(s: str) -> str:
                # Make a category name compatible with dummy col names lifelines/pandas created
                return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(s))

            def _encoded_cols_for(original_col: str, model_cols: List[str]) -> List[str]:
                pref = f"{original_col}_"
                return [c for c in model_cols if c.startswith(pref)]

            def _baseline_row(df_final: pd.DataFrame, model_cols: List[str], strata_col: Optional[str], strata_val: Optional[Any]) -> pd.DataFrame:
                # zeros for indicators; medians for numeric columns present in model; include strata if needed
                row = {col: 0.0 for col in model_cols}
                for col in model_cols:
                    if col in df_final.columns and pd.api.types.is_numeric_dtype(df_final[col]):
                        med = df_final[col].median()
                        if np.isfinite(med):
                            row[col] = float(med)
                X = pd.DataFrame([row], columns=model_cols)
                if strata_col is not None:
                    X[strata_col] = strata_val
                return X

            def _levels_to_plot(df: pd.DataFrame, col: str, max_levels: int = 8) -> List[Any]:
                if col not in df.columns:
                    return []
                # Keep it readable
                vals = pd.Series(df[col].dropna().unique()).tolist()
                # sort for deterministic legend
                try:
                    return sorted(vals)[:max_levels]
                except Exception:
                    return vals[:max_levels]

            def _apply_category(X: pd.DataFrame, original_col: str, model_cols: List[str], chosen_level: Any, encoding: str):
                enc_cols = _encoded_cols_for(original_col, model_cols)
                if not enc_cols:
                    # Either continuous var or not encoded (no-op here)
                    return
                # zero all indicators first
                for c in enc_cols:
                    X[c] = 0.0
                # find matching indicator for chosen level (if any)
                wanted = f"{original_col}_{chosen_level}"
                if wanted not in enc_cols:
                    safe = f"{original_col}_{_sanitize_level(chosen_level)}"
                    if safe in enc_cols:
                        wanted = safe
                    else:
                        # fallback: suffix match
                        matches = [c for c in enc_cols if c.lower().endswith(str(chosen_level).lower())]
                        if matches:
                            wanted = matches[0]
                        else:
                            wanted = None
                # dummy: reference level == all zeros; one-hot: set its column to 1.0
                if wanted is not None:
                    X[wanted] = 1.0

            def _apply_continuous(X: pd.DataFrame, model_cols: List[str], col: str, val: float):
                # if the original column survived as-is
                if col in model_cols:
                    X[col] = float(val)

            try:
                if plot_covariate:
                    with PlotUtils.setup_plotting():
                        fig, ax = plt.subplots(figsize=(10, 7))
                        mcols = _model_cols(cph)

                        # choose a stratum to visualize, if any
                        chosen_stratum = None
                        if strata_col is not None and strata_col in df_final.columns:
                            chosen_stratum = df_final[strata_col].value_counts(dropna=False).index[0]

                        base = _baseline_row(df_final, mcols, strata_col, chosen_stratum)

                        # category vs continuous
                        is_cat = (pd.api.types.is_object_dtype(df[plot_covariate]) or
                                  pd.api.types.is_categorical_dtype(df[plot_covariate]) or
                                  pd.api.types.is_bool_dtype(df[plot_covariate]))

                        if is_cat:
                            levels = _levels_to_plot(df, plot_covariate, max_levels=8)
                            if not levels:
                                raise ValueError(f"No levels to plot for {plot_covariate}")
                            for lvl in levels:
                                X = base.copy()
                                _apply_category(X, plot_covariate, mcols, lvl, encoding)
                                # ensure all params columns exist (in case dummy ref → no col)
                                for c in mcols:
                                    if c not in X.columns:
                                        X[c] = 0.0
                                X = X[[c for c in mcols] + ([strata_col] if strata_col else [])]
                                sf = cph.predict_survival_function(X)
                                ax.plot(sf.index.values, sf.iloc[:, 0].values, label=str(lvl))
                            ax.legend(title=plot_covariate, fontsize=9)
                            title_extra = f" (stratum={chosen_stratum})" if strata_col else ""
                            ax.set_title(f"Predicted Survival by {plot_covariate}{title_extra}")
                        else:
                            # continuous: show at low/median/high (10th/50th/90th)
                            series = pd.to_numeric(df[plot_covariate], errors="coerce").dropna()
                            if series.empty:
                                raise ValueError(f"No numeric data for {plot_covariate}")
                            q_vals = np.percentile(series.values, [10, 50, 90])
                            labels = ["p10", "p50", "p90"]
                            for val, lbl in zip(q_vals, labels):
                                X = base.copy()
                                _apply_continuous(X, mcols, plot_covariate, val)
                                # ensure complete column set/order
                                for c in mcols:
                                    if c not in X.columns:
                                        X[c] = 0.0
                                X = X[[c for c in mcols] + ([strata_col] if strata_col else [])]
                                sf = cph.predict_survival_function(X)
                                ax.plot(sf.index.values, sf.iloc[:, 0].values, label=f"{lbl}={self._fmt_num(val, nd=2)}")
                            ax.legend(title=plot_covariate, fontsize=9)
                            title_extra = f" (stratum={chosen_stratum})" if strata_col else ""
                            ax.set_title(f"Predicted Survival by {plot_covariate}{title_extra}")

                        ax.set_xlabel("Time")
                        ax.set_ylabel("Survival probability")
                        ax.grid(True, alpha=0.3)

                        artifacts.append({
                            "type": "plot",
                            "id": "cox_ph_survival_curves",
                            "title": f"Predicted Survival Curves by {plot_covariate}",
                            "content": PlotUtils.fig_to_base64(fig),
                        })
                        plt.close(fig)
            except Exception as plot_ex:
                sections.append({
                    "type": "text",
                    "title": f"Survival Curve Plot Failed for {plot_covariate}" if plot_covariate else "Survival Curve Plot Failed",
                    "content": str(plot_ex),
                })

            # === Model fit stats ===
            fit_summary = [
                ["Number of Observations", int(cph.durations.shape[0])],
                ["Number of Events", int(np.nan_to_num(cph.event_observed).sum())],
                ["Partial Log-Likelihood", self._fmt_num(cph.log_likelihood_, nd=3)],
                ["Concordance Index", self._fmt_num(cph.concordance_index_, nd=3)],
            ]
            sections.append({
                "type": "table",
                "title": "Model Fit Statistics",
                "headers": ["Statistic", "Value"],
                "data": fit_summary
            })

            # === Proportional Hazards tests (programmatic, not just printed) ===
            try:
                # Use rank-based time transform (common choice); alternative: "km", "identity"
                ph_test = proportional_hazard_test(cph, df_final, time_transform="rank")
                ph_df = ph_test.summary.reset_index().rename(columns={"index": "variable"})
                # Round numeric
                for col in ph_df.columns:
                    if pd.api.types.is_numeric_dtype(ph_df[col]):
                        ph_df[col] = ph_df[col].round(4)

                sections.append({
                    "type": "table",
                    "title": "Proportional Hazards Test (Schoenfeld residuals)",
                    "headers": ph_df.columns.tolist(),
                    "data": ph_df.values.tolist(),
                    "footer": f"P-values < {self._fmt_num(alpha_ui, nd=2)} suggest violation of the PH assumption."
                })

                # Optional diagnostic plots (version-agnostic; no plot_covariate_groups dependency)
                if gen_plots:
                    cov_list = list(cph.summary.index)
                    if cov_list:
                        with PlotUtils.setup_plotting():
                            for cov in cov_list:
                                fig = None
                                try:
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    self._plot_ph_schoenfeld(cph, df_final, cov, ax=ax)
                                    plt.tight_layout()
                                    artifacts.append({
                                        "type": "plot", "id": f"ph_diagnostic_{cov}",
                                        "title": f"PH Assumption Diagnostic for {cov}",
                                        "content": PlotUtils.fig_to_base64(fig)
                                    })
                                finally:
                                    if fig: plt.close(fig)
            except Exception as e:
                sections.append({
                    "type": "text",
                    "title": "PH Test Failed",
                    "content": str(e)
                })

            summary = (
                f"Cox regression completed with {len(df_final)} observations. "
                f"Concordance Index: {self._fmt_num(cph.concordance_index_, nd=3)}."
            )

            return {
                "status": "ok",
                "summary": summary,
                "artifacts": artifacts,
                "sections": sections,
                "meta": {"tool_name": self.name, "parameters": parameters},
            }

        except (ConvergenceError, ValueError) as e:
            # Common: collinearity (esp. with one-hot)
            is_one_hot = kwargs.get("encoding_method") == "one_hot"
            penalizer_str = kwargs.get("penalizer")
            penalizer_val = float(penalizer_str) if penalizer_str else 0.0

            if "collinearity" in str(e).lower() and is_one_hot:
                if penalizer_val == 0.0:
                    # User used one-hot with no penalizer, the most common cause of this error.
                    summary = (
                        "Model fitting failed due to high collinearity. This is common with **One-Hot Encoding** when no penalizer is used.\n\n"
                        "**Suggestions:**\n1. Switch to **Dummy (n-1)** encoding.\n2. Or, keep One-Hot encoding and add a small **Penalizer** value (e.g., 0.01) to handle the collinearity."
                    )
                else:
                    # User tried a penalizer, but it might have been too small.
                    summary = (
                        f"Model fitting failed due to high collinearity, even with a penalizer of {penalizer_val}. This can happen with very correlated data.\n\n"
                        "**Suggestions:**\n1. Try a slightly **larger Penalizer** value (e.g., 0.1).\n2. Switch to **Dummy (n-1)** encoding."
                    )
            else:
                summary = (
                    f"Model fitting failed: {str(e)}"
                )
            return {"status": "error", "summary": summary}
        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Provides a formal interpretation of the Cox Proportional Hazards model results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))

            summary_section = next((s for s in results.get('sections', []) if 'Model Summary' in s.get('title', '')), None)
            fit_stats_section = next((s for s in results.get('sections', []) if 'Model Fit Statistics' in s.get('title', '')), None)
            ph_test_section = next((s for s in results.get('sections', []) if 'Proportional Hazards Test' in s.get('title', '')), None)

            if not summary_section or not fit_stats_section:
                return "Could not find model summary or fit statistics for interpretation."

            # 1. Get Concordance Index
            c_index = "N/A"
            for row in fit_stats_section.get('data', []):
                if row[0] == 'Concordance Index':
                    c_index = row[1]
                    break
            
            interpretation_parts = [f"The model's Concordance Index is {c_index}, indicating its predictive discrimination ability (0.5 is random, 1.0 is perfect)."]

            # 2. Find significant covariates
            headers = summary_section.get('headers', [])
            try:
                # The index column can be named 'variable' or 'covariate'
                var_idx = -1
                if 'variable' in headers: var_idx = headers.index('variable')
                elif 'covariate' in headers: var_idx = headers.index('covariate')
                hr_idx = headers.index('Hazard Ratio')
                p_idx = headers.index('p-value')
            except ValueError:
                return "Could not parse the model summary table for interpretation."

            significant_vars = []
            for row in summary_section.get('data', []):
                p_value = float(row[p_idx])
                if p_value < alpha:
                    var_name = row[var_idx]
                    hr = float(row[hr_idx])
                    effect = "an increased" if hr > 1 else "a decreased"
                    change = f"by a factor of {hr:.2f}" if hr > 1 else f"to {hr:.2f} times the baseline"
                    significant_vars.append(f"'{var_name}' is a significant predictor. A one-unit increase is associated with {effect} hazard of the event ({change}, HR={hr:.3f}).")

            if significant_vars:
                interpretation_parts.append("\nSignificant Predictors:")
                interpretation_parts.extend([f"- {s}" for s in significant_vars])
            else:
                interpretation_parts.append("\nNo statistically significant predictors were found at the α={alpha} level.")

            # 3. Check PH assumption
            if ph_test_section:
                try:
                    ph_headers = ph_test_section.get('headers', [])
                    ph_var_idx = -1
                    if 'variable' in ph_headers: ph_var_idx = ph_headers.index('variable')
                    elif 'covariate' in ph_headers: ph_var_idx = ph_headers.index('covariate')

                    ph_p_idx = ph_headers.index('p')
                    
                    violators = []
                    for row in ph_test_section.get('data', []):
                        if row[ph_var_idx] != 'GLOBAL' and float(row[ph_p_idx]) < alpha:
                            violators.append(row[ph_var_idx])
                    
                    if violators:
                        interpretation_parts.append(f"\nWarning: The Proportional Hazards assumption may be violated for: {', '.join(violators)}. Results for these variables should be interpreted with caution. Consider using stratification.")
                    else:
                        interpretation_parts.append("\nThe Proportional Hazards assumption appears to hold for all covariates.")
                except (ValueError, IndexError):
                    pass # Could not parse PH test results

            return "\n".join(interpretation_parts)

        except (ValueError, TypeError, IndexError, KeyError):
            return "Could not automatically interpret the Cox Proportional Hazards model results."
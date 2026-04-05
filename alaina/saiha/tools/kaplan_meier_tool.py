
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
from django.core.files.storage import default_storage
from typing import Any, Dict, List, Optional

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from quantalytics.ai_agents.tools.plot_utils import PlotUtils


class KaplanMeierTool(BaseAnalysisTool):
    """
    A tool to perform Kaplan-Meier survival analysis.
    """

    @property
    def name(self) -> str:
        return "kaplan_meier"

    @property
    def description(self) -> str:
        return "Performs Kaplan-Meier survival analysis to estimate and compare survival curves."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(
            ToolParameter(
                name="time_column",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Time to Event Column",
                description="Select the column containing the duration (e.g., days, months).",
                required=True,
            )
        )
        params.add_parameter(
            ToolParameter(
                name="event_column",
                parameter_type=ParameterType.COLUMN_SELECT,
                label="Event Observed Column",
                description="Select the column indicating if the event occurred.",
                required=True,
                column_source="all",
            )
        )
        params.add_parameter(
            ToolParameter(
                name="event_value",
                parameter_type=ParameterType.TEXT,
                label="Event Value",
                description="If the event column is not numeric, specify the value that indicates an event (e.g., 'Dead').",
                help_text="Case-sensitive. Required if the event column contains text.",
                required=True,
            )
        )
        params.add_parameter(
            ToolParameter(
                name="group_column",
                parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                label="Group By Column (Optional)",
                description="Optional. Select a column to compare survival curves between groups.",
                required=False,
            )
        )
        params.add_parameter(
            ToolParameter(
                name="alpha",
                parameter_type=ParameterType.SELECT,
                label="Significance Level (α)",
                description="Controls the CI width: 1−α is the confidence level.",
                required=True,
                default_value="0.05",
                options=[
                    {"value": "0.05", "label": "0.05 (95% Confidence)"},
                    {"value": "0.01", "label": "0.01 (99% Confidence)"},
                    {"value": "0.10", "label": "0.10 (90% Confidence)"},
                ],
            )
        )
        return params

    # ----------------- helpers -----------------

    def _fmt_num(self, x, nd: int = 4) -> str:
        """
        Safe number formatting that won’t crash on strings/None/np types.
        Tries float→fixed with nd decimals; otherwise returns str(x).
        """
        try:
            xv = float(x)
            if np.isfinite(xv):
                return f"{xv:.{nd}f}"
        except Exception:
            pass
        return str(x)

    def _prepare_TE(
        self, df: pd.DataFrame, time_col: str, event_col: str, event_value: Optional[str]
    ):
        """
        Coerce and align T (durations) and E (events).
        - Time coerced to numeric, non-negative, finite.
        - Event numeric → non-zero as event=1; else text match against event_value.
        Drops rows with invalid/missing values.
        """
        T = pd.to_numeric(df[time_col], errors="coerce")

        if pd.api.types.is_numeric_dtype(df[event_col]):
            E_raw = pd.to_numeric(df[event_col], errors="coerce").fillna(0)
            E = (E_raw != 0).astype(int)
        else:
            if not event_value:
                return None, None, "The selected event column is not numeric. Please specify the 'Event Value' that indicates an event."
            E = (df[event_col] == event_value).astype(int)

        valid = T.notna() & np.isfinite(T) & (T >= 0) & E.notna()
        T = T[valid]
        E = E[valid]
        return T, E, None

    # ----------------- main -----------------

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            time_col = parameters.get("time_column")
            event_col = parameters.get("event_column")
            # Treat empty string as no group
            group_col = parameters.get("group_column") or None
            event_value = parameters.get("event_value")
            alpha_sig = float(parameters.get("alpha", 0.05))  # e.g., 0.05
            ci_level = max(0.0, min(1.0, 1.0 - alpha_sig))    # lifelines expects confidence level

            if not time_col or not event_col:
                return {"status": "error", "summary": "Time and Event columns are required."}

            # Use efficient column projection loading from BaseAnalysisTool
            cols_to_load = [time_col, event_col]
            if group_col:
                cols_to_load.append(group_col)
            df = self.load_dataset(columns=cols_to_load)

            # Column existence checks
            for col in [time_col, event_col] + ([group_col] if group_col else []):
                if col and col not in df.columns:
                    return {"status": "error", "summary": f"Column not found in dataset: {col}"}

            # Prepare aligned T and E
            T, E, err = self._prepare_TE(df, time_col, event_col, event_value)
            if err:
                return {"status": "error", "summary": err}
            if T is None or E is None or T.empty:
                return {"status": "error", "summary": "No valid rows after cleaning. Check time/event columns for missing or invalid values."}

            kmf = KaplanMeierFitter()
            artifacts: List[Dict[str, Any]] = []
            sections: List[Dict[str, Any]] = []
            summary = "Kaplan-Meier analysis completed."
            statistical_results = {}
            lr_res = None

            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(10, 6))

                if group_col:
                    # Align groups to valid T/E index and drop NaNs
                    groups_series = df.loc[T.index, group_col]
                    valid_groups = groups_series.dropna()
                    unique_groups = valid_groups.unique()

                    if valid_groups.nunique() < 2:
                        # Not enough groups; fallback to overall
                        kmf.fit(T, E, label="Overall Survival", alpha=ci_level)
                        kmf.plot_survival_function(ax=ax, ci_show=True)
                        ax.set_title("Kaplan-Meier Survival Curve (no valid groups)")
                    else:
                        # Plot per group (cap for readability)
                        MAX_GROUPS_PLOT = 10
                        plotted = 0
                        group_summary_records = []
                        for g in unique_groups:
                            # Safely align the boolean index to T and E's index
                            idx_series = (valid_groups == g)
                            # Ensure we only use indices that exist in T and E to prevent Unalignable boolean Series error
                            aligned_idx = idx_series.reindex(T.index, fill_value=False)
                            
                            if aligned_idx.sum() == 0:
                                continue
                            kmf.fit(T[aligned_idx], E[aligned_idx], label=str(g), alpha=ci_level)
                            if plotted < MAX_GROUPS_PLOT:
                                kmf.plot_survival_function(ax=ax, ci_show=True)
                                plotted += 1
                            
                            # Capture group-specific stats
                            median_survival = kmf.median_survival_time_
                            median_survival_display = "Not Reached"
                            if np.isfinite(median_survival):
                                median_survival_display = self._fmt_num(median_survival, nd=2)

                            group_summary_records.append({
                                "Group": str(g),
                                "Observations": int(idx.sum()),
                                "Events": int(E[idx].sum()),
                                "Median Survival Time": median_survival_display,
                            })

                        if plotted >= MAX_GROUPS_PLOT:
                            sections.append({
                                "type": "text",
                                "title": "Note",
                                "content": f"Only the first {MAX_GROUPS_PLOT} groups were plotted for readability."
                            })

                        # Add the group summary table to the sections
                        if group_summary_records:
                            headers = list(group_summary_records[0].keys())
                            sections.append({
                                'type': 'table',
                                'title': 'Survival Summary by Group',
                                'headers': headers,
                                'data': [[rec[h] for h in headers] for rec in group_summary_records]
                            })

                        # Multigroup log-rank test (k >= 2)
                        try:
                            lr_res = multivariate_logrank_test(
                                event_durations=T.values,
                                groups=valid_groups.values,
                                event_observed=E.values,
                            )
                            sections.append({
                                "type": "table",
                                "title": f"Log-Rank Test by {group_col}",
                                "headers": ["Statistic", "Value"],
                                "data": [
                                    ["Test Statistic", self._fmt_num(lr_res.test_statistic, nd=4)],
                                    ["P-Value", self._fmt_num(lr_res.p_value, nd=6)],
                                    ["Is Significant", "Yes" if float(lr_res.p_value) < alpha_sig else "No"],
                                ],
                            })
                            ax.set_title(f"Kaplan-Meier Survival Curves by {group_col}")
                        except Exception as ex:
                            sections.append({
                                "type": "text",
                                "title": "Log-Rank Test Error",
                                "content": f"Could not compute multigroup log-rank test: {str(ex)}"
                            })
                            ax.set_title(f"Kaplan-Meier Survival Curves by {group_col} (no test)")
                else:
                    # Single overall curve
                    kmf.fit(T, E, label="Overall Survival", alpha=ci_level)
                    kmf.plot_survival_function(ax=ax, ci_show=True)
                    ax.set_title("Kaplan-Meier Survival Curve")

                    # Create a summary table for the single group case
                    median_survival = kmf.median_survival_time_
                    median_survival_display = "Not Reached"
                    if np.isfinite(median_survival):
                        median_survival_display = self._fmt_num(median_survival, nd=2)

                    single_group_summary = [{
                        "Group": "Overall",
                        "Observations": int(T.size),
                        "Events": int(E.sum()),
                        "Median Survival Time": median_survival_display,
                    }]

                    headers = list(single_group_summary[0].keys())
                    sections.append({
                        'type': 'table', 'title': 'Survival Summary', 'headers': headers,
                        'data': [[rec[h] for h in headers] for rec in single_group_summary]
                    })

                ax.set_xlabel("Time")
                ax.set_ylabel("Survival Probability")
                ax.legend()
                plt.tight_layout()

                artifacts.append({
                    "type": "plot",
                    "id": "kaplan_meier_curve",
                    "title": "Kaplan-Meier Survival Curve",
                    "content": PlotUtils.fig_to_base64(fig),
                })

            # Median survival (overall only)
            if not group_col:
                ms = kmf.median_survival_time_
                try:
                    msf = float(ms) if ms is not None else None
                except Exception:
                    msf = None

                if msf is not None and np.isfinite(msf):
                    statistical_results['median_survival_time'] = msf
                    summary += f" The median survival time is {self._fmt_num(msf, nd=2)}."
                else:
                    summary += " The median survival time was not reached (more than 50% survival)."
            else:
                if lr_res is not None:
                    summary += f" Groups were compared using a multigroup log-rank test (p={self._fmt_num(lr_res.p_value, nd=4)})."

            return {
                "status": "ok",
                "summary": summary,
                "data": {"statistical_results": statistical_results},
                "artifacts": artifacts,
                "sections": sections,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                },
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Provides a formal interpretation of the Kaplan-Meier analysis results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            group_col = params.get('group_column') or None

            if group_col:
                # Group comparison interpretation
                log_rank_section = next((s for s in results.get('sections', []) if 'Log-Rank Test' in s.get('title', '')), None)
                if not log_rank_section:
                    return "Log-rank test results not found for interpretation."

                p_value = None
                for row in log_rank_section.get('data', []):
                    if row[0] == 'P-Value':
                        p_value = float(row[1])
                        break
                
                if p_value is None:
                    return "Could not determine p-value from log-rank test results."

                if p_value < alpha:
                    return f"Since the p-value from the log-rank test ({p_value:.4f}) is less than α ({alpha}), we reject the null hypothesis. This indicates there is a statistically significant difference in the survival distributions among the groups defined by '{group_col}'."
                else:
                    return f"Since the p-value from the log-rank test ({p_value:.4f}) is not less than α ({alpha}), we fail to reject the null hypothesis. There is not enough evidence to conclude a significant difference in survival distributions among the groups defined by '{group_col}'."
            else:
                # Single group interpretation
                summary_section = next((s for s in results.get('sections', []) if s.get('title') == 'Survival Summary'), None)
                if not summary_section:
                    return "Overall survival summary not found for interpretation."

                median_survival = summary_section['data'][0][3] # 'Median Survival Time' is the 4th column
                if median_survival == "Not Reached":
                    return "The Kaplan-Meier analysis for the overall sample shows that the median survival time was not reached. This means more than 50% of subjects survived past the maximum observation time in the dataset."
                else:
                    return f"The Kaplan-Meier analysis for the overall sample shows a median survival time of {median_survival}. This is the time point at which 50% of the subjects are estimated to have not yet experienced the event."

        except (ValueError, TypeError, IndexError, KeyError):
            return "Could not automatically interpret the Kaplan-Meier analysis results."
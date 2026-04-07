import logging
import json
from typing import List, Dict, Any, Optional
from ..llm_management.gemini_service import gemini_service
from ..analysis_tools.registry import tool_registry

logger = logging.getLogger(__name__)


class AnalysisPlanner:
    """
    The orchestrator that maps User Query → Tool Intents.

    Architecture (3 layers):
      Layer 0 — Deterministic Pre-Router:
                Intercepts obvious intents via keyword matching.
                Zero LLM calls, ~0ms latency.
      Layer 1 — Smart Tool Filter:
                Classifies query into a category, sends only
                10–20 relevant tools to LLM instead of all 80+.
      Layer 2 — Gemini Planner:
                Handles fuzzy / compound / ambiguous queries.
    """

    # ── LAYER 1 CONFIG ──────────────────────────────────────────────────────
    # Maps category name → exact tool names as registered in the registry.
    _TOOL_CATEGORIES: Dict[str, List[str]] = {
        "stats": [
            "correlation_matrix", "descriptive_statistics",
            "statistical_analysis", "column_analysis", "dataset_overview",
            "reliability_analysis", "effect_size_calculator",
            "precision_analysis",
        ],
        "hypothesis": [
            "one_way_anova", "two_way_anova", "repeated_measures_anova",
            "one_sample_t_test", "two_sample_t_test", "paired_t_test",
            "one_sample_z_test", "two_sample_z_test",
            "chi_square_test", "stratified_chi_square_test",
            "mann_whitney_u_test", "kruskal_wallis_test",
            "wilcoxon_signed_rank_test", "friedman_test",
            "one_sample_ks_test", "two_sample_ks_test",
            "manova_tool", "log_linear_analysis",
        ],
        "viz": [
            "histogram", "box_plot", "scatter_plot", "bar_chart",
            "line_chart", "pair_plot", "mosaic_plot", "qq_plot",
            "outlier_detection", "acf_pacf_plots", "visualization",
        ],
        "ml": [
            "kmeans_clustering", "pca", "decision_tree",
            "logistic_regression", "linear_regression",
            "lasso_ridge_regression", "factor_analysis",
            "multinomial_logistic_regression", "mediation_analysis",
            "monte_carlo_simulation", "sensitivity_analysis",
            "propensity_score_matching", "sample_size_estimator",
            "randomization_generator",
        ],
        "survival": [
            "kaplan_meier", "cox_ph_regression",
        ],
        "causal": [
            "difference_in_differences", "interrupted_time_series",
            "mediation_analysis", "propensity_score_matching",
        ],
        "time_series": [
            "arima_forecasting", "auto_arima",
            "time_series_decomposition", "acf_pacf_plots",
        ],
        "cleaning": [
            "data_quality_assessment", "drop_column", "rename_column",
            "filter_rows", "imputation_tool", "recode_tool",
            "data_type_conversion", "datetime_conversion",
            "variable_transformation", "compute_variable",
            "outlier_treatment",
        ],
        "reporting": [
            "Generate Session Summary", "Smart Recommendations",
            "recommendations", "export",
            "Export PowerPoint Report", "Export Word Report",
        ],
    }

    # Keywords that trigger each category. Longer/more-specific phrases first.
    _CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "stats": [
            "statistic", "mean", "median", "std", "average",
            "describe", "summary", "correlation", "distribution",
            "variance", "spread",
        ],
        "hypothesis": [
            "anova", "t-test", "t test", "chi-square", "chi square",
            "mann whitney", "wilcoxon", "kruskal", "friedman",
            "hypothesis", "p-value", "p value", "significance",
            "normality", "z-test", "z test", "manova",
        ],
        "viz": [
            "plot", "chart", "graph", "visual", "histogram",
            "scatter", "box plot", "bar chart", "qq plot",
            "pair plot", "line chart", "mosaic",
        ],
        "ml": [
            "cluster", "classify", "predict", "regression",
            "pca", "factor", "machine learning", "model",
            "lasso", "ridge", "logistic", "decision tree",
            "random forest", "simulation", "monte carlo",
        ],
        "survival": [
            "survival", "kaplan", "meier", "cox", "hazard",
            "time to event", "censored",
        ],
        "causal": [
            "causal", "difference in differences", "did analysis",
            "interrupted time series", "propensity", "mediation",
            "counterfactual",
        ],
        "time_series": [
            "time series", "forecast", "arima", "trend",
            "seasonal", "acf", "pacf", "decomposition",
        ],
        "cleaning": [
            "clean", "missing", "duplicate", "drop column",
            "rename", "filter rows", "impute", "recode",
            "convert", "transform", "data quality",
        ],
        "reporting": [
            "report", "export", "recommend", "session summary",
            "pptx", "docx", "word", "powerpoint",
        ],
    }

    # ── LAYER 0 CONFIG ──────────────────────────────────────────────────────
    # (keywords, tool_name, base_params)
    # Order matters — more specific/longer phrases must come FIRST.
    _HARD_ROUTES = [
        # Correlation — any of these phrases → always correlation_matrix
        (
            ["correlation matrix", "correlation heatmap", "run correlation",
             "show correlation", "collinear", "pairwise correlation",
             "correlate all", "correlation between", "check correlation",
             "compute correlation"],
            "correlation_matrix", {}
        ),
        # Descriptive statistics — require compound phrases or unambiguous terms
        (
            ["descriptive statistics", "run descriptive", "full statistics",
             "summary statistics", "show skewness", "show kurtosis",
             "skewness", "kurtosis", "show percentiles", "distribution summary",
             "statistical summary"],
            "descriptive_statistics", {}
        ),
        # Dataset overview
        (
            ["dataset overview", "data overview", "show overview",
             "overview of the dataset", "overview of data"],
            "dataset_overview", {}
        ),
        # Data quality
        (
            ["data quality", "check duplicates", "find duplicates",
             "missing value report", "data cleaning check", "quality check"],
            "data_quality_assessment", {}
        ),
        # Outlier detection
        (
            ["detect outliers", "find outliers", "show outliers",
             "outlier detection", "identify outliers", "outlier analysis"],
            "outlier_detection", {}
        ),
        # Session summary
        (
            ["session summary", "summarize session", "summarise session",
             "overall summary", "analysis summary"],
            "Generate Session Summary", {}
        ),
        # Recommendations
        (
            ["recommendations", "what should i do next",
             "smart recommendations", "suggest next steps",
             "what do you recommend"],
            "Smart Recommendations", {}
        ),
    ]

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or gemini_service.model_id

    # ── LAYER 0: DETERMINISTIC PRE-ROUTER ───────────────────────────────────
    def _pre_route(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Pure Python keyword match. Bypasses LLM entirely for obvious intents.
        Returns a tool intent list if matched, None if query needs LLM judgment.
        """
        q = query.lower().strip()
        for keywords, tool_name, base_params in self._HARD_ROUTES:
            if any(kw in q for kw in keywords):
                logger.info(
                    f"[PRE-ROUTER] Hard-routed '{query}' → '{tool_name}'"
                )
                return [{"tool": tool_name, "params": base_params}]
        return None

    # ── LAYER 1: SMART TOOL FILTER ───────────────────────────────────────────
    def _get_relevant_tools(
        self, all_tools: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """
        Classifies query into tool categories and returns only the relevant
        subset instead of all 80+ tools. Dramatically reduces token usage.
        """
        q = query.lower()
        tool_name_to_meta = {t["tool"]: t for t in all_tools}

        # Find matching categories
        matched_categories: set = set()
        for category, keywords in self._CATEGORY_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                matched_categories.add(category)

        # Fallback: if no category matched, send stats + viz (most common)
        if not matched_categories:
            matched_categories = {"stats", "viz"}

        # Collect relevant tool names
        relevant_names: set = set()
        for cat in matched_categories:
            relevant_names.update(self._TOOL_CATEGORIES.get(cat, []))

        # Always include core utility tools so LLM is never totally blind
        relevant_names.update([
            "dataset_overview", "Generate Session Summary",
            "Smart Recommendations",
        ])

        filtered = [
            tool_name_to_meta[n]
            for n in relevant_names
            if n in tool_name_to_meta
        ]

        logger.info(
            f"[TOOL FILTER] Query '{query}' → categories {matched_categories} "
            f"→ {len(filtered)}/{len(all_tools)} tools sent to LLM"
        )
        return filtered

    # ── LAYER 2: LLM PLANNER ────────────────────────────────────────────────
    def create_plan(
        self,
        query: str,
        schema_text: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Maps a query to a list of tool intents using a 3-layer architecture:
          Layer 0 → Deterministic pre-router (hard keyword match, no LLM)
          Layer 1 → Smart tool filter (category-based, reduces token load)
          Layer 2 → Gemini (fuzzy / compound / ambiguous queries only)
        """

        # ── LAYER 0: TRY DETERMINISTIC ROUTING FIRST ───────────────────────
        pre_routed = self._pre_route(query)
        if pre_routed is not None:
            return pre_routed

        # ── LAYER 1: FILTER TOOLS TO RELEVANT SUBSET ───────────────────────
        all_tools_meta = tool_registry.get_all_tool_metadata()
        for t in all_tools_meta:
            t["tool"] = t.pop("name")

        filtered_tools = self._get_relevant_tools(all_tools_meta, query)
        tools_description = json.dumps(filtered_tools, indent=2)

        # ── LAYER 2: GEMINI PLANNER ─────────────────────────────────────────
        # Format history
        history_text = "No previous context."
        if history:
            history_text = "\n".join(
                [f"{m['role'].upper()}: {m['content']}" for m in history]
            )

        system_instruction = f"""
        You are an expert data analyst planner for the ChatFlow system.
        Your task is to identify which analysis tools are needed to satisfy the user's query.

        CONVERSATION HISTORY (Last 10 messages):
        {history_text}

        DATASET SCHEMA (Columns & Types):
        {schema_text}

        AVAILABLE TOOLS (WITH PARAMETERS):
        {tools_description}

        PLANNING RULES (STRICT):
        1. **PARAM-AWARE SATURATION**: If the User Query is already satisfied by a SUCCESSFUL tool execution in the HISTORY with the SAME parameters, you MUST return an empty array `[]`. Do NOT re-run the same tool on the same column.
        2. **LEGITIMATE SEQUENTIAL ANALYSIS**: If the user asks for a DIFFERENT column (e.g., 'Now do Income' after 'Do Age'), you SHOULD run the appropriate tool again with the NEW parameters.
        3. **STRICT PARAMETERS**: Use ONLY the parameter names listed in the tool list above.
        4. **VIZ-READY**: If the user mentions 'graph', 'chart', or 'plot', you MUST select a tool that produces visualizations (e.g. outlier_detection, box_plot, histogram).
        5. **OUTLIERS + GRAPHS**: If outliers are requested, prioritize 'outlier_detection'.
        6. **TOOL MAPPING GUIDELINES**:
           - **Correlation Matrix / Heatmap / Pairwise / Collinearity**: Always use `correlation_matrix`.
           - **Descriptive Statistics (numeric deep-dive)**: Use `descriptive_statistics` for skewness, kurtosis, percentiles, or full numeric summaries.
           - **Column Profiling / Mixed/Categorical Columns**: Use `column_analysis` for frequency tables, missing values, or mixed-type exploration.
           - **Data Quality / Cleaning Checks**: Use `data_quality_assessment`.
           - **Hypothesis Testing (t-test, ANOVA, chi-square etc.)**: Use the specific test tool. Do NOT use `statistical_analysis` for these.
           - **Specific Plots (Hist, Box, Scatter)**: Use the dedicated plot tools.
        7. **FORMAT**: Return ONLY a JSON array.
        8. **STRICT KEY NAMES**: Use `"tool"` for the tool name and `"params"` for parameters.
        9. **FLAT JSON PARAMS**: `"params"` must be a flat JSON object `{{}}`.
        10. **JSON EXAMPLE**: `[{{"tool": "histogram", "params": {{"column_name": "Age"}}}}]`
        11. **DONE SIGNAL**: If the analysis is already complete or no tool fits, return `[]`.
        """

        prompt = (
            f"User Query: '{query}'\n"
            f"Analyze the query vs history and schema. Determine if a NEW tool execution is required."
        )

        try:
            intents = gemini_service.get_intent_json(prompt, system_instruction)

            # Robustness: handle single dict or nested lists
            if isinstance(intents, dict):
                intents = [intents]

            if isinstance(intents, list):
                flattened = []
                for item in intents:
                    if isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(item)
                intents = flattened

            if not isinstance(intents, list):
                logger.error(f"Invalid intent format returned: {intents}")
                return []

            logger.info(f"[LLM PLANNER] Query '{query}' → {intents}")
            return intents

        except Exception as e:
            logger.error(f"Analysis Planning failed: {e}")
            return []


# Global accessor
analysis_planner = AnalysisPlanner()

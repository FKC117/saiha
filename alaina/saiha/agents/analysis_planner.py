import logging
import json
from typing import List, Dict, Any, Optional
from ..llm_management.gemini_service import gemini_service
from ..analysis_tools.registry import tool_registry
from ..models import AnalysisSession
from .context_builder import ContextBuilder

logger = logging.getLogger(__name__)


class AnalysisPlanner:
    """
    The orchestrator that maps User Query → Tool Intents.
    """

    # ── ELITE MODE v3.2 PROTECTIONS ──────────────────────────────────────────
    _FOLLOW_UP_PROMPTS = [
        "it", "this", "that", "those", "them", "they", "its", 
        "why", "how", "weird", "explain", "meaning", "insight",
        "continue", "more", "next", "again"
    ]
    _OVERRIDE_KEYWORDS = ["no", "instead", "not", "change", "use", "replace"]

    def _normalize(self, text: str) -> str:
        """Standardizes text for fuzzy matching (Elite v3.3)."""
        if not text: return ""
        return text.lower().replace("_", " ").strip()

    def _get_dataset_columns(self, session: AnalysisSession) -> List[str]:
        """Atomic fetch of all column names for the current dataset."""
        if not session.dataset: return []
        return list(session.dataset.columns.values_list('column_name', flat=True))

    def _query_has_new_signal(self, query: str, session: AnalysisSession) -> bool:
        """
        Detects if the user is mentioning a column NOT currently in focus (Pivoting).
        Optimized v3.3: Pre-computes normalized maps to avoid nested loop hangs.
        """
        query_norm = self._normalize(query)
        columns = self._get_dataset_columns(session)
        if not columns: return False

        # Get currently active columns for comparison
        wm = session.working_memory or {}
        active = {self._normalize(c) for c in wm.get("active_columns", [])}
        
        # 1. Full Match Detection (O(N))
        # Checks if any column name is found in the query text.
        for col in columns:
            col_norm = self._normalize(col)
            if col_norm in query_norm and col_norm not in active:
                return True
        
        # 2. Sub-token Matching (O(T * N))
        # Only check tokens > 3 chars to avoid noise like 'the', 'and'.
        query_tokens = [t for t in query_norm.split() if len(t) > 3]
        if not query_tokens: return False

        for col in columns:
            col_norm = self._normalize(col)
            if col_norm in active: continue
            if any(t in col_norm for t in query_tokens):
                return True
                
        return False

    def _is_override(self, query: str, session: AnalysisSession) -> bool:
        """
        Context-Aware Override Detection (Elite v3.3).
        Only triggers if an override keyword is used alongside a dataset entity.
        Optimized performance with early exits.
        """
        query_norm = self._normalize(query)
        words = query_norm.split()
        
        has_keyword = any(kw in words for kw in self._OVERRIDE_KEYWORDS)
        if not has_keyword: return False
        
        # Check for entity reference (column name) using pre-normalized signal detection logic
        # Reuses the same logic as _query_has_new_signal but ignores 'active' state
        columns = self._get_dataset_columns(session)
        
        # Exact match anywhere in query
        for col in columns:
            col_norm = self._normalize(col)
            if col_norm in query_norm:
                return True
        
        # Token-based match
        significant_tokens = [t for t in words if len(t) > 3]
        for col in columns:
            col_norm = self._normalize(col)
            if any(t in col_norm for t in significant_tokens):
                return True
        
        return False

    def _is_follow_up(self, query: str, session: AnalysisSession) -> bool:
        """
        Hybrid Intent Detection (Elite v3.3).
        Combines linguistic signals, query length, state presence, and pivot detection.
        Priority: Pivot Detection > Linguistic Signal > Brevity.
        """
        # 1. Pivot Detection (Elite v3.3) - PRIORITY
        # If user mentions a NEW column, it's a fresh query branch, not a follow-up,
        # regardless of how short the query is.
        if self._query_has_new_signal(query, session):
            logger.info("New signal detected in query; treating as fresh branch.")
            return False

        query_low = query.lower()
        words = query_low.split()

        # 2. Linguistic Signals (Pronouns/Demonstratives)
        if any(word in self._FOLLOW_UP_PROMPTS for word in words):
            return True
            
        # 3. Query Length (Human-style brevity usually implies context)
        if len(words) < 5:
            return True
        
        # 4. State Presence (Fallback)
        wm = session.working_memory or {}
        if wm.get("active_columns") or wm.get("last_tool"):
            return True
            
        return False

    """
    Architecture (3 layers):
      Layer 0 - Deterministic Pre-Router:
                Intercepts obvious intents via keyword matching.
                Zero LLM calls, ~0ms latency.
      Layer 1 - Smart Tool Filter:
                Classifies query into a category, sends only
                10-20 relevant tools to LLM instead of all 80+.
      Layer 2 - Gemini Planner:
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
        (
            ["recommendations", "what should i do next",
             "smart recommendations", "suggest next steps",
             "what do you recommend"],
            "Smart Recommendations", {}
        ),
    ]

    _SYSTEM_INSTRUCTION_STATIC = """
    You are an expert data analyst planner for the ChatFlow system.
    Your task is to identify which analysis tools are needed to satisfy the user's query.

    PLANNING RULES (STRICT):
    1. **PARAM-AWARE SATURATION**: If the User Query is already satisfied by a SUCCESSFUL tool execution in the HISTORY with the SAME parameters, return `{"tool": "chat", "params": {"message": "ALREADY_DONE"}}`. Do NOT return an empty array if the intent is clear but finished.
    2. **LEGITIMATE SEQUENTIAL ANALYSIS**: If the user asks for a DIFFERENT column (e.g., 'Now do Income' after 'Do Age'), you SHOULD run the appropriate tool again with the NEW parameters.
    3. **STRICT PARAMETERS**: Use ONLY the parameter names listed in the tool list below.
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
    9. **FLAT JSON PARAMS**: `"params"` must be a flat JSON object `{}`.
    10. **JSON EXAMPLE**: [{"tool": "histogram", "params": {"column_name": "Age"}}]
    11. **DONE SIGNAL**: If you determine that NO new analysis is required (it's already done), you MUST return: [{"tool": "chat", "params": {"message": "ALREADY_DONE"}}]
    12. **MANDATORY ARRAY**: You must ALWAYS return a valid JSON array `[]`, even if it is empty.
    """

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

        relevant_names = sorted(list(relevant_names))
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
        session_id: Optional[str] = None,
        user: Optional[Any] = None,
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
        # 1. Fetch Session object for context & caching
        session = None
        if session_id:
            session = AnalysisSession.objects.filter(id=session_id).first()

        if not session:
            logger.error(f"Cannot create plan: Session {session_id} not found.")
            return []

        # 2. Build Static Context (FOR CACHING)
        static_context = ContextBuilder.build_static_context(
            self._SYSTEM_INSTRUCTION_STATIC,
            schema_text,
            tools_description
        )

        # 3. Get or Create Gemini Context Cache (Deterministic v3)
        cache_id = gemini_service.get_or_create_cache(
            session=session,
            static_context_str=static_context,
            system_instruction=self._SYSTEM_INSTRUCTION_STATIC
        )

        # 4. Intent & Override Processing (Elite v3.3)
        from .memory_manager import MemoryManager
        
        # A. Check for Contextual Override (KW + Entity)
        if self._is_override(query, session):
            logger.info(f"Intent Override detected for session {session.id}. Performing fresh reset.")
            MemoryManager.decay_stale_state(session, force=True)
            include_summary = False
        else:
            # B. Normal Gating
            include_summary = self._is_follow_up(query, session)
            
            # C. Freshness Guard: If metadata was failed/empty, blocking stale fallback if a new signal is present
            if session.last_valid_metadata == {} and self._query_has_new_signal(query, session):
                logger.debug("Blocking stale metadata fallback due to new signal detection.")
                include_summary = False

        # 5. Build Dynamic Prompt (Rolling Memory + Query)
        dynamic_prompt = ContextBuilder.build_planner_context(session, query, include_summary=include_summary)

        try:
            # Call Gemini with Cache support
            intents = gemini_service.get_intent_json(
                prompt=dynamic_prompt,
                system_instruction=self._SYSTEM_INSTRUCTION_STATIC,
                session_id=session_id,
                user=user,
                cache_name=cache_id
            )

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

            # --- Vague / Ambiguous Query Fallback (v3.2) ---
            # If the LLM didn't find specific tools and returned 'chat', but we have no context,
            # we should suggest a clarification fallback.
            if len(intents) == 1 and intents[0].get('tool') == 'chat' and not include_summary:
                # If they just said "hey" or something short with no context, ask what to analyze.
                if len(query.split()) < 4:
                    return [{
                        "tool": "chat",
                        "parameters": {
                            "message": "I'm ready to help! What specific columns or relationships would you like me to analyze?"
                        }
                    }]

            logger.info(f"[LLM PLANNER] Query '{query}' → {intents}")
            return intents

        except Exception as e:
            logger.error(f"Analysis Planning failed: {e}")
            return []


# Global accessor
analysis_planner = AnalysisPlanner()

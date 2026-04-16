import logging
from typing import List, Dict, Any, Optional
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)


class ParamCorrector:
    """
    Schema-Aware Parameter Correction Layer.

    Upgraded from keyword-heuristic to schema-driven.
    Uses ToolParameterSet (the tool's own contract) to:
      1. Detect column parameters via `column_source` (not key-name guessing)
      2. Fuzzy-correct enum values against the `options` list
      3. Auto-fill missing params from `default_value`
      4. Type-coerce values to match `ParameterType`
      5. Raise structured errors for missing required params (fail fast, fail loud)

    Falls back to legacy keyword heuristic for any tool that
    doesn't provide a schema (graceful degradation).
    """

    def __init__(self, available_columns: List[str]):
        self.available_columns = available_columns

    # ── PUBLIC API ───────────────────────────────────────────────────────────

    def correct_column_names(
        self, proposed_columns: List[str], threshold: int = 70
    ) -> List[str]:
        """
        Fuzzy-match proposed column names against the actual dataset schema.
        Ex: ['smoking habit'] → ['Smoking']
        """
        corrected = []
        col_lower_map = {c.lower(): c for c in self.available_columns}

        for col in proposed_columns:
            if not col or not isinstance(col, str):
                continue

            # 1. Exact match (fastest)
            if col in self.available_columns:
                corrected.append(col)
                continue

            # 2. Case-insensitive exact match (LLM uses "age" when schema has "Age")
            ci_match = col_lower_map.get(col.lower())
            if ci_match:
                logger.info(f"[ParamCorrector] Case-insensitive match: '{col}' → '{ci_match}'")
                corrected.append(ci_match)
                continue

            # 3. Fuzzy match for typos and partial names
            match = process.extractOne(col, self.available_columns, scorer=fuzz.WRatio)
            if match and match[1] >= threshold:
                logger.info(
                    f"[ParamCorrector] Fuzzy column match: '{col}' → '{match[0]}' "
                    f"(score={match[1]})"
                )
                corrected.append(match[0])
            else:
                logger.warning(
                    f"[ParamCorrector] No column match for '{col}' "
                    f"(threshold={threshold}). Keeping original."
                )
                corrected.append(col)
        return corrected

    def apply_to_params(
        self,
        params: Dict[str, Any],
        schema,  # ToolParameterSet | None
    ) -> Dict[str, Any]:
        """
        Apply schema-driven correction to LLM-proposed params.

        If schema is None or empty, falls back to the legacy keyword heuristic
        so no existing tool breaks.
        """
        # Guard: fall back to legacy if no schema provided
        if schema is None or not hasattr(schema, "parameters") or not schema.parameters:
            logger.debug("[ParamCorrector] No schema — using legacy keyword heuristic.")
            return self._legacy_apply(params)

        # Lazy import to avoid circular imports at module load time
        try:
            from ..analysis_tools.tool_parameters import ParameterType
        except ImportError:
            logger.warning("[ParamCorrector] Could not import ParameterType — legacy fallback.")
            return self._legacy_apply(params)

        corrected = params.copy()
        validation_errors = []

        for param_def in schema.parameters:
            name = param_def.name
            value = corrected.get(name)

            # ── STEP 1: Auto-fill missing params from default_value ──────────
            # ── STEP 1: Auto-fill/Synonym Map missing params ──────────
            if value is None:
                # A. Synonym Mapping (Hardened Elite v3.3)
                # If a required column parameter is missing, search for common hallucinations.
                # We check both the explicit 'column_source' and the 'parameter_type'.
                is_column_param = (
                    param_def.column_source is not None or 
                    (hasattr(param_def, 'parameter_type') and 
                     str(param_def.parameter_type.value).endswith('column_select'))
                )
                
                if is_column_param:
                    aliases = ["column", "col", "variable", "var", "target", "y", "feature", "x", "column_name"]
                    for alias in aliases:
                        if alias in params:
                            raw_val = params[alias]
                            # Fuzzy correct if it's a string, or a list of strings
                            if isinstance(raw_val, str):
                                value = self.correct_column_names([raw_val])[0]
                            elif isinstance(raw_val, list):
                                value = self.correct_column_names(raw_val)
                            else:
                                value = raw_val
                                
                            corrected[name] = value
                            logger.info(f"[ParamCorrector] Hardened synonym mapped & corrected: '{alias}' → '{name}' for value '{value}'")
                            break
                
                # B. Default Value Fallback
                if value is None:
                    if param_def.default_value is not None:
                        corrected[name] = param_def.default_value
                        logger.debug(
                            f"[ParamCorrector] '{name}' missing → filled default: "
                            f"{param_def.default_value}"
                        )
                    elif param_def.required:
                        # Required param with no default and no value — record error
                        validation_errors.append(
                            f"Required parameter '{name}' is missing and has no default."
                        )
                    continue  # Nothing more to correct for a missing/defaulted param

            # ── STEP 2: Column params — use column_source, not key name ──────
            if param_def.column_source is not None:
                if isinstance(value, str):
                    corrected[name] = self.correct_column_names([value])[0]
                elif isinstance(value, list):
                    corrected[name] = self.correct_column_names(value)
                # Type already handled — skip further processing
                continue

            # ── STEP 3: Enum correction via options list ──────────────────────
            ptype = param_def.parameter_type
            if ptype == ParameterType.SELECT and param_def.options:
                valid_values = [o.get("value", "") for o in param_def.options]
                if isinstance(value, str) and value not in valid_values:
                    match = process.extractOne(value, valid_values, scorer=fuzz.WRatio)
                    if match and match[1] >= 70:
                        logger.info(
                            f"[ParamCorrector] Enum corrected: '{name}'='{value}' "
                            f"→ '{match[0]}' (score={match[1]})"
                        )
                        corrected[name] = match[0]
                    elif param_def.default_value is not None:
                        logger.warning(
                            f"[ParamCorrector] Invalid enum '{value}' for '{name}' "
                            f"— falling back to default: '{param_def.default_value}'"
                        )
                        corrected[name] = param_def.default_value
                continue

            # ── STEP 4: Type coercion ─────────────────────────────────────────
            if ptype == ParameterType.CHECKBOX:
                # LLM often returns "true" / "false" as strings or 0/1 as ints
                if isinstance(value, str):
                    corrected[name] = value.lower() in ("true", "on", "1", "yes")
                elif isinstance(value, int):
                    corrected[name] = bool(value)

            elif ptype == ParameterType.NUMBER:
                try:
                    corrected[name] = float(value)
                except (ValueError, TypeError):
                    if param_def.default_value is not None:
                        logger.warning(
                            f"[ParamCorrector] Could not coerce '{name}'='{value}' "
                            f"to number — using default: {param_def.default_value}"
                        )
                        corrected[name] = param_def.default_value

            elif ptype == ParameterType.MULTISELECT:
                # LLM sometimes returns a comma-separated string instead of a list
                if isinstance(value, str):
                    corrected[name] = [v.strip() for v in value.split(",") if v.strip()]

        # ── STEP 5: Fail fast (loud) for truly missing required params ────────
        if validation_errors:
            logger.error(
                f"[ParamCorrector] Validation errors: {validation_errors}"
            )
            # Raise so the agent can broadcast a clear error to the UI
            # instead of silently dispatching a broken task to Celery.
            raise ValueError(
                f"Parameter validation failed: {'; '.join(validation_errors)}"
            )

        return corrected

    # ── PRIVATE: LEGACY FALLBACK ─────────────────────────────────────────────

    def _legacy_apply(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Original keyword-heuristic approach.
        Used only when a tool provides no ToolParameterSet schema.
        """
        corrected = params.copy()
        for key, value in params.items():
            if (
                "column" in key.lower()
                or "variable" in key.lower()
                or "var" in key.lower()
                or key.lower() in ("x", "y", "target", "feature",
                                   "dependent", "independent", "outcome",
                                   "time_col", "event_col", "group")
            ):
                if isinstance(value, str):
                    corrected[key] = self.correct_column_names([value])[0]
                elif isinstance(value, list):
                    corrected[key] = self.correct_column_names(value)
        return corrected

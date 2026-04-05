import logging
from typing import List, Dict, Any, Tuple
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

class ParamCorrector:
    """
    The 'Hybrid Safety Layer' for Parameter Extraction.
    LLM proposes names (e.g., 'price') → System corrects them (e.g., 'price_usd')
    against the actual dataset schema.
    """
    def __init__(self, available_columns: List[str]):
        self.available_columns = available_columns

    def correct_column_names(self, proposed_columns: List[str], threshold: int = 70) -> List[str]:
        """
        Maps fuzzy column names to actual dataset columns.
        Ex: ['price'] -> ['price_usd']
        """
        corrected = []
        for col in proposed_columns:
            # Check for exact match first
            if col in self.available_columns:
                corrected.append(col)
                continue
            
            # Fuzzy match
            match = process.extractOne(col, self.available_columns, scorer=fuzz.WRatio)
            if match and match[1] >= threshold:
                logger.info(f"Fuzzy matched '{col}' to '{match[0]}' (Score: {match[1]})")
                corrected.append(match[0])
            else:
                logger.warning(f"Could not find a match for column: {col}")
                # We keep the original and let the tool's Pydantic validation catch the error later
                corrected.append(col)
        return corrected

    def apply_to_params(self, params: Dict[str, Any], tool_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Iterates through tool parameters and applies fuzzy correction 
        to any field that represents a column or list of columns.
        """
        corrected_params = params.copy()
        for key, value in params.items():
            # Heuristic: If key contains 'column' or 'variable' and value is str/list
            if 'column' in key.lower() or 'variable' in key.lower() or 'var' in key.lower():
                if isinstance(value, str):
                    corrected_params[key] = self.correct_column_names([value])[0]
                elif isinstance(value, list):
                    corrected_params[key] = self.correct_column_names(value)
        
        return corrected_params

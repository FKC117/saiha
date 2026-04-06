import pandas as pd
from typing import Any, Dict, List

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType

class DropColumnTool(BaseAnalysisTool):
    """
    A tool to drop one or more columns from the dataset.
    """

    @property
    def name(self) -> str:
        return "drop_column"

    @property
    def description(self) -> str:
        return "Remove unwanted columns from the dataset."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="columns_to_drop",
            parameter_type=ParameterType.MULTISELECT,
            label="Columns to Drop",
            description="Select one or more columns to remove.",
            required=True,
            column_source="all"
        ))
        params.add_parameter(ToolParameter(
            name="save_as_new_dataset",
            parameter_type=ParameterType.CHECKBOX,
            label="Save as New Dataset",
            description="Create a new dataset with the changes instead of modifying the current one.",
            required=False,
            default_value=False
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        import logging
        logger = logging.getLogger(__name__)
        try:
            cols = kwargs.get("columns_to_drop", [])
            if isinstance(cols, str):
                cols = [cols]
            
            if not cols:
                return {"status": "error", "summary": "No columns selected to drop."}

            df = self.load_dataset()
            
            # Identify which exist
            # 3. Validate column existence
            # `cols` is the list of columns requested to be dropped.
            # `cols_to_remove` will be the subset of `cols` that actually exist in the DataFrame.
            # `missing` will be the subset of `cols` that do NOT exist in the DataFrame.
            cols_to_remove = [col for col in cols if col in df.columns]
            missing = [col for col in cols if col not in df.columns]
            
            if not cols_to_remove: # If none of the requested columns exist in the DataFrame
                logger.error(f"DropColumnTool: None of the columns {cols} were found in dataset {self.dataset.id}.")
                return {"status": "error", "summary": f"None of the selected columns were found in the dataset. Missing: {', '.join(missing)}"}

            if missing: # If some, but not all, requested columns are missing
                logger.warning(f"DropColumnTool: Some columns not found: {missing}")
                # `cols_to_remove` already contains only the existing ones, so no further filtering needed here.

            # Perform Drop
            logger.error(f"DEBUG: Before drop columns: {list(df.columns)}")
            df.drop(columns=cols_to_remove, inplace=True)
            logger.error(f"DEBUG: After drop columns: {list(df.columns)}")
            
            summary = f"Successfully dropped {len(cols_to_remove)} columns: {', '.join(cols_to_remove)}. (v2)"
            if missing:
                summary += f" (Note: {len(missing)} columns were not found/already dropped)."

            # Check for Save As New
            save_as_new = kwargs.get("save_as_new_dataset", False)
            if isinstance(save_as_new, str):
                save_as_new = save_as_new.lower() == 'true'
            
            if save_as_new:
                 from ...dataset_utils import save_dataframe_as_dataset
                 suffix = "Dropped"
                 new_dataset = save_dataframe_as_dataset(df, self.dataset, suffix)
                 summary += f" Saved as new dataset: **{new_dataset.name}**"
                 
                 return {
                    "status": "ok",
                    "summary": summary,
                    "data": {"new_dataset_id": str(new_dataset.id)},
                    "sections": [
                        {
                            "type": "text",
                            "title": "Operation Successful",
                            "icon": "bi bi-check-circle-fill",
                            "content": summary
                        }
                    ],
                    "artifacts": []
                }
            else:
                # Save
                self.save_dataset(df)

                return {
                    "status": "ok",
                    "summary": summary,
                    "data": {},
                    "sections": [
                        {
                            "type": "text",
                            "title": "Operation Successful",
                            "icon": "bi bi-check-circle-fill",
                            "content": summary
                        }
                    ],
                    "artifacts": []
                }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An error occurred while dropping columns: {str(e)}"}

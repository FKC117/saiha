import pandas as pd
from typing import Any, Dict

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType

class RenameColumnTool(BaseAnalysisTool):
    """
    A tool to rename a specific column in the dataset.
    """

    @property
    def name(self) -> str:
        return "rename_column"

    @property
    def description(self) -> str:
        return "Rename an existing column to a new name."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="column_to_rename",
            parameter_type=ParameterType.COLUMN_SELECT,
            label="Column to Rename",
            description="Select the column you want to rename.",
            required=True,
            column_source="all"
        ))
        params.add_parameter(ToolParameter(
            name="new_name",
            parameter_type=ParameterType.TEXT,
            label="New Name",
            description="Enter the new name for the column.",
            required=True,
            help_text="Must be unique (cannot exist already)."
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
        try:
            old_name = kwargs.get("column_to_rename")
            new_name = kwargs.get("new_name")

            if not old_name or not new_name:
                return {"status": "error", "summary": "Both 'Column to Rename' and 'New Name' are required."}

            # Validations
            if old_name == new_name:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"RenameColumnTool: Old and new names are identical ('{old_name}').")
                return {"status": "error", "summary": "New name is identical to the old name. No changes made."}

            df = self.load_dataset()
            
            if old_name not in df.columns:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"RenameColumnTool: Column '{old_name}' not found in dataset {self.dataset.id}.")
                return {"status": "error", "summary": f"Column '{old_name}' not found in dataset."}
            
            if new_name in df.columns:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"RenameColumnTool: New name '{new_name}' already exists in dataset {self.dataset.id}.")
                return {"status": "error", "summary": f"Column '{new_name}' already exists. Please choose a unique name."}

            # Perform Rename
            df.rename(columns={old_name: new_name}, inplace=True)
            
            # Check for Save As New
            save_as_new = kwargs.get("save_as_new_dataset", False)
            if isinstance(save_as_new, str):
                save_as_new = save_as_new.lower() == 'true'
            
            message = ""
            if save_as_new:
                 from ...dataset_utils import save_dataframe_as_dataset
                 # suffix = f"Renamed {old_name} to {new_name}" # Might be too long
                 suffix = "Renamed"
                 new_dataset = save_dataframe_as_dataset(df, self.dataset, suffix)
                 
                 message = f"Successfully renamed column '{old_name}' to '{new_name}'.\nSaved as new dataset: **{new_dataset.name}**"
                 
                 return {
                    "status": "ok",
                    "summary": message,
                    "data": {"new_dataset_id": str(new_dataset.id)},
                    "sections": [
                        {
                            "type": "text",
                            "title": "Operation Successful",
                            "icon": "bi bi-check-circle-fill",
                            "content": message
                        }
                    ],
                    "artifacts": []
                }
            else:
                # Save (Overwrite)
                self.save_dataset(df)

                message = f"Successfully renamed column '{old_name}' to '{new_name}' in the current dataset."

                return {
                    "status": "ok",
                    "summary": message,
                    "data": {},
                    "sections": [
                        {
                            "type": "text",
                            "title": "Operation Successful",
                            "icon": "bi bi-check-circle-fill",
                            "content": message
                        }
                    ],
                    "artifacts": []
                }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An error occurred while renaming: {str(e)}"}

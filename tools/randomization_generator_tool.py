from typing import Any, Dict, List
import random
import pandas as pd
from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType

class RandomizationGeneratorTool(BaseAnalysisTool):
    """
    A tool to generate randomization schedules for experimental studies.
    """

    @property
    def name(self) -> str:
        return "randomization_generator"

    @property
    def description(self) -> str:
        return "Generate a randomized assignment list for participants (Simple or Block)."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        
        params.add_parameter(ToolParameter(
            name="total_participants", parameter_type=ParameterType.NUMBER,
            label="Total Participants",
            description="The total number of subjects to randomize.",
            required=True, default_value=100,
            validation_rules={"min": 1, "step": 1}
        ))
        
        params.add_parameter(ToolParameter(
            name="group_names", parameter_type=ParameterType.TEXT,
            label="Group Names",
            description="Comma-separated list of groups (e.g., 'Treatment, Control' or 'A, B, C').",
            required=True, default_value="Treatment, Control"
        ))
        
        params.add_parameter(ToolParameter(
            name="randomization_type", parameter_type=ParameterType.SELECT,
            label="Randomization Method",
            description="Choose the method of randomization.",
            required=True, default_value="simple",
            options=[
                {"value": "simple", "label": "Simple Randomization (Coin Flip)"},
                {"value": "block", "label": "Block Randomization (Balanced Groups)"},
            ]
        ))
        
        params.add_parameter(ToolParameter(
            name="block_size", parameter_type=ParameterType.SELECT,
            label="Block Size (for Block Randomization)",
            description="Size of each block. Must be a multiple of the number of groups.",
            required=False, default_value="4",
            options=[
                {"value": "2", "label": "2"},
                {"value": "4", "label": "4"},
                {"value": "6", "label": "6"},
                {"value": "8", "label": "8"},
            ],
            help_text="Only used if Block Randomization is selected."
        ))

        params.add_parameter(ToolParameter(
            name="seed", parameter_type=ParameterType.NUMBER,
            label="Random Seed (Optional)",
            description="Set a seed for reproducibility.",
            required=False
        ))

        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            params = kwargs
            total_n = int(params.get("total_participants") or 100)
            group_input = params.get("group_names", "Treatment, Control")
            method = params.get("randomization_type", "simple")
            block_size_str = params.get("block_size", "4")
            seed_val = params.get("seed")

            # Parse groups
            groups = [g.strip() for g in group_input.split(',') if g.strip()]
            if not groups:
                groups = ["Group A", "Group B"]
            n_groups = len(groups)

            # Set seed
            if seed_val:
                try:
                    random.seed(int(seed_val))
                except ValueError:
                    pass

            assignments = []
            
            if method == "block":
                try:
                    block_size = int(block_size_str)
                except ValueError:
                    block_size = 4
                
                # Validation: Block size must be divisible by number of groups
                if block_size % n_groups != 0:
                    return {"status": "error", "summary": f"Block size ({block_size}) must be a multiple of the number of groups ({n_groups})."}
                
                if block_size < n_groups:
                     return {"status": "error", "summary": f"Block size ({block_size}) cannot be smaller than the number of groups ({n_groups})."}

                # Generate blocks
                num_full_blocks = total_n // block_size
                remainder = total_n % block_size
                
                subjects_per_group_per_block = block_size // n_groups
                base_block = []
                for g in groups:
                    base_block.extend([g] * subjects_per_group_per_block)
                
                for _ in range(num_full_blocks):
                    current_block = base_block.copy()
                    random.shuffle(current_block)
                    assignments.extend(current_block)
                
                # Handle remainder (simple radomization for last few? or partial block? 
                # Standard practice usually implies total_n should fit blocks, but we'll fill remainder randomly from groups to reach total_n)
                if remainder > 0:
                    remainder_pool = []
                    # Try to keep remainder balanced if possible, but it's small
                    while len(remainder_pool) < remainder:
                         remainder_pool.append(random.choice(groups))
                    assignments.extend(remainder_pool)
                    
            else: # Simple Randomization
                for _ in range(total_n):
                    assignments.append(random.choice(groups))

            # Create DataFrame
            df = pd.DataFrame({
                'Subject ID': range(1, total_n + 1),
                'Group': assignments
            })
            
            # Summary Stats
            counts = df['Group'].value_counts()
            
            sections = []
            
            # 1. Summary Table
            summary_data = [[g, counts.get(g, 0), f"{(counts.get(g, 0)/total_n)*100:.1f}%"] for g in groups]
            sections.append({
                'type': 'table', 'title': 'Allocation Summary',
                'headers': ['Group', 'Count', 'Percentage'],
                'data': summary_data
            })

            # 2. Preview Table (First 10 rows)
            file_preview_data = df.head(10).values.tolist()
            sections.append({
                'type': 'table', 'title': 'Schedule Preview (First 10)',
                'headers': ['Subject ID', 'Group Assignment'],
                'data': file_preview_data
            })
            
            # Note: We aren't saving a CSV file to disk here as per 'requires_dataset=False' usually implying no input dataset.
            # But the user might want to download this. 
            # ideally we'd save this as an artifact or return a download link. 
            # For now, we return the data which the frontend can display.
            
            summary_text = f"Generated valid randomization schedule for {total_n} participants across {n_groups} groups using {method.title().replace('_', ' ')} Randomization."
            if method == 'block':
                summary_text += f" (Block Size: {block_size})"

            return {
                "status": "ok",
                "summary": summary_text,
                "sections": sections,
                "artifacts": [], # Could add CSV export here in future
                "meta": {"tool_name": self.name, "parameters": params}
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"Randomization error: {str(e)}"}

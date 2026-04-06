"""
Two-Way ANOVA Tool
Performs a Two-Way Analysis of Variance on a numeric dependent variable and two categorical factors.
"""
from typing import Dict, Any, List
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from django.core.files.storage import default_storage
import warnings

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from quantalytics.ai_agents.tools.plot_utils import PlotUtils


class TwoWayAnovaTool(BaseAnalysisTool):
    """
    A tool to perform a Two-Way ANOVA to examine the influence of two different
    categorical independent variables on one continuous dependent variable.
    """

    @property
    def name(self) -> str:
        return "two_way_anova"

    @property
    def description(self) -> str:
        return "Performs a Two-Way ANOVA to test the main effects and interaction of two categorical factors."

    def get_parameters_schema(self) -> ToolParameterSet:
        """Defines the parameters for the Two-Way ANOVA tool."""
        params = ToolParameterSet(tool_name="two_way_anova")
        params.add_parameter(
            ToolParameter(
                name="dependent_variable",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Dependent Variable (Numeric)",
                description="The continuous variable you are measuring.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="factor_a",
                parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                label="Factor A (Categorical)",
                description="The first independent categorical variable.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="factor_b",
                parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                label="Factor B (Categorical)",
                description="The second independent categorical variable.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="include_interaction",
                parameter_type=ParameterType.CHECKBOX,
                label="Include Interaction Effect",
                description="Include the interaction term (Factor A * Factor B) in the model.",
                required=False,
                default_value=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="perform_posthoc",
                parameter_type=ParameterType.CHECKBOX,
                label="Perform Post-Hoc Tests (Tukey's HSD)",
                description="If any main or interaction effects are significant, perform pairwise comparisons.",
                required=False,
                default_value=False
            )
        )
        params.add_parameter(
            ToolParameter(
                name="alpha",
                parameter_type=ParameterType.SELECT,
                label="Significance Level (α)",
                description="The threshold for statistical significance.",
                required=True,
                default_value="0.05",
                options=[
                    {"value": "0.01", "label": "0.01"},
                    {"value": "0.05", "label": "0.05"},
                    {"value": "0.10", "label": "0.10"}
                ]
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Executes the Two-Way ANOVA and returns a modern Result Envelope."""
        try:
            parameters = kwargs
            dependent_var = parameters.get("dependent_variable")
            factor_a = parameters.get("factor_a")
            factor_b = parameters.get("factor_b")
            include_interaction = str(parameters.get("include_interaction", "true")).lower() in ('true', 'on', '1')
            perform_posthoc = str(parameters.get("perform_posthoc", "false")).lower() in ('true', 'on', '1')
            alpha = float(parameters.get("alpha", 0.05))
            
            warnings_found = []

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            # Clean column names for the formula
            clean_dep, clean_a, clean_b = self.clean_column_names([dependent_var, factor_a, factor_b])
            df.rename(columns={dependent_var: clean_dep, factor_a: clean_a, factor_b: clean_b}, inplace=True)

            # Build the formula string
            formula = f"Q('{clean_dep}') ~ C(Q('{clean_a}')) + C(Q('{clean_b}'))"
            if include_interaction:
                formula += f" + C(Q('{clean_a}')):C(Q('{clean_b}'))"

            # Perform ANOVA, catching potential warnings about data structure
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model = ols(formula, data=df).fit()
                for warning_message in w:
                    # Check the category name as a string to avoid NameError for undefined warning types
                    category_name = warning_message.category.__name__
                    if category_name in ['UserWarning', 'ValueWarning', 'RuntimeWarning']:
                        warnings_found.append(str(warning_message.message))

            anova_table = sm.stats.anova_lm(model, typ=2)
            anova_table.reset_index(inplace=True)
            anova_table.rename(columns={'index': 'Source', 'PR(>F)': 'p_value'}, inplace=True)
            
            sections: List[Dict[str, Any]] = []
            artifacts: List[Dict[str, Any]] = []

            # Add main ANOVA table to sections
            sections.append({
                'type': 'table', 'title': 'Two-Way ANOVA Results',
                'headers': anova_table.columns.tolist(),
                'data': anova_table.round(4).to_numpy().tolist()
            })

            # Perform post-hoc test if any effect is significant
            is_any_significant = any(anova_table['p_value'] < alpha)
            if is_any_significant and perform_posthoc:
                try:
                    # Combine factors to test all interactions
                    df['interaction_group'] = df[clean_a].astype(str) + " & " + df[clean_b].astype(str)
                    tukey = pairwise_tukeyhsd(endog=df[clean_dep], groups=df['interaction_group'], alpha=alpha)
                    post_hoc_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                    
                    sections.append({
                        'type': 'table', 'title': "Tukey's HSD Post-Hoc Test (for all group combinations)",
                        'headers': post_hoc_df.columns.tolist(),
                        'data': post_hoc_df.to_numpy().tolist(),
                        "footer": "Shows pairwise comparisons between combined factor groups."
                    })

                    # Generate heatmap visualization from the tukey results
                    heatmap_df = post_hoc_df.pivot(index='group1', columns='group2', values='p-adj')
                    fig_heatmap, ax_heatmap = plt.subplots(figsize=(max(10, len(heatmap_df.columns)), max(8, len(heatmap_df.index))))
                    sns.heatmap(heatmap_df, annot=True, cmap='coolwarm_r', fmt=".4f", ax=ax_heatmap, cbar_kws={'label': 'Adjusted P-value'})
                    ax_heatmap.set_title("Post-Hoc P-Value Heatmap (Tukey's HSD)")
                    plt.tight_layout()
                    artifacts.append({
                        "type": "plot", "id": "twoway_anova_posthoc_heatmap", "title": "Post-Hoc Heatmap",
                        "content": PlotUtils.fig_to_base64(fig_heatmap)
                    })
                except Exception as posthoc_ex:
                    sections.append({"type": "text", "title": "Post-Hoc Test Failed", "content": str(posthoc_ex)})

            # Generate Visualizations
            with PlotUtils.setup_plotting():
                # Interaction Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.pointplot(data=df, x=clean_a, y=clean_dep, hue=clean_b, ax=ax, dodge=True)
                ax.set_title(f'Interaction Plot of {factor_a} and {factor_b} on {dependent_var}')
                ax.set_xlabel(factor_a)
                ax.set_ylabel(f'Mean of {dependent_var}')
                plt.xticks(rotation=45, ha='right')
                artifacts.append({
                    "type": "plot",
                    "id": "interaction_plot",
                    "title": "Interaction Plot",
                    "content": PlotUtils.fig_to_base64(fig)
                })

                # Conditionally generate Violin Plot to avoid performance issues with high cardinality.
                max_x_categories = 20
                max_hue_categories = 10
                num_x_categories = df[clean_a].nunique()
                num_hue_categories = df[clean_b].nunique()

                if num_x_categories <= max_x_categories and num_hue_categories <= max_hue_categories:
                    fig_violin, ax_violin = plt.subplots(figsize=(12, 7))
                    # The 'split' parameter only works when the 'hue' variable has exactly two levels.
                    use_split = num_hue_categories == 2
                    sns.violinplot(data=df, x=clean_a, y=clean_dep, hue=clean_b, ax=ax_violin, split=use_split, inner="quartile")
                    ax_violin.set_title(f'Distribution of {dependent_var} by {factor_a} and {factor_b}')
                    ax_violin.set_xlabel(factor_a)
                    ax_violin.set_ylabel(dependent_var)
                    plt.xticks(rotation=45, ha='right')
                    artifacts.append({
                        "type": "plot",
                        "id": "violin_plot",
                        "title": "Violin Plot",
                        "content": PlotUtils.fig_to_base64(fig_violin)
                    })
                else:
                    # Add a note to warnings if the plot is skipped.
                    warnings_found.append(f"Violin plot was skipped because one or both factors have too many unique categories (Factor A: {num_x_categories}, Factor B: {num_hue_categories}), which would be slow and unreadable.")

            # Construct the Result Envelope
            summary = f"Two-Way ANOVA completed for '{dependent_var}' with factors '{factor_a}' and '{factor_b}'."
            # Note: Complex warning handling has been removed for simplicity and stability.
            
            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": parameters}
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> str:
        """
        Provides a formal interpretation of the Two-Way ANOVA results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            params = results.get('meta', {}).get('parameters', {})
            alpha = float(params.get('alpha', 0.05))
            dep_var = params.get('dependent_variable', 'the dependent variable')
            factor_a = params.get('factor_a', 'Factor A')
            factor_b = params.get('factor_b', 'Factor B')

            anova_section = next((s for s in results.get('sections', []) if s.get('title') == 'Two-Way ANOVA Results'), None)
            if not anova_section:
                return "ANOVA results table not found for interpretation."

            headers = anova_section.get('headers', [])
            p_value_idx = headers.index('p_value')
            source_idx = headers.index('Source')

            p_values = {row[source_idx]: row[p_value_idx] for row in anova_section.get('data', []) if 'C(Q(' in row[source_idx]}

            findings = []
            interaction_term = f"C(Q('{self.clean_column_names([factor_a])[0]}')):C(Q('{self.clean_column_names([factor_b])[0]}'))"
            factor_a_term = f"C(Q('{self.clean_column_names([factor_a])[0]}'))"
            factor_b_term = f"C(Q('{self.clean_column_names([factor_b])[0]}'))"

            # Interpret Interaction Effect
            if interaction_term in p_values:
                p_interaction = p_values[interaction_term]
                if p_interaction < alpha:
                    findings.append(f"There is a statistically significant interaction effect between '{factor_a}' and '{factor_b}' (p={p_interaction:.4f}). This means the effect of one factor on '{dep_var}' depends on the level of the other factor.")
                else:
                    findings.append(f"There is no significant interaction effect between the factors (p={p_interaction:.4f}).")

            # Interpret Main Effect of Factor A
            if factor_a_term in p_values:
                p_a = p_values[factor_a_term]
                findings.append(f"The main effect for '{factor_a}' is {'statistically significant' if p_a < alpha else 'not statistically significant'} (p={p_a:.4f}).")

            # Interpret Main Effect of Factor B
            if factor_b_term in p_values:
                p_b = p_values[factor_b_term]
                findings.append(f"The main effect for '{factor_b}' is {'statistically significant' if p_b < alpha else 'not statistically significant'} (p={p_b:.4f}).")

            # Check for post-hoc results
            tukey_results = next((s for s in results.get('sections', []) if "Tukey's HSD" in s.get('title', '')), None)
            if tukey_results:
                significant_pairs = []
                try:
                    headers = tukey_results.get('headers', [])
                    reject_idx = headers.index('reject')
                    group1_idx = headers.index('group1')
                    group2_idx = headers.index('group2')
                    for row in tukey_results.get('data', []):
                        if row[reject_idx] == True:
                            significant_pairs.append(f"'{row[group1_idx]}' and '{row[group2_idx]}'")
                except (ValueError, IndexError):
                    pass # If columns aren't found, we just won't list pairs.

                if significant_pairs:
                    findings.append(f"The post-hoc test (Tukey's HSD) shows significant differences between the following group pairings: {', '.join(significant_pairs)}.")

            return " ".join(findings)

        except (ValueError, TypeError, IndexError, KeyError) as e:
            return f"Could not automatically interpret the Two-Way ANOVA results due to an error: {e}"
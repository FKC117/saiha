"""
Propensity Score Matching (PSM) Tool
Reduces selection bias in observational studies by matching treated units with similar control units based on propensity scores.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class PropensityScoreMatchingTool(BaseAnalysisTool):
    """Tool for Propensity Score Matching (PSM)."""

    @property
    def name(self) -> str:
        return "propensity_score_matching"

    @property
    def description(self) -> str:
        return "Estimate causal effects by reducing selection bias via Propensity Score Matching."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="propensity_score_matching")
        params.add_parameter(
            ToolParameter(
                name="outcome_variable",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Outcome Variable",
                description="The dependent variable to analyze.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="treatment_variable",
                parameter_type=ParameterType.COLUMN_SELECT,
                label="Treatment Variable (Binary)",
                description="Binary variable indicating Treatment (1) vs Control (0).",
                required=True,
                column_source="numeric,categorical"
            )
        )
        params.add_parameter(
            ToolParameter(
                name="covariates",
                parameter_type=ParameterType.MULTISELECT,
                label="Covariates (Confounders)",
                description="Variables to match on (that predict treatment assignment).",
                required=True,
                column_source="numeric,categorical"
            )
        )
        params.add_parameter(
            ToolParameter(
                name="encoding_method",
                parameter_type=ParameterType.SELECT,
                label="Categorical Encoding",
                description="How to transform text variables for the model. One-Hot is usually safer for this model.",
                required=True,
                default_value="one_hot",
                options=[
                    {"value": "one_hot", "label": "One-Hot Encoding (Dummies)"},
                    {"value": "label", "label": "Label Encoding (Ordinal)"}
                ]
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters
            parameters = kwargs
            outcome_col = parameters.get("outcome_variable")
            treatment_col = parameters.get("treatment_variable")
            covariates = parameters.get("covariates", [])
            
            if isinstance(covariates, str):
                covariates = [covariates]

            if not all([outcome_col, treatment_col, covariates]):
                return {"status": "error", "summary": "Missing required variables."}

            if len(covariates) < 1:
                return {"status": "error", "summary": "Select at least one covariate for matching."}

            columns_needed = [outcome_col, treatment_col] + covariates
            df = self.load_dataset(columns=columns_needed)
            df_clean = df.dropna()

            if df_clean.empty:
                return {"status": "error", "summary": "No data remaining after removing missing values."}

            # 2. Calculate Propensity Scores (Logistic Regression)
            # Identify categorical columns
            cat_cols = [c for c in covariates if df_clean[c].dtype in ['object', 'category', 'bool']]
            encoding_method = parameters.get("encoding_method", "one_hot")

            if encoding_method == "label":
                from sklearn.preprocessing import LabelEncoder
                X_encoded = df_clean[covariates].copy()
                for col in cat_cols:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            else:
                # Default One-Hot
                X_encoded = pd.get_dummies(df_clean[covariates], columns=cat_cols, drop_first=True)
            
            y = df_clean[treatment_col]
            X_encoded = X_encoded.astype(float)
            
            # Simple check if treatment is binary 0/1
            if not set(y.unique()).issubset({0, 1}):
                 # If user passed text (Yes/No), we encode it.
                 if y.dtype == object or str(y.dtype) == 'category' or y.dtype == bool:
                     # Map to 0/1 (treating first alphabetical as 0, second as 1)
                     y = pd.factorize(y, sort=True)[0]
                 else:
                     # Coerce to numeric then factorize if still not 0/1
                     y = pd.factorize(y, sort=True)[0]
            
            log_reg = LogisticRegression(solver='liblinear', random_state=42)
            log_reg.fit(X_encoded, y)
            
            ps_scores = log_reg.predict_proba(X_encoded)[:, 1]
            df_clean['propensity_score'] = ps_scores
            
            # 3. Matching (Nearest Neighbor)
            # Ensure we split by the numeric 0/1 representation we just created if needed, 
            # OR make sure we use the same index logic
            
            # Re-assign y to df just in case
            df_clean['treatment_numeric'] = y
            
            treated = df_clean[df_clean['treatment_numeric'] == 1]
            control = df_clean[df_clean['treatment_numeric'] == 0]
            
            if treated.empty or control.empty:
                return {"status": "error", "summary": "Treatment or Control group is empty."}
            
            # Fit NN on Control group
            control_ps = control[['propensity_score']].values
            treated_ps = treated[['propensity_score']].values
            
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control_ps)
            distances, indices = nbrs.kneighbors(treated_ps)
            
            # Create Matched DataFrame
            # indices is (n_treated, 1) array of indices in 'control' dataframe
            matched_control_indices = control.iloc[indices.flatten()].index
            matched_control = control.loc[matched_control_indices]
            
            # Combine Treated + Matched Control
            matched_df = pd.concat([treated, matched_control])
            
            # 4. Estimation (ATT)
            # Compare means of Outcome in matched sample
            mean_treated = treated[outcome_col].mean()
            mean_control_matched = matched_control[outcome_col].mean()
            att = mean_treated - mean_control_matched
            
            # T-test for significance
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(treated[outcome_col], matched_control[outcome_col], equal_var=False)
            
            # 5. Covariate Balance Check (Standardized Mean Differences)
            smd_data = []
            for col in covariates:
                if col in X_encoded.columns:
                    # Pre-match
                    t_pre = treated[col].mean()
                    c_pre = control[col].mean()
                    pool_std_pre = np.sqrt((treated[col].var() + control[col].var())/2)
                    smd_pre = (t_pre - c_pre) / pool_std_pre if pool_std_pre > 0 else 0
                    
                    # Post-match
                    t_post = treated[col].mean() 
                    c_post = matched_control[col].mean()
                    pool_std_post = np.sqrt((treated[col].var() + matched_control[col].var())/2)
                    smd_post = (t_post - c_post) / pool_std_post if pool_std_post > 0 else 0
                    
                    smd_data.append({
                        'Covariate': col,
                        'SMD Pre': smd_pre,
                        'SMD Post': smd_post
                    })
                else:
                    # For categorical variables that were one-hot encoded
                    matching_dummies = [c for c in X_encoded.columns if c.startswith(f"{col}_")]
                    for d_col in matching_dummies:
                        t_pre = treated[d_col].mean()
                        c_pre = control[d_col].mean()
                        pool_std_pre = np.sqrt((treated[d_col].var() + control[d_col].var())/2)
                        smd_pre = (t_pre - c_pre) / pool_std_pre if pool_std_pre > 0 else 0
                        
                        t_post = treated[d_col].mean()
                        c_post = matched_control[d_col].mean()
                        pool_std_post = np.sqrt((treated[d_col].var() + matched_control[d_col].var())/2)
                        smd_post = (t_post - c_post) / pool_std_post if pool_std_post > 0 else 0
                        
                        smd_data.append({
                            'Covariate': d_col,
                            'SMD Pre': smd_pre,
                            'SMD Post': smd_post
                        })
            
            smd_df = pd.DataFrame(smd_data)

            # 6. Artifacts
            artifacts = []
            sections = []
            
            summary_text = f"Propensity Score Matching performed.\nMatched {len(treated)} treated units with {len(matched_control)} control units.\n"
            summary_text += f"Average Treatment Effect on Treated (ATT): {att:.4f} (p={p_val:.4f}).\n"
            
            if p_val < 0.05:
                summary_text += "Result is Statistically Significant."
            else:
                summary_text += "Result is Not Significant."
            
            # Balance Table
            if not smd_df.empty:
                sections.append({
                    'type': 'table',
                    'title': 'Covariate Balance (SMD)',
                    'headers': ['Covariate', 'SMD Before Match', 'SMD After Match'],
                    'data': [[r['Covariate'], f"{r['SMD Pre']:.3f}", f"{r['SMD Post']:.3f}"] for i, r in smd_df.iterrows()]
                })
                
                # Love Plot
                with PlotUtils.setup_plotting():
                    fig, ax = plt.subplots(figsize=(8, max(4, len(covariates)*0.5)))
                    y_pos = np.arange(len(smd_df))
                    ax.plot(smd_df['SMD Pre'], y_pos, 'o', label='Unmatched', color='red', alpha=0.6)
                    ax.plot(smd_df['SMD Post'], y_pos, 'o', label='Matched', color='blue', alpha=0.8)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(smd_df['Covariate'])
                    ax.axvline(0, color='gray', linestyle='--')
                    ax.axvline(0.1, color='gray', linestyle=':', label='Threshold (0.1)')
                    ax.axvline(-0.1, color='gray', linestyle=':')
                    ax.set_xlabel('Standardized Mean Difference (SMD)')
                    ax.set_title('Covariate Balance (Love Plot)')
                    ax.legend()
                    plt.tight_layout()
                    
                    artifacts.append({
                        "type": "plot",
                        "id": "love_plot",
                        "title": "Covariate Balance (Love Plot)",
                        "content": PlotUtils.fig_to_base64(fig)
                    })
                    plt.close(fig)
            
            # Propensity Score Distribution
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.kdeplot(treated['propensity_score'], shade=True, color='blue', label='Treated', ax=ax)
                sns.kdeplot(control['propensity_score'], shade=True, color='red', label='Control (Unmatched)', ax=ax)
                sns.kdeplot(matched_control['propensity_score'], shade=True, color='green', linestyle='--', label='Control (Matched)', ax=ax)
                ax.set_title("Propensity Score Distribution")
                ax.set_xlabel("Propensity Score")
                ax.legend()
                plt.tight_layout()
                artifacts.append({
                    "type": "plot",
                    "id": "ps_distribution",
                    "title": "Propensity Scores",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)

            return {
                "status": "ok",
                "summary": summary_text,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                }
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get('status') != 'ok':
            return None
        return results.get('summary', "PSM Analysis Completed.")
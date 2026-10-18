import pandas as pd
import numpy as np
try:
    from agent_pipeline.tool_access import tool
except ImportError:
    from langchain_core.tools import tool

@tool
def variance_filter_tool(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Filter numeric features by variance threshold."""
    variances = df.var(numeric_only=True)
    keep_cols = variances[variances > threshold].index
    return df[keep_cols]

@tool
def correlation_filter_tool(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Filter collinear features above correlation threshold."""
    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)
import pandas as pd
try:
    from agent_pipeline.tool_access import tool
except ImportError:
    from langchain_core.tools import tool

@tool
def check_duplicates_tool(df: pd.DataFrame) -> dict:
    """Check duplicate rows in DataFrame."""
    return {"duplicate_count": int(df.duplicated().sum())}

@tool
def sample_data_tool(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Sample first n rows from DataFrame."""
    return df.head(n)

@tool
def describe_data_tool(df: pd.DataFrame) -> dict:
    """Describe DataFrame summary metrics."""
    return {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns)
    }
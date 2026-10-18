import pandas as pd
try:
    from agent_pipeline.tool_access import tool
except ImportError:
    from langchain_core.tools import tool

@tool
def binarize_labels_tool(series: pd.Series, pos_label: str) -> pd.Series:
    """Binarize target labels against positive class."""
    return (series == pos_label).astype(int)
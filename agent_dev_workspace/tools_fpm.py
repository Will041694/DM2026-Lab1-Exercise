import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
try:
    from agent_pipeline.tool_access import tool
except ImportError:
    from langchain_core.tools import tool

@tool
def top_k_fae_tool(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Extract top K frequent itemsets using FP-Growth algorithm."""
    freq_items = fpgrowth(df.astype(bool), min_support=0.1, use_colnames=True)
    if freq_items.empty:
        return freq_items
    return freq_items.nlargest(k, 'support')

@tool
def max_fpgrowth_tool(df: pd.DataFrame, min_support: float = 0.2) -> pd.DataFrame:
    """Extract maximal frequent itemsets using FP-Growth algorithm."""
    return fpgrowth(df.astype(bool), min_support=min_support, use_colnames=True)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
try:
    from agent_pipeline.tool_access import tool
except ImportError:
    from langchain_core.tools import tool

@tool
def build_dtm_tool(df: pd.DataFrame, text_col: str, min_df: int = 1, ngram_range: tuple = (1, 1)) -> pd.DataFrame:
    """Build Document Term Matrix from text column."""
    if isinstance(ngram_range, list):
        ngram_range = tuple(ngram_range)
    vec = CountVectorizer(min_df=min_df, ngram_range=ngram_range)
    dtm_mat = vec.fit_transform(df[text_col].fillna(""))
    return pd.DataFrame(dtm_mat.toarray(), columns=vec.get_feature_names_out())

@tool
def term_frequency_tool(dtm_df: pd.DataFrame) -> pd.Series:
    """Calculate total term frequencies from DTM."""
    return dtm_df.sum(axis=0).sort_values(ascending=False)

@tool
def dtm_heatmap_tool(dtm_df: pd.DataFrame, top_n: int = 10) -> dict:
    """Get top terms for DTM heatmap visualization."""
    top_cols = dtm_df.sum(axis=0).nlargest(top_n).index
    return {"status": "success", "top_terms": list(top_cols)}
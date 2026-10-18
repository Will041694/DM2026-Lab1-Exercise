import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
try:
    from agent_pipeline.tool_access import tool
except ImportError:
    from langchain_core.tools import tool

@tool
def pca_tool(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Apply PCA dimensionality reduction."""
    pca = PCA(n_components=n_components)
    res = pca.fit_transform(df.select_dtypes(include=['number']))
    return pd.DataFrame(res, columns=[f"PC{i+1}" for i in range(n_components)])

@tool
def tsne_tool(df: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0) -> pd.DataFrame:
    """Apply t-SNE dimensionality reduction."""
    num_df = df.select_dtypes(include=['number'])
    actual_perplexity = min(perplexity, max(1.0, float(len(num_df) - 1)))
    tsne = TSNE(n_components=n_components, perplexity=actual_perplexity, random_state=42)
    res = tsne.fit_transform(num_df)
    return pd.DataFrame(res, columns=[f"TSNE{i+1}" for i in range(n_components)])

@tool
def umap_tool(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Apply UMAP dimensionality reduction."""
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    res = reducer.fit_transform(df.select_dtypes(include=['number']))
    return pd.DataFrame(res, columns=[f"UMAP{i+1}" for i in range(n_components)])
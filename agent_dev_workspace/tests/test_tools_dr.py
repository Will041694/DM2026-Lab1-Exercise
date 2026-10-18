import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE = _ROOT / "agent_dev_workspace"
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
if str(_WORKSPACE) not in sys.path: sys.path.insert(0, str(_WORKSPACE))

import pandas as pd
from tools_dr import pca_tool, tsne_tool, umap_tool

def test_tools_dr():
    df = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0, 5.0], "feat2": [5.0, 4.0, 3.0, 2.0, 1.0]})
    pca_fn = getattr(pca_tool, "func", pca_tool)
    tsne_fn = getattr(tsne_tool, "func", tsne_tool)
    umap_fn = getattr(umap_tool, "func", umap_tool)

    assert pca_fn(df, n_components=2).shape == (5, 2)
    assert tsne_fn(df, n_components=2, perplexity=2.0).shape == (5, 2)
    assert umap_fn(df, n_components=2).shape == (5, 2)
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE = _ROOT / "agent_dev_workspace"
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
if str(_WORKSPACE) not in sys.path: sys.path.insert(0, str(_WORKSPACE))

import pandas as pd
from tools_dtm import build_dtm_tool, term_frequency_tool, dtm_heatmap_tool

def test_tools_dtm():
    df = pd.DataFrame({"text": ["cat sat mat", "cat cat dog"]})
    build_fn = getattr(build_dtm_tool, "func", build_dtm_tool)
    tf_fn = getattr(term_frequency_tool, "func", term_frequency_tool)
    heatmap_fn = getattr(dtm_heatmap_tool, "func", dtm_heatmap_tool)

    dtm = build_fn(df, text_col="text")
    assert "cat" in dtm.columns
    tf = tf_fn(dtm)
    assert tf["cat"] == 3
    heatmap = heatmap_fn(dtm, top_n=2)
    assert heatmap["status"] == "success"
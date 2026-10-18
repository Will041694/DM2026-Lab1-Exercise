import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE = _ROOT / "agent_dev_workspace"
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
if str(_WORKSPACE) not in sys.path: sys.path.insert(0, str(_WORKSPACE))

import pandas as pd
from tools_filtering import variance_filter_tool, correlation_filter_tool

def test_tools_filtering():
    df = pd.DataFrame({"const": [1, 1, 1], "var": [1, 2, 3], "var_copy": [1, 2, 3]})
    var_fn = getattr(variance_filter_tool, "func", variance_filter_tool)
    corr_fn = getattr(correlation_filter_tool, "func", correlation_filter_tool)

    var_res = var_fn(df, threshold=0.0)
    assert "const" not in var_res.columns and "var" in var_res.columns
    corr_res = corr_fn(df, threshold=0.9)
    assert len(corr_res.columns) < len(df.columns)
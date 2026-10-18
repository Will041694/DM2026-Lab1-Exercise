import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE = _ROOT / "agent_dev_workspace"
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
if str(_WORKSPACE) not in sys.path: sys.path.insert(0, str(_WORKSPACE))

import pandas as pd
from tools_data import check_duplicates_tool, sample_data_tool, describe_data_tool

def test_tools_data():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    check_fn = getattr(check_duplicates_tool, "func", check_duplicates_tool)
    sample_fn = getattr(sample_data_tool, "func", sample_data_tool)
    desc_fn = getattr(describe_data_tool, "func", describe_data_tool)

    assert check_fn(df)["duplicate_count"] == 1
    assert len(sample_fn(df, n=2)) == 2
    desc = desc_fn(df)
    assert desc["num_rows"] == 3 and desc["num_cols"] == 2
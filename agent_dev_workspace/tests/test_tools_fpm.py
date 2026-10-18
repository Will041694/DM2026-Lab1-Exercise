import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE = _ROOT / "agent_dev_workspace"
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
if str(_WORKSPACE) not in sys.path: sys.path.insert(0, str(_WORKSPACE))

import pandas as pd
from tools_fpm import top_k_fae_tool, max_fpgrowth_tool

def test_tools_fpm():
    df = pd.DataFrame({"item1": [True, True, False, True], "item2": [True, True, True, False]})
    max_fn = getattr(max_fpgrowth_tool, "func", max_fpgrowth_tool)
    topk_fn = getattr(top_k_fae_tool, "func", top_k_fae_tool)

    assert not max_fn(df, min_support=0.2).empty
    assert len(topk_fn(df, k=1)) <= 1
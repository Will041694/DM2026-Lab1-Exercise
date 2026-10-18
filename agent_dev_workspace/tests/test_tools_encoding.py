import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE = _ROOT / "agent_dev_workspace"
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
if str(_WORKSPACE) not in sys.path: sys.path.insert(0, str(_WORKSPACE))

import pandas as pd
from tools_encoding import binarize_labels_tool

def test_binarize_labels():
    s = pd.Series(["cat", "dog", "cat"])
    bin_fn = getattr(binarize_labels_tool, "func", binarize_labels_tool)

    assert list(bin_fn(s, pos_label="cat")) == [1, 0, 1]
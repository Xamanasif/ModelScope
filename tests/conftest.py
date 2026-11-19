"""Point every path at a throw-away directory BEFORE app modules are imported,
so tests never touch data/raw, artifacts/ or the real registry."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="modelscope-tests-"))
os.environ["MODELSCOPE_DATA_DIR"] = str(_TMP / "data")
os.environ["MODELSCOPE_ARTIFACT_DIR"] = str(_TMP / "artifacts")
os.environ["MODELSCOPE_DB"] = str(_TMP / "artifacts" / "test.db")
os.environ["MODELSCOPE_POLICY"] = str(_TMP / "policy.json")
(_TMP / "data").mkdir()
(_TMP / "artifacts").mkdir()

from tests.fixtures import small_policy  # noqa: E402

(_TMP / "policy.json").write_text(json.dumps(small_policy()), encoding="utf-8")
TEST_TMP = _TMP

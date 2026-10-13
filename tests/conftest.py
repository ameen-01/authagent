import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("AUTHAGENT_NO_DENSE", "1")  # keep tests fast and offline

import pytest  # noqa: E402

from authagent import common  # noqa: E402


@pytest.fixture(scope="session")
def patients():
    if not common.PATIENT_FILE.exists():
        from authagent.generator import write_all

        write_all()
    return common.load_patients()


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    import authagent.evaluate as ev
    import authagent.runner as rn

    monkeypatch.setattr(rn, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path)
    return tmp_path

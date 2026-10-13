"""Shared paths, constants and small helpers."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("AUTHAGENT_DATA_DIR", ROOT / "data"))
POLICY_DIR = DATA_DIR / "policies"
PATIENT_FILE = DATA_DIR / "patients" / "patients.jsonl"
PAIRS_DIR = DATA_DIR / "verifier_pairs"
CACHE_DIR = Path(os.environ.get("AUTHAGENT_CACHE_DIR", ROOT / "cache"))
RESULTS_DIR = Path(os.environ.get("AUTHAGENT_RESULTS_DIR", ROOT / "results"))
MODELS_DIR = Path(os.environ.get("AUTHAGENT_MODELS_DIR", ROOT / "models"))

STATUSES = ("met", "unmet", "missing")
DECISIONS = ("approve", "pend", "deny")
VERIFIER_LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")

# Fixed "today" for the synthetic world, so date rules are reproducible.
AS_OF = date(2026, 10, 1)


def decide(statuses: Iterable[str]) -> str:
    """Case decision from criterion statuses.

    Any unmet criterion -> deny (likely-denial report, do not submit).
    Otherwise any missing criterion -> pend (ask the clinic for the missing items).
    Otherwise -> approve (draft the request).
    """
    statuses = list(statuses)
    if any(s == "unmet" for s in statuses):
        return "deny"
    if any(s == "missing" for s in statuses):
        return "pend"
    return "approve"


def load_policy(policy_id: str) -> dict:
    return json.loads((POLICY_DIR / f"{policy_id}.json").read_text(encoding="utf-8"))


def list_policies() -> list[str]:
    return sorted(p.stem for p in POLICY_DIR.glob("*.json"))


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_patients(policy_ids: Iterable[str] | None = None) -> list[dict]:
    rows = read_jsonl(PATIENT_FILE)
    if policy_ids:
        wanted = set(policy_ids)
        rows = [r for r in rows if r["chart"]["policy_id"] in wanted]
    return rows

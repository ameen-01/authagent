"""Run a configuration over the patients and save one JSON line per case (resumable)."""

from __future__ import annotations

import json
import time
import traceback

from .agent import run_agent
from .baselines import run_fixed_pipeline, run_single_prompt
from .common import RESULTS_DIR, load_policy, read_jsonl
from .verifier import Cascade, LLMJudge, load_tau, make_local_verifier

CONFIGS = {
    "single_prompt": {"runner": "single", "verifier": None},
    "fixed_pipeline": {"runner": "pipeline", "verifier": "llm"},
    "agent": {"runner": "agent", "verifier": "llm"},
    "agent_cascade": {"runner": "agent", "verifier": "cascade"},
    # ablations
    "agent_no_rules": {"runner": "agent", "verifier": "llm", "use_rules": False},
    "agent_local_only": {"runner": "agent", "verifier": "local"},
}


def build_verifier(kind: str | None, llm, local_kind: str | None = None, tau: float | None = None):
    if kind is None:
        return None
    if kind == "llm":
        return LLMJudge(llm)
    local = make_local_verifier(local_kind)
    if kind == "local":
        return local
    return Cascade(local, LLMJudge(llm), tau if tau is not None else load_tau())


def _meter_diff(after: dict, before: dict) -> dict:
    out = {}
    for k, v in after.items():
        if isinstance(v, (int, float)):
            out[k] = round(v - before.get(k, 0), 3)
        elif isinstance(v, dict):
            out[k] = {kk: vv - before.get(k, {}).get(kk, 0) for kk, vv in v.items() if vv - before.get(k, {}).get(kk, 0)}
    return out


def run_config(config: str, llm, records: list[dict], retrieval_mode: str = "hybrid",
               local_kind: str | None = None, tau: float | None = None, tag: str | None = None,
               resume: bool = True, verbose: bool = True) -> list[dict]:
    spec = CONFIGS[config]
    name = tag or config
    out_path = RESULTS_DIR / "runs" / f"{name}.jsonl"
    trace_dir = RESULTS_DIR / "traces" / name
    trace_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    if resume and out_path.exists():
        done = {r["patient_id"]: r for r in read_jsonl(out_path) if not r.get("error")}
    verifier = build_verifier(spec["verifier"], llm, local_kind, tau)
    rows = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records, 1):
            chart, gold = rec["chart"], rec["gold"]
            pid = chart["patient_id"]
            if pid in done:
                row = done[pid]
            else:
                policy = load_policy(chart["policy_id"])
                before = llm.meter.as_dict()
                t0 = time.time()
                try:
                    if spec["runner"] == "single":
                        res = run_single_prompt(chart, policy, llm)
                    elif spec["runner"] == "pipeline":
                        res = run_fixed_pipeline(chart, policy, llm, verifier, retrieval_mode=retrieval_mode)
                    else:
                        res = run_agent(chart, policy, llm, verifier, retrieval_mode=retrieval_mode,
                                        use_rules=spec.get("use_rules", True))
                    error = None
                except Exception as exc:  # keep going; the row records the failure
                    res, error = None, f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                usage = _meter_diff(llm.meter.as_dict(), before)
                row = {"patient_id": pid, "policy_id": chart["policy_id"], "config": name, "model": llm.name,
                       "retrieval": retrieval_mode, "verifier": getattr(verifier, "name", None),
                       "gold_decision": gold["decision"],
                       "gold": {cid: {k: g[k] for k in ("status", "variant", "hard", "evidence_ids")}
                                for cid, g in gold["criteria"].items()},
                       "wall_s": round(time.time() - t0, 3), **usage}
                if error:
                    row["error"] = error
                else:
                    row.update({"statuses": res["statuses"], "decision": res["decision"],
                                "evidence": {cid: d.get("evidence_ids", []) for cid, d in res["decisions"].items()},
                                "steps": res["steps"], "parse_failures": res["parse_failures"],
                                "stop_reason": res["stop_reason"], "verifier_calls": res["verifier_calls"],
                                "escalations": res["escalations"]})
                    (trace_dir / f"{pid}.json").write_text(json.dumps(
                        {"patient_id": pid, "config": name, "decision": res["decision"], "gold_decision": gold["decision"],
                         "statuses": res["statuses"], "gold": row["gold"], "draft": res["draft"], "trace": res["trace"]},
                        indent=1, ensure_ascii=False), encoding="utf-8")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            if verbose:
                mark = "ERR" if row.get("error") else ("ok " if row.get("decision") == row["gold_decision"] else "XX ")
                print(f"[{name}] {i:>3}/{len(records)} {pid} {mark} pred={row.get('decision')} "
                      f"gold={row['gold_decision']} llm_calls={row.get('llm_calls', 0)}", flush=True)
    return rows

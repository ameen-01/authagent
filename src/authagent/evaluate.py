"""Metrics, confidence intervals and significance tests for the comparison tables."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict

import numpy as np

from .calibrate import metrics as cls_metrics
from .calibrate import risk_coverage
from .common import PAIRS_DIR, RESULTS_DIR, VERIFIER_LABELS, load_policy, read_jsonl
from .index import Retriever, chart_to_docs, get_encoder
from .redact import redact_chart

KINDS = {}


def _kind(policy_id: str, cid: str) -> str:
    if (policy_id, cid) not in KINDS:
        for c in load_policy(policy_id)["criteria"]:
            KINDS[(policy_id, c["id"])] = c["kind"]
    return KINDS[(policy_id, cid)]


# ------------------------------------------------------------------ case / criterion metrics
def case_metrics(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if not r.get("error")]
    n = len(ok_rows)
    if n == 0:
        return {"n": 0, "errors": len(rows)}
    correct = [r["decision"] == r["gold_decision"] for r in ok_rows]
    not_approvable = [r for r in ok_rows if r["gold_decision"] != "approve"]
    approvable = [r for r in ok_rows if r["gold_decision"] == "approve"]
    crit = [(r, cid, g) for r in ok_rows for cid, g in r["gold"].items()]
    crit_ok = [r["statuses"][cid] == g["status"] for r, cid, g in crit]
    hard = [ok for (r, cid, g), ok in zip(crit, crit_ok) if g["hard"]]
    missing = [r["statuses"][cid] == "missing" for r, cid, g in crit if g["status"] == "missing"]
    by_kind = defaultdict(list)
    for (r, cid, g), ok in zip(crit, crit_ok):
        by_kind[_kind(r["policy_id"], cid)].append(ok)
    cites, cited_ok = 0, 0
    for r, cid, g in crit:
        pred_ev = set(r.get("evidence", {}).get(cid, []))
        if r["statuses"][cid] == g["status"] and g["status"] in ("met", "unmet") and g["evidence_ids"] and pred_ev:
            cites += 1
            cited_ok += bool(pred_ev & set(g["evidence_ids"]))
    vcalls = sum(r.get("verifier_calls", 0) for r in ok_rows)
    tokens = [r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in ok_rows]
    return {
        "n": n,
        "errors": len(rows) - n,
        "decision_accuracy": float(np.mean(correct)),
        "false_approval_rate": (sum(r["decision"] == "approve" for r in not_approvable) / len(not_approvable)
                                if not_approvable else None),
        "auto_approval_recall": (sum(r["decision"] == "approve" for r in approvable) / len(approvable)
                                 if approvable else None),
        "criterion_accuracy": float(np.mean(crit_ok)),
        "hard_criterion_accuracy": float(np.mean(hard)) if hard else None,
        "missing_detection_recall": float(np.mean(missing)) if missing else None,
        "citation_precision": cited_ok / cites if cites else None,
        "by_kind": {k: float(np.mean(v)) for k, v in sorted(by_kind.items())},
        "llm_calls_per_case": float(np.mean([r.get("llm_calls", 0) for r in ok_rows])),
        "tokens_per_case": float(np.mean(tokens)),
        "llm_latency_s_per_case": float(np.mean([r.get("llm_latency_s", 0) for r in ok_rows])),
        "verifier_calls": vcalls,
        "escalation_share": (sum(r.get("escalations", 0) for r in ok_rows) / vcalls) if vcalls else None,
        "steps_per_case": float(np.mean([r.get("steps", 0) for r in ok_rows])),
        "parse_failures": int(sum(r.get("parse_failures", 0) for r in ok_rows)),
        "unfinished": int(sum(r.get("stop_reason") not in ("finished", "single_call", "pipeline") for r in ok_rows)),
    }


def bootstrap_ci(rows: list[dict], key: str, iters: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    ok_rows = [r for r in rows if not r.get("error")]
    if not ok_rows:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        sample = [ok_rows[rng.randrange(len(ok_rows))] for _ in ok_rows]
        v = case_metrics(sample).get(key)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Exact McNemar test on per-case decision correctness (paired by patient)."""
    a = {r["patient_id"]: r.get("decision") == r["gold_decision"] for r in rows_a if not r.get("error")}
    b = {r["patient_id"]: r.get("decision") == r["gold_decision"] for r in rows_b if not r.get("error")}
    common = sorted(set(a) & set(b))
    only_a = sum(a[p] and not b[p] for p in common)
    only_b = sum(b[p] and not a[p] for p in common)
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        k = min(only_a, only_b)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n_pairs": len(common), "a_only_correct": only_a, "b_only_correct": only_b, "p_value": p}


def failures(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("error"):
            out.append({"patient_id": r["patient_id"], "error": r["error"]})
            continue
        for cid, g in r["gold"].items():
            if r["statuses"][cid] != g["status"]:
                out.append({"patient_id": r["patient_id"], "policy_id": r["policy_id"], "criterion": cid,
                            "kind": _kind(r["policy_id"], cid), "gold": g["status"], "variant": g["variant"],
                            "pred": r["statuses"][cid]})
    return out


def _fmt(v, pct=True):
    if v is None:
        return "–"
    return f"{100 * v:.1f}%" if pct else f"{v:.2f}"


def report(configs: list[str], reference: str = "agent") -> dict:
    runs = {}
    for c in configs:
        path = RESULTS_DIR / "runs" / f"{c}.jsonl"
        if path.exists():
            runs[c] = read_jsonl(path)
    out = {"configs": {}, "mcnemar_vs_" + reference: {}}
    lines = ["# AuthAgent results", "",
             "| Config | n | Decision acc. [95% CI] | False approvals [95% CI] | Criterion acc. | Hard criteria | "
             "Missing detected | Citation precision | LLM calls/case | Tokens/case | Checks sent to LLM |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, rows in runs.items():
        m = case_metrics(rows)
        if not m.get("n"):
            continue
        m["decision_accuracy_ci"] = bootstrap_ci(rows, "decision_accuracy")
        m["false_approval_rate_ci"] = bootstrap_ci(rows, "false_approval_rate")
        m["model"] = rows[0].get("model")
        out["configs"][c] = m
        ci = lambda k: f" [{_fmt(m[k][0])}–{_fmt(m[k][1])}]" if m.get(k) else ""  # noqa: E731
        lines.append(f"| {c} | {m['n']} | {_fmt(m['decision_accuracy'])}{ci('decision_accuracy_ci')} | "
                     f"{_fmt(m['false_approval_rate'])}{ci('false_approval_rate_ci')} | {_fmt(m['criterion_accuracy'])} | "
                     f"{_fmt(m['hard_criterion_accuracy'])} | {_fmt(m['missing_detection_recall'])} | "
                     f"{_fmt(m['citation_precision'])} | {m['llm_calls_per_case']:.1f} | {m['tokens_per_case']:.0f} | "
                     f"{_fmt(m['escalation_share'])} |")
    if reference in runs:
        lines += ["", f"## Paired McNemar tests vs `{reference}` (decision correctness)", "",
                  "| Config | pairs | only ref correct | only config correct | p-value |", "|---|---|---|---|---|"]
        for c, rows in runs.items():
            if c == reference:
                continue
            t = mcnemar(runs[reference], rows)
            out["mcnemar_vs_" + reference][c] = t
            lines.append(f"| {c} | {t['n_pairs']} | {t['a_only_correct']} | {t['b_only_correct']} | {t['p_value']:.3f} |")
    lines += ["", "## Accuracy by criterion kind", "", "| Config | " + " | ".join(
        sorted({k for m in out["configs"].values() for k in m["by_kind"]})) + " |"]
    kinds = sorted({k for m in out["configs"].values() for k in m["by_kind"]})
    lines.append("|---|" + "---|" * len(kinds))
    for c, m in out["configs"].items():
        lines.append(f"| {c} | " + " | ".join(_fmt(m["by_kind"].get(k)) for k in kinds) + " |")
    lines += ["", "## Failures (criterion level)", ""]
    for c, rows in runs.items():
        f = failures(rows)
        lines.append(f"**{c}** — {len(f)} wrong criteria")
        for x in f[:25]:
            if "error" in x:
                lines.append(f"- {x['patient_id']}: ERROR {x['error']}")
            else:
                lines.append(f"- {x['patient_id']} {x['policy_id']} {x['criterion']} ({x['kind']}): gold {x['gold']} "
                             f"[{x['variant']}] → predicted {x['pred']}")
        lines.append("")
    out["markdown"] = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "report.md").write_text(out["markdown"], encoding="utf-8")
    (RESULTS_DIR / "report.json").write_text(json.dumps({k: v for k, v in out.items() if k != "markdown"}, indent=1),
                                             encoding="utf-8")
    return out


# ------------------------------------------------------------------ retrieval
def retrieval_eval(records: list[dict], modes: tuple[str, ...] = ("bm25", "dense", "hybrid"),
                   ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    if get_encoder() is None:
        modes = tuple(m for m in modes if m == "bm25")
    out = {}
    for mode in modes:
        hits_at = {k: [] for k in ks}
        rr = []
        for rec in records:
            docs = chart_to_docs(redact_chart(rec["chart"]))
            ret = Retriever(docs, mode=mode)
            policy = load_policy(rec["chart"]["policy_id"])
            for c in policy["criteria"]:
                gold = set(rec["gold"]["criteria"][c["id"]]["evidence_ids"])
                if not gold:
                    continue
                ranked = [h["id"] for h in ret.search(c["text"], k=max(ks + (10,)))]
                for k in ks:
                    hits_at[k].append(bool(gold & set(ranked[:k])))
                rank = next((i + 1 for i, d in enumerate(ranked) if d in gold), None)
                rr.append(1 / rank if rank else 0.0)
        out[mode] = {**{f"recall@{k}": float(np.mean(v)) for k, v in hits_at.items()}, "mrr@10": float(np.mean(rr)),
                     "queries": len(rr)}
    return out


# ------------------------------------------------------------------ verifier on labelled pairs
def verifier_eval(verifier, split: str = "test", limit: int | None = None, seed: int = 0) -> dict:
    rows = read_jsonl(PAIRS_DIR / f"{split}.jsonl")
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    labels = list(VERIFIER_LABELS)
    pairs = [(r["criterion"], r["evidence"]) for r in rows]
    if hasattr(verifier, "predict_batch"):
        preds = verifier.predict_batch(pairs)
    else:
        preds = [verifier.predict(c, e) for c, e in pairs]
    probs = np.array([[p["probs"][lab] for lab in labels] for p in preds])
    y = np.array([labels.index(r["label"]) for r in rows])
    return {"verifier": getattr(verifier, "name", "?"), "split": split, **cls_metrics(probs, y),
            "risk_coverage": risk_coverage(probs, y)}

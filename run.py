"""AuthAgent command line. Run `python run.py -h` for all commands.

  python run.py selftest                          # 30-second check: imports, data, rules, offline agent
  python run.py generate                          # (re)build patients and verifier pairs
  python run.py run --config agent --limit 5      # run one configuration (real LLM from .env)
  python run.py run --config all                  # all main configurations
  python run.py report                            # metrics tables -> results/report.md
  python run.py retrieval                         # recall@k for BM25 / dense / hybrid
  python run.py verifier-eval --verifier nli      # score a verifier on the held-out policy pairs
  python run.py case P-007 --config agent         # run one case and print the trace + output
  python run.py demo                              # Gradio app (shareable link on Colab)
  python run.py api                               # FastAPI server on :8000

Use --llm mock to exercise everything offline with a scripted stand-in (plumbing only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

MAIN_CONFIGS = ["single_prompt", "fixed_pipeline", "agent", "agent_cascade"]


def cmd_selftest(_args) -> int:
    t0 = time.time()
    ok = True

    def check(name, fn):
        nonlocal ok
        try:
            msg = fn()
            print(f"  PASS  {name}" + (f" — {msg}" if msg else ""))
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")

    print("AuthAgent self-test")
    from authagent import common

    def data():
        if not common.PATIENT_FILE.exists():
            from authagent.generator import write_all

            write_all()
        rows = common.load_patients()
        return f"{len(rows)} patients, policies {common.list_policies()}"

    def rules():
        from authagent.rules import evaluate

        crit = {c["id"]: c for c in common.load_policy("POL-GLP1")["criteria"]}
        assert evaluate(crit["C3"], {"value": 7.0, "date": "2026-09-20"})["status"] == "met"
        assert evaluate(crit["C3"], {"value": 6.9, "date": "2026-09-20"})["status"] == "unmet"
        assert evaluate(crit["C3"], {"value": 9.0, "date": "2026-01-02"})["status"] == "missing"
        assert evaluate(crit["C4"], {"drug": "metformin", "start": "2026-01-01", "end": "2026-03-31"})["status"] == "unmet"
        return "thresholds, staleness and durations behave"

    def redaction():
        from authagent.redact import find_leaks, redact_chart

        rows = common.load_patients()
        leaks = sum(bool(find_leaks(json.dumps(redact_chart(r["chart"])), r["chart"]["phi"])) for r in rows)
        assert leaks == 0, f"{leaks} charts leak identifiers"
        return "no identifiers in any redacted chart"

    def retrieval():
        from authagent.index import get_encoder

        return "dense encoder loaded" if get_encoder() is not None else "BM25 only (sentence-transformers not available)"

    def offline_agent():
        from authagent.agent import run_agent
        from authagent.mock_llm import MockLLM
        from authagent.verifier import LexicalVerifier, LLMJudge

        rec = common.load_patients()[0]
        llm = MockLLM()
        res = run_agent(rec["chart"], common.load_policy(rec["chart"]["policy_id"]), llm, LLMJudge(llm),
                        retrieval_mode="bm25")
        assert res["stop_reason"] == "finished", res["stop_reason"]
        _ = LexicalVerifier
        return f"{res['steps']} steps, decision={res['decision']}"

    def llm_config():
        from authagent.llm import LLMClient

        c = LLMClient()
        cached = len(c.cache)
        key = "set" if c.api_key else "not set"
        return f"model={c.model} base_url={c.base_url} api_key={key} cached_responses={cached}"

    check("data", data)
    check("rule engine", rules)
    check("PHI redaction", redaction)
    check("retrieval", retrieval)
    check("agent loop (offline mock)", offline_agent)
    check("LLM config", llm_config)
    print(f"{'OK' if ok else 'FAILED'} in {time.time() - t0:.1f}s")
    return 0 if ok else 1


def cmd_generate(args) -> int:
    from authagent.generator import write_all

    print(write_all(n_per_policy=args.n_per_policy, seed=args.seed))
    return 0


def _records(args):
    from authagent.common import load_patients

    recs = load_patients(args.policy.split(",") if args.policy else None)
    if args.patients:
        wanted = set(args.patients.split(","))
        recs = [r for r in recs if r["chart"]["patient_id"] in wanted]
    if getattr(args, "per_policy", None):
        # Balanced subset: the first N patients of each policy (same subset for every config and person).
        seen: dict[str, int] = {}
        keep = []
        for r in recs:
            pol = r["chart"]["policy_id"]
            if seen.get(pol, 0) < args.per_policy:
                keep.append(r)
                seen[pol] = seen.get(pol, 0) + 1
        recs = keep
    if args.limit:
        recs = recs[: args.limit]
    return recs


def cmd_run(args) -> int:
    from authagent.llm import get_llm
    from authagent.runner import CONFIGS, run_config

    configs = MAIN_CONFIGS if args.config == "all" else args.config.split(",")
    for c in configs:
        if c not in CONFIGS:
            print(f"unknown config {c}; choose from {list(CONFIGS)} or 'all'")
            return 2
    recs = _records(args)
    for c in configs:
        llm = get_llm(args.llm)
        tag = f"{c}{args.tag}" if args.tag else None
        run_config(c, llm, recs, retrieval_mode=args.retrieval, local_kind=args.verifier, tau=args.tau, tag=tag,
                   resume=not args.no_resume)
    return 0


def cmd_report(args) -> int:
    from authagent.common import RESULTS_DIR
    from authagent.evaluate import report

    configs = args.configs.split(",") if args.configs else sorted(p.stem for p in (RESULTS_DIR / "runs").glob("*.jsonl"))
    out = report(configs, reference=args.reference)
    print(out["markdown"])
    return 0


def cmd_retrieval(args) -> int:
    from authagent.evaluate import retrieval_eval

    res = retrieval_eval(_records(args))
    print(json.dumps(res, indent=1))
    return 0


def cmd_verifier_eval(args) -> int:
    from authagent.common import RESULTS_DIR
    from authagent.evaluate import verifier_eval
    from authagent.llm import get_llm
    from authagent.verifier import LLMJudge, make_local_verifier

    v = LLMJudge(get_llm(args.llm)) if args.verifier == "llm" else make_local_verifier(args.verifier)
    res = verifier_eval(v, split=args.split, limit=args.limit)
    rc = res.pop("risk_coverage")
    print(json.dumps(res, indent=1))
    path = RESULTS_DIR / f"verifier_eval_{res['verifier']}_{args.split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**res, "risk_coverage": rc}, indent=1))
    return 0


def cmd_case(args) -> int:
    from authagent.agent import run_agent
    from authagent.baselines import run_fixed_pipeline, run_single_prompt
    from authagent.common import load_patients, load_policy
    from authagent.llm import get_llm
    from authagent.runner import CONFIGS, build_verifier

    rec = next(r for r in load_patients() if r["chart"]["patient_id"] == args.patient_id)
    chart, policy = rec["chart"], load_policy(rec["chart"]["policy_id"])
    llm = get_llm(args.llm)
    spec = CONFIGS[args.config]
    verifier = build_verifier(spec["verifier"], llm, args.verifier, args.tau)
    if spec["runner"] == "single":
        res = run_single_prompt(chart, policy, llm)
    elif spec["runner"] == "pipeline":
        res = run_fixed_pipeline(chart, policy, llm, verifier, retrieval_mode=args.retrieval)
    else:
        res = run_agent(chart, policy, llm, verifier, retrieval_mode=args.retrieval, use_rules=spec.get("use_rules", True))
    for s in res["trace"]:
        if "tool" in s:
            print(f"{s.get('step', '-'):>3} {s['tool']:16s} {json.dumps(s.get('args', s.get('values', {})))[:90]}")
            print(f"    -> {json.dumps(s['observation'])[:160]}")
        else:
            print(f"{s.get('step', '-'):>3} ERROR {s.get('error')}")
    print("\n" + res["draft"])
    print(f"\npredicted={res['decision']} gold={rec['gold']['decision']} | llm: {llm.meter.as_dict()}")
    return 0


def cmd_demo(args) -> int:
    from authagent.app import launch

    launch(share=args.share, llm_name=args.llm)
    return 0


def cmd_api(args) -> int:
    import uvicorn

    uvicorn.run("authagent.api:app", host="0.0.0.0", port=args.port, reload=False)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_opts(p):
        p.add_argument("--llm", default=os.environ.get("AUTHAGENT_LLM", "api"), help="'api' (default) or 'mock'")
        p.add_argument("--retrieval", default="hybrid", choices=["bm25", "dense", "hybrid"])
        p.add_argument("--verifier", default=None, help="local verifier: finetuned | nli | lexical (default: auto)")
        p.add_argument("--tau", type=float, default=None, help="cascade threshold (default: from calibration.json)")

    def select_opts(p):
        p.add_argument("--policy", default=None, help="comma-separated policy ids")
        p.add_argument("--patients", default=None, help="comma-separated patient ids")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--per-policy", type=int, default=None, help="balanced subset: first N patients per policy")

    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    p = sub.add_parser("generate")
    p.add_argument("--n-per-policy", type=int, default=15)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("run")
    p.add_argument("--config", required=True, help="single_prompt, fixed_pipeline, agent, agent_cascade, "
                                                   "agent_no_rules, agent_local_only, or 'all'")
    p.add_argument("--tag", default="", help="suffix for the results file, e.g. _bm25")
    p.add_argument("--no-resume", action="store_true")
    common_opts(p)
    select_opts(p)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("--configs", default=None)
    p.add_argument("--reference", default="agent")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("retrieval")
    select_opts(p)
    p.set_defaults(fn=cmd_retrieval)
    p = sub.add_parser("verifier-eval")
    p.add_argument("--verifier", default="nli", help="finetuned | nli | lexical | llm")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--llm", default=os.environ.get("AUTHAGENT_LLM", "api"))
    p.set_defaults(fn=cmd_verifier_eval)
    p = sub.add_parser("case")
    p.add_argument("patient_id")
    p.add_argument("--config", default="agent")
    common_opts(p)
    p.set_defaults(fn=cmd_case)
    p = sub.add_parser("demo")
    p.add_argument("--share", action="store_true")
    p.add_argument("--llm", default=os.environ.get("AUTHAGENT_LLM", "api"))
    p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("api")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_api)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

"""One entry point used by the demo UI and the API: run a configuration on one chart."""

from __future__ import annotations

from functools import lru_cache

from .agent import run_agent
from .baselines import run_fixed_pipeline, run_single_prompt
from .common import load_patients, load_policy
from .llm import get_llm
from .runner import CONFIGS, build_verifier


@lru_cache(maxsize=4)
def _llm(name: str):
    return get_llm(name)


@lru_cache(maxsize=8)
def _verifier(config: str, llm_name: str):
    return build_verifier(CONFIGS[config]["verifier"], _llm(llm_name))


def patient_ids() -> list[str]:
    return [f"{r['chart']['patient_id']} ({r['chart']['policy_id']})" for r in load_patients()]


def get_record(patient_id: str) -> dict:
    pid = patient_id.split(" ")[0]
    return next(r for r in load_patients() if r["chart"]["patient_id"] == pid)


def authorize(chart: dict, config: str = "agent_cascade", llm_name: str = "api", retrieval: str = "hybrid") -> dict:
    policy = load_policy(chart["policy_id"])
    llm = _llm(llm_name)
    before = llm.meter.as_dict()
    spec = CONFIGS[config]
    verifier = _verifier(config, llm_name)
    if spec["runner"] == "single":
        res = run_single_prompt(chart, policy, llm)
    elif spec["runner"] == "pipeline":
        res = run_fixed_pipeline(chart, policy, llm, verifier, retrieval_mode=retrieval)
    else:
        res = run_agent(chart, policy, llm, verifier, retrieval_mode=retrieval, use_rules=spec.get("use_rules", True))
    after = llm.meter.as_dict()
    res["usage"] = {k: round(after[k] - before[k], 3) for k in after if isinstance(after[k], (int, float))}
    res["policy"] = policy
    return res

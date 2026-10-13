"""FastAPI service: `python run.py api`, then open http://localhost:8000/docs."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .common import list_policies, load_patients
from .runner import CONFIGS
from .service import authorize

app = FastAPI(title="AuthAgent", version="0.1.0")
LLM = os.environ.get("AUTHAGENT_LLM", "api")


class ByPatient(BaseModel):
    patient_id: str
    config: str = "agent_cascade"


class ByChart(BaseModel):
    chart: dict
    config: str = "agent_cascade"


def _public(res: dict) -> dict:
    return {"decision": res["decision"], "statuses": res["statuses"], "decisions": res["decisions"],
            "draft": res["draft"], "steps": res["steps"], "verifier_calls": res["verifier_calls"],
            "escalations": res["escalations"], "usage": res["usage"]}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "policies": list_policies(), "configs": list(CONFIGS)}


@app.get("/patients")
def patients() -> list[dict]:
    return [{"patient_id": r["chart"]["patient_id"], "policy_id": r["chart"]["policy_id"]} for r in load_patients()]


@app.post("/authorize")
def authorize_patient(req: ByPatient) -> dict:
    if req.config not in CONFIGS:
        raise HTTPException(400, f"config must be one of {list(CONFIGS)}")
    rec = next((r for r in load_patients() if r["chart"]["patient_id"] == req.patient_id), None)
    if rec is None:
        raise HTTPException(404, "unknown patient_id")
    return _public(authorize(rec["chart"], req.config, LLM))


@app.post("/authorize-chart")
def authorize_chart(req: ByChart) -> dict:
    if req.chart.get("policy_id") not in list_policies():
        raise HTTPException(400, "chart.policy_id must be a known policy")
    if req.config not in CONFIGS:
        raise HTTPException(400, f"config must be one of {list(CONFIGS)}")
    return _public(authorize(req.chart, req.config, LLM))

"""Evidence verifiers and the cost-aware cascade.

All verifiers answer one question: does this chart text SUPPORT, REFUTE, or give
NOT_ENOUGH_INFO for this criterion? They return a label, class probabilities and a
confidence, so they can be compared and calibrated on the same footing.

  LexicalVerifier   keyword/negation heuristic. Weak baseline; also used in offline tests.
  NLIVerifier       off-the-shelf NLI cross-encoder (no training), the "before" model.
  FineTunedVerifier your DeBERTa-small fine-tuned on the generated pairs, with temperature.
  LLMJudge          the expensive LLM, asked for a label.
  Cascade           local verifier first; escalate to the LLM only below the confidence threshold.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from .common import AS_OF, MODELS_DIR, VERIFIER_LABELS
from .llm import parse_json

LABELS = list(VERIFIER_LABELS)


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    z = [x / temperature for x in logits]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def _result(probs: list[float], source: str) -> dict:
    i = max(range(len(probs)), key=lambda k: probs[k])
    return {"label": LABELS[i], "confidence": round(probs[i], 4),
            "probs": {lab: round(p, 4) for lab, p in zip(LABELS, probs)}, "source": source}


class LexicalVerifier:
    """Overlap + negation cues. Deliberately simple; a floor for the comparison table."""

    name = "lexical"
    NEG = re.compile(r"\b(no|not|denies|negative|ruled out|without|well controlled|resolved|good response|"
                     r"skipping|stopped|gaps|revised|excluded|does not)\b", re.IGNORECASE)

    def predict(self, criterion: str, evidence: str) -> dict:
        c = set(re.findall(r"[a-z]{4,}", criterion.lower()))
        e = set(re.findall(r"[a-z]{4,}", evidence.lower()))
        overlap = len(c & e) / max(1, len(c))
        neg = bool(self.NEG.search(evidence))
        crit_neg = bool(re.search(r"\b(absence|no|without)\b", criterion.lower()))
        if overlap < 0.2:
            probs = [0.15, 0.15, 0.70]
        elif neg != crit_neg:
            probs = [0.25, 0.55, 0.20]
        else:
            probs = [0.55, 0.20, 0.25]
        return _result(probs, self.name)

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[dict]:
        return [self.predict(c, e) for c, e in pairs]


class _HFVerifier:
    """Shared code for transformer cross-encoders (premise=evidence, hypothesis=criterion)."""

    name = "hf"

    def __init__(self, model_dir: str, label_map: dict[str, str], temperature: float = 1.0, max_len: int = 160):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        # Order the model's outputs as SUPPORTS, REFUTES, NOT_ENOUGH_INFO.
        self.order = [next(i for i, lab in id2label.items() if label_map.get(lab.lower(), lab) == target)
                      for target in LABELS]
        self.temperature = temperature
        self.max_len = max_len

    def logits_batch(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[list[float]]:
        out = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            enc = self.tok([e for _, e in chunk], [c for c, _ in chunk], truncation=True,
                           max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits.float().cpu().tolist()
            out.extend([[row[j] for j in self.order] for row in logits])
        return out

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[dict]:
        return [_result(softmax(lg, self.temperature), self.name) for lg in self.logits_batch(pairs)]

    def predict(self, criterion: str, evidence: str) -> dict:
        return self.predict_batch([(criterion, evidence)])[0]


class NLIVerifier(_HFVerifier):
    name = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        super().__init__(model_name, {"entailment": "SUPPORTS", "contradiction": "REFUTES",
                                      "neutral": "NOT_ENOUGH_INFO"})


class FineTunedVerifier(_HFVerifier):
    name = "finetuned"

    def __init__(self, model_dir: str | Path | None = None):
        model_dir = Path(model_dir or MODELS_DIR / "verifier")
        calib = model_dir / "calibration.json"
        t = json.loads(calib.read_text())["temperature"] if calib.exists() else 1.0
        super().__init__(str(model_dir), {}, temperature=t)


JUDGE_SYSTEM = """You are a careful clinical documentation reviewer for prior authorization.
Given ONE criterion and ONE piece of chart text, decide whether the text, on its own:
- SUPPORTS: shows the criterion is satisfied
- REFUTES: shows the criterion is NOT satisfied
- NOT_ENOUGH_INFO: is related but does not settle it (other drug, family member, a plan or order, a different condition, missing values)
Reply with JSON only: {"label": "SUPPORTS" | "REFUTES" | "NOT_ENOUGH_INFO", "rationale": "<one short sentence>"}"""


class LLMJudge:
    name = "llm"

    def __init__(self, llm):
        self.llm = llm

    def predict(self, criterion: str, evidence: str, phi: dict | None = None) -> dict:
        messages = [{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": f"Today's date: {AS_OF.isoformat()}\nCriterion: {criterion}\n"
                                                f"Chart text: {evidence}"}]
        r = self.llm.chat(messages, purpose="judge", phi=phi)
        try:
            label = str(parse_json(r.text).get("label", "")).upper().strip()
        except ValueError:
            label = ""
        if label not in LABELS:
            label = "NOT_ENOUGH_INFO"
        probs = [1.0 if lab == label else 0.0 for lab in LABELS]
        out = _result(probs, self.name)
        out["confidence"] = None  # an LLM label is not a calibrated probability
        return out


class Cascade:
    """Local verifier first; escalate to the LLM judge when confidence < tau."""

    def __init__(self, local, judge: LLMJudge, tau: float):
        self.local, self.judge, self.tau = local, judge, tau
        self.name = f"cascade({local.name}->llm@{tau})"

    def predict(self, criterion: str, evidence: str, phi: dict | None = None) -> dict:
        res = self.local.predict(criterion, evidence)
        if res["confidence"] >= self.tau:
            res["escalated"] = False
            return res
        out = self.judge.predict(criterion, evidence, phi=phi)
        out["escalated"] = True
        out["local"] = res
        return out


def load_tau(default: float = 0.9) -> float:
    calib = MODELS_DIR / "verifier" / "calibration.json"
    if calib.exists():
        return float(json.loads(calib.read_text()).get("tau", default))
    return float(os.environ.get("AUTHAGENT_TAU", default))


def make_local_verifier(kind: str | None = None):
    """'finetuned' if a trained model exists, else 'nli' if transformers is installed, else 'lexical'."""
    kind = kind or os.environ.get("AUTHAGENT_VERIFIER", "auto")
    if kind == "lexical":
        return LexicalVerifier()
    if kind in ("finetuned", "auto") and (MODELS_DIR / "verifier" / "config.json").exists():
        return FineTunedVerifier()
    if kind in ("nli", "auto", "finetuned"):
        try:
            return NLIVerifier()
        except Exception:  # noqa: BLE001
            if kind != "auto":
                raise
    return LexicalVerifier()

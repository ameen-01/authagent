import numpy as np

from authagent.calibrate import choose_tau, ece, fit_temperature, metrics, softmax


def overconfident(n=3000, seed=0):
    """Logits from a model that is right ~80% of the time but far too sure of itself."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    pred = np.where(rng.random(n) < 0.8, y, (y + rng.integers(1, 3, n)) % 3)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), pred] += 6.0
    return logits, y


def test_temperature_scaling_reduces_ece():
    logits, y = overconfident()
    t = fit_temperature(logits, y)
    assert t > 1.0  # overconfident -> soften
    assert ece(softmax(logits, t), y) < ece(softmax(logits), y) / 2


def test_tau_keeps_selective_risk_under_target():
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.integers(0, 3, n)
    conf = rng.uniform(0.34, 1.0, n)
    correct = rng.random(n) < conf  # calibrated by construction
    pred = np.where(correct, y, (y + 1) % 3)
    probs = np.full((n, 3), 0.0)
    probs[np.arange(n), pred] = conf
    probs[np.arange(n), (pred + 1) % 3] = (1 - conf) / 2
    probs[np.arange(n), (pred + 2) % 3] = (1 - conf) / 2
    tau = choose_tau(probs, y, target_risk=0.05)
    m = probs.max(axis=1) >= tau
    assert (probs[m].argmax(axis=1) != y[m]).mean() <= 0.05
    assert 0.5 < tau < 1.0


def test_metrics_shape():
    logits, y = overconfident(300)
    m = metrics(softmax(logits), y)
    assert set(m) == {"n", "accuracy", "macro_f1", "ece"} and 0.7 < m["accuracy"] < 0.9


def test_cascade_always_escalates_not_enough_info():
    from authagent.verifier import Cascade

    class Local:
        name = "stub"

        def __init__(self, label):
            self.label = label

        def predict(self, c, e):
            return {"label": self.label, "confidence": 0.999, "source": "stub"}

    class Judge:
        def predict(self, c, e, phi=None):
            return {"label": "SUPPORTS", "confidence": None, "source": "llm"}

    assert Cascade(Local("NOT_ENOUGH_INFO"), Judge(), 0.95).predict("c", "e")["escalated"] is True
    assert Cascade(Local("SUPPORTS"), Judge(), 0.95).predict("c", "e")["escalated"] is False
    assert Cascade(Local("NOT_ENOUGH_INFO"), Judge(), 0.95, escalate_nei=False).predict("c", "e")["escalated"] is False

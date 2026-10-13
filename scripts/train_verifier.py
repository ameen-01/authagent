"""Fine-tune the evidence verifier (DeBERTa-v3-small by default) on the generated pairs.

Run on a GPU (Colab T4: about 3-5 minutes):
    python scripts/train_verifier.py
    python scripts/train_verifier.py --compare-nli     # also score the off-the-shelf NLI model

Steps: train on RA + GLP-1 pairs -> pick the best epoch on validation (macro-F1) -> fit a
temperature on validation -> choose the cascade threshold tau on validation -> report on the
held-out PCSK9 policy, which the model never saw.
Writes models/verifier/ (weights, tokenizer, calibration.json) and results/verifier_report.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authagent.calibrate import choose_tau, fit_temperature, metrics, reliability, risk_coverage, softmax  # noqa: E402
from authagent.common import MODELS_DIR, PAIRS_DIR, RESULTS_DIR, VERIFIER_LABELS, read_jsonl  # noqa: E402

LABELS = list(VERIFIER_LABELS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="microsoft/deberta-v3-small")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--target-risk", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(MODELS_DIR / "verifier"))
    ap.add_argument("--compare-nli", action="store_true")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} base={args.base}")

    data = {s: read_jsonl(PAIRS_DIR / f"{s}.jsonl") for s in ("train", "val", "test")}
    print({s: len(v) for s, v in data.items()})

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
        ignore_mismatched_sizes=True).to(device)

    def batches(rows, shuffle):
        idx = list(range(len(rows)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), args.batch_size):
            chunk = [rows[j] for j in idx[i:i + args.batch_size]]
            enc = tok([r["evidence"] for r in chunk], [r["criterion"] for r in chunk], truncation=True,
                      max_length=args.max_len, padding=True, return_tensors="pt")
            enc["labels"] = torch.tensor([LABELS.index(r["label"]) for r in chunk])
            yield {k: v.to(device) for k, v in enc.items()}

    @torch.no_grad()
    def logits_of(rows):
        model.eval()
        out = []
        for b in batches(rows, shuffle=False):
            b.pop("labels")
            out.append(model(**b).logits.float().cpu().numpy())
        return np.concatenate(out)

    def y_of(rows):
        return np.array([LABELS.index(r["label"]) for r in rows])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(data["train"]) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best_f1, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for b in batches(data["train"], shuffle=True):
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        val = metrics(softmax(logits_of(data["val"])), y_of(data["val"]))
        print(f"epoch {epoch} loss={total:.2f} val_acc={val['accuracy']:.3f} val_f1={val['macro_f1']:.3f} "
              f"({time.time() - t0:.0f}s)")
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # Calibrate on validation, report on the held-out policy.
    val_logits, test_logits = logits_of(data["val"]), logits_of(data["test"])
    y_val, y_test = y_of(data["val"]), y_of(data["test"])
    temperature = fit_temperature(val_logits, y_val)
    tau = choose_tau(softmax(val_logits, temperature), y_val, target_risk=args.target_risk)
    test_probs = softmax(test_logits, temperature)
    rc = risk_coverage(test_probs, y_test)
    at_tau = next(r for r in rc if abs(r["tau"] - round(tau, 3)) < 1e-6) if any(
        abs(r["tau"] - round(tau, 3)) < 1e-6 for r in rc) else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "calibration.json").write_text(json.dumps(
        {"temperature": temperature, "tau": tau, "target_risk": args.target_risk, "base": args.base}, indent=1))

    report = {
        "base": args.base, "epochs": args.epochs, "train_seconds": round(time.time() - t0, 1),
        "temperature": temperature, "tau": tau, "target_risk": args.target_risk,
        "val_uncalibrated": metrics(softmax(val_logits), y_val),
        "val_calibrated": metrics(softmax(val_logits, temperature), y_val),
        "test_heldout_policy_uncalibrated": metrics(softmax(test_logits), y_test),
        "test_heldout_policy_calibrated": metrics(test_probs, y_test),
        "test_at_tau": at_tau,
        "test_reliability": reliability(test_probs, y_test),
        "test_risk_coverage": rc,
    }

    if args.compare_nli:
        sys.path.insert(0, str(ROOT / "src"))
        from authagent.verifier import NLIVerifier

        nli = NLIVerifier()
        preds = nli.predict_batch([(r["criterion"], r["evidence"]) for r in data["test"]])
        probs = np.array([[p["probs"][lab] for lab in LABELS] for p in preds])
        report["nli_offtheshelf_test"] = metrics(probs, y_test)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "verifier_report.json").write_text(json.dumps(report, indent=1))
    t = report["test_heldout_policy_calibrated"]
    print(f"\nHeld-out policy (PCSK9): acc={t['accuracy']:.3f} macroF1={t['macro_f1']:.3f} ECE={t['ece']:.3f}")
    print(f"temperature={temperature:.2f} tau={tau:.3f} test@tau={at_tau}")
    if "nli_offtheshelf_test" in report:
        n = report["nli_offtheshelf_test"]
        print(f"Off-the-shelf NLI on the same test: acc={n['accuracy']:.3f} macroF1={n['macro_f1']:.3f}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()

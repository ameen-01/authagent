"""Synthetic patient charts with gold labels by construction, plus verifier training pairs.

Each patient is generated from a spec that decides, per criterion, whether it is met, unmet
or missing, and how hard the evidence is to find (structured row vs buried in a note,
values at a threshold, stale labs, near-miss drugs, look-alike sentences). Because the spec
is known, the correct answer for every criterion and every case is known exactly.
"""

from __future__ import annotations

import random
from datetime import timedelta

from .common import AS_OF, PAIRS_DIR, PATIENT_FILE, decide, load_policy, write_jsonl
from .domain import (CITIES, CRITERION_PARAPHRASES, DOMAIN, FIRST_NAMES, GENERIC_DISTRACTORS,
                     LAST_NAMES, PREFIXES, STREETS, SUFFIXES)

DOSES = {
    "methotrexate": "15 mg weekly",
    "metformin": "1000 mg twice daily",
    "atorvastatin": "80 mg daily",
    "rosuvastatin": "20 mg daily",
}
OTHER_MEDS = [("lisinopril", "10 mg daily"), ("vitamin D3", "1000 IU daily"), ("omeprazole", "20 mg daily"),
              ("amlodipine", "5 mg daily"), ("folic acid", "1 mg daily")]
OTHER_DX = [("I10", "Essential hypertension"), ("E55.9", "Vitamin D deficiency, unspecified"),
            ("K21.9", "Gastro-esophageal reflux disease without esophagitis")]

MED_NOTE_TEMPLATES = [
    "{Drug} {dose} was taken from {start} to {end}.",
    "The patient was on {drug} from {start} through {end}.",
    "{Drug} was started on {start} and stopped on {end}.",
]
MED_NOTE_ONGOING = [
    "{Drug} {dose} was started on {start} and is being continued.",
    "Has been taking {drug} since {start}.",
]
LAB_NOTE_NUMERIC = ["{lab} on {date} was {value} {unit}.", "Most recent {lab} ({date}): {value} {unit}."]
LAB_NOTE_CATEGORICAL = ["{lab} drawn on {date} was {value}.", "{lab} result from {date}: {value}."]


def _d(days_ago: int) -> str:
    return (AS_OF - timedelta(days=days_ago)).isoformat()


def compose(rng: random.Random, core: str) -> str:
    """Wrap a core sentence with an optional prefix and suffix to vary surface form."""
    prefix = rng.choice(PREFIXES)
    suffix = rng.choice(SUFFIXES)
    body = core
    if prefix and body and not body.split()[0].isupper():
        body = body[0].lower() + body[1:]
    return f"{prefix}{body}{suffix}"


class _Chart:
    """Mutable builder for one patient's chart."""

    def __init__(self) -> None:
        self.problem_list: list[dict] = []
        self.medications: list[dict] = []
        self.labs: list[dict] = []
        # (sentence, criterion_id or None, carries a date that must precede the note)
        self.note_sentences: list[tuple[str, str | None, bool]] = []

    def add_dx(self, code: str, desc: str) -> str:
        rid = f"DX-{len(self.problem_list) + 1}"
        self.problem_list.append({"id": rid, "code": code, "description": desc})
        return rid

    def add_med(self, name: str, dose: str, start: str, end: str | None) -> str:
        rid = f"MED-{len(self.medications) + 1}"
        self.medications.append({"id": rid, "name": name, "dose": dose, "start": start, "end": end})
        return rid

    def add_lab(self, name: str, value, unit: str | None, on: str) -> str:
        rid = f"LAB-{len(self.labs) + 1}"
        self.labs.append({"id": rid, "name": name, "value": value, "unit": unit, "date": on})
        return rid

    def add_sentence(self, text: str, cid: str | None, dated: bool = False) -> None:
        self.note_sentences.append((text, cid, dated))


def _sample_statuses(rng: random.Random, criteria: list[dict], target: str) -> dict[str, str]:
    kinds = {c["id"]: c["kind"] for c in criteria}
    statuses = {cid: "met" for cid in kinds}
    soft = [cid for cid, k in kinds.items() if k in ("lab", "med_duration", "documentation")]
    if target == "pend":
        for cid in rng.sample(soft, rng.choice([1, 1, 2])):
            statuses[cid] = "missing"
    elif target == "deny":
        weights = {"diagnosis": 0.15, "age": 0.1, "lab": 0.25, "med_duration": 0.25, "documentation": 0.25}
        ids = list(kinds)
        unmet = rng.choices(ids, weights=[weights[kinds[i]] for i in ids])[0]
        statuses[unmet] = "unmet"
        if rng.random() < 0.3:
            others = [cid for cid in soft if cid != unmet]
            statuses[rng.choice(others)] = "missing"
    return statuses


def _diagnosis(rng, ch: _Chart, crit, status, dom):
    cid = crit["id"]
    cores = dom["cores"][cid]
    if rng.random() < 0.3:  # look-alike sentence that settles nothing
        ch.add_sentence(compose(rng, rng.choice(cores["NOT_ENOUGH_INFO"])), None)
    if status == "met":
        if rng.random() < 0.6:
            code, desc = rng.choice(dom["dx_structured"])
            return {"variant": "structured", "hard": False, "evidence_ids": [ch.add_dx(code, desc)]}
        ch.add_sentence(compose(rng, rng.choice(cores["SUPPORTS"])), cid)
        return {"variant": "note_only", "hard": True, "evidence_ids": []}
    code, desc = rng.choice(dom["dx_other"])
    ch.add_dx(code, desc)
    ch.add_sentence(compose(rng, rng.choice(cores["REFUTES"])), cid)
    return {"variant": "refuted_in_note", "hard": False, "evidence_ids": []}


def _age(rng, status):
    if status == "met":
        if rng.random() < 0.15:
            return 18, {"variant": "boundary_18", "hard": True}
        return rng.randint(25, 80), {"variant": "adult", "hard": False}
    age = rng.choice([16, 17])
    return age, {"variant": f"minor_{age}", "hard": age == 17}


def _lab(rng, ch: _Chart, crit, status, dom):
    cid = crit["id"]
    lab = dom["lab"]
    name = lab["name"]
    unit = lab.get("unit")
    recent = lambda: rng.randint(*lab["recent_days"])  # noqa: E731
    stale = lambda: rng.randint(*lab["stale_days"])  # noqa: E731

    def numeric_value(good: bool, boundary: bool):
        thr, step = lab["threshold"], lab["step"]
        if boundary:
            return thr if good else round(thr - step, 1)
        lo, hi = lab["good_range"] if good else lab["bad_range"]
        return round(rng.uniform(lo, hi), 1) if step < 1 else rng.randint(int(lo), int(hi))

    def place(value, days_ago, note_only: bool):
        if note_only:
            tmpl = rng.choice(LAB_NOTE_NUMERIC if lab["type"] == "numeric" else LAB_NOTE_CATEGORICAL)
            ch.add_sentence(tmpl.format(lab=name, date=_d(days_ago), value=value, unit=unit or "").replace(" .", "."), cid, dated=True)
            return []
        return [ch.add_lab(name, value, unit, _d(days_ago))]

    if status == "missing":
        if rng.random() < 0.4:
            good = lab["good"] if lab["type"] == "categorical" else numeric_value(True, False)
            ids = place(good, stale(), note_only=False)
            return {"variant": "stale", "hard": True, "evidence_ids": ids}
        return {"variant": "absent", "hard": False, "evidence_ids": []}

    good = status == "met"
    if lab["type"] == "categorical":
        value, boundary = (lab["good"] if good else lab["bad"]), False
    else:
        boundary = rng.random() < 0.2
        value = numeric_value(good, boundary)
    note_only = rng.random() < 0.3
    days = recent()
    variant = "boundary" if boundary else ("note_only" if note_only else "structured")
    # Sometimes add an older result on the other side of the threshold: only the most recent counts.
    older = None
    if lab["type"] == "numeric" and not note_only and rng.random() < 0.25:
        older = numeric_value(not good, False)
        ch.add_lab(name, older, unit, _d(days + rng.randint(120, 240)))
        variant += "+older_conflicting"
    ids = place(value, days, note_only)
    return {"variant": variant, "hard": boundary or note_only or older is not None, "evidence_ids": ids}


def _med(rng, ch: _Chart, crit, status, dom):
    cid = crit["id"]
    drug = rng.choice(crit["params"]["drugs"])
    dose = DOSES[drug]
    min_days = crit["params"]["min_days"]
    if status == "missing":
        if rng.random() < 0.5:
            nm = dom["med_nearmiss"]
            start = rng.randint(150, 600)
            ch.add_med(nm["name"], nm["dose"], _d(start), None)
            return {"variant": "nearmiss_drug_only", "hard": True, "evidence_ids": []}
        return {"variant": "absent", "hard": False, "evidence_ids": []}

    if status == "met":
        roll = rng.random()
        if roll < 0.15:
            duration, variant, hard = min_days, "boundary", True
        elif roll < 0.25:
            start = rng.randint(200, 900)
            if rng.random() < 0.5:
                rid = ch.add_med(drug, dose, _d(start), None)
                return {"variant": "ongoing", "hard": False, "evidence_ids": [rid]}
            tmpl = rng.choice(MED_NOTE_ONGOING)
            ch.add_sentence(tmpl.format(Drug=drug.capitalize(), drug=drug, dose=dose, start=_d(start)), cid, dated=True)
            return {"variant": "ongoing_note_only", "hard": True, "evidence_ids": []}
        else:
            duration, variant, hard = rng.randint(100, 400), "normal", False
    else:
        if rng.random() < 0.3:
            duration, variant, hard = min_days - 1, "boundary", True
        else:
            duration, variant, hard = rng.randint(30, 75), "short_trial", False

    end_ago = rng.randint(20, 200)
    start, end = _d(end_ago + duration), _d(end_ago)
    if variant != "boundary" and rng.random() < 0.3:
        tmpl = rng.choice(MED_NOTE_TEMPLATES)
        ch.add_sentence(tmpl.format(Drug=drug.capitalize(), drug=drug, dose=dose, start=start, end=end), cid, dated=True)
        return {"variant": variant + "+note_only", "hard": True, "evidence_ids": []}
    return {"variant": variant, "hard": hard, "evidence_ids": [ch.add_med(drug, dose, start, end)]}


def _documentation(rng, ch: _Chart, crit, status, dom):
    cid = crit["id"]
    cores = dom["cores"][cid]
    if status == "met":
        ch.add_sentence(compose(rng, rng.choice(cores["SUPPORTS"])), cid)
        if rng.random() < 0.2:
            ch.add_sentence(compose(rng, rng.choice(cores["NOT_ENOUGH_INFO"])), None)
            return {"variant": "supported+lookalike", "hard": True, "evidence_ids": []}
        return {"variant": "supported", "hard": False, "evidence_ids": []}
    if status == "unmet":
        ch.add_sentence(compose(rng, rng.choice(cores["REFUTES"])), cid)
        return {"variant": "refuted", "hard": False, "evidence_ids": []}
    if rng.random() < 0.5:
        ch.add_sentence(compose(rng, rng.choice(cores["NOT_ENOUGH_INFO"])), None)
        return {"variant": "lookalike_only", "hard": True, "evidence_ids": []}
    return {"variant": "absent", "hard": False, "evidence_ids": []}


def generate_patient(rng: random.Random, policy: dict, pid: str, target: str) -> dict:
    dom = DOMAIN[policy["policy_id"]]
    criteria = policy["criteria"]
    statuses = _sample_statuses(rng, criteria, target)
    ch = _Chart()
    gold: dict[str, dict] = {}
    age = 50

    for crit in criteria:
        cid, kind, status = crit["id"], crit["kind"], statuses[crit["id"]]
        if kind == "diagnosis":
            info = _diagnosis(rng, ch, crit, status, dom)
        elif kind == "age":
            age, info = _age(rng, status)
            info["evidence_ids"] = ["DEMO"]
        elif kind == "lab":
            info = _lab(rng, ch, crit, status, dom)
        elif kind == "med_duration":
            info = _med(rng, ch, crit, status, dom)
        else:
            info = _documentation(rng, ch, crit, status, dom)
        gold[cid] = {"status": status, **info}

    # Unrelated structured rows for realism.
    for code, desc in rng.sample(OTHER_DX, rng.randint(0, 2)):
        ch.add_dx(code, desc)
    for name, dose in rng.sample(OTHER_MEDS, rng.randint(1, 3)):
        ch.add_med(name, dose, _d(rng.randint(100, 900)), None)

    # PHI.
    first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
    sex = rng.choice(["F", "M"])
    dob = (AS_OF - timedelta(days=age * 365 + rng.randint(10, 300))).isoformat()
    phi = {
        "name": f"{first} {last}",
        "first_name": first,
        "last_name": last,
        "dob": dob,
        "mrn": f"MRN-{rng.randint(100000, 999999)}",
        "phone": f"(555) 01{rng.randint(0, 9)}-{rng.randint(1000, 9999)}",
        "address": f"{rng.randint(10, 999)} {rng.choice(STREETS)}, {rng.choice(CITIES)}",
    }

    # Notes: evidence sentences scattered among distractors.
    distractors = GENERIC_DISTRACTORS + dom["domain_distractors"]
    sentences = list(ch.note_sentences)
    rng.shuffle(sentences)
    # Dated evidence (labs, medication dates) only goes into notes written after it: N2 is
    # 2-4 days old and N3 1 day old, while the newest dated evidence is at least 5 days old.
    note_specs = [("N1", "Primary care", rng.randint(60, 120)),
                  ("N2", dom["specialty"], rng.randint(2, 4)),
                  ("N3", "Nursing phone call", 1)]
    buckets: dict[str, list[tuple[str, str | None]]] = {n: [] for n, _, _ in note_specs}
    for text, cid, dated in sentences:
        options = ["N2", "N2", "N3"] if dated else ["N1", "N2", "N2", "N3"]
        buckets[rng.choice(options)].append((text, cid))
    sex_word = "woman" if sex == "F" else "man"
    title = ("Ms." if sex == "F" else "Mr.") + f" {last}"
    notes = []
    for note_id, author, days_ago in note_specs:
        body = buckets[note_id] + [(d, None) for d in rng.sample(distractors, rng.randint(2, 4))]
        rng.shuffle(body)
        if note_id == "N3":
            opener = f"Spoke with {first} by phone at {phi['phone']}."
        else:
            opener = f"{first} {last} ({phi['mrn']}) is a {age}-year-old {sex_word} seen for follow-up."
        closer = f"{title} will follow up as scheduled."
        sents = [opener] + [s for s, _ in body] + [closer]
        notes.append({"note_id": note_id, "date": _d(days_ago), "author": author, "sentences": sents})
        for i, (s, cid) in enumerate(body, start=2):
            if cid:
                gold[cid]["evidence_ids"].append(f"{note_id}.s{i}")

    chart = {
        "patient_id": pid,
        "policy_id": policy["policy_id"],
        "as_of": AS_OF.isoformat(),
        "requested_drug": dom["requested_drug"],
        "prescriber_specialty": dom["specialty"],
        "phi": phi,
        "demographics": {"sex": sex, "dob": dob, "address": phi["address"]},
        "problem_list": ch.problem_list,
        "medications": ch.medications,
        "labs": ch.labs,
        "notes": notes,
    }
    decision = decide(g["status"] for g in gold.values())
    return {"chart": chart, "gold": {"criteria": gold, "decision": decision, "target": target}}


def generate_patients(n_per_policy: int = 15, seed: int = 7,
                      policies: tuple[str, ...] = ("POL-RA", "POL-GLP1", "POL-PCSK9")) -> list[dict]:
    rng = random.Random(seed)
    rows, k = [], 0
    for pol_id in policies:
        policy = load_policy(pol_id)
        n_approve = round(n_per_policy * 0.4)
        n_pend = round(n_per_policy * 0.3)
        targets = ["approve"] * n_approve + ["pend"] * n_pend + ["deny"] * (n_per_policy - n_approve - n_pend)
        rng.shuffle(targets)
        for target in targets:
            k += 1
            rows.append(generate_patient(rng, policy, f"P-{k:03d}", target))
    return rows


# ---------------------------------------------------------------- verifier training pairs

TRAIN_POLICIES = ("POL-RA", "POL-GLP1")
TEST_POLICIES = ("POL-PCSK9",)


def make_pairs(seed: int = 11, per_core: int = 40, val_frac: float = 0.2) -> dict[str, list[dict]]:
    """Criterion-evidence pairs labelled SUPPORTS / REFUTES / NOT_ENOUGH_INFO.

    Train and validation come from the RA and GLP-1 policies, split by core sentence so the
    validation set contains sentences the model never saw. Test is the PCSK9 policy, which
    the model never sees at all, to measure generalization to a new policy.
    """
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    def variants(core: str, crit_texts: list[str]) -> list[tuple[str, str]]:
        combos = [(c, p, s) for c in crit_texts for p in PREFIXES for s in SUFFIXES]
        picked = rng.sample(combos, min(per_core, len(combos)))
        out = []
        for crit_text, p, s in picked:
            body = core if not p or core.split()[0].isupper() else core[0].lower() + core[1:]
            out.append((crit_text, f"{p}{body}{s}"))
        return out

    for (pol_id, cid), crit_texts in CRITERION_PARAPHRASES.items():
        split_base = "test" if pol_id in TEST_POLICIES else "train"
        cores = DOMAIN[pol_id]["cores"][cid]
        for label, core_list in cores.items():
            idx = list(range(len(core_list)))
            rng.shuffle(idx)
            n_val = max(1, round(len(idx) * val_frac)) if split_base == "train" else 0
            val_idx = set(idx[:n_val])
            for i, core in enumerate(core_list):
                split = split_base if i not in val_idx else "val"
                for crit_text, evidence in variants(core, crit_texts):
                    splits[split].append({"policy_id": pol_id, "criterion_id": cid, "criterion": crit_text,
                                          "evidence": evidence, "label": label, "core_id": f"{pol_id}:{cid}:{label}:{i}"})
        # Plain distractors are NOT_ENOUGH_INFO for every criterion.
        distractors = GENERIC_DISTRACTORS + DOMAIN[pol_id]["domain_distractors"]
        for j, d in enumerate(distractors):
            split = split_base if (split_base == "test" or j % 5) else "val"
            for crit_text in crit_texts:
                splits[split].append({"policy_id": pol_id, "criterion_id": cid, "criterion": crit_text,
                                      "evidence": d, "label": "NOT_ENOUGH_INFO", "core_id": f"{pol_id}:{cid}:distractor:{j}"})
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def write_all(n_per_policy: int = 15, seed: int = 7) -> dict:
    patients = generate_patients(n_per_policy=n_per_policy, seed=seed)
    write_jsonl(PATIENT_FILE, patients)
    pairs = make_pairs()
    for split, rows in pairs.items():
        write_jsonl(PAIRS_DIR / f"{split}.jsonl", rows)
    decisions: dict[str, int] = {}
    for p in patients:
        decisions[p["gold"]["decision"]] = decisions.get(p["gold"]["decision"], 0) + 1
    return {"patients": len(patients), "decisions": decisions, **{f"pairs_{k}": len(v) for k, v in pairs.items()}}

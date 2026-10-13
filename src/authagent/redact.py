"""PHI redaction: identifiers never reach the LLM or the logs.

Because the charts are synthetic, we know exactly which fields are identifiers, so redaction
is deterministic. Regex patterns for phone numbers and MRNs act as a safety net, and
`find_leaks` is checked by the LLM client before every call.
"""

from __future__ import annotations

import copy
import re
from datetime import date

PHONE_RE = re.compile(r"\(?\b\d{3}\)?[ .-]?\d{3}[ .-]\d{4}\b")
MRN_RE = re.compile(r"\bMRN[- ]?\d{4,}\b", re.IGNORECASE)


def _age_on(dob: str, as_of: str) -> int:
    b, t = date.fromisoformat(dob), date.fromisoformat(as_of)
    return t.year - b.year - ((t.month, t.day) < (b.month, b.day))


class Redactor:
    def __init__(self, phi: dict):
        self.phi = phi
        pairs = [
            (phi["name"], "[PATIENT]"),
            (f"Ms. {phi['last_name']}", "[PATIENT]"),
            (f"Mr. {phi['last_name']}", "[PATIENT]"),
            (phi["mrn"], "[MRN]"),
            (phi["phone"], "[PHONE]"),
            (phi["address"], "[ADDRESS]"),
            (phi["dob"], "[DOB]"),
            (phi["first_name"], "[PATIENT]"),
            (phi["last_name"], "[PATIENT]"),
        ]
        # Longest first so "Aisha Miller" is replaced before "Aisha".
        pairs.sort(key=lambda p: -len(p[0]))
        self.patterns = [(re.compile(r"(?<!\w)" + re.escape(v) + r"(?!\w)", re.IGNORECASE), tok) for v, tok in pairs]

    def text(self, s: str) -> str:
        for pat, tok in self.patterns:
            s = pat.sub(tok, s)
        s = MRN_RE.sub("[MRN]", s)
        s = PHONE_RE.sub("[PHONE]", s)
        return s


def redact_chart(chart: dict) -> dict:
    """Return a copy of the chart with identifiers removed and DOB replaced by age."""
    r = Redactor(chart["phi"])
    out = copy.deepcopy(chart)
    out.pop("phi")
    demo = out.pop("demographics")
    out["demographics"] = {"sex": demo["sex"], "age": _age_on(demo["dob"], chart["as_of"])}
    for note in out["notes"]:
        note["sentences"] = [r.text(s) for s in note["sentences"]]
    return out


def find_leaks(text: str, phi: dict) -> list[str]:
    """Identifier values that appear in `text` (used as a guard before any LLM call)."""
    leaks = []
    for key in ("name", "first_name", "last_name", "mrn", "phone", "address", "dob"):
        v = phi.get(key)
        if v and re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", text, re.IGNORECASE):
            leaks.append(key)
    return leaks

"""Chart chunking and hybrid retrieval (BM25 + dense embeddings, fused with RRF).

Every chunk has a stable id (DEMO, DX-1, MED-2, LAB-1, N2.s4) so that every decision can cite
the exact chart line it rests on. BM25 is implemented here (about 40 lines) so the
retrieval maths is fully visible; dense retrieval uses sentence-transformers when installed
and is skipped otherwise.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter

STOPWORDS = set("""a an the of and or to in on for with by at from as is was were be been are this that it its
patient has have had per not no""".split())
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    toks = []
    for raw in TOKEN_RE.findall(text.lower()):
        # Keep "quantiferon-tb" and also its parts, so "TB" or "QuantiFERON" queries match it.
        parts = [raw] + ([p for p in re.split(r"[.\-]", raw) if len(p) > 1] if re.search(r"[.\-]", raw) else [])
        for t in parts:
            if t in STOPWORDS:
                continue
            if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
                t = t[:-1]  # crude plural folding
            toks.append(t)
    return toks


def chart_to_docs(chart: dict) -> list[dict]:
    """Turn a (redacted) chart into retrievable chunks with stable ids."""
    docs = []
    demo = chart["demographics"]
    age = demo.get("age")
    docs.append({"id": "DEMO", "source": "structured",
                 "text": f"Demographics: age {age} years, sex {demo['sex']}."})
    for dx in chart["problem_list"]:
        docs.append({"id": dx["id"], "source": "structured",
                     "text": f"Problem list diagnosis: {dx['description']} (ICD-10 {dx['code']})."})
    for m in chart["medications"]:
        end = f"stopped {m['end']}" if m["end"] else "ongoing"
        docs.append({"id": m["id"], "source": "structured",
                     "text": f"Medication: {m['name']} {m['dose']}, started {m['start']}, {end}."})
    for lab in chart["labs"]:
        unit = f" {lab['unit']}" if lab.get("unit") else ""
        docs.append({"id": lab["id"], "source": "structured",
                     "text": f"Lab result: {lab['name']} = {lab['value']}{unit} on {lab['date']}."})
    for note in chart["notes"]:
        for i, s in enumerate(note["sentences"], start=1):
            docs.append({"id": f"{note['note_id']}.s{i}", "source": "note",
                         "note": f"{note['author']} note, {note['date']}", "text": s})
    return docs


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(t) for t in texts]
        self.lens = [len(d) for d in self.docs]
        self.avgdl = sum(self.lens) / max(1, len(self.lens))
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.lens):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


_ENCODER = None
_ENCODER_FAILED = False
DENSE_MODEL = os.environ.get("AUTHAGENT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def get_encoder():
    """Load the sentence-transformers encoder once; return None if unavailable."""
    global _ENCODER, _ENCODER_FAILED
    if _ENCODER is None and not _ENCODER_FAILED and os.environ.get("AUTHAGENT_NO_DENSE") != "1":
        try:
            from sentence_transformers import SentenceTransformer
            _ENCODER = SentenceTransformer(DENSE_MODEL, device="cpu")
        except Exception:  # noqa: BLE001 - optional dependency or no network
            _ENCODER_FAILED = True
    return _ENCODER


class Retriever:
    """mode: 'bm25', 'dense' or 'hybrid' (reciprocal rank fusion of both)."""

    def __init__(self, docs: list[dict], mode: str = "hybrid", rrf_k: int = 60):
        self.docs = docs
        self.by_id = {d["id"]: d for d in docs}
        self.bm25 = BM25([d["text"] for d in docs])
        self.rrf_k = rrf_k
        self.encoder = get_encoder() if mode in ("dense", "hybrid") else None
        self.mode = mode if (self.encoder is not None or mode == "bm25") else "bm25"
        self.emb = None
        if self.encoder is not None:
            self.emb = self.encoder.encode([d["text"] for d in docs], normalize_embeddings=True)

    def _dense_scores(self, query: str) -> list[float]:
        import numpy as np

        q = self.encoder.encode([query], normalize_embeddings=True)[0]
        return (np.asarray(self.emb) @ np.asarray(q)).tolist()  # cosine: embeddings are normalized

    def search(self, query: str, k: int = 5) -> list[dict]:
        rankings = []
        if self.mode in ("bm25", "hybrid"):
            rankings.append(self.bm25.scores(query))
        if self.mode in ("dense", "hybrid"):
            rankings.append(self._dense_scores(query))
        fused = [0.0] * len(self.docs)
        for scores in rankings:
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            for rank, i in enumerate(order):
                if len(rankings) == 1:
                    fused[i] = scores[i]
                else:
                    fused[i] += 1.0 / (self.rrf_k + rank + 1)
        order = sorted(range(len(fused)), key=lambda i: -fused[i])
        hits = []
        for i in order[:k]:
            if len(rankings) == 1 and self.mode == "bm25" and fused[i] <= 0:
                break  # no lexical overlap at all
            d = self.docs[i]
            hits.append({"id": d["id"], "text": d["text"], "score": round(fused[i], 4),
                         **({"note": d["note"]} if "note" in d else {})})
        return hits

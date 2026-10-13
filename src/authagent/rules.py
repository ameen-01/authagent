"""Deterministic rule engine: numbers and dates are decided in code, never by the LLM.

The LLM's job is to find and read the values; this module compares them with the policy.
Every result is one of:
  met / unmet / missing   -> a final status the agent may record
  no_match                -> the value does not belong to this criterion (wrong drug, wrong code)
  error                   -> bad input; the agent should fix the call
"""

from __future__ import annotations

from datetime import date

from .common import AS_OF


def _parse_date(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip()[:10])


def _result(status: str, reason: str, **computed) -> dict:
    return {"status": status, "reason": reason, "computed": computed}


def _matches(name: str, accepted: list[str]) -> bool:
    name = name.lower()
    return any(a.lower() in name or name in a.lower() for a in accepted)


def evaluate(criterion: dict, values: dict, as_of: date = AS_OF) -> dict:
    kind = criterion["kind"]
    params = criterion.get("params", {})
    try:
        if kind == "age":
            return _age(params, values)
        if kind == "diagnosis":
            return _diagnosis(params, values)
        if kind == "lab":
            return _lab(params, values, as_of)
        if kind == "med_duration":
            return _med_duration(params, values, as_of)
        return _result("error", f"'{kind}' criteria are judged from text: use verify_evidence instead.")
    except (KeyError, TypeError, ValueError) as exc:
        return _result("error", f"Bad values for a {kind} rule: {exc!r}. Expected: {EXPECTED[kind]}")


EXPECTED = {
    "age": '{"age": 54}',
    "diagnosis": '{"code": "M06.9"}',
    "lab": '{"name": "HbA1c", "measurements": [{"value": 7.8, "date": "2026-08-12"}, ...]}',
    "med_duration": '{"drug": "metformin", "start": "2026-01-10", "end": "2026-05-02" or null if ongoing}',
}


def _age(params: dict, values: dict) -> dict:
    age = int(values["age"])
    need = params["min_age"]
    if age >= need:
        return _result("met", f"Age {age} >= {need}.", age=age)
    return _result("unmet", f"Age {age} < {need}.", age=age)


def _diagnosis(params: dict, values: dict) -> dict:
    code = str(values["code"]).upper().strip()
    prefixes = params["code_prefixes"]
    if any(code.startswith(p.upper()) for p in prefixes):
        return _result("met", f"ICD-10 {code} matches accepted codes {prefixes}.", code=code)
    return _result("no_match", f"ICD-10 {code} is not one of {prefixes}; it does not settle this criterion on its own.",
                   code=code)


def _lab(params: dict, values: dict, as_of: date) -> dict:
    name = values.get("name")
    if name and not _matches(name, params["labs"]):
        return _result("no_match", f"'{name}' is not an accepted test for this criterion ({params['labs']}).")
    measurements = values.get("measurements") or [{"value": values["value"], "date": values["date"]}]
    latest = max(measurements, key=lambda m: _parse_date(m["date"]))
    on = _parse_date(latest["date"])
    age_days = (as_of - on).days
    if age_days > params["max_age_days"]:
        return _result("missing", f"Most recent result is from {on} ({age_days} days old); a result within "
                                  f"{params['max_age_days']} days is required.", date=str(on), age_days=age_days)
    value = latest["value"]
    if "expected" in params:
        ok = str(value).strip().lower() == params["expected"].lower()
        status = "met" if ok else "unmet"
        return _result(status, f"Result '{value}' on {on}; required '{params['expected']}'.",
                       value=value, date=str(on))
    v, thr, op = float(value), float(params["threshold"]), params["op"]
    ok = v >= thr if op == ">=" else v > thr if op == ">" else v <= thr if op == "<=" else v < thr
    status = "met" if ok else "unmet"
    return _result(status, f"Most recent value {v} {params.get('unit', '')} on {on}; required {op} {thr}.",
                   value=v, date=str(on), age_days=age_days)


def _med_duration(params: dict, values: dict, as_of: date) -> dict:
    drug = str(values["drug"])
    if not _matches(drug, params["drugs"]):
        return _result("no_match", f"'{drug}' is not an accepted drug for this criterion ({params['drugs']}).")
    start = _parse_date(values["start"])
    end = _parse_date(values["end"]) if values.get("end") else as_of
    days = (end - start).days
    need = params["min_days"]
    status = "met" if days >= need else "unmet"
    return _result(status, f"{drug} from {start} to {end}: {days} days; required >= {need}.",
                   days=days, start=str(start), end=str(end))

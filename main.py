"""
Dave Backend — FastAPI application (v7.1)

Endpoints:
  GET  /api/health             — liveness + chart_in / local_rules flags
  POST /api/analyze            — note and/or chart (local deterministic always; Claude optional)
  GET  /api/vim/launch         — Vim OAuth authorize entry
  POST /api/vim/callback       — Vim token exchange
  POST /api/vim/launch-log     — non-PHI operational beacon
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vim_auth import router as vim_router

# ---- Logging (never log note text / PHI) ------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("dave")

APP_VERSION = "7.1"  # Step 7: /api/analyze accepts note and/or chart (Evidence Data Layer bridge)

# ---- App --------------------------------------------------------------------
app = FastAPI(
    title="Dave Coding Intelligence Backend",
    version=APP_VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# ---- CORS -------------------------------------------------------------------
# FRONTEND_ORIGIN + CORS_ORIGINS; always allow Vim SDK origins.
_cors_env = os.getenv("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
_fo = os.getenv("FRONTEND_ORIGIN", "").strip()
if _fo and _fo not in _cors_origins:
    _cors_origins.append(_fo)
# Safe defaults for NyCal Vercel app if env not set yet
for _default in (
    "https://nycal-ai-billing.vercel.app",
    "http://127.0.0.1:8788",
    "http://localhost:8788",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
):
    if _default not in _cors_origins:
        _cors_origins.append(_default)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"https://.*\.getvim\.com$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)

app.include_router(vim_router)

# Simple in-process rate limit (per instance): max N analyzes / window
_RATE_WINDOW_SEC = 60
_RATE_MAX = int(os.getenv("ANALYZE_RATE_MAX", "30"))
_rate_hits: list[float] = []


def _rate_ok() -> bool:
    now = time.time()
    while _rate_hits and _rate_hits[0] < now - _RATE_WINDOW_SEC:
        _rate_hits.pop(0)
    if len(_rate_hits) >= _RATE_MAX:
        return False
    _rate_hits.append(now)
    return True


# ---- Health -----------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "claude": bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
        "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        "local_rules": True,
        "chart_in": True,
        "hybrid": True,
        "vim_configured": bool(
            os.getenv("VIM_CLIENT_ID", "").strip()
            and os.getenv("VIM_CLIENT_SECRET", "").strip()
        ),
    }


# ---- Analyze (Step 7) -------------------------------------------------------
class ChartObservation(BaseModel):
    name: str
    value: Any = None
    unit: Optional[str] = None
    date: Optional[str] = None
    loinc: Optional[str] = None


class ChartMedication(BaseModel):
    name: str
    rxnorm: Optional[str] = None
    status: Optional[str] = "active"


class ChartCondition(BaseModel):
    code: Optional[str] = None
    display: Optional[str] = None
    onset: Optional[str] = None
    status: Optional[str] = "active"


class ChartDocument(BaseModel):
    type: Optional[str] = "consult"
    specialty: Optional[str] = None
    date: Optional[str] = None
    summary: Optional[str] = None


class ChartEncounter(BaseModel):
    date: Optional[str] = None
    type: Optional[str] = None
    modality: Optional[str] = "unknown"


class ChartBody(BaseModel):
    """Minimal internal chart shape (DAVE-EVIDENCE-DATA-LAYER.md Step 3)."""
    encounter: Optional[ChartEncounter] = None
    conditions: Optional[List[ChartCondition]] = None
    observations: Optional[List[ChartObservation]] = None
    medications: Optional[List[ChartMedication]] = None
    documents: Optional[List[ChartDocument]] = None


class AnalyzeBody(BaseModel):
    """Step 7: note and/or chart. At least one required."""
    note: Optional[str] = None
    chart: Optional[ChartBody] = None
    age_band: Optional[str] = None
    coded_diagnoses: Optional[list] = None
    payer_hint: Optional[str] = None  # medicare | medicaid | commercial | dual | medicare_ma
    payer: Optional[str] = None  # alias
    visit_hints: Optional[dict] = None
    prefer_local: Optional[bool] = False  # force local rules even if Claude configured


@app.post("/api/analyze")
async def analyze(body: AnalyzeBody, request: Request):
    """
    Dave analyze — note-in, chart-in, or hybrid.
    - Local deterministic path always available (Evidence Graph + recommendation-only).
    - Claude optional when ANTHROPIC_API_KEY set and prefer_local is false.
    Frontend may also run pure JS localAnalyze; non-2xx → fall back to local UI engine.
    """
    note = (body.note or "").strip()
    chart_dict = body.chart.model_dump(exclude_none=True) if body.chart else None
    has_note = bool(note)
    has_chart = bool(chart_dict) and any(
        chart_dict.get(k)
        for k in ("conditions", "observations", "medications", "documents", "encounter")
    )
    if not has_note and not has_chart:
        raise HTTPException(400, "Provide note and/or chart")
    if has_note and len(note) > 50_000:
        raise HTTPException(413, "note too large (max 50,000 chars)")

    if not _rate_ok():
        raise HTTPException(429, "rate limit — try again shortly or use local engine")

    payer = (body.payer or body.payer_hint or "").strip().lower() or None
    t0 = time.time()

    # Prefer Claude when configured unless prefer_local or chart-only without note
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    use_claude = bool(api_key) and not body.prefer_local and has_note

    if use_claude:
        try:
            out = await _analyze_claude(
                note=note,
                chart=chart_dict,
                age_band=body.age_band,
                coded=body.coded_diagnoses,
                payer_hint=payer,
            )
            elapsed_ms = int((time.time() - t0) * 1000)
            log.info(
                "analyze claude ok note_chars=%s chart=%s ms=%s icd=%s programs=%s",
                len(note),
                has_chart,
                elapsed_ms,
                len(out.get("icd") or []),
                len(out.get("programs") or []),
            )
            return JSONResponse(out)
        except HTTPException:
            # Fall through to local if Claude fails
            log.warning("claude path failed — falling back to local rules")

    # Local deterministic path (chart and/or note) — no PHI logged
    out = _analyze_local(
        note=note or "",
        chart=chart_dict,
        payer=payer,
        visit_hints=body.visit_hints,
        coded=body.coded_diagnoses,
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    log.info(
        "analyze local ok note_chars=%s chart=%s ms=%s icd=%s programs=%s evidence=%s",
        len(note),
        has_chart,
        elapsed_ms,
        len(out.get("icd") or []),
        len(out.get("programs") or []),
        (out.get("evidenceGraph") or {}).get("count") or 0,
    )
    return JSONResponse(out)


async def _analyze_claude(
    note: str,
    chart: Optional[dict],
    age_band: Optional[str],
    coded: Optional[list],
    payer_hint: Optional[str],
) -> dict:
    try:
        import anthropic
    except ImportError:
        raise HTTPException(501, "anthropic SDK not installed")

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5").strip() or "claude-sonnet-4-5"
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "").strip())
    user_content = _build_user_message(note, age_band, coded, payer_hint, chart)

    create_kwargs = dict(
        model=model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    try:
        try:
            resp = client.messages.create(**create_kwargs, timeout=45.0)
        except TypeError:
            resp = client.messages.create(**create_kwargs)
    except Exception as e:
        log.warning("Anthropic call failed: %s", type(e).__name__)
        raise HTTPException(502, "AI analysis failed — frontend should fall back")

    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    parsed = _parse_json_payload(text)
    if parsed is None:
        raise HTTPException(502, "AI returned unparseable output")
    out = _normalize_result(parsed)
    # Attach local evidence graph when chart present (Claude may omit it)
    if chart:
        eg = _build_evidence_from_chart(chart)
        out["evidenceGraph"] = eg
        out["evidence_graph_ref"] = eg.get("graph_id")
    return out


def _chart_to_synthetic_note(chart: dict) -> str:
    """Turn minimal chart JSON into a short structured note for Claude / local."""
    lines: list[str] = []
    enc = chart.get("encounter") or {}
    if enc:
        lines.append(
            f"Encounter: {enc.get('type') or 'office'} · modality {enc.get('modality') or 'unknown'}"
            + (f" · date {enc.get('date')}" if enc.get("date") else "")
        )
    for c in chart.get("conditions") or []:
        if str(c.get("status") or "active").lower() in ("inactive", "resolved", "remission"):
            continue
        lines.append(f"Problem: {c.get('display') or c.get('code') or 'condition'} ({c.get('code') or ''})")
    for o in chart.get("observations") or []:
        lines.append(
            f"Lab/vital: {o.get('name')} {o.get('value')} {o.get('unit') or ''}".strip()
            + (f" on {o.get('date')}" if o.get("date") else "")
        )
    for m in chart.get("medications") or []:
        if str(m.get("status") or "active").lower() in ("stopped", "inactive", "cancelled"):
            continue
        lines.append(f"Medication: {m.get('name')} active")
    for d in chart.get("documents") or []:
        lines.append(
            f"{d.get('type') or 'consult'}: {d.get('specialty') or ''} — {d.get('summary') or ''}".strip()
        )
    return "\n".join(lines) if lines else ""


def _build_evidence_from_chart(chart: dict) -> dict:
    """Python mirror of buildGraphFromChart — atomic facts for audit / UI."""
    import time as _time

    evidence: list[dict] = []
    seq = 0

    def add(ev: dict) -> str:
        nonlocal seq
        seq += 1
        eid = f"ev_{ev.get('type', 'x')}_{seq}"
        fact = {
            "evidence_id": eid,
            "confidence": ev.get("confidence", 0.9),
            "provenance": "structured_fhir",
            "links_to": ev.get("links_to") or [],
            **ev,
            "evidence_id": eid,
        }
        evidence.append(fact)
        return eid

    for obs in chart.get("observations") or []:
        name = str(obs.get("name") or "").strip()
        if not name:
            continue
        raw = obs.get("value")
        try:
            num = float(raw)
            is_num = True
        except (TypeError, ValueError):
            num = None
            is_num = False
        links: list[str] = []
        clinical_read = None
        typ = "lab"
        display = name
        unit = obs.get("unit")
        nl = name.lower().replace(" ", "")
        if nl in ("a1c", "hba1c") or "a1c" in nl:
            display, links = "A1c", ["cond_dm"]
            unit = unit or "%"
            if is_num:
                clinical_read = (
                    "markedly_elevated"
                    if num >= 9
                    else "uncontrolled_range"
                    if num >= 7.5
                    else "diabetes_range"
                    if num >= 6.5
                    else "prediabetes_range"
                    if num >= 5.7
                    else "normal"
                )
        elif "egfr" in nl:
            display, links = "eGFR", ["cond_ckd"]
            unit = unit or "mL/min"
            if is_num:
                stage = (
                    "5" if num < 15 else "4" if num < 30 else "3b" if num < 45 else "3a" if num < 60 else "2" if num < 90 else "1"
                )
                clinical_read = f"stage_{stage}_range"
        elif nl in ("bp", "bloodpressure") or "bloodpressure" in nl:
            typ, display, links = "vital", "BP", ["cond_htn"]
            unit = unit or "mmHg"
            bp = str(raw or "")
            m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", bp)
            if m:
                s, d = int(m.group(1)), int(m.group(2))
                raw = f"{s}/{d}"
                is_num = False
                clinical_read = (
                    "above_goal" if s >= 140 or d >= 90 else "elevated" if s >= 130 or d >= 80 else "at_goal"
                )
        elif "bmi" in nl:
            typ, display, links = "vital", "BMI", ["cond_obesity"]
            unit = unit or "kg/m2"
            if is_num:
                clinical_read = (
                    "class3" if num >= 40 else "class2" if num >= 35 else "class1" if num >= 30 else "overweight" if num >= 25 else "normal"
                )
        elif "phq" in nl:
            typ, display, links = "score", "PHQ-9", ["cond_depression"]
            if is_num:
                clinical_read = (
                    "severe" if num >= 20 else "moderately_severe" if num >= 15 else "moderate" if num >= 10 else "mild" if num >= 5 else "minimal"
                )
        elif "ldl" in nl:
            display, links = "LDL", ["cond_lipid"]
            unit = unit or "mg/dL"
        elif "ef" == nl or "ejection" in nl:
            typ, display, links = "imaging", "EF", ["cond_hf"]
            unit = unit or "%"
            if is_num:
                clinical_read = "HFrEF_range" if num <= 40 else "HFpEF_range" if num >= 50 else "HFmrEF_range"
        add(
            {
                "type": typ,
                "name": display,
                "value": num if is_num else raw,
                "unit": unit,
                "observed_at": obs.get("date"),
                "code": obs.get("loinc"),
                "code_system": "LOINC" if obs.get("loinc") else None,
                "links_to": links,
                "clinical_read": clinical_read,
                "confidence": 0.95,
            }
        )

    for med in chart.get("medications") or []:
        status = str(med.get("status") or "active").lower()
        if status in ("stopped", "inactive", "completed", "cancelled"):
            continue
        n = str(med.get("name") or "").lower()
        if not n:
            continue
        links, label = [], "med"
        if re.search(r"metformin|semaglutide|empagliflozin|insulin|ozempic|jardiance|mounjaro", n):
            links, label = ["cond_dm"], "dm_agent"
        elif re.search(r"sacubitril|entresto|furosemide|spironolactone", n):
            links, label = ["cond_hf"], "hf_agent"
        elif re.search(r"carvedilol|metoprolol succ|bisoprolol", n):
            links, label = ["cond_hf", "cond_htn"], "bb_hf_htn"
        elif re.search(r"lisinopril|losartan|amlodipine|valsartan|hctz", n):
            links, label = ["cond_htn"], "htn_agent"
        elif re.search(r"atorvastatin|rosuvastatin|statin", n):
            links, label = ["cond_lipid"], "statin"
        elif re.search(r"sertraline|escitalopram|fluoxetine|bupropion", n):
            links, label = ["cond_depression"], "antidepressant"
        add(
            {
                "type": "medication",
                "name": label,
                "value": med.get("name"),
                "links_to": links,
                "code": med.get("rxnorm"),
                "code_system": "RxNorm" if med.get("rxnorm") else None,
                "confidence": 0.9,
            }
        )

    for c in chart.get("conditions") or []:
        status = str(c.get("status") or "active").lower()
        if status in ("inactive", "resolved", "remission", "entered-in-error"):
            continue
        code = str(c.get("code") or "")
        display = c.get("display") or code or "condition"
        cid = "cond_other"
        if re.match(r"^E11", code, re.I) or re.search(r"diabet", display, re.I):
            cid = "cond_dm"
        elif re.match(r"^N18", code, re.I) or re.search(r"ckd|chronic kidney", display, re.I):
            cid = "cond_ckd"
        elif re.match(r"^I1[0-3]", code, re.I) or re.search(r"hypertens", display, re.I):
            cid = "cond_htn"
        elif re.match(r"^I50", code, re.I) or re.search(r"heart failure|chf", display, re.I):
            cid = "cond_hf"
        elif re.match(r"^F3[23]", code, re.I) or re.search(r"depress", display, re.I):
            cid = "cond_depression"
        elif re.match(r"^E66", code, re.I) or re.search(r"obesity", display, re.I):
            cid = "cond_obesity"
        add(
            {
                "type": "problem_or_ap",
                "name": cid,
                "value": display,
                "code": code or None,
                "code_system": "ICD10" if code else None,
                "links_to": [cid],
                "evidence_class": "suggested",
                "confidence": 0.82,
                "clinical_read": "provider_addressed_or_listed",
            }
        )

    for d in chart.get("documents") or []:
        typ = str(d.get("type") or "consult").lower()
        specialty = str(d.get("specialty") or typ).lower()
        links = []
        if "nephrolog" in specialty:
            links.append("cond_ckd")
        if "cardiolog" in specialty:
            links.extend(["cond_hf", "cond_htn"])
        if "endocrin" in specialty:
            links.append("cond_dm")
        add(
            {
                "type": "imaging" if typ == "imaging" else "consult",
                "name": specialty,
                "value": d.get("summary") or d.get("specialty") or typ,
                "links_to": links,
                "observed_at": d.get("date"),
                "clinical_read": "imaging_report" if typ == "imaging" else "specialty_involved",
                "confidence": 0.85,
            }
        )

    gid = f"eg_{int(_time.time())}_{seq}"
    enc = chart.get("encounter") or {}
    return {
        "graph_id": gid,
        "version": "0.1",
        "input_mode": "chart_in",
        "count": len(evidence),
        "evidence": evidence,
        "encounter": enc if enc else None,
        "decision_traces": [
            {
                "trace_id": "tr_1",
                "actor": "adapter_chart",
                "action": "added_facts",
                "detail": f"Chart-in mapped {len(evidence)} facts",
                "evidence_ids": [e["evidence_id"] for e in evidence],
            }
        ],
    }


def _analyze_local(
    note: str,
    chart: Optional[dict],
    payer: Optional[str],
    visit_hints: Optional[dict],
    coded: Optional[list],
) -> dict:
    """
    Deterministic local analyze — recommendation only.
    Builds Evidence Graph from chart (and simple note labs if present),
    ICD suggestions from active conditions, opportunities from labs/meds.
    """
    chart = chart or {}
    synthetic = _chart_to_synthetic_note(chart) if chart else ""
    combined = "\n".join(x for x in (note, synthetic) if x)

    evidence_graph = _build_evidence_from_chart(chart) if chart else {
        "graph_id": None,
        "version": "0.1",
        "input_mode": "note_in",
        "count": 0,
        "evidence": [],
        "decision_traces": [],
    }

    # Pull key values from evidence
    vals: dict[str, Any] = {}
    for e in evidence_graph.get("evidence") or []:
        if e.get("name") in ("A1c", "eGFR", "BP", "BMI", "PHQ-9", "EF", "LDL"):
            vals[e["name"]] = e.get("value")

    # Simple note regex fallbacks
    if note:
        m = re.search(r"a1c[^0-9]{0,12}(\d{1,2}(?:\.\d)?)", note, re.I)
        if m and "A1c" not in vals:
            vals["A1c"] = float(m.group(1))
        m = re.search(r"egfr[^0-9]{0,10}(\d{1,3})", note, re.I)
        if m and "eGFR" not in vals:
            vals["eGFR"] = float(m.group(1))
        m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", note)
        if m and "BP" not in vals:
            vals["BP"] = f"{m.group(1)}/{m.group(2)}"

    icd: list[dict] = []
    programs: list[dict] = []

    def ids_for(*names: str) -> list[str]:
        out = []
        for e in evidence_graph.get("evidence") or []:
            if e.get("name") in names or any(l in names for l in (e.get("links_to") or [])):
                out.append(e["evidence_id"])
        return out[:4]

    # Conditions → ICD (suggested / captured — recommendation only)
    for c in chart.get("conditions") or []:
        status = str(c.get("status") or "active").lower()
        if status in ("inactive", "resolved", "remission"):
            continue
        code = str(c.get("code") or "").strip()
        display = c.get("display") or code
        if not code and not display:
            continue
        kind = "captured"
        note_txt = "From chart problem list — confirm MEAT before claiming. Recommendation only."
        # Specificity coach
        if re.match(r"^E11\.9", code, re.I):
            egfr = vals.get("eGFR")
            if egfr is not None and float(egfr) < 60:
                icd.append(
                    {
                        "code": "E11.22",
                        "to": "E11.22",
                        "from": code,
                        "desc": "Type 2 DM with diabetic CKD",
                        "kind": "upgrade",
                        "note": "Chart has DM + eGFR in CKD range — prefer E11.22 + N18.x when both assessed (MEAT). Recommendation only.",
                        "evidence_ids": ids_for("A1c", "eGFR", "cond_dm", "cond_ckd"),
                        "raf": 0.3,
                    }
                )
                kind = "flag"
                note_txt = "Unspecified DM on list while CKD-range eGFR present — consider E11.22 path."
        link_names: list = []
        cu = code.upper()
        if "E11" in cu or "E10" in cu:
            link_names = ["cond_dm", "A1c", "dm_agent"]
        elif "N18" in cu:
            link_names = ["cond_ckd", "eGFR"]
        elif re.match(r"I1[0-3]", code, re.I):
            link_names = ["cond_htn", "BP", "htn_agent"]
        elif "I50" in cu:
            link_names = ["cond_hf", "EF", "hf_agent"]
        elif re.match(r"F3[23]", code, re.I):
            link_names = ["cond_depression", "PHQ-9"]
        elif "E66" in cu:
            link_names = ["cond_obesity", "BMI"]
        icd.append(
            {
                "code": code or "UNSPEC",
                "to": code or "UNSPEC",
                "desc": display,
                "kind": kind,
                "note": note_txt,
                "evidence_ids": ids_for(*link_names) if link_names else [],
            }
        )

    # Opportunities from labs
    a1c = vals.get("A1c")
    try:
        a1c_f = float(a1c) if a1c is not None else None
    except (TypeError, ValueError):
        a1c_f = None
    if a1c_f is not None and a1c_f >= 7.5:
        programs.append(
            {
                "code": "95250 / 95251",
                "desc": "CGM fit in office + monthly interpretation",
                "note": "Uncontrolled DM signal from A1c — consider in-office CGM next visit. Recommendation only.",
                "evidence_ids": ids_for("A1c", "cond_dm", "dm_agent"),
                "timing": "next_visit",
            }
        )
    bp = str(vals.get("BP") or "")
    high_bp = False
    m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", bp)
    if m and (int(m.group(1)) >= 140 or int(m.group(2)) >= 90):
        high_bp = True
    has_htn = any(
        re.match(r"^I1[0-3]", str(c.get("code") or ""), re.I)
        or re.search(r"hypertens", str(c.get("display") or ""), re.I)
        for c in (chart.get("conditions") or [])
    )
    if has_htn or high_bp:
        programs.append(
            {
                "code": "99453 / 99454 / 99457",
                "desc": "RPM — fit BP cuff + remote monitoring",
                "note": "HTN context — RPM BP path is a found-value opportunity. Recommendation only.",
                "evidence_ids": ids_for("BP", "cond_htn", "htn_agent"),
                "timing": "monthly",
            }
        )

    # Payer gate note
    allow_g = bool(payer and any(x in payer for x in ("medicare", "ma", "dual")))
    if not payer:
        programs.append(
            {
                "code": "Confirm payer",
                "desc": "Payer not clear — confirm before any G-codes",
                "note": "G-codes are Medicare FFS / MA / Dual only. Ask provider if payer unclear. Recommendation only.",
            }
        )

    # Number opportunities
    opp_n = 0
    for p in programs:
        if str(p.get("code") or "").lower().startswith("confirm"):
            continue
        opp_n += 1
        p["oppNumber"] = opp_n

    em_level = "99214" if len(icd) >= 2 else "99213"
    return {
        "icd": icd[:24],
        "em": {
            "level": em_level,
            "basis": "Local rules from chart/note complexity (recommendation only)",
            "value": 0,
            "baseline": "99213",
            "baseValue": 0,
        },
        "programs": programs[:20],
        "q": [],
        "v": {str(k): str(v)[:48] for k, v in list(vals.items())[:20]},
        "rafGain": 0.0,
        "emGain": 0,
        "payer": payer,
        "product": None,
        "payerFindings": [],
        "payerMeta": {
            "allowGCodes": allow_g,
            "payerUncertain": not bool(payer),
        },
        "evidenceGraph": {
            "version": evidence_graph.get("version"),
            "count": evidence_graph.get("count"),
            "evidence": (evidence_graph.get("evidence") or [])[:40],
            "input_mode": evidence_graph.get("input_mode"),
        },
        "evidence_graph_ref": evidence_graph.get("graph_id"),
        "engine": "local-chart" if chart and not note else "local-hybrid" if chart else "local-note",
        "version": APP_VERSION,
    }


def _parse_json_payload(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        # salvage first { ... } block
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def _normalize_result(parsed: dict) -> dict:
    """Ensure frontend-safe shape matching localAnalyze output."""
    icd = parsed.get("icd") if isinstance(parsed.get("icd"), list) else []
    programs = parsed.get("programs") if isinstance(parsed.get("programs"), list) else []
    q = parsed.get("q") if isinstance(parsed.get("q"), list) else []
    em = parsed.get("em") if isinstance(parsed.get("em"), dict) else {}
    v = parsed.get("v") if isinstance(parsed.get("v"), dict) else {}

    clean_icd = []
    for i in icd[:24]:
        if not isinstance(i, dict):
            continue
        code = str(i.get("code") or i.get("to") or "").strip()
        if not code:
            continue
        clean_icd.append(
            {
                "code": code,
                "desc": str(i.get("desc") or code)[:160],
                "kind": str(i.get("kind") or "flag")[:20],
                "hcc": i.get("hcc"),
                "raf": float(i["raf"]) if isinstance(i.get("raf"), (int, float)) else 0,
                "note": str(i.get("note") or "")[:500],
            }
        )

    clean_programs = []
    for p in programs[:20]:
        if not isinstance(p, dict) or not p.get("code"):
            continue
        clean_programs.append(
            {
                "code": str(p.get("code"))[:80],
                "desc": str(p.get("desc") or "")[:200],
                "note": str(p.get("note") or "")[:800],
            }
        )

    clean_q = []
    for item in q[:20]:
        if not isinstance(item, dict) or not item.get("code"):
            continue
        clean_q.append(
            {
                "code": str(item.get("code"))[:20],
                "desc": str(item.get("desc") or "")[:160],
                "note": str(item.get("note") or "")[:400],
            }
        )

    level = str(em.get("level") or "99213").split()[0]
    return {
        "icd": clean_icd,
        "em": {
            "level": level,
            "basis": str(em.get("basis") or "")[:400],
            "value": em.get("value") or 0,
            "baseline": str(em.get("baseline") or "99213"),
            "baseValue": em.get("baseValue") or 0,
            "tcm": bool(em.get("tcm")),
            "timeNote": em.get("timeNote"),
        },
        "programs": clean_programs,
        "q": clean_q,
        "v": {str(k)[:32]: str(val)[:48] for k, val in list(v.items())[:20]},
        "rafGain": float(parsed["rafGain"]) if isinstance(parsed.get("rafGain"), (int, float)) else 0.0,
        "emGain": float(parsed["emGain"]) if isinstance(parsed.get("emGain"), (int, float)) else 0,
        "payer": parsed.get("payer"),
        "product": parsed.get("product"),
        "payerFindings": parsed.get("payerFindings") if isinstance(parsed.get("payerFindings"), list) else [],
        "engine": "claude",
        "version": APP_VERSION,
    }


# ---- Claude system prompt (aligned with local Dave rules) -------------------
_SYSTEM_PROMPT = """You are Dave, a Family Medicine ICD-10 coding copilot embedded in a clinical app (NyCal.ai).

Your job: review the provider note, problem list, meds, labs, and any currently selected ICD-10 codes, then:
• Identify when a more specific ICD-10 code is appropriate.
• Discourage unspecified codes (.9 / "unspecified") when documentation supports specificity.
• Never invent diagnoses not clearly supported by documentation.
• Stay recommendation-only — final code selection rests with the treating physician and coding staff.

Return STRICT JSON only — no markdown fences, no prose outside JSON:

{
  "icd": [
    {"code": "E11.22", "desc": "Type 2 DM with diabetic CKD", "kind": "upgrade|captured|flag",
     "hcc": "HCC 37 or null", "raf": 0.302,
     "note": "1-2 sentences + optional provider clarification question. Recommendation only.",
     "from": "E11.9"}
  ],
  "em": {"level": "99214", "basis": "why", "value": 0, "baseline": "99213",
         "baseValue": 0, "tcm": false, "timeNote": null},
  "programs": [{"code": "G2211", "desc": "...", "note": "..."}],
  "q": [{"code": "1111F", "desc": "...", "note": "..."}],
  "v": {"A1c": "8.4%", "eGFR": "52", "BP": "146/88", "BMI": "34.2"},
  "rafGain": 0.0,
  "emGain": 0,
  "payer": null,
  "product": null,
  "payerFindings": [],
  "summary": "One short sentence of what you found (specificity gaps / claim-ready upgrades).",
  "documentation_prompts": [
    "Optional clarifying questions when specificity is possible but not fully documented."
  ]
}

FAMILY MEDICINE CODING COPILOT RULES:
A. Documentation-first: if specificity is possible but NOT documented, do not guess — put a brief clarification in documentation_prompts / note.
B. Unspecified ban: if selected or implied code ends in .9 or is "unspecified" AND the note supports laterality, stage, type, acute/chronic, or complication — suggest the specific code with reason.
C. Diabetes pairs: E11.9 + CKD/nephropathy → E11.22 + N18.x stage; + neuropathy → E11.40/E11.42; + retinopathy → E11.3xx; + foot ulcer → E11.62x. Never invent complication without note support.
D. Labs guide specificity only: eGFR/A1c/BMI are not diagnoses by themselves — require provider assessment wording (MEAT).
E. Never invent diagnoses. Only suggest codes supported by the note, problem list, meds, or labs-in-context.

HARD RULES (billing/programs):
1. Recommendation / found-value ONLY — never orders.
2. Only code what the note supports.
3. E11.A = T2DM in remission only if provider documents remission and no complication codes (Coding Clinic Q4 2025). ADA: A1c ≥6.5 = diabetes range (not prediabetes). Never R73.03 with E08–E13.
4. ICD specificity + MEAT for chronics. Aim ≤12 claim codes.
5. Acute primary coded but severe chronics in note → coach MDM impact IF assessed; do not invent unassessed codes.
6. E/M: 2021 AMA MDM + time. Multi-chronic stable management is usually 99214, not 99215, unless true high risk.
7. TCM: post-ER/hospital → 99495/99496 when criteria met; else office E/M + clear YES/NO TCM prompt. G2211 does not attach to TCM.
8. G-codes only when Medicare/MA/Dual is evident.
9. G2211 only on O/O E/M 99202–99215 — never on AWV line alone.
10. G0444 = depression (PHQ); G0442 = alcohol screen; G0443 = alcohol counseling. Do not swap.
11. Dementia assessed → recommend MoCA/Mini-Cog if no tool; never leave F03.9x when etiology is known.
12. Advance directives updated → 99497 if ≥16 min documented.
13. programs[]: concrete same-day codes; post-claim opportunities labeled clearly (CCM, CGM, RPM, PCM).
14. Concise notes (≤2 sentences). Empty arrays when nothing applies.
"""


def _build_user_message(
    note: str,
    age_band: Optional[str],
    coded: Optional[list],
    payer_hint: Optional[str],
    chart: Optional[dict] = None,
) -> str:
    parts = ["Analyze this progress note:\n\n", note or "(no free-text note)"]
    if chart:
        synthetic = _chart_to_synthetic_note(chart)
        if synthetic:
            parts.append("\n\nStructured chart facts (labs/meds/problems):\n")
            parts.append(synthetic)
    if age_band:
        parts.append(f"\n\nPatient age band: {age_band}")
    if coded:
        parts.append(
            "\n\nAlready coded on encounter: "
            + ", ".join(str(c) for c in coded)
        )
    if payer_hint:
        parts.append(f"\n\nPayer hint from UI: {payer_hint}")
    parts.append("\n\nReturn the JSON object now. Recommendation only.")
    return "".join(parts)


# ---- Entrypoint -------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

"""
api_note.py — Dave Super Listen 2 note writer for the dave-backend (FastAPI on Render).

POST /api/note
  in : { prompt, template_id, template_name, sections: [keys], rules: [str], provider_id, transcript,
         chart: { name, age, sex, visit, cc, problems, meds, allergies, labs, bp, bmi, hx, prev }, source }
  out: { sections: { cc, hpi, meds, allergies, pmh, psh, hosp, fh, sh, ros, exam, data, assessment, plan, next },
         flags: [ { title, text } ], assessmentItems: [ { title, code, assessment, plan } ],
         template_id, model, engine: "render-note" }

Drop-in:
    from api_note import router as note_router
    app.include_router(note_router)

Env:
    ANTHROPIC_API_KEY     required (already set for /api/analyze)
    DAVE_NOTE_MODEL       optional; falls back to ANTHROPIC_MODEL (same model /api/analyze uses)
    DAVE_NOTE_MAX_TOKENS  optional, default 2500

Rules the route enforces regardless of what the model returns:
  - the HPI starts with the Dave consent line
  - nothing is invented: the model is told to use only the transcript and chart; empty sections stay empty
  - questions for the provider come back as flags ("CHECK BEFORE SIGNING"), never settled by the model
  - no PHI is logged; errors return JSON, never a stack trace
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    import anthropic  # pip install anthropic
except Exception:  # pragma: no cover
    anthropic = None

router = APIRouter()

SECTION_KEYS = ["cc", "hpi", "meds", "allergies", "pmh", "psh", "hosp", "fh", "sh", "ros", "exam", "data", "assessment", "plan", "next"]
CONSENT = "Patient consented to the use of Dave to record and transcribe notes during this visit."
MODEL = (os.getenv("DAVE_NOTE_MODEL") or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-5").strip()
MAX_TOKENS = int(os.getenv("DAVE_NOTE_MAX_TOKENS", "2500"))


class NoteIn(BaseModel):
    prompt: str = ""
    template_id: str = "default"
    template_name: str = "Default"
    sections: List[str] = Field(default_factory=lambda: SECTION_KEYS)
    rules: List[str] = Field(default_factory=list)
    provider_id: str = "default"
    transcript: str = ""
    chart: Dict[str, Any] = Field(default_factory=dict)
    source: str = ""

    class Config:
        extra = "allow"


SYSTEM = """You write progress-note sections for a primary-care provider from a visit transcript and the chart on screen.
Hard rules:
1. Use ONLY what is in the transcript and the chart. Never invent vitals, labs, exam findings, medications, history or negatives. If a section has nothing, return an empty string for it.
2. The HPI is one chronological story in plain clinical prose, third person, no headings inside it. Patient-reported symptoms are attributed as reported, never written as objective findings.
3. Assessment and plan are by problem, numbered, and only for problems that are on the chart AND/OR were addressed in the visit. A problem on the chart that was not discussed is left out. A problem discussed but not on the chart gets a flag.
4. Anything the provider must decide (a diagnosis not named, specificity such as CKD stage or diabetic complication, depression episode type, a controlled-substance refill) is NOT decided by you: put it in flags as a short question.
5. Follow the provider's rules exactly when given.
6. Output JSON only, no prose around it, with this shape:
{"sections":{"cc":"","hpi":"","meds":"","allergies":"","pmh":"","psh":"","hosp":"","fh":"","sh":"","ros":"","exam":"","data":"","assessment":"","plan":"","next":""},
 "assessmentItems":[{"title":"","code":"","assessment":"","plan":""}],
 "flags":[{"title":"","text":""}]}
Keys you were not asked for stay empty strings."""


def _chart_text(c: Dict[str, Any]) -> str:
    def j(v: Any) -> str:
        if isinstance(v, list):
            return "; ".join(str(x) for x in v[:60])
        return "" if v is None else str(v)
    lines = []
    for k, label in [("name", "Patient"), ("age", "Age"), ("sex", "Sex"), ("visit", "Visit type"), ("cc", "Chief complaint"), ("bp", "BP"), ("bmi", "BMI"),
                     ("problems", "Problem list"), ("meds", "Medications"), ("allergies", "Allergies"), ("labs", "Labs"), ("hx", "History"), ("prev", "Preventive")]:
        v = j(c.get(k))
        if v:
            lines.append(f"{label}: {v}")
    return "\n".join(lines)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _clean(d: Dict[str, Any], wanted: List[str]) -> Dict[str, Any]:
    sec_in = d.get("sections") if isinstance(d.get("sections"), dict) else d
    sections = {k: (str(sec_in.get(k) or "").strip() if k in wanted else "") for k in SECTION_KEYS}
    hpi = sections.get("hpi", "")
    if hpi and not hpi.lower().startswith(CONSENT.lower()[:30]):
        sections["hpi"] = CONSENT + " " + hpi
    items = []
    for it in (d.get("assessmentItems") or [])[:20]:
        if not isinstance(it, dict):
            continue
        items.append({"title": str(it.get("title") or "")[:160], "code": str(it.get("code") or "")[:12],
                      "assessment": str(it.get("assessment") or "")[:1500], "plan": str(it.get("plan") or "")[:1500]})
    flags = []
    for f in (d.get("flags") or [])[:20]:
        if isinstance(f, str):
            flags.append({"title": "Check", "text": f[:300]})
        elif isinstance(f, dict):
            flags.append({"title": str(f.get("title") or "Check")[:60], "text": str(f.get("text") or "")[:300]})
    return {"sections": sections, "assessmentItems": items, "flags": flags}


@router.post("/api/note")
async def api_note(body: NoteIn):
    if anthropic is None or not os.getenv("ANTHROPIC_API_KEY"):
        return JSONResponse({"error": "note writer not configured"}, status_code=500)
    if not body.transcript.strip():
        return JSONResponse({"error": "empty transcript"}, status_code=400)

    wanted = [k for k in body.sections if k in SECTION_KEYS] or SECTION_KEYS
    rules = "\n".join(f"- {r}" for r in body.rules if str(r).strip())
    user = (
        f"TEMPLATE: {body.template_name} ({body.template_id})\n"
        f"SECTIONS TO WRITE: {', '.join(wanted)}\n\n"
        f"TEMPLATE GUIDANCE AND PROVIDER STYLE:\n{body.prompt.strip()}\n\n"
        + (f"PROVIDER RULES:\n{rules}\n\n" if rules else "")
        + f"CHART ON SCREEN:\n{_chart_text(body.chart)}\n\n"
        f"TRANSCRIPT:\n{body.transcript.strip()[:40000]}\n\n"
        "Return the JSON now."
    )

    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "").strip())
        msg = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(p, "text", "") for p in msg.content)
    except Exception as e:  # never echo the provider's message with PHI in it
        return JSONResponse({"error": "model call failed", "kind": type(e).__name__}, status_code=502)

    parsed = _extract_json(text)
    if not parsed:
        return JSONResponse({"error": "model returned no JSON"}, status_code=502)

    out = _clean(parsed, wanted)
    out.update({"template_id": body.template_id, "model": MODEL, "engine": "render-note"})
    return out

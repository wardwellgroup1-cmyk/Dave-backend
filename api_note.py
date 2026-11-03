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
  - server-side checks add flags for: pronoun vs chart sex, med frequency vs chart, refill on a controlled/high-risk med (PDMP),
    acute chart diagnosis not discussed today, G89 pain code next to an explaining diagnosis; stray "." lines are removed
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
4b. orders: every lab, imaging study, referral or in-office procedure the clinician ORDERED OR PLACED in this visit — never one only discussed, declined, done before, or planned for a later visit. name as said ("A1c", "MRI lumbar spine", "Cardiology"); problem = the problem it is for, as named in assessmentItems; to = the clinician or practice named for a referral, else ""; urgency = "urgent", the time frame as said ("within 2 weeks"), or "routine". No codes. None ordered → [].
5. The chart's medication list is what the patient takes. Never write that a medication was stopped, held, discontinued, started or changed unless the transcript says so, and never summarise the medication list as reviewed or stopped in the plan.
5b. Follow the provider's rules exactly when given.
7. Pronouns and sex come from the chart's Sex field, never from the transcript or the name.
8. Medication dose, route and frequency come from the chart as written. If the transcript states a different frequency or dose (e.g. chart "twice a day", visit "three times daily"), write the chart's version and add a flag naming both.
9. The plan never says something the HPI or transcript contradicts (e.g. HPI "not concerned about supply" vs plan "difficulty obtaining refills"). If the transcript is unclear, flag it instead of choosing.
10. Turn what was said into actions: "running low", "need a refill", "almost out" on a chart medication goes in that problem's plan as "Refill requested: <med as on chart>" and, if the medication is controlled or high-risk (opioids, benzodiazepines, z-drugs, stimulants, testosterone, gabapentin, pregabalin, muscle relaxants), add a flag "PDMP check before refill". A vaccine the patient asks about or the provider plans goes in the plan with its timing as said, and in "next". A follow-up interval said in the visit goes in "next".
11. Exam: write only the findings stated in the visit, one line per body system with the system name first ("Skin: ..."), so it merges into the template's exam line for that system.
12. Coding questions go to flags, never settled by you: a chart diagnosis that is acute (e.g. acute respiratory failure) but was not addressed today; a primary diagnosis that is not the reason for today's visit; a pain code (G89.x) alongside a diagnosis that already explains the pain; a diagnosis whose timing or definition may not fit yet (e.g. post-herpetic neuralgia within weeks of the rash).
13. HPI covers the HPI elements heard: location (side/dermatome), quality, severity (score if said), duration and onset date, timing, context, modifying factors, associated symptoms and pertinent negatives. Do not end any section with a stray "." line.
14. Treatment by diagnosis (the plan): each plan item goes under its MATCHING diagnosis, matched by ICD-10 code first, then by name. Each diagnosis gets one or two plain sentences: no "Assessment:", "Medications:" or "Plan:" labels inside an item. Skip a diagnosis when the provider's clinical notes already cover it. Never restate an order already shown as a structured prescription, lab or referral line. Drop filler such as "Ongoing symptom.". Use only facts from the visit or the chart. Never write a reference range (for example "normal range") as a result. A plan item with no matching diagnosis goes in flags for the provider to review; never put it under an "N/A Other" heading and never add a diagnosis on your own.
6. Output JSON only, no prose around it, with this shape:
{"sections":{"cc":"","hpi":"","meds":"","allergies":"","pmh":"","psh":"","hosp":"","fh":"","sh":"","ros":"","exam":"","data":"","assessment":"","plan":"","next":""},
 "assessmentItems":[{"title":"","code":"","assessment":"","plan":""}],
 "orders":[{"kind":"lab|imaging|referral|procedure","name":"","reason":"","problem":"","to":"","urgency":""}],
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
    orders = []
    for o in (d.get("orders") or [])[:15]:
        if not isinstance(o, dict):
            continue
        kind = str(o.get("kind") or "").lower().strip()
        name = str(o.get("name") or "").strip()[:120]
        if kind in ("lab", "imaging", "referral", "procedure") and name:
            orders.append({"kind": kind, "name": name, "reason": str(o.get("reason") or "")[:200], "problem": str(o.get("problem") or "")[:200],
                           "to": str(o.get("to") or "")[:120], "urgency": str(o.get("urgency") or "")[:40]})
    return {"sections": sections, "assessmentItems": items, "flags": flags, "orders": orders}


# ---- Server-side checks (learned from the Freed sample note review, Nov 2026) ----
# These run on every note regardless of what the model returned, and only ever ADD flags.

_FREQ = [
    ("four times", r"\b(four times (a|per) day|four times daily|qid|4 ?x ?(a )?day)\b"),
    ("three times", r"\b(three times (a|per) day|three times daily|tid|3 ?x ?(a )?day)\b"),
    ("twice", r"\b(twice (a|per) day|twice daily|two times (a|per) day|bid|2 ?x ?(a )?day)\b"),
    ("once", r"\b(once (a|per) day|once daily|daily|qd|every day)\b"),
]
_HIGH_RISK = ["gabapentin", "pregabalin", "tramadol", "oxycodone", "hydrocodone", "morphine", "codeine", "fentanyl",
              "buprenorphine", "methadone", "zolpidem", "eszopiclone", "alprazolam", "lorazepam", "clonazepam", "diazepam",
              "adderall", "amphetamine", "methylphenidate", "testosterone", "baclofen", "carisoprodol", "cyclobenzaprine", "tizanidine"]
_REFILL = r"(refill|running low|run(ning)? out|almost out|low on|need more)"


def _freq_of(text: str) -> str:
    t = text.lower()
    for name, pat in _FREQ:
        if re.search(pat, t):
            return name
    return ""


def _checks(out: Dict[str, Any], chart: Dict[str, Any], transcript: str) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []
    have = " ".join(f.get("text", "").lower() for f in out.get("flags", []))
    sec = out.get("sections", {})
    note = " ".join([sec.get(k, "") for k in ("hpi", "plan", "assessment", "meds")] +
                    [i.get("plan", "") + " " + i.get("assessment", "") for i in out.get("assessmentItems", [])])
    nl = note.lower()

    def add(title: str, text: str) -> None:
        if text.lower()[:40] not in have:
            flags.append({"title": title, "text": text[:300]})

    # 1. pronoun vs chart sex
    sex = str(chart.get("sex") or "").strip().lower()[:1]
    he = len(re.findall(r"\b(he|his|him)\b", nl))
    she = len(re.findall(r"\b(she|her|hers)\b", nl))
    if sex == "f" and he > she:
        add("Pronoun", "Chart sex is female but the note uses he/his. Check pronouns before signing.")
    elif sex == "m" and she > he:
        add("Pronoun", "Chart sex is male but the note uses she/her. Check pronouns before signing.")

    meds = chart.get("meds") or []
    if isinstance(meds, str):
        meds = [m for m in re.split(r"[;\n]", meds) if m.strip()]
    tl = (transcript or "").lower()
    refill_talk = bool(re.search(_REFILL, tl))
    for m in meds[:60]:
        ms = str(m)
        name = (re.match(r"\s*([A-Za-z][A-Za-z\-]+)", ms) or [None, ""])[1].lower()
        if len(name) < 4:
            continue
        # 2. frequency in the note vs the chart's medication list
        chart_f = _freq_of(ms)
        for hit in re.finditer(re.escape(name), nl):
            note_f = _freq_of(nl[hit.start(): hit.start() + 80])
            if chart_f and note_f and note_f != chart_f:
                add("Medication", f"{name.title()}: chart says {chart_f} a day, note says {note_f} a day. Confirm dose and frequency before signing.")
                break
        # 3. refill heard on a controlled / high-risk med -> PDMP before refill
        if refill_talk and name in tl and any(h in name for h in _HIGH_RISK):
            add("PDMP", f"Refill talk heard for {name.title()} (controlled/high-risk). Run the PDMP check; Dave does not stage this refill.")

    # 4. acute chart diagnosis carried into today's assessment without being discussed
    for it in out.get("assessmentItems", []):
        code = (it.get("code") or "").upper()
        title = (it.get("title") or "").lower()
        if (code.startswith(("J96.0", "I21", "N17", "J96.2")) or "acute respiratory failure" in title) and not any(w in tl for w in re.findall(r"[a-z]{6,}", title) if w not in ("acute", "chronic", "unspecified", "without")):
            add("Coding", f"{it.get('title') or code} is an acute diagnosis not discussed today. Remove or confirm before billing.")

    # 5. pain code next to a diagnosis that explains the pain
    codes = [(i.get("code") or "").upper() for i in out.get("assessmentItems", [])]
    if any(c.startswith("G89") for c in codes) and any(c.startswith(("B02", "M", "G50", "G54", "G56", "G57")) for c in codes):
        add("Coding", "A G89 pain code is listed with a diagnosis that already explains the pain. Provider to confirm the G89 code is needed.")

    # 7. reference ranges are never results, and filler lines are dropped (push spec, Treatment rule)
    for k in ("plan", "assessment"):
        v = sec.get(k, "")
        if v:
            kept = [ln for ln in v.splitlines() if not re.search(r"normal range|reference range|ongoing symptom\.?\s*$", ln, re.I)]
            sec[k] = "\n".join(kept).rstrip()
    # 6. strip stray "." lines the model may leave
    for k, v in sec.items():
        if v:
            sec[k] = re.sub(r"(?m)^\s*\.\s*$\n?", "", v).rstrip()
    return flags


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
            model=MODEL, max_tokens=MAX_TOKENS, temperature=0, system=SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(p, "text", "") for p in msg.content)
    except Exception as e:  # never echo the provider's message with PHI in it
        return JSONResponse({"error": "model call failed", "kind": type(e).__name__}, status_code=502)

    parsed = _extract_json(text)
    if not parsed:
        return JSONResponse({"error": "model returned no JSON"}, status_code=502)

    out = _clean(parsed, wanted)
    out["flags"].extend(_checks(out, body.chart or {}, body.transcript))
    out["flags"] = out["flags"][:25]
    out.update({"template_id": body.template_id, "model": MODEL, "engine": "render-note"})
    return out

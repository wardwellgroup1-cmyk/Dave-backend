"""
Dave Backend — FastAPI application.

Endpoints:
  GET  /api/health             — liveness probe (frontend uses this to
                                  decide whether to call /api/analyze)
  POST /api/analyze            — Claude-powered note analysis (optional)
  GET  /api/vim/launch         — Vim OAuth authorize entry (Vim calls this)
  POST /api/vim/callback       — Vim token exchange (SDK POSTs code here)
  POST /api/vim/launch-log     — operational telemetry beacon
"""

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vim_auth import router as vim_router

# ---- Logging (no PHI ever) --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("dave")

# ---- App --------------------------------------------------------------------
app = FastAPI(
    title="Dave Coding Intelligence Backend",
    version="6.5",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# ---- CORS ------------------------------------------------------------------
# Explicit origins from env, plus regex allowance for all Vim subdomains
# (per Vim spec — the SDK calls our token endpoint from *.getvim.com).
_cors_env = os.getenv("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
if not _cors_origins:
    fo = os.getenv("FRONTEND_ORIGIN", "").strip()
    if fo:
        _cors_origins = [fo]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or [],
    allow_origin_regex=r"https://.*\.getvim\.com$",  # Vim SDK cross-origin calls
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)

# ---- Vim OAuth routes -------------------------------------------------------
app.include_router(vim_router)


# ---- Health -----------------------------------------------------------------
@app.get("/api/health")
async def health():
    """Non-PHI liveness probe. The frontend uses this to decide whether to
    call /api/analyze or fall back to the local rules engine."""
    return {
        "status":         "ok",
        "version":        "6.5",
        "claude":         bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
        "vim_configured": bool(os.getenv("VIM_CLIENT_ID", "").strip()
                               and os.getenv("VIM_CLIENT_SECRET", "").strip()),
    }


# ---- Analyze ----------------------------------------------------------------
class AnalyzeBody(BaseModel):
    note: str
    age_band: Optional[str] = None
    coded_diagnoses: Optional[list] = None


@app.post("/api/analyze")
async def analyze(body: AnalyzeBody):
    """AI-powered analysis using Claude. Frontend falls back to local rules
    if this endpoint returns anything non-200."""
    if not body.note or not body.note.strip():
        raise HTTPException(400, "note is required")
    if len(body.note) > 50_000:
        raise HTTPException(413, "note too large (max 50,000 chars)")

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(501, "AI backend not configured — using local rules engine")

    try:
        import anthropic
    except ImportError:
        raise HTTPException(501, "anthropic SDK not installed")

    client = anthropic.Anthropic(api_key=api_key)
    user_content = _build_user_message(body.note, body.age_band, body.coded_diagnoses)

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception as e:
        log.warning("Anthropic call failed: %s", type(e).__name__)
        raise HTTPException(502, "AI analysis failed — frontend should fall back")

    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    import json
    try:
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
    except Exception:
        raise HTTPException(502, "AI returned unparseable output")

    if not isinstance(parsed, dict) or "icd" not in parsed:
        raise HTTPException(502, "AI output missing required fields")

    return JSONResponse(parsed)


# ---- Claude prompt scaffolding ----------------------------------------------
_SYSTEM_PROMPT = """You are Dave, a coding intelligence agent for primary care.

You analyze a progress note and return STRICT JSON matching this shape:

{
  "icd": [
    {"code": "E11.22", "desc": "...", "kind": "upgrade|captured|flag",
     "hcc": "HCC 37", "raf": 0.302, "note": "..."}
  ],
  "em": {"level": "99214", "basis": "...", "value": 132, "baseline": "99213",
         "baseValue": 93, "tcm": false, "timeNote": null},
  "programs": [{"code": "G2211", "desc": "...", "note": "..."}],
  "q": [{"code": "1111F", "desc": "...", "note": "..."}],
  "v": {"A1c": "8.4%", "eGFR": "52", "BP": "146/88", "BMI": "34.2"},
  "rafGain": 0.302,
  "emGain": 39,
  "payerFindings": []
}

Rules:
- Every finding must cite what's actually documented in the note.
- Never invent findings the chart doesn't support.
- E/M leveling: use 2021 AMA MDM + time-based override.
- TCM 99495/99496 are PRIMARY post-discharge codes — never route to 99215
  when ToC requirements are met.
- If nothing to report in a category, return an empty array.
- Return JSON only, no prose, no markdown fences.
"""


def _build_user_message(note: str, age_band: Optional[str], coded: Optional[list]) -> str:
    parts = ["Analyze this progress note:\n\n", note]
    if age_band:
        parts.append(f"\n\nPatient age band: {age_band}")
    if coded:
        parts.append(f"\n\nAlready coded on encounter: {', '.join(str(c) for c in coded)}")
    parts.append("\n\nReturn the JSON now.")
    return "".join(parts)


# ---- Entrypoint -------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

"""
Dave Backend — NyCal.ai
One FastAPI app, three jobs:
  1. /api/vim/launch  — Vim Canvas OAuth step 1 (redirect to Vim authorize)
  2. /api/vim/token   — Vim Canvas OAuth step 2 (code -> token exchange; secret stays server-side)
  3. /api/analyze     — Claude-powered coding intelligence

Run locally:  uvicorn main:app --reload --port 8788
Deploy:       any US-hosted platform (Render / Railway / Fly.io / AWS).
              Vim requires the app server to be US-hosted.

PHI WARNING: encounter notes are PHI. Before sending real patient notes
through this endpoint, execute a BAA with Anthropic (available for the
Claude API — contact Anthropic sales) and confirm your hosting platform
BAA. Until then, test with de-identified notes only.
"""

import json
import logging
import os
from urllib.parse import urlencode

import httpx
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from coding_prompt import CODING_SYSTEM_PROMPT

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dave")

# ---------------------------------------------------------------- config
VIM_CLIENT_ID = os.getenv("VIM_CLIENT_ID", "")
VIM_CLIENT_SECRET = os.getenv("VIM_CLIENT_SECRET", "")
# Your app's frontend URL — must match the redirect URI registered in the Vim Console manifest
VIM_REDIRECT_URI = os.getenv("VIM_REDIRECT_URI", "https://your-app.example.com")
# Verify these two against the implementation guide in your Vim Console —
# they follow Vim's documented OAuth flow (api.getvim.com/v1/oauth/*)
VIM_AUTHORIZE_URL = os.getenv("VIM_AUTHORIZE_URL", "https://api.getvim.com/v1/oauth/authorize")
VIM_TOKEN_URL = os.getenv("VIM_TOKEN_URL", "https://api.getvim.com/v1/oauth/token")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

app = FastAPI(title="Dave Backend — NyCal.ai", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


# ---------------------------------------------------------------- health
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "claude": bool(anthropic_client),
        "vim_configured": bool(VIM_CLIENT_ID and VIM_CLIENT_SECRET),
    }


# ---------------------------------------------------------------- Vim OAuth
@app.get("/api/vim/launch")
async def vim_launch(launch_id: str, vim_organization_id: str | None = None, ehr_url: str | None = None):
    """
    Step 1 of Vim's auth flow. VimOS calls this with a launch_id when it
    injects your iframe. We redirect to Vim's authorize endpoint, which
    issues an authorization code back to the redirect_uri (your frontend).
    vim_organization_id / ehr_url are available for multi-tenant routing
    or EHR allow/deny lists if you need them later.
    """
    if not VIM_CLIENT_ID:
        raise HTTPException(500, "VIM_CLIENT_ID not configured")
    params = urlencode({
        "launch_id": launch_id,
        "client_id": VIM_CLIENT_ID,
        "redirect_uri": VIM_REDIRECT_URI,
        "response_type": "code",
    })
    log.info("Vim launch: org=%s", vim_organization_id)
    return RedirectResponse(f"{VIM_AUTHORIZE_URL}?{params}")


class TokenRequest(BaseModel):
    code: str


@app.post("/api/vim/token")
async def vim_token(body: TokenRequest):
    """
    Step 2. The VimOS SDK (frontend) sends us the authorization code;
    we exchange it — WITH the client secret, which never leaves this
    server — for the access token + id token the SDK needs.
    """
    if not (VIM_CLIENT_ID and VIM_CLIENT_SECRET):
        raise HTTPException(500, "Vim credentials not configured")
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            VIM_TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "code": body.code,
                "client_id": VIM_CLIENT_ID,
                "client_secret": VIM_CLIENT_SECRET,
                "redirect_uri": VIM_REDIRECT_URI,
            },
        )
    if resp.status_code != 200:
        log.error("Vim token exchange failed: %s %s", resp.status_code, resp.text[:300])
        raise HTTPException(resp.status_code, "Vim token exchange failed")
    return JSONResponse(resp.json())


# ---------------------------------------------------------------- Claude analyze
class AnalyzeRequest(BaseModel):
    note: str = Field(..., min_length=10, max_length=50_000)
    # Optional structured context from the Vim EHR subscription —
    # existing coded diagnoses, patient age band, etc. (avoid direct identifiers)
    coded_diagnoses: list[str] | None = None
    payer_type: str | None = None  # e.g. "Medicare Advantage"


EMPTY_ANALYSIS = {
    "extractedValues": [], "icdRecommendations": [], "emRecommendation": None,
    "cptII": [], "carePrograms": [], "negationsNoted": [],
    "documentationGaps": [], "complianceNote": "Provider confirmation required for all codes.",
}


@app.post("/api/analyze")
async def analyze(body: AnalyzeRequest):
    if not anthropic_client:
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured")

    user_content = f"Analyze this encounter note:\n\n<note>\n{body.note}\n</note>"
    if body.coded_diagnoses:
        user_content += f"\n\nDiagnoses the provider has already coded: {', '.join(body.coded_diagnoses)}"
    if body.payer_type:
        user_content += f"\nPayer type: {body.payer_type}"

    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            temperature=0,  # deterministic coding recommendations
            system=CODING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception as e:  # network/auth/ratelimit
        log.exception("Claude API call failed")
        raise HTTPException(502, f"Analysis service unavailable: {type(e).__name__}")

    raw = "".join(block.text for block in msg.content if block.type == "text").strip()
    # Strip accidental markdown fences, then parse
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.error("Claude returned non-JSON: %s", raw[:300])
        return JSONResponse({**EMPTY_ANALYSIS, "documentationGaps": ["Analysis engine returned an unreadable response — try again."]}, status_code=200)

    # Fill any missing keys so the frontend never breaks
    return JSONResponse({**EMPTY_ANALYSIS, **data})

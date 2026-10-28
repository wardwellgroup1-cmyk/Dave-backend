"""
api_listen.py — Dave Listen phone-mic token (moved here from the never-deployed Vercel listen-token.js).

POST /api/listen-token  ->  { token, expires_in_seconds: 60 }
Hands the Dave extension a 60-second AssemblyAI streaming token. Never sees audio or
transcript text; the AssemblyAI key never leaves Render.

Render -> dave-backend-1 -> Environment:
  ASSEMBLYAI_API_KEY      your AssemblyAI key
  LISTEN_ALLOWED_ORIGINS  chrome-extension://<extension id>[,chrome-extension://<another id>]
"""
from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

_NO_STORE = {"Cache-Control": "no-store"}


def _allowed() -> list[str]:
    return [o.strip() for o in os.getenv("LISTEN_ALLOWED_ORIGINS", "").split(",") if o.strip()]


@router.post("/api/listen-token")
async def listen_token(request: Request):
    origin = request.headers.get("origin", "")
    allowed = _allowed()
    if not allowed or origin not in allowed:
        return JSONResponse({"error": "origin not allowed"}, status_code=403, headers=_NO_STORE)
    key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
    if not key:
        return JSONResponse({"error": "not configured"}, status_code=500, headers=_NO_STORE)
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://streaming.assemblyai.com/v3/token",
                params={"expires_in_seconds": 60, "max_session_duration_seconds": 3600},
                headers={"Authorization": key},
            )
        if r.status_code != 200:
            return JSONResponse({"error": "token service unavailable"}, status_code=502, headers=_NO_STORE)
        return JSONResponse({"token": r.json().get("token"), "expires_in_seconds": 60}, headers=_NO_STORE)
    except Exception:
        return JSONResponse({"error": "token service unavailable"}, status_code=502, headers=_NO_STORE)

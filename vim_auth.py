"""
Vim Canvas OAuth2 + OIDC flow — server-side glue.
==================================================

Implements Vim's official spec:
https://docs.getvim.com/vim-os-js/authentication/step-by-step-implementation

Two endpoints:

1. Launch endpoint (GET /api/vim/launch)
   VimOS calls this with launch_id. We redirect to Vim's authorize endpoint,
   which bounces the browser back to the frontend URL with a `code` query param.

2. Token endpoint (POST /api/vim/callback)
   The VimOS SDK POSTs {code} here. We exchange it with Vim's token endpoint,
   verify the returned ID token JWT signature/issuer/audience against Vim's
   JWKS, and return {access_token, id_token} to the SDK as JSON.

Env vars required on Render:
  VIM_CLIENT_ID
  VIM_CLIENT_SECRET
  FRONTEND_ORIGIN       = https://clinisys-bill-ai-vim.vercel.app

Env vars optional (defaults shown):
  VIM_AUTHORIZE_URL     = https://api.getvim.com/v1/oauth/authorize
  VIM_TOKEN_URL         = https://api.getvim.com/v1/oauth/token
  VIM_JWKS_URL          = https://auth.getvim.com/.well-known/jwks.json
  VIM_ISSUER            = https://auth.getvim.com/
"""

import os
import logging
from urllib.parse import urlencode
from typing import Optional

import httpx
import jwt
from jwt import PyJWKClient
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel

log = logging.getLogger("vim_auth")
router = APIRouter(prefix="/api/vim", tags=["vim"])

# --- Configuration ----------------------------------------------------------
VIM_CLIENT_ID     = os.getenv("VIM_CLIENT_ID", "")
VIM_CLIENT_SECRET = os.getenv("VIM_CLIENT_SECRET", "")
VIM_AUTHORIZE_URL = os.getenv("VIM_AUTHORIZE_URL", "https://api.getvim.com/v1/oauth/authorize")
VIM_TOKEN_URL     = os.getenv("VIM_TOKEN_URL",     "https://api.getvim.com/v1/oauth/token")
VIM_JWKS_URL      = os.getenv("VIM_JWKS_URL",      "https://auth.getvim.com/.well-known/jwks.json")
VIM_ISSUER        = os.getenv("VIM_ISSUER",        "https://auth.getvim.com/")
FRONTEND_ORIGIN   = os.getenv("FRONTEND_ORIGIN",   "")

# JWKS client — PyJWKClient auto-fetches Vim's public keys and caches them.
_jwks_client = PyJWKClient(VIM_JWKS_URL, cache_keys=True, lifespan=3600)


# --- 1. Launch endpoint -----------------------------------------------------
@router.get("/launch")
async def vim_launch(
    launch_id: str = Query(..., min_length=1),
    vim_organization_id: Optional[str] = Query(None),
    ehr_url: Optional[str] = Query(None),
):
    """
    Step 1 of the OIDC flow. VimOS's frontend layer hits this endpoint with a
    launch_id when the iframe is injected into an EHR. We construct the
    authorize URL and 302-redirect the browser there.

    Vim then handles user authentication and redirects the browser to
    FRONTEND_ORIGIN with `?code=...` in the query string. The SDK on the
    frontend extracts the code and POSTs it to /api/vim/callback below.
    """
    if not VIM_CLIENT_ID:
        raise HTTPException(500, "VIM_CLIENT_ID not configured on backend")
    if not FRONTEND_ORIGIN:
        raise HTTPException(500, "FRONTEND_ORIGIN not configured on backend")

    params = urlencode({
        "launch_id":     launch_id,
        "client_id":     VIM_CLIENT_ID,
        # Per Vim spec: redirect_uri must match the app manifest base URL,
        # which is the frontend iframe origin — NOT the backend callback.
        "redirect_uri":  FRONTEND_ORIGIN,
        "response_type": "code",
    })
    log.info("Vim launch: org=%s ehr_url_present=%s",
             vim_organization_id, bool(ehr_url))
    return RedirectResponse(f"{VIM_AUTHORIZE_URL}?{params}", status_code=302)


# --- 2. Token endpoint ------------------------------------------------------
class TokenExchangeBody(BaseModel):
    code: str


@router.post("/callback")
async def vim_token(body: TokenExchangeBody):
    """
    Step 2 of the OIDC flow. The VimOS SDK POSTs the authorization code here.

    We:
      1. POST code + client credentials to Vim's token endpoint
      2. Receive {access_token, id_token} back
      3. Verify the id_token JWT signature against Vim's JWKS
      4. Verify issuer + audience claims
      5. Apply application-level authorization
      6. Return the tokens to the SDK (or 403 if not authorized)
    """
    if not (VIM_CLIENT_ID and VIM_CLIENT_SECRET):
        raise HTTPException(500, "Vim credentials not configured on backend")

    # --- Exchange authorization code for tokens ---
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                VIM_TOKEN_URL,
                json={
                    "grant_type":    "authorization_code",
                    "code":          body.code,
                    "client_id":     VIM_CLIENT_ID,
                    "client_secret": VIM_CLIENT_SECRET,
                },
                headers={"Content-Type": "application/json"},
            )
        except httpx.HTTPError as e:
            log.error("Vim token endpoint unreachable: %s", type(e).__name__)
            raise HTTPException(502, "Vim token endpoint unreachable")

    if resp.status_code != 200:
        log.error("Vim token exchange failed: status=%s body=%s",
                  resp.status_code, resp.text[:500])
        raise HTTPException(
            status_code=502,
            detail=f"Vim token exchange returned {resp.status_code}",
        )

    tokens = resp.json()
    id_token     = tokens.get("id_token")
    access_token = tokens.get("access_token")

    if not id_token or not access_token:
        log.error("Vim token response missing tokens: keys=%s", list(tokens.keys()))
        raise HTTPException(502, "Malformed token response from Vim")

    # --- Verify ID token per Vim spec ---
    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(id_token)
        claims = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=VIM_CLIENT_ID,
            issuer=VIM_ISSUER,
            options={"verify_exp": True, "verify_iat": True},
        )
    except jwt.ExpiredSignatureError:
        log.warning("ID token expired")
        raise HTTPException(401, "ID token expired")
    except jwt.InvalidAudienceError:
        log.warning("ID token audience mismatch")
        raise HTTPException(401, "ID token audience mismatch")
    except jwt.InvalidIssuerError:
        log.warning("ID token issuer mismatch")
        raise HTTPException(401, "ID token issuer mismatch")
    except jwt.InvalidSignatureError:
        log.warning("ID token signature invalid")
        raise HTTPException(401, "ID token signature invalid")
    except Exception as e:
        log.error("ID token verification failed: %s: %s",
                  type(e).__name__, str(e)[:200])
        raise HTTPException(401, "ID token verification failed")

    # --- Application-level authorization ---
    if not _is_user_authorized(claims):
        log.info("User not authorized: sub=%s email=%s",
                 claims.get("sub"), claims.get("email"))
        raise HTTPException(403, "User not authorized for this application")

    # --- Success — return tokens to the SDK ---
    log.info("Vim auth success: sub=%s email=%s billing=%s trial_days=%s",
             claims.get("sub"),
             claims.get("email"),
             claims.get("application_billing_plan"),
             claims.get("days_until_end_of_trial"))
    return JSONResponse({
        "access_token": access_token,
        "id_token":     id_token,
    })


def _is_user_authorized(claims: dict) -> bool:
    """
    Application-level authorization check.

    Sandbox + early UAT: accept every Vim-verified user (Vim already
    authenticated them upstream). Post-launch, extend this to query the
    NyCal.ai user database and enforce practice-level access rules.

    Also enforces Vim's free-trial expiration signal from the ID token
    (per Vim spec — returning authorized after trial expires while our
    activation status is Enabled results in Vim billing us for the user).
    """
    if claims.get("application_billing_plan") == "free_trial":
        if claims.get("is_free_trial") is False:
            log.info("Free trial expired for user %s", claims.get("sub"))
            return False
    # TODO post-launch: check against NyCal.ai user database here.
    return True


# --- 3. Operational telemetry (frontend beacon on SDK init) -----------------
class LaunchLogBody(BaseModel):
    ehrOrigin: Optional[str] = None
    orgId: Optional[str] = None
    ts: Optional[int] = None


@router.post("/launch-log")
async def vim_launch_log(body: LaunchLogBody):
    """
    Non-PHI operational telemetry. The frontend fires a keepalive beacon on
    SDK init so we can track which EHR domains launch us in production
    (per the CSP `frame-ancestors *` reactive-monitoring pattern
    recommended by Vim's dev-platform PM).
    """
    log.info("Vim launch origin: ehr=%s org=%s ts=%s",
             body.ehrOrigin, body.orgId, body.ts)
    return {"logged": True}


@router.get("/launch-log")
async def vim_launch_log_get():
    """Placeholder GET so you can eyeball the endpoint in a browser."""
    return {"ok": True, "expects": "POST with {ehrOrigin, orgId, ts}"}

# Dave Backend — NyCal.ai

One FastAPI app: the Claude-powered coding intelligence endpoint (/api/analyze),
the Dave Super Listen 2 note writer (/api/note) and phone-mic token (/api/listen-token).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in your keys
uvicorn main:app --reload --port 8788
```

Test the analyzer:

```bash
curl -s localhost:8788/api/analyze -H 'Content-Type: application/json' -d '{
  "note": "MA pt 71M. Coded E11.9 and I10. A1c 9.4, eGFR 48, creatinine 1.6. BP 152/94 home avg. Denies neuropathy. BMI 41.2. Lisinopril titrated. 32 minutes total.",
  "coded_diagnoses": ["E11.9", "I10"],
  "payer_type": "Medicare Advantage"
}' | python -m json.tool
```

Watch for: E11.22 + N18.31 recommended, E11.42 **NOT** recommended
("denies neuropathy" honored in `negationsNoted`).

## Wire up the frontend

In `clinisys-billai-v6-demo.html`, set:

```js
const BACKEND_URL = "https://your-backend.example.com";
```

The frontend calls `/api/analyze` and falls back to its built-in local
engine automatically if the backend is unreachable.

## Deploy

Any US-hosted platform: Render, Railway,
Fly.io, AWS. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## PHI / compliance

Encounter notes are PHI. **Before real patient data flows through
`/api/analyze`:** execute a BAA with Anthropic for the Claude API
(contact Anthropic sales), confirm your hosting platform BAA, and keep
`temperature=0` + no logging of note bodies (already configured). Test
with de-identified notes until then.

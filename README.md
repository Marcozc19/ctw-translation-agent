# CTW Multi-Language Translation Agent

> AI Product Manager Technical Assessment — Marco, June 2026

A chat-based translation agent that accepts a CSV of Chinese content and returns a fully translated output in one or more target languages, powered by a four-agent pipeline.

## Demo

| Layer | Service |
|-------|---------|
| Frontend | Vercel |
| Backend | Railway |

## Architecture

```
User (chat UI)
     │
     ▼
FastAPI backend
     │
     ├── Agent 1: Column Identifier (rule-based CJK detection)
     │
     ├── Agent 2: Low-Cost Translator (DeepSeek V3)
     │       ↓
     ├── Agent 3: Evaluator (Gemini 2.5 Flash, LLM-as-judge)
     │       ↓ (if score < 0.75)
     └── Agent 4: High-Cost Translator (Claude Haiku 4.5)
                 ↓ (second eval pass)
              done | review
```

Rows are processed in **batches of 10**, up to **5 concurrent batches**. Each row tracks its own state: `pending → translating → evaluating → done | escalating → re-evaluating → done | review`.

### Confidence scoring

| Score | Result |
|-------|--------|
| ≥ 0.75 | `high` — accepted |
| 0.55–0.74 | `low` — escalated to Haiku |
| < 0.55 | `review` — hard-flagged, best attempt kept |

**Gemini** scores each translation directly against the original Chinese (meaning, completeness, tone/register) — a different model family from the translators, which avoids self-grading bias.

## Local Development

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in DEEPSEEK_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY

uvicorn app.main:app --reload
# → http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

Vite proxies `/api/*` → `http://localhost:8000` in dev, so no CORS config needed locally.

## Deployment

### Backend → Railway

1. Connect this repo in Railway, set root to `/backend`
2. Add env vars: `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`
3. Railway auto-detects `railway.toml` and runs uvicorn

### Frontend → Vercel

1. Connect this repo in Vercel (root-level `vercel.json` already configured)
2. Add env var: `VITE_API_URL=https://your-railway-app.railway.app`
3. Deploy

## API Keys Required

| Key | Where to get |
|-----|-------------|
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |

## Cost Estimate (100-row CSV)

| Scenario | Estimate |
|----------|----------|
| Best case (100% DeepSeek pass) | ~$0.003 |
| Typical (25% escalation) | ~$0.02 |
| Worst case (100% escalation) | ~$0.07 |

## Key Design Decisions

- **DeepSeek V3** for first-pass: best Chinese→X benchmark performance per dollar
- **Gemini as evaluator**: different model family prevents self-grading bias
- **Direct LLM scoring over back-translation + embeddings**: a single Gemini call judges meaning/completeness/tone directly, which is faster and avoids the heavy `sentence-transformers`/`torch` dependency (~1-2GB) entirely
- **Hard cap at 2 escalations**: no infinite loops; flagged rows get best attempt
- **Polling over SSE**: simpler infra, sufficient for <60s jobs

import io
import uuid
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load API keys from backend/.env into the process environment before any
# agent module reads os.getenv(...) for its API clients.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd

from .agents import detect_chinese_columns, chat_agent, SUPPORTED_LANGUAGES
from .orchestrator import run_translation_job

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="CTW Translation Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# In-memory stores (stateless demo — cleared on restart)
sessions: dict = {}
jobs: dict = {}


# ── Health ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Languages ─────────────────────────────────────────────────────────────

@app.get("/languages")
async def list_languages():
    return SUPPORTED_LANGUAGES


# ── Chat ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    phase: str
    target_languages: list[str] = []


@app.post("/chat")
async def chat(req: ChatRequest):
    session = sessions.get(req.session_id) if req.session_id else None
    context = {
        "phase": req.phase,
        "filename": session["filename"] if session else None,
        "detected_columns": session["detected_columns"] if session else [],
        "row_count": len(session["df"]) if session else 0,
        "current_target_languages": req.target_languages,
    }
    return await chat_agent(req.message, context)


# ── Upload ────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    raw = await file.read()

    df = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312", "big5"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except Exception:
            continue

    if df is None:
        raise HTTPException(400, "Could not parse CSV. Please check the file encoding.")
    if df.empty:
        raise HTTPException(400, "CSV file is empty.")

    detected = detect_chinese_columns(df)
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "df": df,
        "detected_columns": detected,
        "filename": file.filename,
    }

    return {
        "session_id": session_id,
        "detected_columns": detected,
        "row_count": len(df),
        "all_columns": list(df.columns),
    }


# ── Translate ─────────────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    session_id: str
    target_languages: list[str]
    confirmed_columns: list[str]


@app.post("/translate")
async def start_translation(req: TranslateRequest, background_tasks: BackgroundTasks):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found — please re-upload your CSV.")
    if not req.target_languages:
        raise HTTPException(400, "No target languages specified.")
    if not req.confirmed_columns:
        raise HTTPException(400, "No columns selected for translation.")

    session = sessions[req.session_id]
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "processed": 0,
        "total": len(session["df"]),
        "flagged": 0,
        "batch_processed": 0,
        "batch_total": None,
        "result_df": None,
        "error": None,
    }

    background_tasks.add_task(
        run_translation_job,
        job_id,
        jobs,
        session["df"].copy(),
        req.confirmed_columns,
        req.target_languages,
    )

    return {"job_id": job_id}


# ── Status ────────────────────────────────────────────────────────────────

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")
    j = jobs[job_id]
    return {
        "status": j["status"],
        "processed": j["processed"],
        "total": j["total"],
        "flagged": j["flagged"],
        "batch_processed": j["batch_processed"],
        "batch_total": j["batch_total"],
        "error": j["error"],
    }


# ── Download ──────────────────────────────────────────────────────────────

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")
    j = jobs[job_id]
    if j["status"] != "completed":
        raise HTTPException(400, f"Job not completed (status: {j['status']}).")
    if j["result_df"] is None:
        raise HTTPException(500, "Result unavailable.")

    buf = io.StringIO()
    j["result_df"].to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8-sig")

    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="translated_{job_id[:8]}.csv"'
        },
    )

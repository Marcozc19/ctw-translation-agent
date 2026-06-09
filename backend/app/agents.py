"""
Four-agent translation pipeline.

Agent 1: detect_chinese_columns   — Rule-based CJK column identification
Agent 2: translate_batch_deepseek — Low-cost first-pass (DeepSeek V3)
Agent 3: evaluate_translation     — Quality eval (Gemini + sentence-transformers)
Agent 4: translate_batch_haiku    — High-quality fallback (Claude Haiku 4.5)
"""
import os
import json
import re
import asyncio
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy singletons — loaded on first use to avoid slowing startup
_embedding_model = None
_gemini_model = None
_deepseek_client = None
_claude_client = None


def _get_deepseek():
    global _deepseek_client
    if _deepseek_client is None:
        from openai import AsyncOpenAI
        _deepseek_client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )
    return _deepseek_client


def _get_claude():
    global _claude_client
    if _claude_client is None:
        from anthropic import AsyncAnthropic
        _claude_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    return _claude_client


def _get_gemini():
    global _gemini_model
    if _gemini_model is None:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
        _gemini_model = genai.GenerativeModel(
            os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        )
    return _gemini_model


def _get_embedder():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _embedding_model


# ---------------------------------------------------------------------------
# Agent 1 — Column Identifier
# ---------------------------------------------------------------------------

CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")


def detect_chinese_columns(df: pd.DataFrame) -> list[str]:
    """Return column names where ≥30 % of sampled values contain CJK chars."""
    result = []
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(20)
        if len(sample) == 0:
            continue
        hits = sum(1 for t in sample if CJK_RE.search(t))
        if hits / len(sample) >= 0.3:
            result.append(col)
    return result


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> list | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def _lang_name(code: str) -> str:
    names = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "ja": "Japanese", "ko": "Korean", "pt": "Portuguese", "vi": "Vietnamese",
        "th": "Thai", "id": "Indonesian", "ar": "Arabic", "hi": "Hindi",
        "ru": "Russian", "it": "Italian", "nl": "Dutch",
    }
    return names.get(code, code)


def _output_keys(source_cols: list[str], target_langs: list[str]) -> list[str]:
    return [f"{col}_{lang}" for col in source_cols for lang in target_langs]


# ---------------------------------------------------------------------------
# Agent 2 — Low-Cost Translator (DeepSeek V3)
# ---------------------------------------------------------------------------

async def translate_batch_deepseek(
    batch: list[dict],
    source_cols: list[str],
    target_langs: list[str],
) -> list[dict] | None:
    lang_list = ", ".join(_lang_name(l) for l in target_langs)
    out_keys = _output_keys(source_cols, target_langs)

    system = (
        "You are a professional translator specialising in Chinese. "
        "Your output must be ONLY a valid JSON array — no commentary, no markdown."
    )
    user = (
        f"Translate the Chinese fields to {lang_list}.\n"
        f"Return a JSON array. Each object must have 'id' (unchanged) plus: {json.dumps(out_keys)}.\n"
        f"Input rows:\n{json.dumps(batch)}"
    )

    for attempt in range(2):
        try:
            resp = await _get_deepseek().chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            result = _parse_json(resp.choices[0].message.content)
            if result:
                return result
        except Exception as exc:
            logger.warning("DeepSeek attempt %d failed: %s", attempt + 1, exc)
            if attempt == 0:
                await asyncio.sleep(1)

    return None


# ---------------------------------------------------------------------------
# Agent 3 — Evaluator (Gemini 2.0 Flash + sentence-transformers)
# ---------------------------------------------------------------------------

async def back_translate_gemini(text: str, target_lang: str) -> str | None:
    """Translate `text` (in target_lang) back to Simplified Chinese."""
    prompt = (
        f"Translate the following {_lang_name(target_lang)} text back to Simplified Chinese. "
        f"Return ONLY the Chinese translation, nothing else.\n\nText: {text}"
    )
    try:
        model = await asyncio.to_thread(_get_gemini)
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.strip()
    except Exception as exc:
        logger.warning("Gemini back-translation failed: %s", exc)
        return None


def compute_similarity(a: str, b: str) -> float:
    """Cosine similarity between two strings using multilingual embeddings."""
    try:
        from sentence_transformers import util
        model = _get_embedder()
        embs = model.encode([a, b])
        return float(util.cos_sim(embs[0], embs[1]))
    except Exception as exc:
        logger.warning("Similarity failed: %s", exc)
        return 0.6  # neutral fallback — won't force escalation


async def evaluate_translation(original_zh: str, translated: str, target_lang: str) -> tuple[float, bool]:
    """
    Returns (score, hard_flagged).
    score >= 0.75  → high confidence (pass)
    0.55–0.74      → low confidence (escalate)
    < 0.55         → hard flag + escalate
    """
    back = await back_translate_gemini(translated, target_lang)
    if back is None:
        return 0.65, False  # can't evaluate; give benefit of the doubt

    score = await asyncio.to_thread(compute_similarity, original_zh, back)
    return score, score < 0.55


# ---------------------------------------------------------------------------
# Agent 4 — High-Cost Translator (Claude Haiku 4.5)
# ---------------------------------------------------------------------------

async def translate_batch_haiku(
    batch: list[dict],
    source_cols: list[str],
    target_langs: list[str],
    prev_translations: dict,  # row_id -> {col_lang: text}
) -> list[dict] | None:
    lang_list = ", ".join(_lang_name(l) for l in target_langs)
    out_keys = _output_keys(source_cols, target_langs)

    enriched = []
    for row in batch:
        r = dict(row)
        if row["id"] in prev_translations:
            r["_previous_attempt"] = prev_translations[row["id"]]
        enriched.append(r)

    system = "You are an expert translator. Return ONLY a valid JSON array — no commentary, no markdown."
    user = (
        f"These rows failed quality evaluation. Provide higher-quality translations to {lang_list}.\n"
        f"Output keys per object: 'id' + {json.dumps(out_keys)}.\n"
        f"'_previous_attempt' shows what failed — improve upon it.\n"
        f"Rows:\n{json.dumps(enriched)}"
    )

    try:
        resp = await _get_claude().messages.create(
            model=os.getenv("HAIKU_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return _parse_json(resp.content[0].text)
    except Exception as exc:
        logger.error("Haiku translation failed: %s", exc)
        return None

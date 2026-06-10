"""
Four-agent translation pipeline.

Agent 1: detect_chinese_columns   — Rule-based CJK column identification
Agent 2: translate_batch_deepseek — Low-cost first-pass (DeepSeek V3)
Agent 3: evaluate_translation     — Quality eval (Gemini, LLM-as-judge)
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
        from google import genai
        _gemini_model = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))
    return _gemini_model


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

def _parse_json(text: str):
    """Parse a JSON array or object out of an LLM response, tolerating
    markdown code fences and surrounding commentary."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


# Languages the pipeline knows how to translate into. Shared with the chat
# agent (for free-text language matching) and exposed to the frontend via
# GET /languages so the UI doesn't need its own hardcoded copy.
SUPPORTED_LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "ja": "Japanese", "ko": "Korean", "pt": "Portuguese", "vi": "Vietnamese",
    "th": "Thai", "id": "Indonesian", "ar": "Arabic", "hi": "Hindi",
    "ru": "Russian", "it": "Italian", "nl": "Dutch", "ms": "Malay",
    "tr": "Turkish", "pl": "Polish", "sv": "Swedish", "zh-TW": "Traditional Chinese",
}


def _lang_name(code: str) -> str:
    return SUPPORTED_LANGUAGES.get(code, code)


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
        f"Some rows include an optional '_context' field with background notes "
        f"about the row (e.g. its purpose, audience, or tone). Use it only to "
        f"inform word choice, tone, and register — do NOT translate it and do "
        f"NOT include '_context' or '_idx' in your output.\n"
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
# Agent 3 — Evaluator (Gemini 2.5 Flash, LLM-as-judge)
# ---------------------------------------------------------------------------

async def evaluate_translation(original_zh: str, translated: str, target_lang: str) -> tuple[float, bool]:
    """
    Ask Gemini to directly judge translation quality (meaning, completeness,
    tone/register) on a 0.0-1.0 scale, instead of back-translating and
    comparing embeddings. A different model family than the translators
    still avoids self-grading bias, without the extra round-trip or the
    sentence-transformers/torch dependency.

    Returns (score, hard_flagged).
    score >= 0.75  → high confidence (pass)
    0.55–0.74      → low confidence (escalate)
    < 0.55         → hard flag + escalate
    """
    prompt = (
        "You are a translation quality judge. Compare the ORIGINAL Simplified "
        "Chinese text with its TRANSLATION and score how faithfully the "
        "translation preserves the original's meaning, completeness, and "
        "tone/register.\n\n"
        f"ORIGINAL (Simplified Chinese): {original_zh}\n"
        f"TRANSLATION ({_lang_name(target_lang)}): {translated}\n\n"
        "Score from 0.0 (completely wrong or missing) to 1.0 (fully accurate "
        "and natural). Set \"flagged\" to true only for serious errors: "
        "mistranslation, missing content, or nonsensical output.\n\n"
        "Respond with ONLY a JSON object, no markdown, no explanation:\n"
        '{"score": <number 0.0-1.0>, "flagged": <true|false>}'
    )
    try:
        from google.genai import types

        client = _get_gemini()
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        result = _parse_json(response.text)
        if result is None:
            raise ValueError(f"Could not parse evaluator response: {response.text!r}")

        score = float(result.get("score", 0.65))
        score = max(0.0, min(1.0, score))
        flagged = bool(result.get("flagged", False)) or score < 0.55
        return score, flagged
    except Exception as exc:
        logger.warning("Gemini evaluation failed: %s", exc)
        return 0.65, False  # can't evaluate; give benefit of the doubt


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
        f"Some rows include an optional '_context' field with background notes "
        f"about the row (purpose, audience, tone) — use it only to inform word "
        f"choice and register; do NOT translate it or include '_context' in "
        f"your output.\n"
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


# ---------------------------------------------------------------------------
# Agent 5 — Chat Concierge (Claude Haiku 4.5)
# ---------------------------------------------------------------------------
#
# Front-of-house conversational agent. Replaces the old client-side regex
# language parser: it reads the user's free-text message (which may contain
# typos, language names in any language, abbreviations, or be totally
# unrelated to translation) and decides whether the user is selecting target
# languages or just chatting/asking a question.

_FALLBACK_CHAT_REPLY = (
    "Sorry, I had trouble processing that just now. Could you try rephrasing — "
    "for example, tell me which languages you'd like (e.g. \"English and Spanish\")?"
)


async def chat_agent(message: str, context: dict) -> dict:
    """
    Returns: {"intent": "set_languages" | "chat", "languages": [...], "reply": "..."}

    - intent == "set_languages": `languages` is a non-empty list of ISO codes
      (validated against SUPPORTED_LANGUAGES) the user wants to translate to.
    - intent == "chat": general conversation — `reply` is shown to the user
      and no translation job is started. `languages` is always [].
    """
    lang_options = ", ".join(f"{code}={name}" for code, name in SUPPORTED_LANGUAGES.items())

    system = (
        "You are the CTW Translation Agent — a friendly, helpful chat assistant "
        "embedded in a tool that translates CSV files containing Chinese text "
        "into other languages.\n\n"
        f"Supported target languages (ISO code = name): {lang_options}\n\n"
        "Given the conversation context (JSON) and the user's latest message, decide:\n"
        "1. Is the user specifying which language(s) to translate the file into? "
        "This can be phrased many ways, in any language, with abbreviations, or "
        "with typos/misspellings (e.g. 'Spnish' -> es, 'Janpanese' -> ja, "
        "'日本語' or '日文' -> ja, 'castellano' -> es). If so, set "
        "intent='set_languages' and list the matching ISO codes from the "
        "supported list in 'languages'.\n"
        "2. Otherwise, this is general conversation — greetings, small talk, "
        "questions about how the tool works, what languages are supported, "
        "the status of the file, or anything else. Set intent='chat' and "
        "write a short, friendly, helpful reply. If the user asks something "
        "totally unrelated to translation, you can still chat briefly, but "
        "gently steer back to the task when relevant.\n\n"
        "Rules:\n"
        "- If context.phase is 'idle' (no file uploaded yet) and the user "
        "names languages, acknowledge it conversationally but set "
        "intent='chat' (there's nothing to translate yet) and remind them to "
        "upload a CSV first.\n"
        "- If context.phase is 'translating', set intent='chat' — don't start "
        "a new job while one is running.\n"
        "- If you genuinely cannot match any language to the supported list, "
        "set intent='chat' and ask for clarification, listing 2-3 example "
        "languages.\n"
        "- Keep 'reply' concise (1-3 sentences), warm, and conversational. "
        "It is shown directly to the user in a chat bubble.\n\n"
        "Respond with ONLY a JSON object, no markdown, no commentary:\n"
        '{"intent": "set_languages" | "chat", "languages": ["en", "es", ...], "reply": "..."}'
    )

    user = f"Context: {json.dumps(context)}\n\nUser message: {message}"

    try:
        resp = await _get_claude().messages.create(
            model=os.getenv("HAIKU_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parsed = _parse_json(resp.content[0].text)
        if isinstance(parsed, dict) and "reply" in parsed:
            languages = [
                l for l in parsed.get("languages", []) or []
                if l in SUPPORTED_LANGUAGES
            ]
            intent = "set_languages" if (parsed.get("intent") == "set_languages" and languages) else "chat"
            return {"intent": intent, "languages": languages, "reply": parsed["reply"]}
    except Exception as exc:
        logger.error("Chat agent failed: %s", exc)

    return {"intent": "chat", "languages": [], "reply": _FALLBACK_CHAT_REPLY}

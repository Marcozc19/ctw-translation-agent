"""
Pipeline orchestrator — manages row-level state machine and concurrent batches.

Row states: pending → translating → evaluating → done | escalating → re-evaluating → done | review
"""
import asyncio
import logging
import pandas as pd

from .agents import (
    translate_batch_deepseek,
    evaluate_batch_gemini,
    translate_batch_haiku,
    _output_keys,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
MAX_CONCURRENT = 10
PASS_THRESHOLD = 0.75   # score >= this → high confidence
ESCALATE_FLOOR = 0.55   # score < this → hard flag regardless of Haiku result


async def _evaluate_rows(
    rows: list[dict],
    results: dict,
    eval_col: str,
    target_langs: list[str],
) -> dict:
    """
    Evaluate translation quality for `rows` using Agent 3 (Gemini), batched
    per target language — one call covers every row for that language, run
    concurrently across languages.

    Returns {idx: (worst_score_across_langs, hard_flagged)}.
    """
    row_by_idx = {r["_idx"]: r for r in rows}
    scores = {idx: 1.0 for idx in row_by_idx}
    hard_flags = {idx: False for idx in row_by_idx}
    evaluable = []

    for idx, r in row_by_idx.items():
        original = r["_source"].get(eval_col, "")
        translations = results[idx]["translations"]
        if not original or any(
            not translations.get(f"{eval_col}_{lang}", "") for lang in target_langs
        ):
            scores[idx] = 0.0
            continue
        evaluable.append(idx)

    async def eval_lang(lang: str):
        items = [
            {
                "id": idx,
                "original": row_by_idx[idx]["_source"][eval_col],
                "translation": results[idx]["translations"][f"{eval_col}_{lang}"],
            }
            for idx in evaluable
        ]
        if not items:
            return []
        return await evaluate_batch_gemini(items, lang)

    for lang_result in await asyncio.gather(*[eval_lang(lang) for lang in target_langs]):
        if lang_result is None:
            # Whole-language eval failed after retries — give the benefit of
            # the doubt but still escalate (matches the prior single-row fallback).
            for idx in evaluable:
                scores[idx] = min(scores[idx], 0.65)
            continue
        for item in lang_result:
            idx = item.get("id")
            if idx not in scores:
                continue
            score = max(0.0, min(1.0, float(item.get("score", 0.65))))
            if item.get("flagged") or score < ESCALATE_FLOOR:
                hard_flags[idx] = True
            scores[idx] = min(scores[idx], score)

    return {idx: (scores[idx], hard_flags[idx]) for idx in row_by_idx}


async def _process_batch(
    rows: list[dict],
    source_cols: list[str],
    target_langs: list[str],
) -> dict:
    """
    Process one batch through the full pipeline.
    Returns {row_idx: {translations, confidence, flagged}}.
    """
    # Build API payload (id + one key per source col + optional _context hint)
    batch = [
        {
            "id": r["_idx"],
            **{col: r["_source"].get(col, "") for col in source_cols},
            **({"_context": r["_context"]} if r.get("_context") else {}),
        }
        for r in rows
    ]

    results = {
        r["_idx"]: {"translations": {}, "confidence": "high", "flagged": False}
        for r in rows
    }

    # ── Stage 1: DeepSeek ──────────────────────────────────────────────────
    translated_data = await translate_batch_deepseek(batch, source_cols, target_langs)

    if translated_data is None:
        # DeepSeek entirely failed — fall through to Haiku immediately
        haiku_data = await translate_batch_haiku(batch, source_cols, target_langs, {})
        if haiku_data:
            for item in haiku_data:
                idx = item.get("id")
                if idx in results:
                    for key in _output_keys(source_cols, target_langs):
                        results[idx]["translations"][key] = item.get(key, "")
                    results[idx]["confidence"] = "low"
        return results

    for item in translated_data:
        idx = item.get("id")
        if idx in results:
            for key in _output_keys(source_cols, target_langs):
                results[idx]["translations"][key] = item.get(key, "")

    # ── Stage 2: Evaluate (first col × all langs as signal) ────────────────
    eval_col = source_cols[0]

    eval_results = await _evaluate_rows(rows, results, eval_col, target_langs)

    escalate_ids = []
    prev_translations = {}

    for idx, (score, hard_flag) in eval_results.items():
        if score < PASS_THRESHOLD:
            escalate_ids.append(idx)
            prev_translations[idx] = results[idx]["translations"].copy()
            results[idx]["confidence"] = "low"
            if hard_flag:
                results[idx]["flagged"] = True

    if not escalate_ids:
        return results

    # ── Stage 3: Haiku for escalated rows ─────────────────────────────────
    esc_batch = []
    for idx in escalate_ids:
        row_data = next(r for r in rows if r["_idx"] == idx)
        esc_batch.append({
            "id": idx,
            **{col: row_data["_source"].get(col, "") for col in source_cols},
            **({"_context": row_data["_context"]} if row_data.get("_context") else {}),
        })

    haiku_data = await translate_batch_haiku(esc_batch, source_cols, target_langs, prev_translations)

    if haiku_data:
        for item in haiku_data:
            idx = item.get("id")
            if idx not in escalate_ids:
                continue
            for key in _output_keys(source_cols, target_langs):
                val = item.get(key, "")
                if val:
                    results[idx]["translations"][key] = val

        # ── Stage 4: Re-evaluate Haiku output ─────────────────────────────
        # Hard-flagged rows are already destined for "review" regardless of
        # the re-eval outcome, so skip them entirely and save the call.
        re_eval_ids = [idx for idx in escalate_ids if not results[idx]["flagged"]]

        if re_eval_ids:
            re_eval_rows = [r for r in rows if r["_idx"] in re_eval_ids]
            re_evals = await _evaluate_rows(re_eval_rows, results, eval_col, target_langs)

            for idx, (score, _) in re_evals.items():
                if score >= PASS_THRESHOLD:
                    results[idx]["confidence"] = "high"

    return results


async def run_translation_job(
    job_id: str,
    jobs: dict,
    df: pd.DataFrame,
    source_columns: list[str],
    target_langs: list[str],
    context_column: str | None = None,
) -> None:
    """Entry point called by FastAPI BackgroundTasks.

    `context_column`, if provided, names a column (e.g. "description")
    containing per-row background notes. It's passed to the translation
    agents as a hint to inform tone/register, but is dropped from the
    final output CSV — it isn't translated content itself.
    """
    jobs[job_id]["status"] = "running"
    total = len(df)

    try:
        # Flatten rows
        all_rows = [
            {
                "_idx": idx,
                "_source": {
                    col: (str(row[col]) if pd.notna(row[col]) else "")
                    for col in source_columns
                },
                "_context": (
                    str(row[context_column])
                    if context_column and pd.notna(row.get(context_column, None))
                    else ""
                ),
            }
            for idx, row in df.iterrows()
        ]

        batches = [all_rows[i: i + BATCH_SIZE] for i in range(0, len(all_rows), BATCH_SIZE)]
        jobs[job_id]["batch_total"] = len(batches)

        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        all_results: dict = {}

        async def run_batch(batch: list, batch_num: int):
            async with semaphore:
                logger.info("Job %s: batch %d/%d", job_id, batch_num, len(batches))
                result = await _process_batch(batch, source_columns, target_langs)
                jobs[job_id]["processed"] = min(
                    jobs[job_id]["processed"] + len(batch), total
                )
                jobs[job_id]["batch_processed"] = batch_num
                return result

        batch_results = await asyncio.gather(
            *[run_batch(b, i + 1) for i, b in enumerate(batches)],
            return_exceptions=True,
        )

        for br in batch_results:
            if isinstance(br, Exception):
                logger.error("Batch error: %s", br)
            else:
                all_results.update(br)

        # Write results back into df
        for key in _output_keys(source_columns, target_langs):
            df[key] = ""

        # Write translations back into df. We don't expose the raw
        # "translation_confidence" levels in the output, but rows that came
        # back hard-flagged or stayed at "low" confidence even after the
        # Haiku retry are marked in a trailing "needs_review" column so a
        # human can quickly find them.
        flagged = 0
        for idx, row_result in all_results.items():
            for key, val in row_result["translations"].items():
                if key in df.columns:
                    df.at[idx, key] = val
            if row_result["flagged"]:
                flagged += 1

        # Drop the context/notes column (e.g. "description") from the
        # output — it was only a hint for the translation agents.
        if context_column and context_column in df.columns:
            df = df.drop(columns=[context_column])

        # Add the review flag last so it appears as the rightmost column.
        df["needs_review"] = ""
        for idx, row_result in all_results.items():
            if row_result["flagged"] or row_result["confidence"] == "low":
                df.at[idx, "needs_review"] = "Yes"

        jobs[job_id].update(
            flagged=flagged,
            processed=total,
            result_df=df,
            status="completed",
        )
        logger.info("Job %s completed — %d flagged", job_id, flagged)

    except Exception:
        logger.exception("Job %s failed", job_id)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Internal pipeline error. Check server logs."

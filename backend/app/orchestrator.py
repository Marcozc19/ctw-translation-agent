"""
Pipeline orchestrator — manages row-level state machine and concurrent batches.

Row states: pending → translating → evaluating → done | escalating → re-evaluating → done | review
"""
import asyncio
import logging
import pandas as pd

from .agents import (
    translate_batch_deepseek,
    evaluate_translation,
    translate_batch_haiku,
    _output_keys,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
MAX_CONCURRENT = 5
PASS_THRESHOLD = 0.75   # score >= this → high confidence
ESCALATE_FLOOR = 0.55   # score < this → hard flag regardless of Haiku result


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

    async def eval_row(r: dict):
        idx = r["_idx"]
        original = r["_source"].get(eval_col, "")
        if not original:
            return idx, None

        worst_score = 1.0
        hard_flag = False

        for lang in target_langs:
            trans = results[idx]["translations"].get(f"{eval_col}_{lang}", "")
            if not trans:
                worst_score = 0.0
                break
            score, flagged = await evaluate_translation(original, trans, lang)
            if flagged:
                hard_flag = True
            if score < worst_score:
                worst_score = score

        return idx, (worst_score, hard_flag)

    eval_results = await asyncio.gather(*[eval_row(r) for r in rows], return_exceptions=True)

    escalate_ids = []
    prev_translations = {}

    for res in eval_results:
        if isinstance(res, Exception) or res[1] is None:
            continue
        idx, (score, hard_flag) = res
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
        async def re_eval_row(idx):
            r_data = next((r for r in rows if r["_idx"] == idx), None)
            if r_data is None:
                return idx, None
            original = r_data["_source"].get(eval_col, "")
            if not original:
                return idx, None

            worst_score = 1.0
            for lang in target_langs:
                trans = results[idx]["translations"].get(f"{eval_col}_{lang}", "")
                if not trans:
                    worst_score = 0.0
                    break
                score, _ = await evaluate_translation(original, trans, lang)
                if score < worst_score:
                    worst_score = score
            return idx, worst_score

        re_evals = await asyncio.gather(*[re_eval_row(idx) for idx in escalate_ids], return_exceptions=True)

        for res in re_evals:
            if isinstance(res, Exception) or res[1] is None:
                continue
            idx, score = res
            if score >= PASS_THRESHOLD:
                # Haiku passed — clear low-confidence flag if not hard-flagged
                if not results[idx]["flagged"]:
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

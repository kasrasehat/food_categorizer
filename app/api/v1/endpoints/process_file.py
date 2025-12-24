from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from utils.categorizer import categorize_food_batch
from utils.llm_translate import translate_to_english_batch
from utils.product_name_processing import finalize_processed_name, prepare_translation_inputs
from utils.tabular_io import read_csv_bytes, read_xlsx_bytes, write_csv_bytes, write_xlsx_bytes


router = APIRouter()


def _detect_file_kind(filename: str, content_type: Optional[str]) -> str:
    name = (filename or "").lower()
    if name.endswith(".csv") or (content_type or "").lower() in {"text/csv", "application/csv"}:
        return "csv"
    if name.endswith(".xlsx") or (content_type or "").lower() in {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }:
        return "xlsx"
    return ""


def _read_tabular(file_kind: str, content: bytes) -> Tuple[List[str], List[Dict[str, Any]]]:
    if file_kind == "csv":
        return read_csv_bytes(content)
    if file_kind == "xlsx":
        return read_xlsx_bytes(content)
    raise ValueError(f"Unsupported file kind: {file_kind}")


def _write_tabular(file_kind: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> bytes:
    if file_kind == "csv":
        return write_csv_bytes(fieldnames, rows)
    if file_kind == "xlsx":
        return write_xlsx_bytes(fieldnames, rows)
    raise ValueError(f"Unsupported file kind: {file_kind}")


@router.post("/process-file")
async def process_file(
    file: UploadFile = File(...),
    rag_llm_model: Optional[str] = Query(
        default=None,
        description="LLM model name/alias for translation (e.g., gpt-4o-mini, gpt4o-mini, gpt5-low).",
    ),
    translate_with_llm: bool = Query(
        default=True,
        description="If true, non-English names (or names without any English tokens) are translated via LLM.",
    ),
) -> StreamingResponse:
    """
    Upload CSV/XLSX containing ACTUAL_PRODUCT_NAME column.
    Returns the same file type with all original columns plus:
    - PROCESSED_PRODUCT_NAME
    - PRODUCT_LABEL
    - LABEL_PROBABILITY
    """
    file_kind = _detect_file_kind(file.filename or "", file.content_type)
    if not file_kind:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload .csv or .xlsx")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        fieldnames, rows = _read_tabular(file_kind, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not fieldnames:
        raise HTTPException(status_code=400, detail="No header row found")

    # Column name must exist (exact name per requirement).
    if "ACTUAL_PRODUCT_NAME" not in fieldnames:
        raise HTTPException(
            status_code=400,
            detail="Missing required column: ACTUAL_PRODUCT_NAME",
        )

    # Ensure output columns exist in header list (append at end).
    out_fieldnames = list(fieldnames)
    for col in ("PROCESSED_PRODUCT_NAME", "PRODUCT_LABEL", "LABEL_PROBABILITY"):
        if col not in out_fieldnames:
            out_fieldnames.append(col)

    raw_names: List[str] = []
    for r in rows:
        raw_names.append("" if r.get("ACTUAL_PRODUCT_NAME") is None else str(r.get("ACTUAL_PRODUCT_NAME")))

    translation_texts, translation_indices, initial_processed, translation_langs = prepare_translation_inputs(
        raw_names
    )

    # Translate only when needed
    if translate_with_llm and translation_texts:
        # cfg format is used to match user's snippet style
        cfg = {"rag_llm_model": rag_llm_model} if rag_llm_model else {"rag_llm_model": "gpt-4o-mini"}
        try:
            translated = translate_to_english_batch(
                translation_texts,
                source_languages=translation_langs,
                cfg=cfg,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"LLM translation failed. Ensure OPENAI_API_KEY is set in .env/env vars. Error: {e}",
            )
        for idx, translated_text in zip(translation_indices, translated):
            initial_processed[idx] = finalize_processed_name(translated_text)
    else:
        # no translation: just finalize anything still empty using best-effort extraction
        for i, val in enumerate(initial_processed):
            if not val:
                initial_processed[i] = finalize_processed_name(raw_names[i])

    # Categorize using *processed* names (not actual names)
    batch_results = await categorize_food_batch(initial_processed)

    # Write back into rows
    for i, r in enumerate(rows):
        r["PROCESSED_PRODUCT_NAME"] = initial_processed[i]
        res = batch_results[i] if i < len(batch_results) else {}
        r["PRODUCT_LABEL"] = str(res.get("category", "other"))
        try:
            r["LABEL_PROBABILITY"] = float(res.get("probability", 0.0))
        except Exception:
            r["LABEL_PROBABILITY"] = 0.0

    out_bytes = _write_tabular(file_kind, out_fieldnames, rows)

    original = file.filename or f"upload.{file_kind}"
    p = Path(original)
    out_name = f"{p.stem}_processed{p.suffix if p.suffix else ('.csv' if file_kind == 'csv' else '.xlsx')}"

    media_type = "text/csv" if file_kind == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    headers = {"Content-Disposition": f'attachment; filename="{out_name}"'}
    return StreamingResponse(iter([out_bytes]), media_type=media_type, headers=headers)



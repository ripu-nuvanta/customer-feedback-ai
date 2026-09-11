import os
import io
import json
import re
import math
import hashlib
import threading
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import Optional, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

ENV_FILE = os.getenv("ENV_FILE", "").strip()
env_candidates = [
    os.path.join(BASE_DIR, ".env"),
    os.path.join(PROJECT_DIR, ".env"),
    os.path.join(os.getcwd(), ".env"),
]
if ENV_FILE:
    env_candidates.insert(0, ENV_FILE)

for env_path in dict.fromkeys(env_candidates):
    if os.path.isfile(env_path):
        load_dotenv(env_path, override=True)

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY", "")
    .strip()
    .strip('"')
    .strip("'")
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()

AI_BATCH_SIZE = int(os.getenv("AI_BATCH_SIZE", "30"))
AI_TIMEOUT = float(os.getenv("AI_TIMEOUT", "75"))
MAX_ASK_SOURCES = int(os.getenv("MAX_ASK_SOURCES", "18"))
AI_BATCH_MAX_OUTPUT_TOKENS = int(
    os.getenv("AI_BATCH_MAX_OUTPUT_TOKENS", "3500")
)
AI_FINAL_MAX_OUTPUT_TOKENS = int(
    os.getenv("AI_FINAL_MAX_OUTPUT_TOKENS", "1800")
)
AI_ASK_MAX_OUTPUT_TOKENS = int(
    os.getenv("AI_ASK_MAX_OUTPUT_TOKENS", "3000")
)

# File-based persistence. Each completed upload gets its own CSV snapshot
# and JSON summary. Previous uploads are never replaced or deleted.
ANALYSIS_STORAGE_DIR = os.getenv(
    "NUVANTA_ANALYSIS_DIR",
    os.path.join(BASE_DIR, "analysis_store")
)
ANALYSIS_RUNS_INDEX = os.path.join(
    ANALYSIS_STORAGE_DIR,
    "analysis_runs.json"
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Customer Feedback AI by Nuvanta AI",
    version="6.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL STATE
# ============================================================

DATA: list[dict[str, Any]] = []
ANALYSIS: Optional[dict[str, Any]] = None
LAST_ASK: Optional[dict[str, Any]] = None
CURRENT_ANALYSIS_RUN_ID: Optional[int] = None

ANALYSIS_STATUS: dict[str, Any] = {
    "status": "idle",
    "message": "Upload customer feedback to begin AI analysis.",
    "error": None,
    "analysis": None,
    "total_records": 0,
    "processed_records": 0,
    "batch_size": AI_BATCH_SIZE,
    "completed_batches": 0,
    "total_batches": 0,
    "analysis_run_id": None,
    "dataset_name": None,
}

ANALYSIS_LOCK = threading.Lock()


# ============================================================
# FILE-BASED ANALYSIS PERSISTENCE
# ============================================================

ANALYSIS_CSV_COLUMNS = [
    "id",
    "record_fingerprint",
    "customer_id",
    "customer_email",
    "customer_name",
    "date",
    "channel",
    "subject",
    "message",
    "original_message",
    "plan",
    "customer_type",
    "revenue",
    "sentiment",
    "feature_request",
    "churn_signal",
    "unresolved_issue",
    "topic",
    "problem",
    "feature_reason",
    "churn_reason",
    "feedback_summary",
]


def make_record_fingerprint(row: dict[str, Any]) -> str:
    """Stable identity for a customer-feedback record across CSV uploads."""
    payload = {
        "customer_id": clean_text(row.get("customer_id")),
        "customer_email": clean_text(row.get("customer_email")),
        "customer_name": clean_text(row.get("customer_name")),
        "date": clean_text(row.get("date")),
        "channel": clean_text(row.get("channel")),
        "subject": clean_text(row.get("subject")),
        "original_message": clean_text(
            row.get("original_message", row.get("message", ""))
        ),
        "plan": clean_text(row.get("plan")),
        "customer_type": clean_text(row.get("customer_type")),
        "revenue": parse_revenue(row.get("revenue", 0)),
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_dataset_fingerprint(rows: list[dict[str, Any]]) -> str:
    fingerprints = sorted(
        make_record_fingerprint(row)
        for row in rows
    )
    raw = json.dumps(
        fingerprints,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def init_storage() -> None:
    """Create the local file store without using a database engine."""
    os.makedirs(ANALYSIS_STORAGE_DIR, exist_ok=True)

    if not os.path.isfile(ANALYSIS_RUNS_INDEX):
        with open(
            ANALYSIS_RUNS_INDEX,
            "w",
            encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "next_run_id": 1,
                    "runs": [],
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )


def _read_storage_index() -> dict[str, Any]:
    init_storage()
    try:
        with open(
            ANALYSIS_RUNS_INDEX,
            "r",
            encoding="utf-8"
        ) as handle:
            data = json.load(handle)
    except Exception:
        data = {"next_run_id": 1, "runs": []}

    if not isinstance(data, dict):
        data = {"next_run_id": 1, "runs": []}

    if not isinstance(data.get("runs"), list):
        data["runs"] = []

    try:
        data["next_run_id"] = max(
            1,
            int(data.get("next_run_id", 1))
        )
    except Exception:
        data["next_run_id"] = 1

    return data


def _write_storage_index(data: dict[str, Any]) -> None:
    init_storage()
    temp_path = f"{ANALYSIS_RUNS_INDEX}.tmp"
    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2,
        )
    os.replace(temp_path, ANALYSIS_RUNS_INDEX)


def _safe_filename(value: str) -> str:
    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        clean_text(value)
    ).strip("._")
    return value[:80] or "uploaded"


def _find_run(run_id: int) -> Optional[dict[str, Any]]:
    data = _read_storage_index()
    for run in data["runs"]:
        try:
            if int(run.get("id")) == int(run_id):
                return run
        except Exception:
            continue
    return None


def _run_csv_path(run: dict[str, Any]) -> str:
    return os.path.join(
        ANALYSIS_STORAGE_DIR,
        clean_text(run.get("csv_file"))
    )


def _run_summary_path(run: dict[str, Any]) -> str:
    return os.path.join(
        ANALYSIS_STORAGE_DIR,
        clean_text(run.get("summary_file"))
    )


def _row_to_saved_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": clean_text(row.get("id")),
        "record_fingerprint": make_record_fingerprint(row),
        "customer_id": clean_text(row.get("customer_id")),
        "customer_email": clean_text(row.get("customer_email")),
        "customer_name": clean_text(row.get("customer_name")) or "Unknown customer",
        "date": clean_text(row.get("date")),
        "channel": clean_text(row.get("channel")) or "chat",
        "subject": clean_text(row.get("subject")),
        "message": clean_text(row.get("message")),
        "original_message": clean_text(row.get("original_message")),
        "plan": clean_text(row.get("plan")),
        "customer_type": clean_text(row.get("customer_type")),
        "revenue": parse_revenue(row.get("revenue", 0)),
        "sentiment": clean_text(row.get("sentiment")) or "neutral",
        "feature_request": int(bool(row.get("feature_request"))),
        "churn_signal": int(bool(row.get("churn_signal"))),
        "unresolved_issue": int(bool(row.get("unresolved_issue"))),
        "topic": clean_text(row.get("topic")),
        "problem": clean_text(row.get("problem")),
        "feature_reason": clean_text(row.get("feature_reason")),
        "churn_reason": clean_text(row.get("churn_reason")),
        "feedback_summary": clean_text(row.get("feedback_summary")),
    }


def _saved_record_to_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": clean_text(record.get("id")),
        "customer_id": clean_text(record.get("customer_id")),
        "customer_email": clean_text(record.get("customer_email")),
        "customer_name": clean_text(record.get("customer_name")) or "Unknown customer",
        "date": clean_text(record.get("date")),
        "channel": clean_text(record.get("channel")) or "chat",
        "subject": clean_text(record.get("subject")),
        "message": clean_text(record.get("message")),
        "original_message": clean_text(record.get("original_message")),
        "plan": clean_text(record.get("plan")),
        "customer_type": clean_text(record.get("customer_type")),
        "revenue": parse_revenue(record.get("revenue", 0)),
        "sentiment": clean_text(record.get("sentiment")) or "neutral",
        "feature_request": str(record.get("feature_request", "0")).strip().lower()
        in {"1", "true", "yes"},
        "churn_signal": str(record.get("churn_signal", "0")).strip().lower()
        in {"1", "true", "yes"},
        "unresolved_issue": str(record.get("unresolved_issue", "0")).strip().lower()
        in {"1", "true", "yes"},
        "topic": clean_text(record.get("topic")),
        "problem": clean_text(record.get("problem")),
        "feature_reason": clean_text(record.get("feature_reason")),
        "churn_reason": clean_text(record.get("churn_reason")),
        "feedback_summary": clean_text(record.get("feedback_summary")),
    }


def create_analysis_run(
    dataset_name: str,
    total_records: int,
    dataset_fingerprint: str = ""
) -> int:
    """Create a unique run and a unique CSV filename; never overwrite an old run."""
    data = _read_storage_index()
    run_id = int(data.get("next_run_id", 1))
    data["next_run_id"] = run_id + 1

    created_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_filename(
        os.path.splitext(dataset_name or "uploaded")[0]
    )

    csv_file = (
        f"analysis_run_{run_id:06d}_{stamp}_{stem}.csv"
    )
    summary_file = (
        f"analysis_run_{run_id:06d}_{stamp}_{stem}.json"
    )

    data["runs"].append(
        {
            "id": run_id,
            "dataset_name": dataset_name or "uploaded.csv",
            "created_at": created_at,
            "completed_at": None,
            "total_records": int(total_records),
            "status": "processing",
            "error": None,
            "dataset_fingerprint": dataset_fingerprint or None,
            "csv_file": csv_file,
            "summary_file": summary_file,
        }
    )
    _write_storage_index(data)

    return run_id


def save_analysis_records(
    run_id: int,
    rows: list[dict[str, Any]]
) -> None:
    """Persist the current run as its own CSV snapshot.

    Repeated calls update only the current run's file. A later upload receives
    a different run id and therefore a different CSV file. Previous run files
    are never replaced or deleted.
    """
    run = _find_run(run_id)
    if run is None:
        raise RuntimeError(f"Analysis run {run_id} was not found in file storage.")

    records = [
        _row_to_saved_record(row)
        for row in rows
    ]

    # A run may be saved after every successful AI batch. Merge the new
    # batch into the current run file so earlier batches are retained.
    # This never touches any older run because every upload has its own
    # unique CSV filename.
    path = _run_csv_path(run)
    existing_records: list[dict[str, Any]] = []
    if os.path.isfile(path):
        try:
            existing_frame = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
            )
            existing_records = existing_frame.to_dict(orient="records")
        except Exception as exc:
            print(
                f"[STORAGE] Could not merge existing run file {path}: {exc!r}"
            )

    merged = existing_records + records
    deduped: dict[str, dict[str, Any]] = {}
    for record in merged:
        key = clean_text(record.get("record_fingerprint")) or clean_text(record.get("id"))
        if key:
            deduped[key] = record

    frame = pd.DataFrame(
        list(deduped.values()),
        columns=ANALYSIS_CSV_COLUMNS
    )
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )


def _iter_saved_runs() -> list[dict[str, Any]]:
    data = _read_storage_index()
    runs = [
        run
        for run in data.get("runs", [])
        if isinstance(run, dict)
    ]
    runs.sort(
        key=lambda item: int(item.get("id", 0)),
        reverse=True
    )
    return runs


def _read_run_records(run: dict[str, Any]) -> list[dict[str, Any]]:
    path = _run_csv_path(run)
    if not path or not os.path.isfile(path):
        return []

    try:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        print(
            f"[STORAGE] Could not read {path}: {exc!r}"
        )
        return []

    return frame.to_dict(orient="records")


def hydrate_cached_analysis(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Reuse AI classifications stored in previous analysis CSV files."""
    if not rows:
        return [], 0

    wanted = {
        make_record_fingerprint(row): row
        for row in rows
    }
    cached: dict[str, dict[str, Any]] = {}

    # Newest completed run wins if the same feedback appears in multiple uploads.
    for run in _iter_saved_runs():
        if run.get("status") != "complete":
            continue

        for record in _read_run_records(run):
            fp = clean_text(record.get("record_fingerprint"))
            if fp and fp in wanted and fp not in cached:
                cached[fp] = record

        if len(cached) == len(wanted):
            break

    fresh_rows: list[dict[str, Any]] = []
    reused = 0

    for row in rows:
        fp = make_record_fingerprint(row)
        record = cached.get(fp)
        if record is None:
            fresh_rows.append(row)
            continue

        # Preserve the newly uploaded source fields, but reuse only the saved AI fields.
        row["sentiment"] = clean_text(record.get("sentiment")) or "neutral"
        row["feature_request"] = str(record.get("feature_request", "0")).strip().lower() in {"1", "true", "yes"}
        row["churn_signal"] = str(record.get("churn_signal", "0")).strip().lower() in {"1", "true", "yes"}
        row["unresolved_issue"] = str(record.get("unresolved_issue", "0")).strip().lower() in {"1", "true", "yes"}
        row["topic"] = clean_text(record.get("topic"))
        row["problem"] = clean_text(record.get("problem"))
        row["feature_reason"] = clean_text(record.get("feature_reason"))
        row["churn_reason"] = clean_text(record.get("churn_reason"))
        row["feedback_summary"] = clean_text(record.get("feedback_summary"))
        reused += 1

    return fresh_rows, reused


def _load_summary_for_run(run: dict[str, Any]) -> dict[str, Any]:
    path = _run_summary_path(run)
    if not path or not os.path.isfile(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_analysis_by_dataset_fingerprint(
    dataset_fingerprint: str
) -> Optional[dict[str, Any]]:
    if not dataset_fingerprint:
        return None

    for run in _iter_saved_runs():
        if (
            run.get("status") == "complete"
            and clean_text(run.get("dataset_fingerprint")) == dataset_fingerprint
        ):
            return {
                "run_id": int(run["id"]),
                "dataset_name": run.get("dataset_name", "uploaded.csv"),
                "created_at": run.get("created_at"),
                "completed_at": run.get("completed_at"),
                "analysis": _load_summary_for_run(run),
            }

    return None


def mark_analysis_complete(
    run_id: int,
    analysis: dict[str, Any]
) -> None:
    data = _read_storage_index()
    completed_at = datetime.now(timezone.utc).isoformat()

    for run in data["runs"]:
        if int(run.get("id", 0)) == int(run_id):
            run["status"] = "complete"
            run["completed_at"] = completed_at
            run["error"] = None

            summary_path = _run_summary_path(run)
            with open(
                summary_path,
                "w",
                encoding="utf-8"
            ) as handle:
                json.dump(
                    analysis,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            break

    _write_storage_index(data)


def mark_analysis_error(
    run_id: int,
    error: str
) -> None:
    data = _read_storage_index()
    for run in data["runs"]:
        if int(run.get("id", 0)) == int(run_id):
            run["status"] = "error"
            run["error"] = error[:4000]
            break
    _write_storage_index(data)


def load_saved_analysis(
    run_id: Optional[int] = None
) -> Optional[dict[str, Any]]:
    runs = _iter_saved_runs()

    selected = None
    for run in runs:
        if run.get("status") != "complete":
            continue
        if run_id is None or int(run.get("id", 0)) == int(run_id):
            selected = run
            break

    if selected is None:
        return None

    analysis = _load_summary_for_run(selected)
    record_rows = _read_run_records(selected)
    rows = [
        _saved_record_to_row(record)
        for record in record_rows
    ]

    # Keep the same compatibility behavior as the old persistence layer.
    row_by_id = {str(row["id"]): row for row in rows}
    for key in ("feature_requests", "churn_signals"):
        items = analysis.get(key) if isinstance(analysis, dict) else None
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            source_id = clean_text(item.get("id"))
            row = row_by_id.get(source_id)
            if row is None:
                continue
            item["sentiment"] = row.get("sentiment", "neutral")
            item["topic"] = row.get("topic", "")
            if key == "churn_signals":
                item["unresolved_issue"] = bool(row.get("unresolved_issue"))

    return {
        "run_id": int(selected["id"]),
        "dataset_name": selected.get("dataset_name", "uploaded.csv"),
        "created_at": selected.get("created_at"),
        "completed_at": selected.get("completed_at"),
        "analysis": analysis,
        "rows": rows,
    }


def load_latest_saved_analysis_into_memory() -> None:
    global DATA
    global ANALYSIS
    global CURRENT_ANALYSIS_RUN_ID
    global ANALYSIS_STATUS

    saved = load_saved_analysis()
    if not saved:
        return

    DATA = saved["rows"]
    ANALYSIS = saved["analysis"]
    CURRENT_ANALYSIS_RUN_ID = saved["run_id"]

    total = len(DATA)
    total_batches = math.ceil(total / AI_BATCH_SIZE) if total else 0

    ANALYSIS_STATUS = {
        "status": "complete",
        "message": (
            f"Loaded saved analysis for all {total:,} records."
        ),
        "error": None,
        "analysis": ANALYSIS,
        "total_records": total,
        "processed_records": total,
        "batch_size": AI_BATCH_SIZE,
        "completed_batches": total_batches,
        "total_batches": total_batches,
        "analysis_run_id": CURRENT_ANALYSIS_RUN_ID,
        "dataset_name": saved["dataset_name"],
        "created_at": saved["created_at"],
        "completed_at": saved["completed_at"],
    }

# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalize_key(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        clean_text(value).lower()
    ).strip("_")


def safe_date(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    try:
        return pd.to_datetime(text).strftime("%Y-%m-%d")
    except Exception:
        return text


def parse_revenue(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        try:
            number = float(value)
            return 0.0 if math.isnan(number) else number
        except Exception:
            return 0.0

    text = (
        clean_text(value)
        .replace("$", "")
        .replace(",", "")
        .replace("USD", "")
        .strip()
    )
    try:
        return float(text) if text else 0.0
    except Exception:
        return 0.0


def parse_json(text: str) -> Any:
    text = clean_text(text)
    if not text:
        return None

    candidates = [text]

    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        re.DOTALL | re.IGNORECASE
    )
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj >= 0 and last_obj > first_obj:
        candidates.append(text[first_obj:last_obj + 1])

    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr >= 0 and last_arr > first_arr:
        candidates.append(text[first_arr:last_arr + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    return None


def response_text(response: Any) -> str:
    direct = clean_text(getattr(response, "output_text", ""))
    if direct:
        return direct

    chunks = []
    output = getattr(response, "output", None) or []

    for item in output:
        content = getattr(item, "content", None) or []
        for part in content:
            text = getattr(part, "text", None)

            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
                continue

            value = getattr(text, "value", None)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())
                continue

            parsed = getattr(part, "parsed", None)
            if parsed is not None:
                try:
                    return json.dumps(
                        parsed,
                        ensure_ascii=False
                    )
                except Exception:
                    pass

    return "\n".join(chunks).strip()


# ============================================================
# OPENAI
# ============================================================

def get_openai_client() -> Optional[Any]:
    if OpenAI is None:
        return None

    key = (
        os.getenv("OPENAI_API_KEY", "")
        .strip()
        .strip('"')
        .strip("'")
    )
    if not key:
        return None

    try:
        return OpenAI(
            api_key=key,
            timeout=AI_TIMEOUT,
            max_retries=1
        )
    except Exception as exc:
        print(f"[AI] client initialization failed: {exc!r}")
        return None


def require_openai() -> Any:
    client = get_openai_client()
    if client is None:
        if OpenAI is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The OpenAI Python package is not installed. "
                    "Run: pip install openai"
                )
            )
        raise HTTPException(
            status_code=503,
            detail=(
                "OpenAI AI is not configured. "
                "Set OPENAI_API_KEY in backend/.env and restart."
            )
        )
    return client


class AIOutputTokenLimitError(RuntimeError):
    """Raised when an OpenAI response cannot finish within its output budget."""


def ai_json(
    prompt: str,
    schema_name: str,
    schema: dict[str, Any],
    label: str,
    max_output_tokens: int
) -> Any:
    client = require_openai()

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=max_output_tokens,
            store=False,
        )
    except Exception as exc:
        print(f"[AI] ERROR {label}: {exc!r}")
        raise RuntimeError(
            f"OpenAI request failed during {label}: {exc}"
        ) from exc

    raw = response_text(response)
    parsed = parse_json(raw)
    if parsed is not None:
        return parsed

    incomplete = getattr(
        response,
        "incomplete_details",
        None
    )
    reason = ""
    if incomplete:
        reason = clean_text(
            getattr(incomplete, "reason", "")
        )

    if reason == "max_output_tokens":
        raise AIOutputTokenLimitError(
            f"OpenAI response reached the output-token limit during {label}."
        )

    raise RuntimeError(
        "OpenAI returned invalid JSON for "
        f"{label}. status="
        f"{getattr(response, 'status', 'unknown')}"
    )


# ============================================================
# CSV NORMALIZATION
# ============================================================

def find_column(
    columns: list[str],
    candidates: list[str]
) -> Optional[str]:
    normalized = {
        normalize_key(c): c
        for c in columns
    }

    for candidate in candidates:
        key = normalize_key(candidate)
        if key in normalized:
            return normalized[key]

    for candidate in candidates:
        key = normalize_key(candidate)
        for normalized_name, original in normalized.items():
            if key and (
                key in normalized_name
                or normalized_name in key
            ):
                return original

    return None


def normalize_channel(value: Any) -> str:
    text = clean_text(value).lower()

    if any(x in text for x in ["email", "mail"]):
        return "email"

    if any(
        x in text
        for x in [
            "call",
            "phone",
            "telephone",
            "voice",
            "transcript"
        ]
    ):
        return "call"

    if any(
        x in text
        for x in [
            "chat",
            "whatsapp",
            "message",
            "messaging",
            "slack",
            "tweet"
        ]
    ):
        return "chat"

    return "chat"


def normalize_dataframe(
    df: pd.DataFrame
) -> list[dict[str, Any]]:
    columns = list(df.columns)

    customer_id_col = find_column(
        columns,
        [
            "customer_id",
            "customer id",
            "customerid",
            "account_id",
            "account id"
        ]
    )
    customer_email_col = find_column(
        columns,
        [
            "customer_email",
            "customer email",
            "email",
            "email_address",
            "email address"
        ]
    )
    customer_name_col = find_column(
        columns,
        [
            "customer_name",
            "customer name",
            "customer",
            "company",
            "account_name",
            "account",
            "organization",
            "name"
        ]
    )
    date_col = find_column(
        columns,
        [
            "date",
            "created_at",
            "created",
            "timestamp",
            "time",
            "conversation_date",
            "submission_date"
        ]
    )
    channel_col = find_column(
        columns,
        [
            "channel",
            "source",
            "type",
            "medium",
            "ticket_channel"
        ]
    )
    message_col = find_column(
        columns,
        [
            "message",
            "text",
            "feedback",
            "conversation",
            "comment",
            "content",
            "body",
            "transcript",
            "ticket_description",
            "description"
        ]
    )
    subject_col = find_column(
        columns,
        [
            "subject",
            "ticket_subject",
            "email_subject"
        ]
    )
    plan_col = find_column(
        columns,
        [
            "plan",
            "subscription",
            "tier",
            "package"
        ]
    )
    customer_type_col = find_column(
        columns,
        [
            "customer_type",
            "customer type",
            "segment",
            "segment_name",
            "customer_segment"
        ]
    )
    revenue_col = find_column(
        columns,
        [
            "revenue",
            "arr",
            "annual_revenue",
            "customer_value",
            "value",
            "mrr",
            "contract_value"
        ]
    )

    if not customer_name_col:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing customer name/company column. "
                "Expected customer_name, customer, company, "
                "account, organization, or name."
            )
        )

    if not message_col:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing conversation/message column. "
                "Expected message, text, feedback, conversation, "
                "transcript, ticket_description, description, "
                "or content."
            )
        )

    rows = []

    for index, raw in df.iterrows():
        message = clean_text(
            raw.get(message_col, "")
        )
        if not message:
            continue

        customer_name = clean_text(
            raw.get(customer_name_col, "")
        ) or "Unknown customer"

        customer_email = (
            clean_text(raw.get(customer_email_col, ""))
            if customer_email_col
            else ""
        )

        explicit_customer_id = (
            clean_text(raw.get(customer_id_col, ""))
            if customer_id_col
            else ""
        )

        customer_id = (
            explicit_customer_id
            or customer_email
            or normalize_key(customer_name)
            or f"customer_{index + 1}"
        )

        rows.append(
            {
                "id": str(len(rows) + 1),
                "customer_id": customer_id,
                "customer_email": customer_email,
                "customer_name": customer_name,
                "date": (
                    safe_date(raw.get(date_col, ""))
                    if date_col else ""
                ),
                "channel": (
                    normalize_channel(raw.get(channel_col, ""))
                    if channel_col else "chat"
                ),
                "subject": (
                    clean_text(raw.get(subject_col, ""))
                    if subject_col else ""
                ),
                "message": message,
                "original_message": message,
                "plan": (
                    clean_text(raw.get(plan_col, ""))
                    if plan_col else ""
                ),
                "customer_type": (
                    clean_text(raw.get(customer_type_col, ""))
                    if customer_type_col else ""
                ),
                "revenue": (
                    parse_revenue(raw.get(revenue_col, 0))
                    if revenue_col else 0.0
                ),
                "sentiment": "neutral",
                "feature_request": False,
                "churn_signal": False,
                "unresolved_issue": False,
                "topic": "",
                "problem": "",
                "feature_reason": "",
                "churn_reason": "",
                "feedback_summary": "",
            }
        )

    return rows


# ============================================================
# CONVERSATION HELPERS
# ============================================================

def grouped_rows(
    rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    groups = defaultdict(list)

    for row in rows:
        key = str(
            row.get("customer_id")
            or row.get("customer_name")
            or row["id"]
        )
        groups[key].append(row)

    for group in groups.values():
        group.sort(
            key=lambda r: (
                r.get("date", ""),
                r.get("id", "")
            )
        )

    return groups


def source_record(
    row: dict[str, Any],
    group: Optional[list[dict[str, Any]]] = None
) -> dict[str, Any]:
    turns = []

    if group and len(group) > 1:
        for item in group:
            turns.append(
                {
                    "id": item["id"],
                    "date": item.get("date", ""),
                    "channel": item.get("channel", "chat"),
                    "subject": item.get("subject", ""),
                    "message": item.get(
                        "original_message",
                        item.get("message", "")
                    ),
                }
            )

    return {
        "id": row["id"],
        "customer_id": row.get("customer_id", ""),
        "customer_email": row.get("customer_email", ""),
        "customer_name": row.get(
            "customer_name",
            "Unknown customer"
        ),
        "customer": row.get(
            "customer_name",
            "Unknown customer"
        ),
        "date": row.get("date", ""),
        "channel": row.get("channel", "chat"),
        "subject": row.get("subject", ""),
        "message": row.get(
            "original_message",
            row.get("message", "")
        ),
        "original_message": row.get(
            "original_message",
            row.get("message", "")
        ),
        "feedback_summary": row.get(
            "feedback_summary",
            ""
        ),
        "ai_generated_summary": bool(
            row.get("feedback_summary")
        ),
        "plan": row.get("plan", ""),
        "customer_type": row.get(
            "customer_type",
            ""
        ),
        "revenue": row.get("revenue", 0),
        "topic": row.get("topic", ""),
        "problem": row.get("problem", ""),
        "sentiment": row.get(
            "sentiment",
            "neutral"
        ),
        "feature_request": bool(
            row.get("feature_request")
        ),
        "churn_signal": bool(
            row.get("churn_signal")
        ),
        "unresolved_issue": bool(
            row.get("unresolved_issue")
        ),
        "feature_reason": row.get(
            "feature_reason",
            ""
        ),
        "churn_reason": row.get(
            "churn_reason",
            ""
        ),
        "turns": turns,
    }


# ============================================================
# AI BATCH CLASSIFICATION
# ============================================================

BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "sentiment": {
                        "type": "string",
                        "enum": [
                            "positive",
                            "negative",
                            "neutral"
                        ]
                    },
                    "feature_request": {
                        "type": "boolean"
                    },
                    "churn_signal": {
                        "type": "boolean"
                    },
                    "unresolved_issue": {
                        "type": "boolean"
                    },
                    "topic": {"type": "string"},
                    "problem": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": [
                    "id",
                    "sentiment",
                    "feature_request",
                    "churn_signal",
                    "unresolved_issue",
                    "topic",
                    "problem",
                    "summary",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["records"],
    "additionalProperties": False,
}


def analyze_batch(
    batch: list[dict[str, Any]],
    batch_number: int,
    total_batches: int,
    dataset_total: int
) -> dict[str, Any]:
    payload = [
        {
            "id": row["id"],
            "customer": row["customer_name"],
            "channel": row["channel"],
            "date": row["date"],
            "plan": row["plan"],
            "customer_type": row["customer_type"],
            "revenue": row["revenue"],
            "message": row["original_message"],
        }
        for row in batch
    ]

    prompt = f"""
You are Customer Feedback AI's customer-intelligence classifier, a product from Nuvanta AI.

You are analyzing batch {batch_number} of {total_batches}.
The complete uploaded dataset contains {dataset_total} records.
This batch contains {len(batch)} records because the backend processes
the complete dataset in small internal batches.

Classify EVERY supplied record. Return exactly one result per ID.
Never skip, merge, invent, or rewrite customer facts.

For every record determine:

1. sentiment:
positive, negative, or neutral.

2. feature_request:
true only when the customer explicitly or implicitly asks for a
missing capability, integration, feature, action, or functionality.
A question such as "Do you support X?" can be a feature request when
the context indicates the capability is missing or desired.

3. churn_signal:
true only when there is meaningful retention risk. Consider evidence
such as cancellation intent, renewal concern, repeated unresolved
problems, blocked workflows, alternatives, declining value, or
important unmet needs.

Negative sentiment alone is NOT churn.

4. unresolved_issue:
true when the record indicates a customer problem remains unresolved.

5. topic:
Use a short consistent label such as billing, pricing, login,
support, integration, reporting, performance, reliability,
onboarding, feature request, product usability, account management,
delivery, or other.

6. problem:
Use a short consistent problem category when a meaningful problem exists.
Otherwise return an empty string.

7. summary:
Write 2 to 4 concise sentences explaining what this conversation reveals.
Combine the customer's actual need, problem, expectation, and business implication
when the evidence supports them. Keep it specific to this conversation.
Do not mention AI, classification, sentiment labels, or these instructions.
Do not repeat the source message word-for-word.

Use only supplied customer evidence.
Do not infer unsupported facts.
Do not use outside information.

RECORDS:
{json.dumps(payload, ensure_ascii=False)}
"""

    print(
        f"[AI] START batch {batch_number}/{total_batches}: "
        f"records={len(batch)}"
    )

    result = ai_json(
        prompt,
        "customer_feedback_batch",
        BATCH_SCHEMA,
        f"batch {batch_number}/{total_batches}",
        AI_BATCH_MAX_OUTPUT_TOKENS
    )

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Batch {batch_number} returned invalid JSON."
        )

    records = result.get("records")
    if not isinstance(records, list):
        raise RuntimeError(
            f"Batch {batch_number} did not return records."
        )

    expected_ids = {
        str(row["id"])
        for row in batch
    }
    returned_ids = {
        clean_text(item.get("id"))
        for item in records
        if isinstance(item, dict)
    }

    missing = expected_ids - returned_ids
    if missing:
        raise RuntimeError(
            f"Batch {batch_number} missed record IDs: "
            f"{sorted(missing)[:10]}"
        )

    print(
        f"[AI] COMPLETE batch {batch_number}/{total_batches}"
    )
    return result


def apply_batch_result(
    rows: list[dict[str, Any]],
    result: dict[str, Any]
) -> None:
    by_id = {
        str(row["id"]): row
        for row in rows
    }

    for item in result.get("records", []):
        if not isinstance(item, dict):
            continue

        sid = clean_text(item.get("id"))
        row = by_id.get(sid)
        if row is None:
            continue

        sentiment = clean_text(
            item.get("sentiment")
        ).lower()
        if sentiment not in {
            "positive",
            "negative",
            "neutral"
        }:
            sentiment = "neutral"

        row["sentiment"] = sentiment
        row["feature_request"] = bool(
            item.get("feature_request", False)
        )
        row["churn_signal"] = bool(
            item.get("churn_signal", False)
        )
        row["unresolved_issue"] = bool(
            item.get("unresolved_issue", False)
        )
        row["topic"] = clean_text(
            item.get("topic")
        )[:80]
        row["problem"] = clean_text(
            item.get("problem")
        )[:100]
        # The UI intentionally presents one combined data-backed reading.
        # Keep the legacy reason columns empty for compatibility with the
        # existing database schema rather than generating duplicate prose.
        row["feature_reason"] = ""
        row["churn_reason"] = ""
        row["feedback_summary"] = clean_text(
            item.get("summary")
        )[:700]


# ============================================================
# AGGREGATION
# ============================================================

def calculate_aggregates(
    rows: list[dict[str, Any]]
) -> dict[str, Any]:
    total = len(rows)

    positive = sum(
        row.get("sentiment") == "positive"
        for row in rows
    )
    negative = sum(
        row.get("sentiment") == "negative"
        for row in rows
    )
    neutral = sum(
        row.get("sentiment") == "neutral"
        for row in rows
    )
    feature_count = sum(
        bool(row.get("feature_request"))
        for row in rows
    )
    churn_count = sum(
        bool(row.get("churn_signal"))
        for row in rows
    )
    unresolved_count = sum(
        bool(row.get("unresolved_issue"))
        for row in rows
    )

    topic_counter = Counter()
    problem_counter = Counter()
    feature_topic_counter = Counter()
    churn_topic_counter = Counter()
    channel_counter = Counter()
    plan_counter = Counter()
    customer_type_counter = Counter()

    for row in rows:
        topic = clean_text(row.get("topic")).lower()
        problem = clean_text(row.get("problem")).lower()

        if topic:
            topic_counter[topic] += 1
        if problem:
            problem_counter[problem] += 1

        if row.get("feature_request") and topic:
            feature_topic_counter[topic] += 1

        if row.get("churn_signal") and topic:
            churn_topic_counter[topic] += 1

        channel = clean_text(row.get("channel")).lower()
        if channel:
            channel_counter[channel] += 1

        plan = clean_text(row.get("plan"))
        if plan:
            plan_counter[plan] += 1

        customer_type = clean_text(
            row.get("customer_type")
        )
        if customer_type:
            customer_type_counter[customer_type] += 1

    return {
        "total_records": total,
        "sentiment": {
            "positive": int(positive),
            "negative": int(negative),
            "neutral": int(neutral),
            "percentages": {
                "positive": round(
                    positive * 100 / max(total, 1), 1
                ),
                "negative": round(
                    negative * 100 / max(total, 1), 1
                ),
                "neutral": round(
                    neutral * 100 / max(total, 1), 1
                ),
            },
        },
        "feature_requests": {
            "count": int(feature_count),
            "percentage": round(
                feature_count * 100 / max(total, 1), 1
            ),
        },
        "churn_signals": {
            "count": int(churn_count),
            "percentage": round(
                churn_count * 100 / max(total, 1), 1
            ),
        },
        "unresolved_issues": {
            "count": int(unresolved_count),
            "percentage": round(
                unresolved_count * 100 / max(total, 1), 1
            ),
        },
        "top_topics": [
            {
                "name": name,
                "count": count,
                "percentage": round(
                    count * 100 / max(total, 1), 1
                ),
            }
            for name, count in topic_counter.most_common(10)
        ],
        "top_problems": [
            {
                "name": name,
                "count": count,
                "percentage": round(
                    count * 100 / max(total, 1), 1
                ),
            }
            for name, count in problem_counter.most_common(10)
        ],
        "feature_topics": [
            {
                "name": name,
                "count": count,
                "percentage": round(
                    count * 100 / max(total, 1), 1
                ),
            }
            for name, count in feature_topic_counter.most_common(10)
        ],
        "churn_topics": [
            {
                "name": name,
                "count": count,
                "percentage": round(
                    count * 100 / max(total, 1), 1
                ),
            }
            for name, count in churn_topic_counter.most_common(10)
        ],
        "channels": dict(channel_counter),
        "plans": [
            {"name": name, "count": count}
            for name, count in plan_counter.most_common(10)
        ],
        "customer_types": [
            {"name": name, "count": count}
            for name, count
            in customer_type_counter.most_common(10)
        ],
    }


# ============================================================
# FINAL AI SUMMARY
# ============================================================

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "priority": {"type": "string"},
        "top_problem_summary": {"type": "string"},
        "customer_theme_summary": {"type": "string"},
        "feature_request_insight": {"type": "string"},
        "churn_insight": {"type": "string"},
    },
    "required": [
        "headline",
        "summary",
        "priority",
        "top_problem_summary",
        "customer_theme_summary",
        "feature_request_insight",
        "churn_insight",
    ],
    "additionalProperties": False,
}


RECOMMENDED_QUESTIONS = [
    "Which feature are customers most complaining about?",
    "Which features should we add to the roadmap, especially where high-profile customers are asking even though request volume is lower?",
    "Which clients are likely to churn and why? Prioritize high-profile customers first.",
    "Which customer problems are causing the most unresolved friction and what should we fix first?",
    "Which customer segments or accounts show the strongest combination of business impact and recurring product pain?",
]


def generate_final_summary(
    rows: list[dict[str, Any]],
    aggregates: dict[str, Any]
) -> dict[str, Any]:
    total = len(rows)

    prompt = f"""
You are Customer Feedback AI's senior customer-intelligence analyst, a product from Nuvanta AI.

The backend has classified EVERY ONE of the {total} uploaded records.

Create a concise founder-level summary using ONLY these complete-dataset
aggregates. Do not invent or alter counts.

Return:
headline: one concise business headline.
summary: 4 to 6 sentences about the overall customer situation.
priority: the single most important business action.
top_problem_summary: explain the most important recurring problem.
customer_theme_summary: explain the strongest customer themes.
feature_request_insight: 2 to 4 concise sentences explaining what
the customer feedback reveals about feature demand. Treat subtle
capability questions as requests when the evidence supports it, for
example "Do you support X?" followed by an indication that X is not
currently supported. Do not invent a feature request.
churn_insight: 2 to 4 concise sentences explaining subtle retention
risk from the evidence. Consider recent unresolved questions, repeated
friction, blocked workflows, declining confidence, alternatives, or
unmet needs. Negative sentiment alone is not churn. If the evidence
does not show meaningful retention risk, say that clearly.

These feature and churn insights must be analytical conclusions from
the supplied customer records, not scripted or generic dashboard copy.
Use the evidence included in the aggregates when explaining the pattern,
but never invent facts beyond the supplied data.

Never say sample, sampled, subset, representative, or imply that only
some records were analyzed.

COMPLETE DATASET:
{total} records

AGGREGATES AND EVIDENCE:
{json.dumps(aggregates, ensure_ascii=False)}
"""

    return ai_json(
        prompt,
        "customer_feedback_final_summary",
        FINAL_SCHEMA,
        "final full-dataset summary",
        AI_FINAL_MAX_OUTPUT_TOKENS
    )


# ============================================================
# FINAL ANALYSIS OBJECT
# ============================================================

def build_final_analysis(
    rows: list[dict[str, Any]],
    final_summary: dict[str, Any],
    aggregates: dict[str, Any],
    completed_batches: int
) -> dict[str, Any]:
    total = len(rows)
    groups = grouped_rows(rows)

    feature_records = []
    for row in rows:
        if not row.get("feature_request"):
            continue

        feature_records.append(
            {
                "id": row["id"],
                "customer": row["customer_name"],
                "message": row["original_message"],
                "summary": row.get("feedback_summary", ""),
                "reason": row.get("feature_reason", ""),
                "channel": row.get("channel", "chat"),
                "date": row.get("date", ""),
                "revenue": parse_revenue(
                    row.get("revenue", 0)
                ),
                "customer_type": row.get(
                    "customer_type",
                    ""
                ),
                "sentiment": row.get("sentiment", "neutral"),
                "topic": row.get("topic", ""),
                "source_ids": [row["id"]],
            }
        )

    # Highest-value customers first, then evidence strength.
    feature_records.sort(
        key=lambda item: (
            item["revenue"],
            len(item.get("reason", "")),
        ),
        reverse=True
    )

    churn_records = []
    for row in rows:
        if not row.get("churn_signal"):
            continue

        severity = "medium"
        if (
            row.get("unresolved_issue")
            and row.get("sentiment") == "negative"
        ):
            severity = "high"

        churn_records.append(
            {
                "id": row["id"],
                "customer": row["customer_name"],
                "message": row["original_message"],
                "evidence": row.get(
                    "feedback_summary",
                    ""
                ),
                "reason": row.get(
                    "churn_reason",
                    ""
                ),
                "severity": severity,
                "plan": row.get("plan", ""),
                "channel": row.get("channel", "chat"),
                "date": row.get("date", ""),
                "revenue": parse_revenue(
                    row.get("revenue", 0)
                ),
                "customer_type": row.get(
                    "customer_type",
                    ""
                ),
                "sentiment": row.get("sentiment", "neutral"),
                "topic": row.get("topic", ""),
                "unresolved_issue": bool(row.get("unresolved_issue")),
                "source_ids": [row["id"]],
            }
        )

    severity_rank = {
        "high": 2,
        "medium": 1,
        "low": 0,
    }

    # High-profile customers are intentionally prioritized.
    churn_records.sort(
        key=lambda item: (
            item["revenue"],
            severity_rank.get(item["severity"], 0),
        ),
        reverse=True
    )

    high_value = []
    for row in sorted(
        rows,
        key=lambda r: parse_revenue(
            r.get("revenue", 0)
        ),
        reverse=True
    ):
        revenue = parse_revenue(row.get("revenue", 0))
        if revenue <= 0:
            continue

        high_value.append(
            {
                "id": row["id"],
                "customer": row["customer_name"],
                "revenue": revenue,
                "message": row["original_message"],
                "evidence": row.get(
                    "feedback_summary",
                    ""
                ),
            }
        )

        if len(high_value) >= 20:
            break

    segment_counter = Counter()
    for row in rows:
        segment = (
            clean_text(row.get("customer_type"))
            or clean_text(row.get("plan"))
            or clean_text(row.get("channel"))
            or "Other"
        )
        segment_counter[segment] += 1

    segments = []
    for segment, count in segment_counter.most_common(10):
        matching = [
            row for row in rows
            if (
                clean_text(row.get("customer_type"))
                or clean_text(row.get("plan"))
                or clean_text(row.get("channel"))
                or "Other"
            ) == segment
        ]

        negative = sum(
            row.get("sentiment") == "negative"
            for row in matching
        )
        churn = sum(
            bool(row.get("churn_signal"))
            for row in matching
        )
        revenue = sum(
            parse_revenue(row.get("revenue", 0))
            for row in matching
        )

        segments.append(
            {
                "segment": segment,
                "count": count,
                "percentage": round(
                    count * 100 / max(total, 1), 1
                ),
                "negative": int(negative),
                "churn": int(churn),
                "revenue": revenue,
            }
        )

    groups_count = len(groups)
    source_counts = Counter(
        row.get("channel", "chat")
        for row in rows
    )

    def evidence_date(row: dict[str, Any]) -> pd.Timestamp:
        try:
            value = pd.to_datetime(
                row.get("date", ""),
                errors="coerce"
            )
            if pd.isna(value):
                return pd.Timestamp("1900-01-01")
            return value
        except Exception:
            return pd.Timestamp("1900-01-01")

    return {
        "total": total,
        "total_records": total,
        "analyzed_records": total,
        "analyzed_all_records": True,
        "analysis_complete": True,
        "analysis_batch_size": AI_BATCH_SIZE,
        "analysis_batches": completed_batches,
        "unique_customers": groups_count,

        "overview": {
            "headline": clean_text(
                final_summary.get("headline")
            ),
            "summary": clean_text(
                final_summary.get("summary")
            ),
            "priority": clean_text(
                final_summary.get("priority")
            ),
            "source_ids": [],
            "ai_generated": True,
            "coverage": f"Analyzed all {total:,} records",
        },

        "sentiment": aggregates["sentiment"],

        "problems": [
            {
                "name": item["name"],
                "mentions": item["count"],
                "percentage": item["percentage"],
                "summary": (
                    final_summary.get(
                        "top_problem_summary",
                        ""
                    )
                    if index == 0
                    else (
                        f"This issue appears in "
                        f"{item['count']:,} of "
                        f"{total:,} records."
                    )
                ),
                "source_ids": [],
            }
            for index, item
            in enumerate(
                aggregates["top_problems"][:10]
            )
        ],

        "feature_requests": feature_records[:100],
        "feature_request_summary": aggregates[
            "feature_requests"
        ],

        "churn_signals": churn_records[:100],
        "churn_summary": aggregates[
            "churn_signals"
        ],
        "feature_request_insight": clean_text(
            final_summary.get("feature_request_insight", "")
        ),
        "churn_insight": clean_text(
            final_summary.get("churn_insight", "")
        ),

        "unresolved_issues": aggregates[
            "unresolved_issues"
        ],

        "top_topics": aggregates["top_topics"],
        "high_value": high_value,
        "segments": segments,

        "source_counts": {
            "email": source_counts.get("email", 0),
            "call": source_counts.get("call", 0),
            "chat": source_counts.get("chat", 0),
        },

        "suggested_questions": RECOMMENDED_QUESTIONS,

        "aggregates": aggregates,

        "analysis_method": (
            f"OpenAI analyzed all {total:,} uploaded records "
            f"in small internal batches of up to {AI_BATCH_SIZE} "
            f"records. Classifications and aggregate counts were "
            f"calculated across all analyzed records."
        ),

        "coverage": {
            "total_records": total,
            "analyzed_records": total,
            "remaining_records": 0,
            "percentage": 100,
            "label": f"Analyzed all {total:,} records",
        },
    }


# ============================================================
# FULL DATASET ANALYSIS
# ============================================================

def build_ai_analysis(
    rows: list[dict[str, Any]],
    run_id: int,
    analysis_rows: Optional[list[dict[str, Any]]] = None,
    initially_cached: int = 0,
) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        raise RuntimeError(
            "No records available for analysis."
        )

    # Only records without a stored AI classification are sent to OpenAI.
    # Previously analyzed records are reused from earlier analysis CSV files.
    pending_rows = analysis_rows if analysis_rows is not None else rows

    # Start with the normal configured batch size. If OpenAI cannot
    # finish a batch within the output-token budget, that batch is
    # automatically split into smaller batches and retried.
    pending_total = len(pending_rows)
    initial_batches = [
        pending_rows[start:start + AI_BATCH_SIZE]
        for start in range(
            0,
            pending_total,
            AI_BATCH_SIZE
        )
    ]

    print(
        f"[AI] FULL DATASET: records={total}, "
        f"cached={initially_cached}, new={pending_total}, "
        f"batch_size={AI_BATCH_SIZE}, "
        f"batches={len(initial_batches)}"
    )

    processed = initially_cached
    completed_batches = 0
    pending_batches = list(initial_batches)

    def update_status(message: str) -> None:
        with ANALYSIS_LOCK:
            ANALYSIS_STATUS["processed_records"] = processed
            ANALYSIS_STATUS["completed_batches"] = completed_batches
            ANALYSIS_STATUS["total_batches"] = (
                completed_batches + len(pending_batches)
            )
            ANALYSIS_STATUS["message"] = message

    if not pending_batches:
        with ANALYSIS_LOCK:
            ANALYSIS_STATUS["message"] = (
                f"All {total:,} records were already analyzed; "
                "reusing the stored classifications."
            )

    while pending_batches:
        batch = pending_batches.pop(0)
        batch_size = len(batch)

        # The displayed batch number is the number of successfully
        # completed requests plus the current request. It is only a
        # progress label; record coverage is tracked separately.
        batch_number = completed_batches + 1
        total_batches_for_display = (
            completed_batches + 1 + len(pending_batches)
        )

        try:
            result = analyze_batch(
                batch,
                batch_number,
                total_batches_for_display,
                total
            )
        except AIOutputTokenLimitError as exc:
            if batch_size <= 1:
                raise RuntimeError(
                    "OpenAI could not complete analysis for a single "
                    "record within the configured output-token budget. "
                    "Increase AI_BATCH_MAX_OUTPUT_TOKENS or shorten the "
                    "record output fields."
                ) from exc

            midpoint = max(1, batch_size // 2)
            left = batch[:midpoint]
            right = batch[midpoint:]

            print(
                f"[AI] OUTPUT LIMIT batch records={batch_size}; "
                f"splitting into {len(left)} + {len(right)} and retrying"
            )

            # Put the two smaller batches at the front so the failed
            # batch is retried immediately and progress remains ordered.
            pending_batches.insert(0, right)
            pending_batches.insert(0, left)

            update_status(
                f"Output limit reached for a {batch_size}-record batch. "
                f"Retrying as {len(left)} + {len(right)}..."
            )
            continue

        apply_batch_result(rows, result)

        # Persist each successfully analyzed batch immediately so the
        # analysis CSV contains the AI-enriched feedback even if a later
        # batch or the final summary request fails.
        save_analysis_records(run_id, batch)

        processed += batch_size
        completed_batches += 1

        update_status(
            f"Analyzing all {total:,} records - "
            f"{processed:,}/{total:,} processed"
        )

    aggregates = calculate_aggregates(rows)

    final_summary = generate_final_summary(
        rows,
        aggregates
    )

    return build_final_analysis(
        rows,
        final_summary,
        aggregates,
        completed_batches
    )


# ============================================================
# ANALYSIS WORKER
# ============================================================

def run_analysis(
    rows: list[dict[str, Any]],
    dataset_name: str,
    run_id: int,
    dataset_fingerprint: str
) -> None:
    global DATA
    global ANALYSIS
    global LAST_ASK
    global ANALYSIS_STATUS
    global CURRENT_ANALYSIS_RUN_ID

    try:
        with ANALYSIS_LOCK:
            DATA = rows
            ANALYSIS = None
            LAST_ASK = None
            CURRENT_ANALYSIS_RUN_ID = run_id

            total = len(rows)
            total_batches = math.ceil(total / AI_BATCH_SIZE)

            ANALYSIS_STATUS = {
                "status": "processing",
                "message": (
                    f"Checking {total:,} customer records against stored AI analysis..."
                ),
                "error": None,
                "analysis": None,
                "total_records": total,
                "processed_records": 0,
                "batch_size": AI_BATCH_SIZE,
                "completed_batches": 0,
                "total_batches": total_batches,
                "analysis_run_id": run_id,
                "dataset_name": dataset_name,
            }

        # Reuse classifications that are already persisted. Only genuinely
        # new feedback records are sent to OpenAI.
        new_rows, reused_count = hydrate_cached_analysis(rows)

        with ANALYSIS_LOCK:
            ANALYSIS_STATUS["processed_records"] = reused_count
            ANALYSIS_STATUS["message"] = (
                f"Reused stored AI analysis for {reused_count:,} records. "
                f"{len(new_rows):,} new records require analysis."
            )

        exact_saved = None
        if not new_rows:
            exact_saved = load_analysis_by_dataset_fingerprint(
                dataset_fingerprint
            )

        if exact_saved is not None:
            final = exact_saved["analysis"]
            completed_batches = int(
                final.get("analysis_batches", 0)
                if isinstance(final, dict)
                else 0
            )
            save_analysis_records(run_id, rows)
            mark_analysis_complete(run_id, final)

            with ANALYSIS_LOCK:
                ANALYSIS = final
                ANALYSIS_STATUS = {
                    "status": "complete",
                    "message": (
                        f"Loaded stored analysis - all {len(rows):,} "
                        f"records were already analyzed."
                    ),
                    "error": None,
                    "analysis": final,
                    "total_records": len(rows),
                    "processed_records": len(rows),
                    "batch_size": AI_BATCH_SIZE,
                    "completed_batches": completed_batches,
                    "total_batches": completed_batches,
                    "analysis_run_id": run_id,
                    "dataset_name": dataset_name,
                }

            print(
                f"[STORAGE] REUSED COMPLETE DATASET: run={run_id}, "
                f"records={len(rows)}"
            )
            return

        final = build_ai_analysis(
            rows,
            run_id,
            analysis_rows=new_rows,
            initially_cached=reused_count,
        )

        # Persist the complete snapshot for this run. The per-batch writes in
        # build_ai_analysis also protect against losing successfully classified
        # records if a later request fails.
        save_analysis_records(run_id, rows)
        mark_analysis_complete(run_id, final)

        with ANALYSIS_LOCK:
            ANALYSIS = final
            ANALYSIS_STATUS = {
                "status": "complete",
                "message": (
                    f"Analysis complete - {len(rows):,} records stored in the analysis CSV."
                ),
                "error": None,
                "analysis": final,
                "total_records": len(rows),
                "processed_records": len(rows),
                "batch_size": AI_BATCH_SIZE,
                "completed_batches": final.get(
                    "analysis_batches", 0
                ),
                "total_batches": final.get(
                    "analysis_batches", 0
                ),
                "analysis_run_id": run_id,
                "dataset_name": dataset_name,
            }

        print(
            f"[AI] COMPLETE AND SAVED: run={run_id}, "
            f"records={len(rows)}, reused={reused_count}, new={len(new_rows)}"
        )

    except Exception as exc:
        print(
            f"[AI] ERROR run={run_id}: {exc!r}"
        )

        try:
            mark_analysis_error(
                run_id,
                str(exc)
            )
        except Exception as storage_exc:
            print(
                f"[STORAGE] Could not mark analysis error: "
                f"{storage_exc!r}"
            )

        with ANALYSIS_LOCK:
            ANALYSIS_STATUS = {
                "status": "error",
                "message": "AI analysis failed.",
                "error": str(exc),
                "analysis": None,
                "total_records": len(rows),
                "processed_records": 0,
                "batch_size": AI_BATCH_SIZE,
                "completed_batches": 0,
                "total_batches": math.ceil(
                    len(rows) / AI_BATCH_SIZE
                ),
                "analysis_run_id": run_id,
                "dataset_name": dataset_name,
            }


def start_analysis(
    rows: list[dict[str, Any]],
    dataset_name: str
) -> dict[str, Any]:
    global DATA
    global ANALYSIS
    global LAST_ASK
    global ANALYSIS_STATUS
    global CURRENT_ANALYSIS_RUN_ID

    total = len(rows)
    dataset_fingerprint = make_dataset_fingerprint(rows)
    run_id = create_analysis_run(
        dataset_name,
        total,
        dataset_fingerprint,
    )

    with ANALYSIS_LOCK:
        DATA = rows
        ANALYSIS = None
        LAST_ASK = None
        CURRENT_ANALYSIS_RUN_ID = run_id

        total_batches = math.ceil(
            total / AI_BATCH_SIZE
        )

        ANALYSIS_STATUS = {
            "status": "processing",
            "message": (
                f"AI analysis is starting across "
                f"all {total:,} records..."
            ),
            "error": None,
            "analysis": None,
            "total_records": total,
            "processed_records": 0,
            "batch_size": AI_BATCH_SIZE,
            "completed_batches": 0,
            "total_batches": total_batches,
            "analysis_run_id": run_id,
            "dataset_name": dataset_name,
        }

    thread = threading.Thread(
        target=run_analysis,
        args=(rows, dataset_name, run_id, dataset_fingerprint),
        daemon=True
    )
    thread.start()

    return {
        "status": "processing",
        "message": (
            f"AI analysis is starting across "
            f"all {total:,} records..."
        ),
        "total": total,
        "total_records": total,
        "analyzed_records": 0,
        "analyzed_all_records": False,
        "batch_size": AI_BATCH_SIZE,
        "total_batches": total_batches,
        "analysis_run_id": run_id,
        "dataset_name": dataset_name,
    }


# ============================================================
# RETRIEVAL FOR ASK YOUR DATA
# ============================================================

def tokenize(text: str) -> set[str]:
    words = re.findall(
        r"[a-zA-Z0-9]+",
        clean_text(text).lower()
    )

    stop = {
        "the", "a", "an", "and", "or", "is", "are", "to",
        "of", "for", "in", "on", "our", "we", "what", "why",
        "how", "do", "does", "me", "show", "customer",
        "customers", "feedback", "please", "can", "could",
        "would", "tell", "about", "from", "with", "most",
        "many", "much", "their", "they", "this", "that",
        "which", "should", "add", "roadmap", "first",
    }

    return {
        word
        for word in words
        if len(word) > 2 and word not in stop
    }


def lexical_score(
    question: str,
    row: dict[str, Any]
) -> float:
    q = clean_text(question).lower()
    qt = tokenize(question)

    enriched = " ".join(
        [
            clean_text(row.get("customer_name")),
            clean_text(row.get("customer_email")),
            clean_text(row.get("channel")),
            clean_text(row.get("plan")),
            clean_text(row.get("customer_type")),
            clean_text(row.get("subject")),
            clean_text(row.get("message")),
            clean_text(row.get("feedback_summary")),
            clean_text(row.get("topic")),
            clean_text(row.get("problem")),
            clean_text(row.get("feature_reason")),
            clean_text(row.get("churn_reason")),
        ]
    )
    rt = tokenize(enriched)

    score = float(len(qt & rt))

    if q and q in enriched.lower():
        score += 8

    if any(
        x in q
        for x in [
            "churn",
            "leaving",
            "cancel",
            "renewal",
            "retention",
            "risk",
            "likely to leave",
        ]
    ):
        if row.get("churn_signal"):
            score += 20
        if row.get("unresolved_issue"):
            score += 5
        score += min(
            parse_revenue(row.get("revenue", 0)) / 25000,
            8
        )

    if any(
        x in q
        for x in [
            "feature",
            "request",
            "integration",
            "api",
            "capability",
            "roadmap",
            "add",
            "support",
        ]
    ):
        if row.get("feature_request"):
            score += 20
        score += min(
            parse_revenue(row.get("revenue", 0)) / 25000,
            8
        )

    if any(
        x in q
        for x in [
            "complaint",
            "problem",
            "issue",
            "frustration",
            "negative",
            "pain",
            "unresolved",
        ]
    ):
        if row.get("unresolved_issue"):
            score += 12
        if row.get("sentiment") == "negative":
            score += 5

    if any(
        x in q
        for x in [
            "high value",
            "high-value",
            "revenue",
            "largest",
            "valuable",
            "important",
            "profile",
        ]
    ):
        score += min(
            parse_revenue(row.get("revenue", 0)) / 10000,
            10
        )

    return score


def retrieve(
    question: str,
    limit: int = MAX_ASK_SOURCES
) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            (lexical_score(question, row), row)
            for row in DATA
        ),
        key=lambda item: item[0],
        reverse=True
    )

    selected = []
    selected_ids = set()

    for score, row in ranked:
        if score > 0 or len(selected) < min(6, limit):
            if row["id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["id"])

        if len(selected) >= limit:
            break

    return selected


# ============================================================
# ASK YOUR DATA AI
# ============================================================

ASK_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "key_findings": {
            "type": "array",
            "items": {"type": "string"}
        },
        "recommendation": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {"type": "string"}
        },
        "confidence": {"type": "string"},
    },
    "required": [
        "answer",
        "key_findings",
        "recommendation",
        "sources",
        "confidence",
    ],
    "additionalProperties": False,
}


def ask_ai(
    question: str,
    sources: list[dict[str, Any]]
) -> dict[str, Any]:
    if not sources:
        raise RuntimeError(
            "There is not enough uploaded customer evidence "
            "to answer this question reliably."
        )

    evidence = [
        {
            "source_id": row["id"],
            "customer": row["customer_name"],
            "customer_id": row.get("customer_id", ""),
            "channel": row["channel"],
            "date": row["date"],
            "plan": row.get("plan", ""),
            "customer_type": row.get(
                "customer_type",
                ""
            ),
            "revenue": row.get("revenue", 0),
            "topic": row.get("topic", ""),
            "problem": row.get("problem", ""),
            "sentiment": row.get(
                "sentiment",
                "neutral"
            ),
            "feature_request": row.get(
                "feature_request",
                False
            ),
            "churn_signal": row.get(
                "churn_signal",
                False
            ),
            "unresolved_issue": row.get(
                "unresolved_issue",
                False
            ),
            "feature_reason": row.get(
                "feature_reason",
                ""
            ),
            "churn_reason": row.get(
                "churn_reason",
                ""
            ),
            "original_message": row.get(
                "original_message",
                row.get("message", "")
            ),
        }
        for row in sources
    ]

    prompt = f"""
You are Customer Feedback AI's customer-intelligence analyst, a product from Nuvanta AI.

The complete uploaded dataset contains {len(DATA):,} records.
The evidence below was retrieved from the analyzed uploaded dataset.

Question:
{question}

Answer ONLY from the supplied evidence.

This is a business intelligence task, not a generic advice task.
The answer must be grounded in actual customer records.

Important rules:
- Never invent customer statements, counts, revenue, trends, or facts.
- Cite evidence using [SOURCE <id>] in the answer or findings.
- Every cited source ID must be one of the supplied source_id values.
- In the returned sources array, include ONLY source IDs that directly support the answer.
- Do not cite a source merely because it is related; the cited conversation must contain the evidence for the claim.
- Negative sentiment alone does not mean churn.
- For churn questions, consider actual churn signals, unresolved
  problems, recency, repeated friction, blocked workflows, renewal
  concerns, alternatives, declining value, and unmet needs.
- For churn questions, prioritize high-value or high-profile customers
  when the evidence supports that priority.
- For roadmap questions, balance request frequency with customer value,
  profile, sentiment, recurring pain, and business impact.
- A feature request may be implicit, such as asking whether a missing
  capability or integration is supported.
- If the evidence is insufficient for a strong conclusion, say so.
- Do not present a negative customer as a churn risk unless the evidence
  supports actual retention risk.

Return:
answer: a direct business answer.
key_findings: 3 to 5 concise evidence-based findings.
recommendation: one practical action based on the evidence.
sources: source IDs that directly support the answer.
confidence: high, medium, or low.

EVIDENCE:
{json.dumps(evidence, ensure_ascii=False)}
"""

    return ai_json(
        prompt,
        "customer_data_answer",
        ASK_SCHEMA,
        "question answering",
        AI_ASK_MAX_OUTPUT_TOKENS
    )


# ============================================================
# API MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str


# ============================================================
# API
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Nuvanta AI",
        "version": "6.0.0",
        "status": "running",
        "conversations": len(DATA),
        "ai": bool(get_openai_client()),
        "analysis_batch_size": AI_BATCH_SIZE,
        "endpoints": [
            "/api/health",
            "/api/upload",
            "/api/analysis/status",
            "/api/analysis",
            "/api/ask",
            "/api/conversations",
            "/api/last-ask",
            "/api/suggestions",
            "/api/customers/{customer_id}",
            "/api/analysis-runs",
            "/api/database/status",
        ],
    }


@app.get("/api/health")
def health():
    with ANALYSIS_LOCK:
        status = dict(ANALYSIS_STATUS)

    return {
        "ok": True,
        "loaded": len(DATA),
        "openai": bool(get_openai_client()),
        "openai_package": OpenAI is not None,
        "model": OPENAI_MODEL,
        "analysis_status": status.get("status"),
        "analysis_total_records": status.get(
            "total_records",
            len(DATA)
        ),
        "analysis_processed_records": status.get(
            "processed_records",
            0
        ),
        "analysis_completed_batches": status.get(
            "completed_batches",
            0
        ),
        "analysis_total_batches": status.get(
            "total_batches",
            0
        ),
        "analysis_storage_dir": ANALYSIS_STORAGE_DIR,
    }


@app.get("/api/analysis/status")
def analysis_status():
    with ANALYSIS_LOCK:
        return dict(ANALYSIS_STATUS)


@app.get("/api/analysis")
def get_analysis():
    with ANALYSIS_LOCK:
        if ANALYSIS is None:
            raise HTTPException(
                status_code=404,
                detail="AI analysis is not complete yet."
            )
        return ANALYSIS


@app.get("/api/analysis-runs")
def get_analysis_runs():
    runs = _iter_saved_runs()[:50]
    return {
        "runs": [
            {
                "id": int(run.get("id", 0)),
                "dataset_name": run.get("dataset_name", "uploaded.csv"),
                "created_at": run.get("created_at"),
                "completed_at": run.get("completed_at"),
                "total_records": int(run.get("total_records", 0)),
                "status": run.get("status", "unknown"),
            }
            for run in runs
        ]
    }


@app.post("/api/upload")
async def upload_csv(
    file: UploadFile = File(...)
):
    require_openai()

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The CSV file is empty."
            )

        df = pd.read_csv(
            io.BytesIO(contents)
        )

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The CSV contains no rows."
            )

        rows = normalize_dataframe(df)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the CSV: {exc}"
        )

    if not rows:
        raise HTTPException(
            status_code=400,
            detail=(
                "No usable customer conversations "
                "were found in the CSV."
            )
        )

    return start_analysis(
        rows,
        file.filename
    )


# ============================================================
# CONVERSATIONS
# ============================================================

@app.get("/api/conversations")
def get_conversations(
    ids: Optional[str] = None
):
    if not DATA:
        raise HTTPException(
            status_code=404,
            detail="Upload customer data first."
        )

    groups = grouped_rows(DATA)

    if ids:
        wanted = {
            x.strip()
            for x in ids.split(",")
            if x.strip()
        }
        return [
            source_record(
                row,
                groups.get(str(row.get("customer_id")))
            )
            for row in DATA
            if row["id"] in wanted
        ]

    # Return every analyzed feedback record. The frontend filters these records
    # by sentiment/category/channel, so the displayed counts always match the
    # full analyzed dataset instead of a small customer-deduplicated subset.
    return [
        source_record(
            row,
            groups.get(str(row.get("customer_id")))
        )
        for row in DATA
    ]

@app.get("/api/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str
):
    groups = grouped_rows(DATA)

    for row in DATA:
        if row["id"] == conversation_id:
            return source_record(
                row,
                groups.get(
                    str(row.get("customer_id"))
                )
            )

    raise HTTPException(
        status_code=404,
        detail="Conversation not found."
    )


# ============================================================
# ASK YOUR DATA
# ============================================================

@app.post("/api/ask")
def ask_customer_data(
    body: AskRequest
):
    global LAST_ASK

    if not DATA:
        raise HTTPException(
            status_code=404,
            detail="Upload customer data first."
        )

    if ANALYSIS is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Wait for the full dataset analysis "
                "to complete first."
            )
        )

    question = clean_text(body.question)
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    require_openai()

    sources = retrieve(question)

    try:
        result = ask_ai(
            question,
            sources
        )
    except Exception as exc:
        print(
            f"[AI] ERROR question answering: {exc!r}"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"The AI could not answer the question: {exc}"
            )
        )

    valid = {
        row["id"]
        for row in sources
    }

    # The supporting-conversation list must be an exact subset of the
    # sources explicitly cited in the generated answer/findings/recommendation.
    # Never fall back to arbitrary retrieved conversations.
    cited_text = " ".join(
        [
            clean_text(result.get("answer", "")),
            " ".join(
                clean_text(x)
                for x in result.get("key_findings", [])
            ),
            clean_text(result.get("recommendation", "")),
        ]
    )

    cited_ids = []
    for match in re.finditer(
        r"\[\s*SOURCE\s+([^\]\s]+)\s*\]",
        cited_text,
        flags=re.IGNORECASE,
    ):
        source_id = clean_text(match.group(1))
        if source_id in valid and source_id not in cited_ids:
            cited_ids.append(source_id)

    # Prefer the exact IDs cited in the answer. The model's sources field is
    # used only for citations that were actually present in the answer text.
    model_source_ids = [
        clean_text(x)
        for x in result.get("sources", [])
        if clean_text(x) in valid
    ]
    source_ids = [
        source_id
        for source_id in cited_ids
        if source_id in model_source_ids or not model_source_ids
    ]

    # If the model's structured sources list is incomplete, still preserve
    # every valid source explicitly cited in the answer.
    for source_id in cited_ids:
        if source_id not in source_ids:
            source_ids.append(source_id)

    source_ids = list(
        dict.fromkeys(source_ids)
    )

    groups = grouped_rows(DATA)
    source_id_set = set(source_ids)

    final_sources = [
        source_record(
            row,
            groups.get(
                str(row.get("customer_id"))
            )
        )
        for row in DATA
        if row["id"] in source_id_set
    ]

    LAST_ASK = {
        "question": question,
        "source_ids": source_ids,
        "sources": final_sources,
    }

    return {
        "question": question,
        "answer": clean_text(
            result.get("answer")
        ),
        "key_findings": [
            clean_text(x)
            for x in result.get(
                "key_findings",
                []
            )
            if clean_text(x)
        ],
        "recommendation": clean_text(
            result.get("recommendation")
        ),
        "confidence": (
            clean_text(
                result.get("confidence")
            )
            or "medium"
        ),
        "ai_used": True,
        "source_count": len(final_sources),
        "source_ids": source_ids,
        "sources": final_sources,
        "dataset_total": len(DATA),
        "analysis_scope": (
            f"All {len(DATA):,} uploaded records"
        ),
        "analysis_run_id": CURRENT_ANALYSIS_RUN_ID,
        "suggested_questions": ANALYSIS.get(
            "suggested_questions",
            RECOMMENDED_QUESTIONS
        ),
    }


@app.get("/api/last-ask")
def last_ask():
    return (
        LAST_ASK
        or {
            "question": "",
            "source_ids": [],
            "sources": [],
        }
    )


@app.get("/api/suggestions")
def suggestions():
    if ANALYSIS is None:
        raise HTTPException(
            status_code=404,
            detail="AI analysis is not complete yet."
        )

    return {
        "questions": ANALYSIS.get(
            "suggested_questions",
            RECOMMENDED_QUESTIONS
        )
    }


# ============================================================
# CUSTOMER
# ============================================================

@app.get("/api/customers/{customer_id}")
def get_customer(
    customer_id: str
):
    matches = [
        row
        for row in DATA
        if str(row.get("customer_id"))
        == str(customer_id)
    ]

    if not matches:
        raise HTTPException(
            status_code=404,
            detail="Customer not found."
        )

    groups = grouped_rows(DATA)

    return {
        "customer_id": customer_id,
        "customer_name": matches[0]["customer_name"],
        "conversation_count": len(matches),
        "conversations": [
            source_record(
                row,
                groups.get(
                    str(row.get("customer_id"))
                )
            )
            for row in matches
        ],
    }


@app.get("/api/database/status")
def database_status():
    """Return a read-only diagnostic for the local CSV/JSON analysis store."""
    runs = _iter_saved_runs()
    complete_runs = [
        run for run in runs
        if run.get("status") == "complete"
    ]

    csv_files = [
        run.get("csv_file")
        for run in runs
        if clean_text(run.get("csv_file"))
        and os.path.isfile(_run_csv_path(run))
    ]

    stored_feedback_records = 0
    for run in complete_runs:
        try:
            stored_feedback_records += len(
                _read_run_records(run)
            )
        except Exception:
            pass

    return {
        "storage_type": "csv_files",
        "storage_directory": ANALYSIS_STORAGE_DIR,
        "storage_index": ANALYSIS_RUNS_INDEX,
        "storage_exists": os.path.isdir(ANALYSIS_STORAGE_DIR),
        "analysis_runs": len(runs),
        "complete_analysis_runs": len(complete_runs),
        "analysis_csv_files": len(csv_files),
        "stored_feedback_records": int(stored_feedback_records),
        "current_memory_records": len(DATA),
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    global ANALYSIS_STATUS

    init_storage()
    print(f"[STORAGE] CSV analysis store: {ANALYSIS_STORAGE_DIR}")

    try:
        load_latest_saved_analysis_into_memory()
    except Exception as exc:
        print(
            f"[STORAGE] Could not load saved analysis: {exc!r}"
        )
        ANALYSIS_STATUS = {
            "status": "idle",
            "message": (
                "Upload customer feedback to begin AI analysis."
            ),
            "error": None,
            "analysis": None,
            "total_records": 0,
            "processed_records": 0,
            "batch_size": AI_BATCH_SIZE,
            "completed_batches": 0,
            "total_batches": 0,
            "analysis_run_id": None,
            "dataset_name": None,
        }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=int(
            os.getenv("PORT", "8000")
        ),
        reload=False
    )

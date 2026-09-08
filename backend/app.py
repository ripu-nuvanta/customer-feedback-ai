import os
import io
import json
import re
import math
import threading
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

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-mini"
).strip()

# IMPORTANT:
# This is the INTERNAL AI batch size.
# It does NOT mean that only this many records are analyzed.
AI_BATCH_SIZE = int(
    os.getenv("AI_BATCH_SIZE", "30")
)

AI_TIMEOUT = float(
    os.getenv("AI_TIMEOUT", "75")
)

MAX_ASK_SOURCES = int(
    os.getenv("MAX_ASK_SOURCES", "14")
)

# Maximum output for a 30-record classification request.
# The response is deliberately tiny.
AI_BATCH_MAX_OUTPUT_TOKENS = int(
    os.getenv("AI_BATCH_MAX_OUTPUT_TOKENS", "3500")
)

# Small final summary request.
AI_FINAL_MAX_OUTPUT_TOKENS = int(
    os.getenv("AI_FINAL_MAX_OUTPUT_TOKENS", "1800")
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Customer Feedback AI",
    version="5.0.0"
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

ANALYSIS_STATUS: dict[str, Any] = {
    "status": "idle",
    "message": "Load customer feedback to begin AI analysis.",
    "error": None,
    "analysis": None,
    "total_records": 0,
    "processed_records": 0,
    "batch_size": AI_BATCH_SIZE,
    "completed_batches": 0,
    "total_batches": 0,
}

ANALYSIS_LOCK = threading.Lock()


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
            return (
                0.0
                if math.isnan(float(value))
                else float(value)
            )
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
        candidates.insert(
            0,
            fenced.group(1).strip()
        )

    first_obj = text.find("{")
    last_obj = text.rfind("}")

    if first_obj >= 0 and last_obj > first_obj:
        candidates.append(
            text[first_obj:last_obj + 1]
        )

    first_arr = text.find("[")
    last_arr = text.rfind("]")

    if first_arr >= 0 and last_arr > first_arr:
        candidates.append(
            text[first_arr:last_arr + 1]
        )

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    return None


def response_text(response: Any) -> str:
    direct = clean_text(
        getattr(response, "output_text", "")
    )

    if direct:
        return direct

    chunks: list[str] = []

    output = getattr(
        response,
        "output",
        None
    ) or []

    for item in output:
        content = getattr(
            item,
            "content",
            None
        ) or []

        for part in content:
            text = getattr(
                part,
                "text",
                None
            )

            if isinstance(text, str):
                if text.strip():
                    chunks.append(text.strip())

            elif text is not None:
                value = getattr(
                    text,
                    "value",
                    None
                )

                if isinstance(value, str):
                    if value.strip():
                        chunks.append(value.strip())

            parsed = getattr(
                part,
                "parsed",
                None
            )

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
        print(
            f"[AI] client initialization failed: {exc!r}"
        )
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
                "Set OPENAI_API_KEY in backend/.env "
                "and restart the backend."
            )
        )

    return client


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

        print(
            f"[AI] ERROR {label} request: {exc!r}"
        )

        raise RuntimeError(
            f"OpenAI request failed: {exc}"
        ) from exc

    raw = response_text(response)

    parsed = parse_json(raw)

    if parsed is not None:
        return parsed

    status = clean_text(
        getattr(response, "status", "")
    )

    incomplete = getattr(
        response,
        "incomplete_details",
        None
    )

    reason = ""

    if incomplete:
        reason = clean_text(
            getattr(
                incomplete,
                "reason",
                ""
            )
        )

    preview = raw[:300].replace(
        "\n",
        " "
    )

    print(
        f"[AI] INVALID JSON {label}: "
        f"status={status!r}, "
        f"reason={reason!r}, "
        f"output_preview={preview!r}"
    )

    if reason == "max_output_tokens":

        raise RuntimeError(
            f"OpenAI response reached the output-token limit "
            f"during {label}. "
            f"The request was too large."
        )

    raise RuntimeError(
        "OpenAI returned a response that could not "
        "be parsed as the required JSON. "
        f"status={status or 'unknown'}"
        + (
            f", reason={reason}"
            if reason
            else ""
        )
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

            if (
                key
                and (
                    key in normalized_name
                    or normalized_name in key
                )
            ):
                return original

    return None


def normalize_channel(value: Any) -> str:

    text = clean_text(value).lower()

    if any(
        word in text
        for word in [
            "email",
            "mail"
        ]
    ):
        return "email"

    if any(
        word in text
        for word in [
            "call",
            "phone",
            "telephone",
            "voice",
            "transcript"
        ]
    ):
        return "call"

    if any(
        word in text
        for word in [
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
                "account, or name."
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

    rows: list[dict[str, Any]] = []

    for index, raw in df.iterrows():

        message = clean_text(
            raw.get(
                message_col,
                ""
            )
        )

        if not message:
            continue

        customer_name = clean_text(
            raw.get(
                customer_name_col,
                ""
            )
        ) or "Unknown customer"

        customer_email = (
            clean_text(
                raw.get(
                    customer_email_col,
                    ""
                )
            )
            if customer_email_col
            else ""
        )

        explicit_customer_id = (
            clean_text(
                raw.get(
                    customer_id_col,
                    ""
                )
            )
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
                "id": str(
                    len(rows) + 1
                ),

                "customer_id": customer_id,

                "customer_email": customer_email,

                "customer_name": customer_name,

                "date": (
                    safe_date(
                        raw.get(
                            date_col,
                            ""
                        )
                    )
                    if date_col
                    else ""
                ),

                "channel": (
                    normalize_channel(
                        raw.get(
                            channel_col,
                            ""
                        )
                    )
                    if channel_col
                    else "chat"
                ),

                "subject": (
                    clean_text(
                        raw.get(
                            subject_col,
                            ""
                        )
                    )
                    if subject_col
                    else ""
                ),

                "message": message,

                "original_message": message,

                "plan": (
                    clean_text(
                        raw.get(
                            plan_col,
                            ""
                        )
                    )
                    if plan_col
                    else ""
                ),

                "customer_type": (
                    clean_text(
                        raw.get(
                            customer_type_col,
                            ""
                        )
                    )
                    if customer_type_col
                    else ""
                ),

                "revenue": (
                    parse_revenue(
                        raw.get(
                            revenue_col,
                            0
                        )
                    )
                    if revenue_col
                    else 0.0
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
# CUSTOMER THREADS
# ============================================================

def grouped_rows(
    rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:

    groups: dict[
        str,
        list[dict[str, Any]]
    ] = defaultdict(list)

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
    group: Optional[
        list[dict[str, Any]]
    ] = None
) -> dict[str, Any]:

    turns = []

    if group and len(group) > 1:

        for item in group:

            turns.append(
                {
                    "id": item["id"],
                    "date": item.get(
                        "date",
                        ""
                    ),
                    "channel": item.get(
                        "channel",
                        "chat"
                    ),
                    "subject": item.get(
                        "subject",
                        ""
                    ),
                    "message": item.get(
                        "original_message",
                        item.get(
                            "message",
                            ""
                        )
                    ),
                }
            )

    return {
        "id": row["id"],

        "customer_id": row.get(
            "customer_id",
            ""
        ),

        "customer_email": row.get(
            "customer_email",
            ""
        ),

        "customer_name": row.get(
            "customer_name",
            "Unknown customer"
        ),

        "customer": row.get(
            "customer_name",
            "Unknown customer"
        ),

        "date": row.get(
            "date",
            ""
        ),

        "channel": row.get(
            "channel",
            "chat"
        ),

        "subject": row.get(
            "subject",
            ""
        ),

        "message": row.get(
            "original_message",
            row.get(
                "message",
                ""
            )
        ),

        "original_message": row.get(
            "original_message",
            row.get(
                "message",
                ""
            )
        ),

        "feedback_summary": row.get(
            "feedback_summary",
            ""
        ),

        "ai_generated_summary": bool(
            row.get(
                "feedback_summary"
            )
        ),

        "plan": row.get(
            "plan",
            ""
        ),

        "customer_type": row.get(
            "customer_type",
            ""
        ),

        "revenue": row.get(
            "revenue",
            0
        ),

        "topic": row.get(
            "topic",
            ""
        ),

        "problem": row.get(
            "problem",
            ""
        ),

        "sentiment": row.get(
            "sentiment",
            "neutral"
        ),

        "feature_request": bool(
            row.get(
                "feature_request",
                False
            )
        ),

        "churn_signal": bool(
            row.get(
                "churn_signal",
                False
            )
        ),

        "unresolved_issue": bool(
            row.get(
                "unresolved_issue",
                False
            )
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
# COMPACT AI BATCH SCHEMA
# ============================================================

BATCH_SCHEMA = {
    "type": "object",

    "properties": {
        "records": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "id": {
                        "type": "string"
                    },

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

                    "topic": {
                        "type": "string"
                    },

                    "problem": {
                        "type": "string"
                    },

                    "feature_reason": {
                        "type": "string"
                    },

                    "churn_reason": {
                        "type": "string"
                    },

                    "summary": {
                        "type": "string"
                    },
                },

                "required": [
                    "id",
                    "sentiment",
                    "feature_request",
                    "churn_signal",
                    "unresolved_issue",
                    "topic",
                    "problem",
                    "feature_reason",
                    "churn_reason",
                    "summary",
                ],

                "additionalProperties": False,
            },
        }
    },

    "required": [
        "records"
    ],

    "additionalProperties": False,
}


# ============================================================
# FINAL SUMMARY SCHEMA
# ============================================================

FINAL_SCHEMA = {
    "type": "object",

    "properties": {

        "headline": {
            "type": "string"
        },

        "summary": {
            "type": "string"
        },

        "priority": {
            "type": "string"
        },

        "top_problem_summary": {
            "type": "string"
        },

        "customer_theme_summary": {
            "type": "string"
        },

        "suggested_questions": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
    },

    "required": [
        "headline",
        "summary",
        "priority",
        "top_problem_summary",
        "customer_theme_summary",
        "suggested_questions",
    ],

    "additionalProperties": False,
}


# ============================================================
# AI BATCH ANALYSIS
# ============================================================

def analyze_batch(
    batch: list[dict[str, Any]],
    batch_number: int,
    total_batches: int,
    dataset_total: int
) -> dict[str, Any]:

    payload = []

    for row in batch:

        payload.append(
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
        )

    prompt = f"""
You are Nuvanta AI's customer-intelligence classifier.

This is batch {batch_number} of {total_batches}.

The complete uploaded dataset contains {dataset_total} records.

This request contains {len(batch)} records only because the backend processes the complete dataset in small internal batches.

IMPORTANT:
- Classify EVERY record in this batch.
- Do not skip records.
- Do not combine records.
- Return exactly one result for every supplied record ID.
- Use semantic understanding.
- Do not invent facts.
- Do not invent customer statements.
- Do not use information from outside the supplied records.

For EACH record determine:

1. sentiment:
   positive, negative, or neutral

2. feature_request:
   true only when the customer explicitly or implicitly asks for a missing capability, integration, feature, or functionality.

3. churn_signal:
   true only when there is meaningful retention risk such as cancellation intent, renewal concern, repeated unresolved problems, blocked workflows, alternatives, declining value, or important unmet needs.

4. unresolved_issue:
   true when the record indicates an unresolved customer problem.

5. topic:
   Use a short consistent topic label such as:
   billing, pricing, login, support, integration, reporting,
   performance, reliability, onboarding, feature request,
   product usability, account management, delivery, other.

6. problem:
   If there is a meaningful customer problem, use a short consistent category.
   Otherwise use an empty string.

7. feature_reason:
   Very short reason if feature_request is true.
   Otherwise empty string.

8. churn_reason:
   Very short reason if churn_signal is true.
   Otherwise empty string.

9. summary:
   One short sentence describing the key meaning of the record.
   Keep it concise.

Do NOT return long explanations.

Return only the required JSON.

RECORDS:
{json.dumps(payload, ensure_ascii=False)}
"""

    print(
        f"[AI] START batch "
        f"{batch_number}/{total_batches}: "
        f"records={len(batch)}, "
        f"dataset_total={dataset_total}"
    )

    result = ai_json(
        prompt=prompt,
        schema_name="customer_feedback_batch",
        schema=BATCH_SCHEMA,
        label=f"batch {batch_number}/{total_batches}",
        max_output_tokens=AI_BATCH_MAX_OUTPUT_TOKENS,
    )

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Batch {batch_number} returned invalid JSON structure."
        )

    records = result.get(
        "records"
    )

    if not isinstance(records, list):
        raise RuntimeError(
            f"Batch {batch_number} did not return a records array."
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

    missing_ids = expected_ids - returned_ids

    extra_ids = returned_ids - expected_ids

    if missing_ids:

        raise RuntimeError(
            f"Batch {batch_number} did not classify "
            f"all records. Missing IDs: "
            f"{sorted(missing_ids)[:10]}"
        )

    if extra_ids:

        print(
            f"[AI] WARNING batch "
            f"{batch_number}: "
            f"ignored unexpected IDs "
            f"{sorted(extra_ids)[:10]}"
        )

    print(
        f"[AI] COMPLETE batch "
        f"{batch_number}/{total_batches}: "
        f"records={len(batch)}"
    )

    return result


# ============================================================
# DETERMINISTIC AGGREGATION
# ============================================================

def apply_batch_result(
    rows: list[dict[str, Any]],
    batch_result: dict[str, Any]
) -> None:

    by_id = {
        str(row["id"]): row
        for row in rows
    }

    records = batch_result.get(
        "records",
        []
    )

    for item in records:

        if not isinstance(item, dict):
            continue

        sid = clean_text(
            item.get("id")
        )

        if sid not in by_id:
            continue

        row = by_id[sid]

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
            item.get(
                "feature_request",
                False
            )
        )

        row["churn_signal"] = bool(
            item.get(
                "churn_signal",
                False
            )
        )

        row["unresolved_issue"] = bool(
            item.get(
                "unresolved_issue",
                False
            )
        )

        row["topic"] = clean_text(
            item.get("topic")
        )[:80]

        row["problem"] = clean_text(
            item.get("problem")
        )[:100]

        row["feature_reason"] = clean_text(
            item.get("feature_reason")
        )[:250]

        row["churn_reason"] = clean_text(
            item.get("churn_reason")
        )[:250]

        row["feedback_summary"] = clean_text(
            item.get("summary")
        )[:500]


# ============================================================
# DETERMINISTIC AGGREGATES
# ============================================================

def calculate_aggregates(
    rows: list[dict[str, Any]]
) -> dict[str, Any]:

    total = len(rows)

    positive = sum(
        1
        for row in rows
        if row.get("sentiment") == "positive"
    )

    negative = sum(
        1
        for row in rows
        if row.get("sentiment") == "negative"
    )

    neutral = sum(
        1
        for row in rows
        if row.get("sentiment") == "neutral"
    )

    feature_count = sum(
        1
        for row in rows
        if row.get("feature_request")
    )

    churn_count = sum(
        1
        for row in rows
        if row.get("churn_signal")
    )

    unresolved_count = sum(
        1
        for row in rows
        if row.get("unresolved_issue")
    )

    topic_counter = Counter()

    problem_counter = Counter()

    feature_topic_counter = Counter()

    churn_topic_counter = Counter()

    channel_counter = Counter()

    plan_counter = Counter()

    customer_type_counter = Counter()

    for row in rows:

        topic = clean_text(
            row.get("topic")
        ).lower()

        problem = clean_text(
            row.get("problem")
        ).lower()

        if topic:
            topic_counter[topic] += 1

        if problem:
            problem_counter[problem] += 1

        if row.get("feature_request"):

            if topic:
                feature_topic_counter[
                    topic
                ] += 1

        if row.get("churn_signal"):

            if topic:
                churn_topic_counter[
                    topic
                ] += 1

        channel = clean_text(
            row.get("channel")
        ).lower()

        if channel:
            channel_counter[
                channel
            ] += 1

        plan = clean_text(
            row.get("plan")
        )

        if plan:
            plan_counter[
                plan
            ] += 1

        customer_type = clean_text(
            row.get("customer_type")
        )

        if customer_type:
            customer_type_counter[
                customer_type
            ] += 1

    sentiment_percentages = {
        "positive": round(
            positive * 100 / max(total, 1),
            1
        ),
        "negative": round(
            negative * 100 / max(total, 1),
            1
        ),
        "neutral": round(
            neutral * 100 / max(total, 1),
            1
        ),
    }

    return {
        "total_records": total,

        "sentiment": {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "percentages": sentiment_percentages,
        },

        "feature_requests": {
            "count": feature_count,
            "percentage": round(
                feature_count * 100 / max(total, 1),
                1
            ),
        },

        "churn_signals": {
            "count": churn_count,
            "percentage": round(
                churn_count * 100 / max(total, 1),
                1
            ),
        },

        "unresolved_issues": {
            "count": unresolved_count,
            "percentage": round(
                unresolved_count * 100 / max(total, 1),
                1
            ),
        },

        "top_topics": [
            {
                "name": name,
                "count": count,
                "percentage": round(
                    count * 100 / max(total, 1),
                    1
                ),
            }
            for name, count
            in topic_counter.most_common(10)
        ],

        "top_problems": [
            {
                "name": name,
                "count": count,
                "percentage": round(
                    count * 100 / max(total, 1),
                    1
                ),
            }
            for name, count
            in problem_counter.most_common(10)
        ],

        "feature_topics": [
            {
                "name": name,
                "count": count,
                "percentage": round(
                    count * 100 / max(total, 1),
                    1
                ),
            }
            for name, count
            in feature_topic_counter.most_common(10)
        ],

        "churn_topics": [
            {
                "name": name,
                "count": count,
                "percentage": round(
                    count * 100 / max(total, 1),
                    1
                ),
            }
            for name, count
            in churn_topic_counter.most_common(10)
        ],

        "channels": dict(
            channel_counter
        ),

        "plans": [
            {
                "name": name,
                "count": count,
            }
            for name, count
            in plan_counter.most_common(10)
        ],

        "customer_types": [
            {
                "name": name,
                "count": count,
            }
            for name, count
            in customer_type_counter.most_common(10)
        ],
    }


# ============================================================
# FINAL AI SUMMARY
# ============================================================

def generate_final_summary(
    rows: list[dict[str, Any]],
    aggregates: dict[str, Any]
) -> dict[str, Any]:

    total = len(rows)

    prompt = f"""
You are Nuvanta AI's senior customer-intelligence analyst.

The backend has already analyzed EVERY ONE of the {total} customer records.

These are deterministic aggregates calculated from all {total} analyzed records.

You are NOT being asked to classify records.
You are NOT being asked to estimate counts.
You are NOT being given a sample.

Your job is only to turn the supplied complete-dataset aggregates into a concise business summary.

IMPORTANT:
- Always refer to the complete dataset.
- Never say sample.
- Never say sampled.
- Never say representative sample.
- Never say "based on a subset".
- Never imply that only some records were analyzed.
- Sentiment numbers come from all {total} records.
- Feature counts come from all {total} records.
- Churn counts come from all {total} records.
- Problem/topic counts come from all {total} records.
- Do not invent statistics.
- Do not change any supplied count.

The final answer should be concise and useful to a founder.

Return:

1. headline:
   One concise business headline.

2. summary:
   4-6 sentences explaining the overall customer situation.
   Explicitly make clear that the analysis covers all {total} records.

3. priority:
   The most important business action.

4. top_problem_summary:
   Explain the most important recurring problem using the supplied aggregate numbers.

5. customer_theme_summary:
   Explain the strongest customer themes using the supplied aggregate numbers.

6. suggested_questions:
   Exactly 5 useful questions a founder could ask next.

Never use the words:
sample
sampled
subset
representative
90 samples
representative evidence

COMPLETE DATASET SIZE:
{total}

AGGREGATES:
{json.dumps(aggregates, ensure_ascii=False)}
"""

    print(
        f"[AI] START final summary: "
        f"all_records={total}"
    )

    result = ai_json(
        prompt=prompt,
        schema_name="customer_feedback_final_summary",
        schema=FINAL_SCHEMA,
        label="final full-dataset summary",
        max_output_tokens=AI_FINAL_MAX_OUTPUT_TOKENS,
    )

    if not isinstance(result, dict):
        raise RuntimeError(
            "Final AI summary returned invalid JSON."
        )

    return result


# ============================================================
# BUILD FINAL ANALYSIS
# ============================================================

def build_final_analysis(
    rows: list[dict[str, Any]],
    final_summary: dict[str, Any],
    aggregates: dict[str, Any],
    completed_batches: int
) -> dict[str, Any]:

    total = len(rows)

    groups = grouped_rows(rows)

    source_counts = Counter(
        row.get(
            "channel",
            "chat"
        )
        for row in rows
    )

    feature_records = []

    for row in rows:

        if not row.get(
            "feature_request"
        ):
            continue

        feature_records.append(
            {
                "id": row["id"],
                "customer": row["customer_name"],
                "message": row["original_message"],
                "summary": row.get(
                    "feedback_summary",
                    ""
                ),
                "reason": row.get(
                    "feature_reason",
                    ""
                ),
                "channel": row.get(
                    "channel",
                    "chat"
                ),
                "date": row.get(
                    "date",
                    ""
                ),
                "source_ids": [
                    row["id"]
                ],
            }
        )

    churn_records = []

    for row in rows:

        if not row.get(
            "churn_signal"
        ):
            continue

        severity = "medium"

        if (
            row.get("sentiment")
            == "negative"
            and row.get(
                "unresolved_issue"
            )
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
                "plan": row.get(
                    "plan",
                    ""
                ),
                "channel": row.get(
                    "channel",
                    "chat"
                ),
                "date": row.get(
                    "date",
                    ""
                ),
                "source_ids": [
                    row["id"]
                ],
            }
        )

    high_value = []

    for row in sorted(
        rows,
        key=lambda r: parse_revenue(
            r.get(
                "revenue",
                0
            )
        ),
        reverse=True
    ):

        revenue = parse_revenue(
            row.get(
                "revenue",
                0
            )
        )

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

    segments = []

    segment_counter = Counter()

    for row in rows:

        segment = (
            clean_text(
                row.get(
                    "customer_type"
                )
            )
            or clean_text(
                row.get(
                    "plan"
                )
            )
            or clean_text(
                row.get(
                    "channel"
                )
            )
            or "Other"
        )

        segment_counter[
            segment
        ] += 1

    for segment, count in segment_counter.most_common(10):

        matching = [
            row
            for row in rows
            if (
                clean_text(
                    row.get(
                        "customer_type"
                    )
                )
                or clean_text(
                    row.get(
                        "plan"
                    )
                )
                or clean_text(
                    row.get(
                        "channel"
                    )
                )
                or "Other"
            ) == segment
        ]

        negative = sum(
            1
            for row in matching
            if row.get(
                "sentiment"
            ) == "negative"
        )

        churn = sum(
            1
            for row in matching
            if row.get(
                "churn_signal"
            )
        )

        segments.append(
            {
                "segment": segment,
                "count": count,
                "percentage": round(
                    count * 100 / max(total, 1),
                    1
                ),
                "negative": negative,
                "churn": churn,
            }
        )

    questions = []

    for question in (
        final_summary.get(
            "suggested_questions",
            []
        )
        if isinstance(
            final_summary.get(
                "suggested_questions",
                []
            ),
            list
        )
        else []
    ):

        q = re.sub(
            r"\s+",
            " ",
            clean_text(question)
        ).strip()

        if (
            q
            and q not in questions
        ):
            questions.append(
                q[:120]
            )

    fallback_questions = [
        "Which customer problems appear most frequently across all records?",
        "Which issues are creating the highest churn risk?",
        "What feature requests appear most often across all customer feedback?",
        "Which customer segments have the most negative feedback?",
        "What should we prioritize based on the full customer feedback dataset?",
    ]

    for question in fallback_questions:

        if len(questions) >= 5:
            break

        if question not in questions:
            questions.append(
                question
            )

    questions = questions[:5]

    analysis = {

        # ----------------------------------------------------
        # COMPLETE DATASET
        # ----------------------------------------------------

        "total": total,

        "total_records": total,

        "analyzed_records": total,

        "analyzed_all_records": True,

        "analysis_complete": True,

        "analysis_batch_size": AI_BATCH_SIZE,

        "analysis_batches": completed_batches,

        "unique_customers": len(
            groups
        ),

        # ----------------------------------------------------
        # OVERVIEW
        # ----------------------------------------------------

        "overview": {

            "headline": clean_text(
                final_summary.get(
                    "headline"
                )
            ),

            "summary": clean_text(
                final_summary.get(
                    "summary"
                )
            ),

            "priority": clean_text(
                final_summary.get(
                    "priority"
                )
            ),

            "source_ids": [],

            "ai_generated": True,

            "coverage": (
                f"Analyzed all {total:,} records"
            ),
        },

        # ----------------------------------------------------
        # SENTIMENT
        # ----------------------------------------------------

        "sentiment": aggregates[
            "sentiment"
        ],

        # ----------------------------------------------------
        # PROBLEMS
        # ----------------------------------------------------

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
                aggregates[
                    "top_problems"
                ][:10]
            )
        ],

        # ----------------------------------------------------
        # FEATURE REQUESTS
        # ----------------------------------------------------

        "feature_requests": feature_records[:100],

        "feature_request_summary": aggregates[
            "feature_requests"
        ],

        # ----------------------------------------------------
        # CHURN
        # ----------------------------------------------------

        "churn_signals": churn_records[:100],

        "churn_summary": aggregates[
            "churn_signals"
        ],

        # ----------------------------------------------------
        # UNRESOLVED
        # ----------------------------------------------------

        "unresolved_issues": aggregates[
            "unresolved_issues"
        ],

        # ----------------------------------------------------
        # TOPICS
        # ----------------------------------------------------

        "top_topics": aggregates[
            "top_topics"
        ],

        # ----------------------------------------------------
        # HIGH VALUE
        # ----------------------------------------------------

        "high_value": high_value,

        # ----------------------------------------------------
        # SEGMENTS
        # ----------------------------------------------------

        "segments": segments,

        # ----------------------------------------------------
        # CHANNELS
        # ----------------------------------------------------

        "source_counts": {
            "email": source_counts.get(
                "email",
                0
            ),
            "call": source_counts.get(
                "call",
                0
            ),
            "chat": source_counts.get(
                "chat",
                0
            ),
        },

        # ----------------------------------------------------
        # QUESTIONS
        # ----------------------------------------------------

        "suggested_questions": questions,

        # ----------------------------------------------------
        # AGGREGATES
        # ----------------------------------------------------

        "aggregates": aggregates,

        # ----------------------------------------------------
        # ANALYSIS METHOD
        # ----------------------------------------------------

        "analysis_method": (
            f"OpenAI analyzed all {total:,} "
            f"uploaded records in small internal "
            f"batches of up to {AI_BATCH_SIZE} records. "
            f"Sentiment, feature-request, churn, "
            f"problem, and topic counts were calculated "
            f"deterministically across all {total:,} "
            f"analyzed records. The internal batch size "
            f"does not represent the number of records "
            f"used for the final analysis."
        ),

        # ----------------------------------------------------
        # COVERAGE
        # ----------------------------------------------------

        "coverage": {
            "total_records": total,
            "analyzed_records": total,
            "remaining_records": 0,
            "percentage": 100,
            "label": (
                f"Analyzed all {total:,} records"
            ),
        },
    }

    return analysis


# ============================================================
# FULL DATASET ANALYSIS
# ============================================================

def build_ai_analysis(
    rows: list[dict[str, Any]]
) -> dict[str, Any]:

    total = len(rows)

    if total == 0:
        raise RuntimeError(
            "No records available for analysis."
        )

    batches = [
        rows[start:start + AI_BATCH_SIZE]
        for start in range(
            0,
            total,
            AI_BATCH_SIZE
        )
    ]

    total_batches = len(batches)

    print(
        f"[AI] FULL DATASET ANALYSIS: "
        f"records={total}, "
        f"batch_size={AI_BATCH_SIZE}, "
        f"batches={total_batches}"
    )

    completed_batches = 0

    for batch_index, batch in enumerate(
        batches,
        start=1
    ):

        result = analyze_batch(
            batch=batch,
            batch_number=batch_index,
            total_batches=total_batches,
            dataset_total=total,
        )

        apply_batch_result(
            rows,
            result
        )

        completed_batches = batch_index

        with ANALYSIS_LOCK:

            ANALYSIS_STATUS[
                "processed_records"
            ] = min(
                completed_batches
                * AI_BATCH_SIZE,
                total
            )

            ANALYSIS_STATUS[
                "completed_batches"
            ] = completed_batches

            ANALYSIS_STATUS[
                "total_batches"
            ] = total_batches

            ANALYSIS_STATUS[
                "message"
            ] = (
                f"Analyzing all {total:,} records "
                f"— "
                f"{min(completed_batches * AI_BATCH_SIZE, total):,}"
                f"/{total:,} processed"
            )

    # --------------------------------------------------------
    # DETERMINISTIC FULL-DATASET AGGREGATION
    # --------------------------------------------------------

    aggregates = calculate_aggregates(
        rows
    )

    print(
        f"[AI] ALL RECORDS CLASSIFIED: "
        f"total={total}, "
        f"positive={aggregates['sentiment']['positive']}, "
        f"negative={aggregates['sentiment']['negative']}, "
        f"neutral={aggregates['sentiment']['neutral']}, "
        f"features={aggregates['feature_requests']['count']}, "
        f"churn={aggregates['churn_signals']['count']}"
    )

    # --------------------------------------------------------
    # SMALL FINAL AI REQUEST
    # --------------------------------------------------------

    final_summary = generate_final_summary(
        rows,
        aggregates
    )

    final = build_final_analysis(
        rows=rows,
        final_summary=final_summary,
        aggregates=aggregates,
        completed_batches=completed_batches,
    )

    print(
        f"[AI] COMPLETE FULL DATASET ANALYSIS: "
        f"total={total}"
    )

    return final


# ============================================================
# ANALYSIS WORKER
# ============================================================

def run_analysis(
    rows: list[dict[str, Any]]
) -> None:

    global DATA
    global ANALYSIS
    global LAST_ASK
    global ANALYSIS_STATUS

    try:

        with ANALYSIS_LOCK:

            DATA = rows

            ANALYSIS = None

            LAST_ASK = None

            total = len(rows)

            total_batches = math.ceil(
                total / AI_BATCH_SIZE
            )

            ANALYSIS_STATUS = {

                "status": "processing",

                "message": (
                    f"Analyzing all {total:,} "
                    f"customer records…"
                ),

                "error": None,

                "analysis": None,

                "total_records": total,

                "processed_records": 0,

                "batch_size": AI_BATCH_SIZE,

                "completed_batches": 0,

                "total_batches": total_batches,
            }

        final = build_ai_analysis(
            rows
        )

        with ANALYSIS_LOCK:

            ANALYSIS = final

            ANALYSIS_STATUS = {

                "status": "complete",

                "message": (
                    f"Analysis complete — "
                    f"all {len(rows):,} records analyzed."
                ),

                "error": None,

                "analysis": final,

                "total_records": len(rows),

                "processed_records": len(rows),

                "batch_size": AI_BATCH_SIZE,

                "completed_batches": math.ceil(
                    len(rows) / AI_BATCH_SIZE
                ),

                "total_batches": math.ceil(
                    len(rows) / AI_BATCH_SIZE
                ),
            }

        print(
            f"[AI] COMPLETE: "
            f"all {len(rows)} records analyzed"
        )

    except Exception as exc:

        print(
            f"[AI] ERROR full dataset analysis: "
            f"{exc!r}"
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
            }


def start_analysis(
    rows: list[dict[str, Any]]
) -> dict[str, Any]:

    global ANALYSIS_STATUS
    global ANALYSIS
    global LAST_ASK
    global DATA

    total = len(rows)

    with ANALYSIS_LOCK:

        DATA = rows

        ANALYSIS = None

        LAST_ASK = None

        ANALYSIS_STATUS = {

            "status": "processing",

            "message": (
                f"Analyzing all {total:,} "
                f"customer records…"
            ),

            "error": None,

            "analysis": None,

            "total_records": total,

            "processed_records": 0,

            "batch_size": AI_BATCH_SIZE,

            "completed_batches": 0,

            "total_batches": math.ceil(
                total / AI_BATCH_SIZE
            ),
        }

    thread = threading.Thread(
        target=run_analysis,
        args=(rows,),
        daemon=True
    )

    thread.start()

    return {

        "status": "processing",

        "message": (
            f"AI analysis is starting across "
            f"all {total:,} records…"
        ),

        "total": total,

        "total_records": total,

        "analyzed_records": 0,

        "analyzed_all_records": False,

        "batch_size": AI_BATCH_SIZE,

        "total_batches": math.ceil(
            total / AI_BATCH_SIZE
        ),
    }


# ============================================================
# RETRIEVAL
# ============================================================

def tokenize(
    text: str
) -> set[str]:

    words = re.findall(
        r"[a-zA-Z0-9]+",
        clean_text(text).lower()
    )

    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "is",
        "are",
        "to",
        "of",
        "for",
        "in",
        "on",
        "our",
        "we",
        "what",
        "why",
        "how",
        "do",
        "does",
        "me",
        "show",
        "customer",
        "customers",
        "feedback",
        "please",
        "can",
        "could",
        "would",
        "tell",
        "about",
        "from",
        "with",
        "most",
        "many",
        "much",
        "their",
        "they",
        "this",
        "that",
    }

    return {
        word
        for word in words
        if len(word) > 2
        and word not in stop
    }


def lexical_score(
    question: str,
    row: dict[str, Any]
) -> float:

    q = question.lower()

    qt = tokenize(
        question
    )

    text = " ".join(
        [
            row.get(
                "customer_name",
                ""
            ),
            row.get(
                "customer_email",
                ""
            ),
            row.get(
                "channel",
                ""
            ),
            row.get(
                "plan",
                ""
            ),
            row.get(
                "customer_type",
                ""
            ),
            row.get(
                "subject",
                ""
            ),
            row.get(
                "message",
                ""
            ),
            row.get(
                "feedback_summary",
                ""
            ),
            row.get(
                "feature_reason",
                ""
            ),
            row.get(
                "churn_reason",
                ""
            ),
        ]
    )

    rt = tokenize(
        text
    )

    score = float(
        len(
            qt & rt
        )
    )

    if q in text.lower():
        score += 8

    if any(
        x in q
        for x in [
            "churn",
            "leaving",
            "cancel",
            "renewal",
            "retention",
            "risk"
        ]
    ):

        if row.get(
            "churn_signal"
        ):
            score += 10

    if any(
        x in q
        for x in [
            "feature",
            "request",
            "integration",
            "api",
            "capability",
            "support"
        ]
    ):

        if row.get(
            "feature_request"
        ):
            score += 10

    if any(
        x in q
        for x in [
            "complaint",
            "problem",
            "issue",
            "frustration",
            "negative"
        ]
    ):

        if row.get(
            "sentiment"
        ) == "negative":
            score += 5

    if any(
        x in q
        for x in [
            "high value",
            "high-value",
            "revenue",
            "largest",
            "valuable"
        ]
    ):

        score += min(
            parse_revenue(
                row.get(
                    "revenue",
                    0
                )
            ) / 10000,
            6
        )

    return score


def retrieve(
    question: str,
    limit: int = MAX_ASK_SOURCES
) -> list[dict[str, Any]]:

    ranked = sorted(
        (
            (
                lexical_score(
                    question,
                    row
                ),
                row
            )
            for row in DATA
        ),
        key=lambda x: x[0],
        reverse=True
    )

    selected = []

    for score, row in ranked:

        if (
            score > 0
            or len(selected)
            < min(5, limit)
        ):

            selected.append(
                row
            )

        if len(selected) >= limit:
            break

    return selected


# ============================================================
# ASK AI
# ============================================================

ASK_SCHEMA = {

    "type": "object",

    "properties": {

        "answer": {
            "type": "string"
        },

        "key_findings": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "recommendation": {
            "type": "string"
        },

        "sources": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "confidence": {
            "type": "string"
        },
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
            "The available customer evidence is "
            "insufficient to answer this question reliably."
        )

    evidence = []

    for row in sources:

        evidence.append(
            {
                "source_id": row["id"],
                "customer": row["customer_name"],
                "channel": row["channel"],
                "date": row["date"],
                "plan": row.get(
                    "plan",
                    ""
                ),
                "customer_type": row.get(
                    "customer_type",
                    ""
                ),
                "revenue": row.get(
                    "revenue",
                    0
                ),
                "topic": row.get(
                    "topic",
                    ""
                ),
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
                    row["message"]
                ),
                "ai_interpretation": row.get(
                    "feedback_summary",
                    ""
                ),
            }
        )

    prompt = f"""
You are Nuvanta AI's customer-intelligence analyst.

The complete uploaded customer dataset contains {len(DATA):,} records.

Question:
{question}

Answer only from the supplied evidence.

The evidence below contains relevant records retrieved from the FULL uploaded dataset.

Do not invent:
- customer statements
- counts
- revenue
- trends
- facts

If a statement is based on the retrieved evidence, cite the supplied source IDs in prose using [SOURCE 12] style.

Return:
- a direct answer
- 3-5 key findings
- a practical recommendation
- supporting source IDs
- confidence

Keep the response concise.

CUSTOMER EVIDENCE:
{json.dumps(evidence, ensure_ascii=False)}
"""

    return ai_json(
        prompt=prompt,
        schema_name="customer_data_answer",
        schema=ASK_SCHEMA,
        label="question answering",
        max_output_tokens=3000,
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

        "name": "Customer Feedback AI",

        "version": "5.0.0",

        "status": "running",

        "conversations": len(DATA),

        "ai": bool(
            get_openai_client()
        ),

        "analysis_batch_size": AI_BATCH_SIZE,

        "endpoints": [
            "/api/health",
            "/api/demo",
            "/api/upload",
            "/api/analysis/status",
            "/api/analysis",
            "/api/ask",
            "/api/conversations",
            "/api/last-ask",
            "/api/suggestions",
        ],
    }


@app.get("/api/health")
def health():

    with ANALYSIS_LOCK:

        status = ANALYSIS_STATUS.get(
            "status"
        )

    return {

        "ok": True,

        "loaded": len(DATA),

        "openai": bool(
            get_openai_client()
        ),

        "openai_package": OpenAI is not None,

        "model": OPENAI_MODEL,

        "analysis_batch_size": AI_BATCH_SIZE,

        "analysis_status": status,

        "analysis_total_records":
            ANALYSIS_STATUS.get(
                "total_records",
                len(DATA)
            ),

        "analysis_processed_records":
            ANALYSIS_STATUS.get(
                "processed_records",
                0
            ),

        "analysis_completed_batches":
            ANALYSIS_STATUS.get(
                "completed_batches",
                0
            ),

        "analysis_total_batches":
            ANALYSIS_STATUS.get(
                "total_batches",
                0
            ),

        "env_file_candidates": [
            path
            for path in dict.fromkeys(
                env_candidates
            )
            if os.path.isfile(path)
        ],
    }


@app.get("/api/analysis/status")
def analysis_status():

    with ANALYSIS_LOCK:
        return dict(
            ANALYSIS_STATUS
        )


@app.get("/api/analysis")
def get_analysis():

    with ANALYSIS_LOCK:

        if ANALYSIS is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "AI analysis is not "
                    "complete yet."
                )
            )

        return ANALYSIS


# ============================================================
# UPLOAD
# ============================================================

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

    if not file.filename.lower().endswith(
        ".csv"
    ):

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

        rows = normalize_dataframe(
            df
        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not process the CSV: "
                f"{exc}"
            )
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
        rows
    )


# ============================================================
# DEMO
# ============================================================

@app.post("/api/demo")
def load_demo():

    require_openai()

    candidates = [
        os.path.join(
            PROJECT_DIR,
            "data",
            "demo_customers.csv"
        ),
        os.path.join(
            BASE_DIR,
            "data",
            "demo_customers.csv"
        ),
    ]

    path = next(
        (
            p
            for p in candidates
            if os.path.isfile(p)
        ),
        None
    )

    if not path:

        raise HTTPException(
            status_code=404,
            detail="Demo CSV not found."
        )

    try:

        df = pd.read_csv(
            path
        )

        rows = normalize_dataframe(
            df
        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not process demo data: "
                f"{exc}"
            )
        )

    return start_analysis(
        rows
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
            detail="Load customer data first."
        )

    groups = grouped_rows(
        DATA
    )

    if ids:

        wanted = {
            x.strip()
            for x in ids.split(",")
            if x.strip()
        }

        return [
            source_record(
                row,
                groups.get(
                    str(
                        row.get(
                            "customer_id"
                        )
                    )
                )
            )
            for row in DATA
            if row["id"] in wanted
        ]

    ranked = sorted(
        DATA,
        key=lambda row: (
            1
            if row.get(
                "feedback_summary"
            )
            else 0,

            len(
                groups.get(
                    str(
                        row.get(
                            "customer_id"
                        )
                    ),
                    []
                )
            ),

            row.get(
                "date",
                ""
            ),
        ),
        reverse=True
    )

    result = []

    seen_customers = set()

    for row in ranked:

        cid = str(
            row.get(
                "customer_id"
            )
        )

        if cid in seen_customers:
            continue

        result.append(
            source_record(
                row,
                groups.get(cid)
            )
        )

        seen_customers.add(
            cid
        )

        if len(result) >= 18:
            break

    return result


@app.get(
    "/api/conversations/{conversation_id}"
)
def get_conversation(
    conversation_id: str
):

    groups = grouped_rows(
        DATA
    )

    for row in DATA:

        if row["id"] == conversation_id:

            return source_record(
                row,
                groups.get(
                    str(
                        row.get(
                            "customer_id"
                        )
                    )
                )
            )

    raise HTTPException(
        status_code=404,
        detail="Conversation not found."
    )


# ============================================================
# ASK
# ============================================================

@app.post("/api/ask")
def ask_customer_data(
    body: AskRequest
):

    global LAST_ASK

    if not DATA:

        raise HTTPException(
            status_code=404,
            detail="Load customer data first."
        )

    if ANALYSIS is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Wait for the full dataset "
                "analysis to complete first."
            )
        )

    question = clean_text(
        body.question
    )

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    require_openai()

    sources = retrieve(
        question
    )

    try:

        result = ask_ai(
            question,
            sources
        )

    except Exception as exc:

        print(
            f"[AI] ERROR question answering: "
            f"{exc!r}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                f"The AI could not answer "
                f"the question: {exc}"
            )
        )

    valid = {
        row["id"]
        for row in sources
    }

    source_ids = [
        clean_text(x)
        for x in result.get(
            "sources",
            []
        )
        if clean_text(x) in valid
    ]

    if not source_ids:

        source_ids = [
            row["id"]
            for row in sources[:8]
        ]

    source_ids = list(
        dict.fromkeys(
            source_ids
        )
    )

    groups = grouped_rows(
        DATA
    )

    final_sources = [
        source_record(
            row,
            groups.get(
                str(
                    row.get(
                        "customer_id"
                    )
                )
            )
        )
        for row in DATA
        if row["id"] in set(
            source_ids
        )
    ]

    LAST_ASK = {
        "question": question,
        "source_ids": source_ids,
        "sources": final_sources,
    }

    return {

        "question": question,

        "answer": clean_text(
            result.get(
                "answer"
            )
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
            result.get(
                "recommendation"
            )
        ),

        "confidence": (
            clean_text(
                result.get(
                    "confidence"
                )
            )
            or "medium"
        ),

        "ai_used": True,

        "source_count": len(
            final_sources
        ),

        "source_ids": source_ids,

        "sources": final_sources,

        "dataset_total": len(
            DATA
        ),

        "analysis_scope": (
            f"All {len(DATA):,} uploaded records"
        ),

        "suggested_questions": ANALYSIS.get(
            "suggested_questions",
            []
        ),
    }


# ============================================================
# LAST ASK
# ============================================================

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


# ============================================================
# SUGGESTIONS
# ============================================================

@app.get("/api/suggestions")
def suggestions():

    if ANALYSIS is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "AI analysis is not "
                "complete yet."
            )
        )

    return {
        "questions": ANALYSIS.get(
            "suggested_questions",
            []
        )
    }


# ============================================================
# CUSTOMER
# ============================================================

@app.get(
    "/api/customers/{customer_id}"
)
def get_customer(
    customer_id: str
):

    matches = [
        row
        for row in DATA
        if str(
            row.get(
                "customer_id"
            )
        ) == str(
            customer_id
        )
    ]

    if not matches:

        raise HTTPException(
            status_code=404,
            detail="Customer not found."
        )

    groups = grouped_rows(
        DATA
    )

    return {

        "customer_id": customer_id,

        "customer_name": matches[0][
            "customer_name"
        ],

        "conversation_count": len(
            matches
        ),

        "conversations": [
            source_record(
                row,
                groups.get(
                    str(
                        row.get(
                            "customer_id"
                        )
                    )
                )
            )
            for row in matches
        ],
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
            os.getenv(
                "PORT",
                "8000"
            )
        ),
        reload=False
    )
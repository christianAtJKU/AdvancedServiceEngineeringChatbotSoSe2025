from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Dict, List, Optional, Set, Tuple

import requests
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# ────────────────────────────── Constants ──────────────────────────────
MAX_LEN: int = 1_000                                    # hard limit for user input
PI_REGEX = re.compile(r"(ignore|disregard|override).*(instruction|previous)", re.I)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama3-8b-8192"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
if not GROQ_API_KEY:
    raise RuntimeError("Environment variable GROQ_API_KEY is missing")

# ─────────────────────────────── Logging ───────────────────────────────
logging.basicConfig(
    filename="requests.log",
    level=logging.INFO,
    format="%(asctime)sZ %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__).info

# ───────────────────────────── spaCy pipeline ───────────────────────────
nlp = spacy.load("en_core_web_sm")

# Custom patterns for e-mail and phone numbers
ruler = nlp.add_pipe("entity_ruler", before="ner", config={"validate": True})
ruler.add_patterns(
    [
        {
            "label": "EMAIL",
            "pattern": [{"TEXT": {"REGEX": r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b"}}],
        },
        {
            "label": "PHONE",
            "pattern": [{"TEXT": {"REGEX": r"\+?\d[\d\s\-]{7,}\d"}}],
        },
    ]
)

# Bi-directional mapping between clear names and pseudonyms
name_to_id: Dict[str, str] = {}

# PII labels we want to replace
PII_LABELS: Set[str] = {
    "PERSON",
    "NORP",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "DATE",
    "TIME",
    "EMAIL",
    "PHONE",
}

# ─────────────────────── Bias masking regex table ──────────────────────
SCRUB_PATTERNS: list[tuple[re.Pattern, str]] = [
    # gender
    (
        re.compile(
            r"\b(he|she|him|her|his|hers|male|female|man|woman|men|women|"
            r"boy|girl|husband|wife|mother|father|son|daughter|brother|sister)\b",
            re.I,
        ),
        "[GENDER]",
    ),
    # religion
    (
        re.compile(
            r"\b(christian|muslim|islamic|jew(ish)?|hindu|buddhist|atheist|"
            r"agnostic|sikh|catholic|protestant)\b",
            re.I,
        ),
        "[RELIGION]",
    ),
    # ethnicity
    (
        re.compile(
            r"\b(black|white|asian|latino|hispanic|arab(ic)?|middle[- ]eastern|"
            r"african[- ]american|native[- ]american|indian)\b",
            re.I,
        ),
        "[ETHNICITY]",
    ),
    # sexual orientation
    (
        re.compile(
            r"\b(gay|lesbian|bisexual|bi\b|queer|lgbtq\+?|trans(gender)?|"
            r"straight|heterosexual|homosexual)\b",
            re.I,
        ),
        "[ORIENTATION]",
    ),
    # disability
    (
        re.compile(
            r"\b(blind|deaf|autistic|autism|disabled|disability|wheelchair|"
            r"paraplegic|schizophrenic|dyslexic)\b",
            re.I,
        ),
        "[DISABILITY]",
    ),
]


# ────────────────────────── Helper functions ────────────────────────────
def scrub_bias(text: str) -> str:
    """Replace bias-relevant terms with neutral placeholders."""
    for pattern, replacement in SCRUB_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def pseudonymise(text: str, active_labels: Set[str]) -> str:
    """Replace PII entities with stable IDs after bias masking."""
    doc = nlp(text)
    spans = [ent for ent in doc.ents if ent.label_ in active_labels]

    # Replace from the end of the string to keep offsets intact
    for ent in sorted(spans, key=lambda s: s.start_char, reverse=True):
        pseudo_id = name_to_id.setdefault(ent.text, f"ID-{uuid.uuid4().hex[:8]}")
        text = f"{text[:ent.start_char]}{pseudo_id}{text[ent.end_char:]}"
    return text


def depseudonymise(text: str) -> str:
    """Convert IDs back to original names for logging."""
    id2name = {pid: name for name, pid in name_to_id.items()}
    return re.sub(r"ID-[0-9a-f]{8}", lambda m: id2name.get(m.group(), m.group()), text)


# ───────────────────────── FastAPI definition ───────────────────────────
app = FastAPI(title="TrustChat")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/", response_class=HTMLResponse)
def root() -> FileResponse:
    """Serve static single-page client."""
    return FileResponse("client.html", media_type="text/html")


# Pydantic models ----------------------------------------------------------------
class ChatRequest(BaseModel):
    history: List[Tuple[str, str]]  # list of (user_ID, bot_ID) pairs
    message: str
    filter: Optional[List[str]] = None  # active PII labels
    bias: bool = True                   # enable / disable bias scrub


class ChatResponse(BaseModel):
    user_pseudo: str
    bot_pseudo: str
    bot_clear: str


# ───────────────────────────── Main endpoint ────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Validate input, mask sensitive content, forward to Groq LLM."""
    # Guard 1 – length check
    if len(req.message) > MAX_LEN:
        raise HTTPException(413, "Message too long.")

    # Guard 2 – prompt-injection check
    if PI_REGEX.search(req.message):
        raise HTTPException(400, "Potential prompt injection detected.")

    active_labels: Set[str] = set(req.filter) if req.filter else PII_LABELS

    # Current user message → bias masking → PII pseudonymisation
    user_clean = scrub_bias(req.message) if req.bias else req.message
    user_pseudo = pseudonymise(user_clean, active_labels)

    # Log only the pseudonymised form
    log(user_pseudo)

    # Transform chat history in the same way
    hist_ids = [
        (
            pseudonymise(scrub_bias(u) if req.bias else u, active_labels),
            pseudonymise(scrub_bias(b) if req.bias else b, active_labels),
        )
        for u, b in req.history
    ]

    # Build the prompt for the LLM
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. All PII has been replaced by IDs and "
                "protected attributes are masked."
            ),
        },
        *(
            {"role": "user", "content": u_id}
            for pair in hist_ids
            for u_id in pair[:1]  # first element
        ),
        *(
            {"role": "assistant", "content": b_id}
            for pair in hist_ids
            for b_id in pair[1:]  # second element
        ),
        {"role": "user", "content": user_pseudo},
    ]

    # Call Groq
    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL, "messages": messages, "temperature": 0.7},
        timeout=60,
    )

    if response.status_code != 200:
        raise HTTPException(response.status_code, response.text)

    bot_pseudo = response.json()["choices"][0]["message"]["content"]
    bot_clear = depseudonymise(bot_pseudo)

    return ChatResponse(
        user_pseudo=user_pseudo,
        bot_pseudo=bot_pseudo,
        bot_clear=bot_clear,
    )
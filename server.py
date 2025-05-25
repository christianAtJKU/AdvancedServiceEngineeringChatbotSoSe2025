"""

  export GROQ_API_KEY="gsk_…"             # dein Key
  uvicorn server:app --reload --port 8000
"""
import os, uuid, re, logging
from typing import List, Tuple, Dict, Optional, Set

import requests
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

MAX_LEN = 1000                     # Zeichenlimit wie im Front-End
PI_REGEX = re.compile(r"(ignore|disregard|override).*(instruction|previous)", re.I)
# ─────────────────────────  LLM  ────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL        = "llama3-8b-8192"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
if not GROQ_API_KEY:
    raise RuntimeError("set GROQ_API_KEY")

# ────────────────────────  Logging  ─────────────────────────────
logging.basicConfig(
    filename="requests.log",
    level=logging.INFO,
    format="%(asctime)sZ %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
def log(msg:str): logging.info(msg)

# ────────────────────────  spaCy  ───────────────────────────────
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner", config={"validate":True})
ruler.add_patterns([
    {"label": "EMAIL", "pattern": [{"TEXT":{"REGEX":r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b"}}]},
    {"label": "PHONE", "pattern": [{"TEXT":{"REGEX":r"\+?\d[\d\s\-]{7,}\d"}}]},
])
name_to_id: Dict[str,str] = {}

PII_LABELS: Set[str] = {
    "PERSON","NORP","ORG","GPE","LOC","FAC","DATE","TIME","EMAIL","PHONE"
}

# ───────────────── Bias-Scrubber  (englisch, regex-basiert) ─────
SCRUB_PATTERNS = [
    (re.compile(r"\b(he|she|him|her|his|hers|male|female|man|woman|men|women|"
                r"boy|girl|husband|wife|mother|father|son|daughter|brother|sister)\b",
                re.I), "[GENDER]"),
    (re.compile(r"\b(christian|muslim|islamic|jew(ish)?|hindu|buddhist|atheist|"
                r"agnostic|sikh|catholic|protestant)\b", re.I), "[RELIGION]"),
    (re.compile(r"\b(black|white|asian|latino|hispanic|arab(ic)?|"
                r"middle[- ]eastern|african[- ]american|native[- ]american|indian)\b",
                re.I), "[ETHNICITY]"),
    (re.compile(r"\b(gay|lesbian|bisexual|bi\b|queer|lgbtq\+?|"
                r"trans(gender)?|straight|heterosexual|homosexual)\b",
                re.I), "[ORIENTATION]"),
    (re.compile(r"\b(blind|deaf|autistic|autism|disabled|disability|wheelchair|"
                r"paraplegic|schizophrenic|dyslexic)\b", re.I), "[DISABILITY]")
]

def scrub_bias(text:str) -> str:
    """Ersetzt geschützte Merkmale durch neutrale Platzhalter."""
    for pat, repl in SCRUB_PATTERNS:
        text = pat.sub(repl, text)
    return text

# ───────────────── Hilfsfunktionen  ─────────────────────────────
def process_text(text:str, active:Set[str]) -> str:
    """PII-Pseudonymisierung nach dem Bias-Scrub."""
    doc   = nlp(text)
    spans = [e for e in doc.ents if e.label_ in active]
    for ent in sorted(spans, key=lambda s:s.start_char, reverse=True):
        pid = name_to_id.setdefault(ent.text, f"ID-{uuid.uuid4().hex[:8]}")
        text = text[:ent.start_char] + pid + text[ent.end_char:]
    return text

def depseudo(text:str) -> str:
    id2name = {pid:name for name,pid in name_to_id.items()}
    return re.sub(r"ID-[0-9a-f]{8}", lambda m: id2name.get(m.group(), m.group()), text)

# ─────────────────── FastAPI  ────────────────────────────────────
app = FastAPI(title="TrustChat")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def root(): return FileResponse("client.html", media_type="text/html")

# ------------  Pydantic Models  ----------------------------------
class ChatRequest(BaseModel):
    history: List[Tuple[str,str]]
    message: str
    filter:  Optional[List[str]] = None
    bias: bool = True

class ChatResponse(BaseModel):
    user_pseudo: str
    bot_pseudo:  str
    bot_clear:   str

# ------------  Endpoint  -----------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # -------- Guard 1: Länge -----------------------------------
    if len(req.message) > MAX_LEN:
        raise HTTPException(413, "Message too long.")

    # -------- Guard 2: Prompt-Injection ------------------------
    if PI_REGEX.search(req.message):
        raise HTTPException(400, "Potential prompt-injection detected. Please rephrase.")

    active = set(req.filter) if req.filter else PII_LABELS

    # 1) Bias-Scrub + Pseudonymisierung für aktuelle Nachricht
    clean_msg   = scrub_bias(req.message) if req.bias else req.message
    user_pseudo = process_text(clean_msg, active)

    # 2) Logging (nur pseudonymisierte User-Eingabe)
    log(user_pseudo)

    # 3) History ebenfalls scrubben + pseudonymisieren
    hist_ids = [
        (process_text(scrub_bias(u) if req.bias else u, active),
         process_text(scrub_bias(b) if req.bias else b, active))
        for u, b in req.history
    ]

    # 4) Kontext fürs LLM
    msgs = [{"role":"system",
             "content":"You are a helpful assistant. All PII is replaced by IDs. "
                       "Protected attributes are masked."}]
    for u_id, b_id in hist_ids:
        msgs += [{"role":"user","content":u_id},{"role":"assistant","content":b_id}]
    msgs.append({"role":"user","content":user_pseudo})

    # 5) Anfrage ans Modell
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization":f"Bearer {GROQ_API_KEY}",
                 "Content-Type":"application/json"},
        json={"model":MODEL,"messages":msgs,"temperature":0.7},
        timeout=60
    )
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.text)

    bot_pseudo = resp.json()["choices"][0]["message"]["content"]
    bot_clear  = depseudo(bot_pseudo)

    return ChatResponse(user_pseudo=user_pseudo,
                        bot_pseudo=bot_pseudo,
                        bot_clear=bot_clear)
import spacy
import uuid
import re
from typing import List, Dict

# ---------------- Pipeline ---------------------------------------------------
nlp = spacy.load("en_core_web_sm")               # Modell bleibt gleich

# (1) EntityRuler jetzt über den Komponentennamen hinzufügen
ruler = nlp.add_pipe(
    "entity_ruler",               #  ← String statt Objekt
    before="ner",
    config={"validate": True}     # gleiche Option wie zuvor
)

# (2) Danach erst die PII-Muster einspielen
ruler.add_patterns([
    {"label": "EMAIL", "pattern": [{"TEXT": {"REGEX": r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b"}}]},
    {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": r"\+?\d[\d\s\-]{7,}\d"}}]},
])

# ---------------- Globale Datenstruktur --------------------------------------
name_to_id: Dict[str, str] = {}
PII_LABELS = {
    "PERSON", "NORP", "ORG", "GPE", "LOC", "FAC",
    "DATE", "TIME", "EMAIL", "PHONE"
}

# ---------------- API-Funktionen (Bezeichner unverändert) --------------------
def process_text(text: str) -> dict:
    doc = nlp(text)
    spans: List = [ent for ent in doc.ents if ent.label_ in PII_LABELS]

    if not spans:
        return {"has_names": False, "pseudonymized_text": text, "names": []}

    pseudonymized_text = text
    original_values: List[str] = []

    for ent in sorted(spans, key=lambda s: s.start_char, reverse=True):
        original_values.append(ent.text)
        pid = name_to_id.setdefault(ent.text, f"ID-{uuid.uuid4().hex[:8]}")
        pseudonymized_text = (
            pseudonymized_text[:ent.start_char] + pid + pseudonymized_text[ent.end_char:]
        )

    return {
        "has_names": True,
        "pseudonymized_text": pseudonymized_text,
        "names": original_values[::-1],
    }


def ent_pseudonymize(text: str) -> str:
    id_to_name = {pid: name for name, pid in name_to_id.items()}
    return re.sub(r"ID-[0-9a-f]{8}", lambda m: id_to_name.get(m.group(), m.group()), text)
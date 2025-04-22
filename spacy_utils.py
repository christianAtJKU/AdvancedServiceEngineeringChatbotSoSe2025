import spacy
import uuid

# spaCy-Modell laden
nlp = spacy.load("en_core_web_sm")

# Globales Mapping für Namen zu IDs
name_to_id = {}

def process_text(text):
    """
    Prüfe, ob Namen vorhanden sind, pseudonymisiere sie und gebe die Ergebnisse zurück.
    """
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if names:
        pseudonymized_text = text
        for name in names:
            if name not in name_to_id:
                name_to_id[name] = f"ID-{uuid.uuid4().hex[:8]}"
            pseudonymized_text = pseudonymized_text.replace(name, name_to_id[name])
        return {"has_names": True, "pseudonymized_text": pseudonymized_text, "names": names}
    return {"has_names": False, "pseudonymized_text": text, "names": []}
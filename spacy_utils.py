import spacy

# spaCy-Modell laden
nlp = spacy.load("en_core_web_sm")

def process_text(text):
    """
    Prüfe, ob Namen vorhanden sind, anonymisiere sie und gebe die Ergebnisse zurück.
    """
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if names:
        anonymized_text = text
        for name in names:
            anonymized_text = anonymized_text.replace(name, "[ANONYM]")
        return {"has_names": True, "anonymized_text": anonymized_text, "names": names}
    return {"has_names": False, "anonymized_text": text, "names": []}
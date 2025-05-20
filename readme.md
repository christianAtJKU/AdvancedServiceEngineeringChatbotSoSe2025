# SafeChat
Ein interaktiver Chatbot mit Pseudonymisierungsfunktion für personenbezogene Daten, realisiert mit [spaCy](https://spacy.io/) und [Gradio](https://gradio.app/).

```bash
# virtuelle Umgebung erstellen
python -m venv .venv

# virtuelle Umgebung aktivieren
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# spaCy-Modell herunterladen
python -m spacy download en_core_web_sm

# Anwendung starten
python gradio_ui.py
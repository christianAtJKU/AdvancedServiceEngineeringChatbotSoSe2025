import requests
import gradio as gr
from datetime import datetime
import json

from spacy_utils import process_text, ent_pseudonymize  # ← neue Funktion importiert


# Key läuft über Konto von Christian Rührlinger. VERTRAULICH BEHANDELN!
GROQ_API_KEY = "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"

# API-Endpunkt
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-8b-8192"

# Datei für das Log der Modellaufrufe
LOG_PATH = "requests.log"


# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────

def log_request(msg: str) -> None:
    """
    Schreibt jede pseudonymisierte Nutzernachricht an das LLM
    als JSON-Zeile in die Log-Datei.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "message": msg,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def chat_with_groq(history, user_message):
    """
    Sendet die Nachricht an die Groq-API und gibt die Antwort zurück.
    user_message muss bereits pseudonymisiert sein.
    """
    # Protokollieren
    log_request(user_message)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Verlauf in OpenAI-kompatibles Format umwandeln
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. All user inputs are pseudonymized "
                "and may contain placeholder IDs instead of real names. Respond accordingly."
            ),
        }
    ]
    for user, bot in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": user_message})

    print("Nachricht an LLM:", messages)  # Debug

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]

    return f"Error: {response.status_code} – {response.text}"


# ──────────────────────────────────────────────────────────────────────────────
# Gradio-Interface
# ──────────────────────────────────────────────────────────────────────────────

with gr.Blocks() as ui:
    gr.Markdown("## TrustChat")

    # Zustände für die Chat-Verläufe
    state_active = gr.State([])  # Original + rückübersetzte Antworten
    state_pseudo = gr.State([])  # Vollständig pseudonymisierter Verlauf

    with gr.Row():
        with gr.Column(scale=1):
            active_chatbot = gr.Chatbot(label="Depseudonymized", height=500)

        with gr.Column(scale=1):
            pseudonymized_chatbot = gr.Chatbot(label="Pseudonymized", height=500)

    # Eingabefeld
    msg = gr.Textbox(label="Message", placeholder="Type your message here…")

    # Button zum Löschen
    clear = gr.ClearButton("Delete")

    # Haupt-Callback
    def respond(active_hist, pseudo_hist, message):
        if not message:
            return active_hist, pseudo_hist, ""

        # 1️⃣ Eingabe pseudonymisieren
        result = process_text(message)
        pseudonymized_message = result["pseudonymized_text"]

        # 2️⃣ An Groq schicken (pseudonymisiert)
        bot_reply = chat_with_groq(pseudo_hist, pseudonymized_message)

        # 3️⃣ Antwort rückpseudonymisieren
        ent_pseudonymized_bot_reply = ent_pseudonymize(bot_reply)

        # 4️⃣ Verläufe aktualisieren
        active_hist.append((message, ent_pseudonymized_bot_reply))
        pseudo_hist.append((pseudonymized_message, bot_reply))

        return active_hist, pseudo_hist, ""

    # Eingabe binden
    msg.submit(
        respond,
        [state_active, state_pseudo, msg],
        [active_chatbot, pseudonymized_chatbot, msg],
    )

    # Clear-Callback
    def clear_chat():
        return [], [], ""

    clear.click(
        clear_chat,
        [],
        [state_active, state_pseudo, msg],
    )

# App starten
ui.launch(share=True)
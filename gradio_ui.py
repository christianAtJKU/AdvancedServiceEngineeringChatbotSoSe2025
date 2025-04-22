import requests
import gradio as gr
from spacy_utils import process_text

# Key läuft über Konto von Christian Rührlinger. VERTRAULICH BEHANDELN!
GROQ_API_KEY = "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"

#  API URL
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL = "llama3-8b-8192"

def chat_with_groq(history, user_message):
    """Send a message to Groq API and return the response."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Convert Gradio chat history to OpenAI API format
    messages = [
        {"role": "system", "content": "You are a helpful assistant. All user inputs are pseudonymized and may contain placeholder IDs instead of real names. Respond accordingly."}
    ]
    for user, bot in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": user_message})

    # Debug: Ausgabe der Nachricht, die an das LLM gesendet wird
    print("Nachricht an LLM:", messages)

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        bot_response = response.json()["choices"][0]["message"]["content"]
        return bot_response
    else:
        return f"Error: {response.status_code} - {response.text}"

def ent_pseudonymize(text):
    """Placeholder function for ent-pseudonymizing text."""
    # Implement the logic for ent-pseudonymization here
    return text

# Gradio Chatbot Interface
with gr.Blocks() as ui:
    gr.Markdown("## Privacy Chatbot - Side-by-Side View")

    # Zustände für die Chat-Verläufe
    state_active = gr.State([])  # Originalnachrichten (inkl. Ent-Pseudonymisierung)
    state_pseudo = gr.State([])  # Pseudonymisierte Nachrichten

    with gr.Row():
        # Linke Spalte: Originalnachrichten (inkl. Ent-Pseudonymisierung)
        with gr.Column(scale=1):
            gr.Markdown("### Active Chat (Original Input and Ent-Pseudonymized Responses)")
            active_chatbot = gr.Chatbot(label="Active Conversation", height=500)

        # Rechte Spalte: Pseudonymisierte Nachrichten
        with gr.Column(scale=1):
            gr.Markdown("### Pseudonymized Log (Read-Only)")
            pseudonymized_chatbot = gr.Chatbot(label="Pseudonymized History", height=500)

    # Eingabefeld für Benutzer
    msg = gr.Textbox(label="Enter Message", placeholder="Type your message here...")

    # Button zum Löschen des Chats
    clear = gr.Button("Delete Chat History")

    # Funktion zur Verarbeitung der Eingabe
    def respond(active_hist, pseudo_hist, message):
        if not message:
            return active_hist, pseudo_hist, ""

        # Pseudonymisierung der Benutzereingabe
        result = process_text(message)
        pseudonymized_message = result["pseudonymized_text"]

        # Nachricht an Groq senden (pseudonymisierte Nachricht)
        bot_reply = chat_with_groq(pseudo_hist, pseudonymized_message)

        # Ent-Pseudonymisierung der Antwort
        ent_pseudonymized_bot_reply = ent_pseudonymize(bot_reply)

        # Verläufe aktualisieren
        active_hist.append((message, ent_pseudonymized_bot_reply))
        pseudo_hist.append((pseudonymized_message, bot_reply))

        # Rückgabe der aktualisierten Verläufe und leeren der Eingabebox
        return active_hist, pseudo_hist, ""

    # Verknüpfen der Eingabe und Anzeigeelemente
    msg.submit(
        respond,
        [state_active, state_pseudo, msg],  # Inputs
        [active_chatbot, pseudonymized_chatbot, msg]  # Outputs
    )

    # Funktion zum Löschen der Chats
    def clear_chat():
        return [], [], ""  # Leert beide Verläufe und die Eingabebox

    clear.click(
        clear_chat,
        [],  # Keine Inputs für die Clear-Funktion
        [state_active, state_pseudo, msg]  # Outputs: Leere beide Verläufe und die Eingabebox
    )

# Launch the app
ui.launch(share=True)
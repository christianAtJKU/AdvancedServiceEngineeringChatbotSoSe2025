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

# Gradio Chatbot Interface
with gr.Blocks() as ui:
    gr.Markdown("## Privacy Chatbot")
    
    # Chatbot-Komponente
    chatbot = gr.Chatbot()
    
    # Eingabefeld für Benutzer
    msg = gr.Textbox(label="Enter Message")
    
    # Textboxen für Original- und pseudonymisierte Nachrichten
    with gr.Row():
        original_message_display = gr.Textbox(label="Original Message (with Names)", interactive=False)
        pseudonymized_message_display = gr.Textbox(label="Pseudonymized Message (with IDs)", interactive=False)
    
    # Button zum Löschen des Chats
    clear = gr.Button("Delete Chat")

    # Funktion zur Verarbeitung der Eingabe
    def respond(history, message):
        # Text verarbeiten (Prüfen und Pseudonymisieren)
        result = process_text(message)
        print(f"Originalnachricht: {message}")
        print(f"Pseudonymisierte Nachricht: {result['pseudonymized_text']}")
        if result["has_names"]:
            print(f"Gefundene Namen: {', '.join(result['names'])}")

        # Nachricht an Groq weiterleiten (pseudonymisierte Nachricht verwenden)
        bot_reply = chat_with_groq(history, result["pseudonymized_text"])

        # Pseudonymisierte Nachricht und Bot-Antwort zum Verlauf hinzufügen
        history.append((result["pseudonymized_text"], bot_reply))

        # Rückgabe der Original- und pseudonymisierten Nachricht für die Anzeige
        return history, "", message, result["pseudonymized_text"]

    # Verknüpfen der Eingabe und Anzeigeelemente
    msg.submit(
        respond, 
        [chatbot, msg], 
        [chatbot, msg, original_message_display, pseudonymized_message_display]
    )
    clear.click(lambda: [], None, chatbot)

# Launch the app
ui.launch(share=True)
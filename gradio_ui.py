import requests
import gradio as gr


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
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user, bot in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": user_message})

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
    gr.Markdown("## Datenschutz Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Nachricht eingeben")
    clear = gr.Button("Chat Löschen")

    def respond(history, message):
        bot_reply = chat_with_groq(history, message)
        history.append((message, bot_reply))
        return history, ""

    msg.submit(respond, [chatbot, msg], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)

# Launch the app
ui.launch(share=True)
import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# Load environment variables
load_dotenv()

# Initialize OpenAI client with API key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initial system message
messages = [
    {"role": "system", "content": "You are a customer care assistant."}
]

def CustomChatGPT(user_input):
    # Add user message
    messages.append({"role": "user", "content": user_input})

    # Call OpenAI Chat API
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # you can also use "gpt-3.5-turbo"
        messages=messages
    )

    # Extract assistant reply
    reply = response.choices[0].message.content

    # Add assistant reply to conversation
    messages.append({"role": "assistant", "content": reply})

    return reply

# Gradio UI
demo = gr.Interface(
    fn=CustomChatGPT,
    inputs="text",
    outputs="text",
    title="Transactional AI chatbot. By A.O. Ozulem",
)

# Launch app
if __name__ == "__main__":
    demo.launch(share=True)  # share=True gives you a temporary public link

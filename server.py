import asyncio
import uuid
import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from typing import Dict

# Import from our modules
from chat_memory import create_chat_session, add_message_to_session, get_formatted_history
from models import call_gpt, call_deepseek
from consensus import verify_and_merge

app = FastAPI()

# --- Core Identity & Business Logic ---

TIWA_PERSONA = (
    "I am TIWA (Task Intelligent Web Agent), a multi-model AI assistant created by Hive Innovation Lab. "
    "My purpose is to provide accurate and helpful responses by orchestrating the strengths of several advanced AI models. "
    "Hive Innovation Lab was co-founded by Abdulkereem O Kereem and Akinola Solmipe, and I was engineered by Abdulkereem. "
    "My intelligence is a synthesis of models like GPT and Deepseek, which I use to reason and respond to your requests."
)

# Stricter identity check triggers
IDENTITY_TRIGGERS = ["who are you", "what is tiwa", "what are you", "tell me about tiwa", "your name"]

def is_identity_question(prompt: str) -> bool:
    """Check if the user's *latest* prompt is a direct question about TIWA's identity."""
    normalized_prompt = prompt.lower().strip()
    # Use a regex to match the triggers as whole phrases to avoid partial matches
    return any(re.search(r"\b" + re.escape(trigger) + r"\b", normalized_prompt) for trigger in IDENTITY_TRIGGERS)

def generate_topic(prompt: str) -> str:
    """Generates a short topic from the user's prompt for the thinking indicator."""
    words = prompt.split()
    topic = " ".join(words[:5])
    if len(words) > 5:
        topic += "..."
    return topic

async def process_single_prompt(websocket: WebSocket, chat_id: str, prompt: str, prompt_id: str):
    """Handles the entire lifecycle of a single prompt, with memory and a strict identity check."""
    try:
        # Step 1: Immediately check if the user's latest input is an identity question.
        if is_identity_question(prompt):
            print(f"Prompt ID {prompt_id}: Identified as an identity question. Responding directly.", flush=True)
            # Add user message and the definitive answer to history
            add_message_to_session(chat_id, "user", prompt)
            add_message_to_session(chat_id, "assistant", TIWA_PERSONA, reasoning="Direct identity response")
            await websocket.send_json({
                "type": "final",
                "chat_id": chat_id,
                "prompt_id": prompt_id,
                "final_source": TIWA_PERSONA
            })
            return # Stop further processing

        # Step 2: For all other questions, proceed with the full memory-enhanced flow.
        add_message_to_session(chat_id, "user", prompt)
        topic = generate_topic(prompt)
        await websocket.send_json({"type": "thinking", "topic": topic, "prompt_id": prompt_id})

        # Step 3: Get conversation history and create a contextual prompt.
        history = get_formatted_history(chat_id)
        contextual_prompt = f"{history}\nUser's current question: {prompt}"

        print(f"Prompt ID {prompt_id}: Processing with context:\n{contextual_prompt}", flush=True)

        # Step 4: Concurrently call models with the full context.
        gpt_task = asyncio.create_task(call_gpt(contextual_prompt))
        deepseek_task = asyncio.create_task(call_deepseek(contextual_prompt))
        gpt_result, deepseek_result = await asyncio.gather(gpt_task, deepseek_task)

        model_outputs = {"gpt": gpt_result, "deepseek": deepseek_result}

        final_data = await verify_and_merge(outputs=model_outputs, evidence=[deepseek_result], prompt=contextual_prompt)

        add_message_to_session(chat_id, "assistant", final_data['final_output'], reasoning=f"Final output after {final_data.get('consensus_method')}")

        await websocket.send_json({
            "type": "final",
            "chat_id": chat_id,
            "prompt_id": prompt_id,
            "final_source": final_data['final_output']
        })

    except Exception as e:
        print(f"Error processing prompt_id {prompt_id}: {e}", flush=True)
        await websocket.send_json({
            "type": "error", "prompt_id": prompt_id, "message": "An error occurred."
        })

@app.get("/")
async def get():
    return FileResponse('index.html')

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    chat_id = str(uuid.uuid4())
    create_chat_session(chat_id)
    print(f"New chat session created for client {client_id}: {chat_id}")

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "message":
                prompt = data.get("prompt")
                prompt_id = data.get("prompt_id")
                if prompt and prompt_id:
                    asyncio.create_task(process_single_prompt(websocket, chat_id, prompt, prompt_id))
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected, chat session {chat_id} closed.")
    except Exception as e:
        print(f"An error occurred in websocket for chat {chat_id}: {e}", flush=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

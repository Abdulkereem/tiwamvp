import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from typing import Dict

# Import from our modules
from chat_memory import create_chat_session, add_message_to_session
from models import call_gpt, call_deepseek
from consensus import verify_and_merge

app = FastAPI()

def generate_topic(prompt: str) -> str:
    """Generates a short topic from the user's prompt for the thinking indicator."""
    words = prompt.split()
    # Take the first 5 words or less
    topic = " ".join(words[:5])
    if len(words) > 5:
        topic += "..."
    return topic

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
            action = data.get("action")
            prompt = data.get("prompt")

            if action == "message" and prompt:
                add_message_to_session(chat_id, "user", prompt)

                # Send a "thinking" status to the frontend immediately
                topic = generate_topic(prompt)
                await websocket.send_json({"type": "thinking", "topic": topic})

                # Run models and get consensus
                gpt_task = asyncio.create_task(call_gpt(prompt))
                deepseek_task = asyncio.create_task(call_deepseek(prompt))
                gpt_result, deepseek_result = await asyncio.gather(gpt_task, deepseek_task)

                model_outputs = {"gpt": gpt_result, "deepseek": deepseek_result}

                final_data = await verify_and_merge(
                    outputs=model_outputs, 
                    evidence=[deepseek_result], 
                    prompt=prompt
                )

                # Log results and store in chat history
                print(f"GPT: {gpt_result}\nDeepseek: {deepseek_result}\nConfidence: {final_data.get('confidence')}\nFinal Source: {final_data.get('final_output')}\nConsensus: {final_data.get('consensus_method')}", flush=True)
                add_message_to_session(chat_id, "assistant", gpt_result, reasoning="GPT output")
                add_message_to_session(chat_id, "assistant", deepseek_result, reasoning="Deepseek output")
                add_message_to_session(chat_id, "assistant", final_data['final_output'], reasoning=f"Final output after {final_data.get('consensus_method')}")

                # Send the final, verified response to the frontend
                await websocket.send_json({
                    "type": "final",
                    "chat_id": chat_id,
                    "final_source": final_data['final_output']
                })

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected, chat session {chat_id} closed.")
    except Exception as e:
        print(f"An error occurred in chat {chat_id}: {e}", flush=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

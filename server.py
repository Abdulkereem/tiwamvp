import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from typing import Dict

# Import from our modules
from chat_memory import create_chat_session, add_message_to_session, chat_sessions
from models import call_gpt, call_claude, call_deepseek
from consensus import verify_and_merge

app = FastAPI()

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

                # 1. Run models concurrently
                gpt_task = asyncio.create_task(call_gpt(prompt))
                claude_task = asyncio.create_task(call_claude(prompt))
                deepseek_task = asyncio.create_task(call_deepseek(prompt))

                gpt_result, claude_result, deepseek_result = await asyncio.gather(
                    gpt_task,
                    claude_task,
                    deepseek_task
                )

                model_outputs = {
                    "gpt": gpt_result,
                    "claude": claude_result,
                    "deepseek": deepseek_result
                }

                # 2. Stream partial results to the client
                for model_name, output in model_outputs.items():
                    await websocket.send_json({
                        "type": "partial",
                        "model": model_name,
                        "chunk": output,
                        "done": True
           git          })
                    await asyncio.sleep(0.1)

                # 3. Get consensus and merged response
                final_data = verify_and_merge(
                    outputs={"gpt": gpt_result, "claude": claude_result},
                    evidence=[deepseek_result] # The evidence should be a list of strings
                )

                final_data["per_model"]["deepseek"] = deepseek_result

                # 4. Store assistant response in memory
                add_message_to_session(chat_id, "assistant", final_data["final_output"])

                # 5. Send final merged response
                await websocket.send_json({
                    "type": "final",
                    "chat_id": chat_id,
                    **final_data
                })

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected, chat session {chat_id} closed.")
        if chat_id in chat_sessions:
            del chat_sessions[chat_id]
    except Exception as e:
        print(f"An error occurred in chat {chat_id}: {e}")
        if chat_id in chat_sessions:
            del chat_sessions[chat_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

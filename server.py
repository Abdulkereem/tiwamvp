import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from typing import Dict

# Import from our modules
from chat_memory import create_chat_session, add_message_to_session, chat_sessions, get_chat_session
from models import call_gpt, call_deepseek
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
                # Chain of thought and TIWA persona logic
                session = get_chat_session(chat_id)
                add_message_to_session(chat_id, "user", prompt)

                # Check if user is asking about TIWA's identity
                if any(x in prompt.lower() for x in ["who are you", "what is tiwa", "your name", "are you tiwa", "who is tiwa"]):
                    tiwa_identity = (
                        "I am TIWA, a Task Intelligent Web Agent built by Hive Innovation Lab, "
                        "based on GPT-4 and Deepseek large language models."
                    )
                    add_message_to_session(chat_id, "assistant", tiwa_identity, reasoning="TIWA persona response")
                    print(f"TIWA Persona: {tiwa_identity}")
                    await websocket.send_json({
                        "type": "final",
                        "chat_id": chat_id,
                        "final_source": tiwa_identity
                    })
                    continue

                # Run both agents and get their outputs
                gpt_task = asyncio.create_task(call_gpt(prompt))
                deepseek_task = asyncio.create_task(call_deepseek(prompt))
                gpt_result, deepseek_result = await asyncio.gather(gpt_task, deepseek_task)

                # Consensus logic: if outputs are similar, use either; if not, use Gemini to arbitrate
                def are_similar(a, b):
                    return a.strip().lower()[:50] == b.strip().lower()[:50]

                if are_similar(gpt_result, deepseek_result):
                    final_source = gpt_result
                    consensus_method = "agreement"
                else:
                    from models import call_gemini_judge
                    final_source = await call_gemini_judge(gpt_result, deepseek_result, prompt)
                    consensus_method = "gemini_judge"

                from consensus import compute_confidence
                confidence = compute_confidence(["gpt", "deepseek"], [deepseek_result])

                # Log all details in backend
                print(f"GPT: {gpt_result}\nDeepseek: {deepseek_result}\nConfidence: {confidence}\nFinal Source: {final_source}\nConsensus method: {consensus_method}")

                # Store agent outputs and reasoning in chain of thought
                add_message_to_session(chat_id, "assistant", gpt_result, reasoning="GPT output")
                add_message_to_session(chat_id, "assistant", deepseek_result, reasoning="Deepseek output")
                add_message_to_session(chat_id, "assistant", final_source, reasoning=f"Final source output ({consensus_method})")

                # Send only the final source to the frontend
                await websocket.send_json({
                    "type": "final",
                    "chat_id": chat_id,
                    "final_source": final_source
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

import asyncio
import uuid
import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from typing import Dict

# Import from our modules
from chat_memory import create_chat_session, add_message_to_session, get_formatted_history
from models import call_gpt, call_deepseek, tool_decider_model
from consensus import verify_and_merge
from tools import tavily_web_search, scrape_url

app = FastAPI()

# --- Core Identity & Business Logic ---

TIWA_PERSONA = (
    "I am TIWA (Task Intelligent Web Agent), a multi-model AI assistant created by Hive Innovation Lab. "
    "My purpose is to provide accurate and helpful responses by orchestrating the strengths of several advanced AI models and accessing live web data. "
    "Hive Innovation Lab was co-founded by Abdulkereem O Kereem and Akinola Solmipe, and I was engineered by Abdulkereem."
)

IDENTITY_TRIGGERS = ["who are you", "what is tiwa", "what are you", "tell me about tiwa", "your name"]

def is_identity_question(prompt: str) -> bool:
    """Check if the user's prompt is a direct question about TIWA's identity."""
    normalized_prompt = prompt.lower().strip()
    return any(re.search(r"\b" + re.escape(trigger) + r"\b", normalized_prompt) for trigger in IDENTITY_TRIGGERS)

def generate_topic(prompt: str) -> str:
    """Generates a short topic from the user's prompt for the thinking indicator."""
    words = prompt.split()
    topic = " ".join(words[:5])
    if len(words) > 5:
        topic += "..."
    return topic

# --- Available Tools ---
AVAILABLE_TOOLS = {
    "tavily_web_search": tavily_web_search,
    "scrape_url": scrape_url,
}

async def process_single_prompt(websocket: WebSocket, chat_id: str, prompt: str, prompt_id: str):
    """Handles prompts dynamically, using a tool-decider model first."""
    try:
        # Step 1: Identity Check (Highest Priority)
        if is_identity_question(prompt):
            print(f"Prompt ID {prompt_id}: Identity question. Responding directly.", flush=True)
            add_message_to_session(chat_id, "user", prompt)
            add_message_to_session(chat_id, "assistant", TIWA_PERSONA, reasoning="Direct identity response")
            await websocket.send_json({"type": "final", "prompt_id": prompt_id, "final_source": TIWA_PERSONA})
            return

        add_message_to_session(chat_id, "user", prompt)
        topic = generate_topic(prompt)
        await websocket.send_json({"type": "thinking", "topic": topic, "prompt_id": prompt_id})

        history = get_formatted_history(chat_id)
        contextual_prompt = f"{history}\nUser's current question: {prompt}"

        # Step 2: Tool Use Decision with Correct Error Handling
        function_call = None
        tool_executed = False

        if tool_decider_model:
            decision_response = await asyncio.to_thread(tool_decider_model.generate_content, contextual_prompt)
            
            try:
                # This is the correct, idiomatic way to check.
                # .text raises ValueError if a function call is present.
                _ = decision_response.text
                print(f"Prompt ID {prompt_id}: AI decided not to use a tool.", flush=True)

            except ValueError:
                # This block is executed ONLY when a function call is likely present.
                print(f"Prompt ID {prompt_id}: AI response not simple text, parsing for tool call.", flush=True)
                try:
                    function_call = decision_response.candidates[0].content.parts[0].function_call
                except Exception as e:
                     print(f"Prompt ID {prompt_id}: Could not parse function call from response: {e}", flush=True)

        if function_call:
            tool_name = function_call.name
            if tool_name in AVAILABLE_TOOLS:
                tool_args = {key: value for key, value in function_call.args.items()}
                print(f"Prompt ID {prompt_id}: AI executing tool: {tool_name} with args: {tool_args}", flush=True)
                
                tool_function = AVAILABLE_TOOLS[tool_name]
                tool_result = await tool_function(**tool_args)
                
                print(f"Prompt ID {prompt_id}: Tool '{tool_name}' executed.", flush=True)
                add_message_to_session(chat_id, "assistant", tool_result, reasoning=f"Direct result from {tool_name}")
                await websocket.send_json({"type": "final", "prompt_id": prompt_id, "final_source": tool_result})
                tool_executed = True

        if not tool_executed:
            # Step 3: Multi-Model Consensus (If no tool is used)
            print(f"Prompt ID {prompt_id}: No tool used. Proceeding with consensus.", flush=True)
            gpt_task = asyncio.create_task(call_gpt(contextual_prompt))
            deepseek_task = asyncio.create_task(call_deepseek(contextual_prompt))
            gpt_result, deepseek_result = await asyncio.gather(gpt_task, deepseek_task)

            model_outputs = {"gpt": gpt_result, "deepseek": deepseek_result}
            final_data = await verify_and_merge(outputs=model_outputs, evidence=[deepseek_result], prompt=contextual_prompt)

            add_message_to_session(chat_id, "assistant", final_data['final_output'], reasoning=f"Final output after {final_data.get('consensus_method')}")
            await websocket.send_json({"type": "final", "prompt_id": prompt_id, "final_source": final_data['final_output']})

    except Exception as e:
        print(f"Error processing prompt_id {prompt_id}: {e}", flush=True)
        await websocket.send_json({"type": "error", "prompt_id": prompt_id, "message": "An error occurred."})


@app.get("/")
async def get():
    return FileResponse('index.html')

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    chat_id = str(uuid.uuid4())
    create_chat_session(chat_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "message":
                prompt, prompt_id = data.get("prompt"), data.get("prompt_id")
                if prompt and prompt_id:
                    asyncio.create_task(process_single_prompt(websocket, chat_id, prompt, prompt_id))
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected.")
    except Exception as e:
        print(f"Websocket error for client {client_id}: {e}", flush=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

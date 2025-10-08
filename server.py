
import asyncio
import uuid
import re
import os
import shutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Optional

# Import from our modules
from chat_memory import create_chat_session, add_message_to_session, get_formatted_history
from models import call_gpt, call_deepseek, tool_decider_model
from consensus import verify_and_merge
from tools import tavily_web_search, scrape_url, generate_image, write_file
from persona import TIWA_PERSONA

app = FastAPI()

# --- Directory Setups ---
os.makedirs("uploads", exist_ok=True)
os.makedirs("generated_files", exist_ok=True)
os.makedirs("static", exist_ok=True)

# --- Static File Mounts ---
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/downloads", StaticFiles(directory="generated_files"), name="downloads")


# --- File & Security Operations ---

def sanitize_filename(filename: str) -> str:
    """Strips dangerous characters from a filename to prevent security vulnerabilities."""
    sanitized = re.sub(r'\.\.[/\\|]|\x00', '', filename)
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', sanitized)
    return sanitized if sanitized else "sanitized_default_name"

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    """Handles file uploads, sanitizes the filename, and saves it."""
    sanitized_filename = sanitize_filename(file.filename)
    upload_path = os.path.join("uploads", sanitized_filename)

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_info = {"filename": sanitized_filename, "path": upload_path}
        
        return JSONResponse(status_code=200, content={
            "message": "File uploaded successfully",
            "file_info": file_info
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Could not upload file: {e}"})


# --- Core Identity & Business Logic ---

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
    "generate_image": generate_image,
    "write_file": write_file,
}

# --- Main Prompt Processing Logic ---

async def process_single_prompt(websocket: WebSocket, chat_id: str, prompt: str, prompt_id: str, file_path: Optional[str] = None):
    """Handles prompts dynamically, including context from uploaded files."""
    try:
        if is_identity_question(prompt):
            # ... (identity logic remains the same)
            return

        add_message_to_session(chat_id, "user", prompt)
        topic = generate_topic(prompt)
        await websocket.send_json({"type": "thinking", "topic": topic, "prompt_id": prompt_id})

        file_content_context = ""
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                file_content_context = f"\n\n--- BEGIN UPLOADED FILE CONTENT ({os.path.basename(file_path)}) ---\n{file_content}\n--- END UPLOADED FILE CONTENT ---\n"
            except Exception as e:
                file_content_context = f"\n\n[System note: Could not read the uploaded file '{os.path.basename(file_path)}'. It might be a binary file.]\n"

        history = get_formatted_history(chat_id)
        contextual_prompt = f"{history}{file_content_context}\nUser's current question: {prompt}"

        # ... (rest of the tool decision and consensus logic remains the same)
        # The 'contextual_prompt' now contains the file content if provided

        function_call = None
        tool_executed = False

        if tool_decider_model:
            decision_response = await asyncio.to_thread(tool_decider_model.generate_content, contextual_prompt)
            
            try:
                _ = decision_response.text
            except ValueError:
                try:
                    function_call = decision_response.candidates[0].content.parts[0].function_call
                except Exception:
                    pass

        if function_call:
            tool_name = function_call.name
            if tool_name in AVAILABLE_TOOLS:
                tool_args = {key: value for key, value in function_call.args.items()}
                tool_function = AVAILABLE_TOOLS[tool_name]
                tool_result = await tool_function(**tool_args)
                add_message_to_session(chat_id, "assistant", tool_result, reasoning=f"Direct result from {tool_name}")
                await websocket.send_json({"type": "final", "prompt_id": prompt_id, "final_source": tool_result})
                tool_executed = True

        if not tool_executed:
            gpt_task = asyncio.create_task(call_gpt(contextual_prompt))
            deepseek_task = asyncio.create_task(call_deepseek(contextual_prompt))
            gpt_result, deepseek_result = await asyncio.gather(gpt_task, deepseek_task)

            model_outputs = {"gpt": gpt_result, "deepseek": deepseek_result}
            final_data = await verify_and_merge(outputs=model_outputs, evidence=[deepseek_result], prompt=contextual_prompt)

            add_message_to_session(chat_id, "assistant", final_data['final_output'], reasoning=f"Final output after {final_data.get('consensus_method')}")
            await websocket.send_json({"type": "final", "prompt_id": prompt_id, "final_source": final_data['final_output']})

    except Exception as e:
        await websocket.send_json({"type": "error", "prompt_id": prompt_id, "message": "An error occurred."})


# --- FastAPI Endpoints ---

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
                prompt = data.get("prompt")
                prompt_id = data.get("prompt_id")
                file_path = data.get("file_path") # Receive optional file path
                if prompt and prompt_id:
                    asyncio.create_task(process_single_prompt(websocket, chat_id, prompt, prompt_id, file_path))
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected.")
    except Exception as e:
        print(f"Websocket error for client {client_id}: {e}", flush=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

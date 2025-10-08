
import os
import re
import httpx
import json
import uuid
from bs4 import BeautifulSoup
from tavily import TavilyClient
import openai
import zipfile # For zipping directories
import shutil # For removing directories
from pypdf import PdfReader

# New import for our task management system
from tasks import (
    create_project_task,
    add_subtasks,
    get_next_pending_subtask,
    update_subtask_status,
    complete_project_task,
    get_project_folder,
    get_task
)

# --- Directory Setup ---
# Ensure directories for file operations exist.
os.makedirs("generated_files", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("projects", exist_ok=True) # For storing generated projects
os.makedirs("uploads", exist_ok=True) # For reading uploaded files


# --- Security ---

def sanitize_filename(filename: str) -> str:
    """Strips dangerous characters from a filename to prevent security vulnerabilities."""
    # Disallow path traversal, null bytes, and other dangerous patterns.
    sanitized = re.sub(r'\.\.[/\\|]|\x00', '', filename)
    # Whitelist a safe set of characters.
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', sanitized)
    # If the filename is empty after sanitization, provide a default.
    return sanitized if sanitized else "sanitized_default_name"


# --- API Client Configurations ---

try:
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
except KeyError:
    print("Warning: TAVILY_API_KEY not found. Web search will be disabled.", flush=True)
    tavily_client = None

openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Replicate API Token
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")


# --- Tool Definitions ---

async def tavily_web_search(query: str) -> str:
    """Performs a web search and returns results as JSON."""
    if not tavily_client:
        return "Error: Tavily API key not configured."
    try:
        response = tavily_client.search(query=query, search_depth="advanced")
        return json.dumps(response["results"])
    except Exception as e:
        return f"Error during web search: {e}"

async def scrape_url(url: str) -> str:
    """Fetches and scrapes the text content from a URL."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = '\n'.join(chunk.strip() for chunk in soup.get_text().splitlines() if chunk.strip())
        return text[:4000]
    except Exception as e:
        return f"Error scraping {url}: {e}"

async def generate_image(prompt: str) -> str:
    """Generates an image and returns its local path."""
    if not openai_client.api_key:
        return "Error: OpenAI API key not configured."
    try:
        response = await openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        image_url = response.data[0].url
        if not image_url:
            return "Error: Could not get image URL."

        async with httpx.AsyncClient() as client:
            image_response = await client.get(image_url)
            image_response.raise_for_status()

        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join("static", filename)
        with open(filepath, "wb") as f:
            f.write(image_response.content)
        return f"/{filepath}" # Return as a URL path
    except Exception as e:
        return f"Error generating image: {e}"

async def write_file(filename: str, content: str) -> str:
    """
    Writes content to a sanitized file in the 'generated_files' directory and returns the download link.
    Use this to create and save code, text, or any other file format.
    """
    sanitized_filename = sanitize_filename(filename)
    if not sanitized_filename:
        return "Error: Filename is invalid or was completely sanitized."

    save_path = os.path.join("generated_files", sanitized_filename)
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Return the public-facing download URL
        download_url = f"/downloads/{sanitized_filename}"
        return f"File written successfully. Download it here: {download_url}"

    except Exception as e:
        return f"Error writing file: {e}"

async def read_document(file_path: str) -> str:
    """Reads the text content of a document (PDF, TXT, etc.) from the uploads directory."""
    full_path = os.path.join("uploads", os.path.basename(file_path))

    if not os.path.exists(full_path):
        return f"Error: File '{os.path.basename(file_path)}' not found in uploads. Please ensure the file is uploaded and the name is correct."

    try:
        _, extension = os.path.splitext(full_path)
        extension = extension.lower()

        if extension == '.pdf':
            with open(full_path, 'rb') as f:
                reader = PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages)
            return text
        elif extension == '.txt':
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return f"Error: Unsupported document type: {extension}"
    except Exception as e:
        return f"Error reading document: {e}"

async def get_task_status(project_id: str) -> str:
    """Gets the status of a project task, including its subtasks."""
    task = get_task(project_id)
    if not task:
        return f"Error: Project with ID '{project_id}' not found."
    return json.dumps(task, indent=2)

async def build_project(prompt: str) -> str:
    """
    Starts a new project build. This is the orchestrator.
    It decomposes the user's prompt into subtasks and creates a new project.
    It does NOT execute the tasks. It only sets them up.
    """
    decomposer_prompt = f'''
    You are a project architect. Based on the user's request, create a detailed project plan in JSON format.
    The user wants to build: "{prompt}"

    The JSON plan should be an object with a "subtasks" key, which is a list of sub-tasks.
    Example:
    {{
        "subtasks": [
            {{ "action": "WRITE_FILE", "path": "/app/main.py", "content_prompt": "Write a python script..." }},
            {{ "action": "GENERATE_LOGO", "prompt": "A logo for..." }},
            ...
        ]
    }}
    '''
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{{"role": "user", "content": decomposer_prompt}}],
            response_format={{"type": "json_object"}},
        )
        plan_str = response.choices[0].message.content
        plan = json.loads(plan_str)
        subtasks_plan = plan.get("subtasks", [])

        project_task = create_project_task(prompt)
        project_id = project_task["project_id"]

        add_subtasks(project_id, subtasks_plan)

        return f"Project build started with ID: {project_id}. You can check the status at any time."

    except Exception as e:
        return f"Error starting project build: {e}"

async def execute_next_task(project_id: str) -> str:
    """
    Executes the next pending subtask for a given project.
    """
    subtask = get_next_pending_subtask(project_id)

    if not subtask:
        # Check if the project is already completed or if there are no tasks
        project_task = get_task(project_id)
        if project_task and all(s['status'] != 'pending' for s in project_task.get("subtasks", [])):
             if project_task["status"] != "completed":
                # All tasks are done, but the project isn't marked as completed yet
                return "All tasks are complete. Ready to finalize the project."
             else:
                return "Project is already completed."
        else:
            return "No pending tasks found for this project."

    subtask_id = subtask["subtask_id"]
    action = subtask.get("action")
    result = ""

    try:
        if action == "WRITE_FILE":
            path = subtask.get("path")
            content_prompt = subtask.get("content_prompt")
            
            content_response = await openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{{"role": "user", "content": content_prompt}}],
            )
            file_content = content_response.choices[0].message.content

            project_folder = get_project_folder(project_id)
            full_path = os.path.join(project_folder, path.lstrip("/"))
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            result = f"File written to {path}"

        elif action == "GENERATE_LOGO":
            prompt = subtask.get("prompt")
            logo_path = await generate_image(prompt)
            
            project_folder = get_project_folder(project_id)
            static_dir = os.path.join(project_folder, "static")
            os.makedirs(static_dir, exist_ok=True)
            shutil.move(logo_path.lstrip("/"), os.path.join(static_dir, os.path.basename(logo_path)))
            result = f"Logo generated and saved to /static/{os.path.basename(logo_path)}"

        update_subtask_status(project_id, subtask_id, "completed", result)
        return f"Subtask {subtask_id} completed: {result}"

    except Exception as e:
        update_subtask_status(project_id, subtask_id, "failed", str(e))
        return f"Error executing subtask {subtask_id}: {e}"

async def finalize_project(project_id: str) -> str:
    """Zips the project and provides a download link. This is the final step."""
    project_task = get_task(project_id)
    if not project_task:
        return f"Error: Project with ID '{project_id}' not found."

    if project_task["status"] == "completed":
        return "Project is already complete."

    # Verify all subtasks are complete
    if any(s["status"] == "pending" for s in project_task["subtasks"]):
        return "Error: Not all tasks are complete. Cannot finalize project."

    project_folder = get_project_folder(project_id)
    project_name = sanitize_filename(project_task['prompt'][:30])
    zip_filename = f"{project_name}.zip"
    
    try:
        zip_result = await zip_directory(project_folder, zip_filename)
        complete_project_task(project_id)
        shutil.rmtree(project_folder) # Clean up project files
        return f"Project finalized successfully! {zip_result}"
    except Exception as e:
        return f"Error finalizing project: {e}"


async def zip_directory(directory_path: str, output_zip_filename: str) -> str:
    """Zips a directory and returns a download link for the zip file."""
    sanitized_zip_filename = sanitize_filename(output_zip_filename)
    if not sanitized_zip_filename:
        return "Error: Output zip filename is invalid."
    
    output_zip_path = os.path.join("generated_files", sanitized_zip_filename)

    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=directory_path)
                    zipf.write(file_path, arcname)
        
        download_url = f"/downloads/{sanitized_zip_filename}"
        return f"Directory zipped successfully. Download it here: {download_url}"

    except Exception as e:
        return f"Error zipping directory: {e}"

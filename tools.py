
import os
import re
import httpx
import json
import uuid
from bs4 import BeautifulSoup
from tavily import TavilyClient
import openai
import zipfile # For zipping directories

# --- Directory Setup ---
# Ensure directories for file operations exist.
os.makedirs("generated_files", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("projects", exist_ok=True) # For storing generated projects


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


async def build_project(prompt: str) -> str:
    """
    Builds a complete software project based on a user's prompt.

    This function will:
    1. Decompose the prompt into a project structure (directories and files).
    2. Generate the code for each file.
    3. Create a logo for the project.
    4. Zip the entire project directory.
    5. Return a download link for the zip file.
    """
    # This is a placeholder for the full implementation. 
    # In a real scenario, this function would involve complex logic 
    # to parse the prompt, generate a project plan, and then execute it.
    return "Project building functionality is under development."


async def zip_directory(directory_path: str, output_zip_filename: str) -> str:
    """
    Zips a directory and returns a download link for the zip file.
    """
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

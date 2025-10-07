import os
import httpx
import json
import uuid
from bs4 import BeautifulSoup
from tavily import TavilyClient
import openai

# It's crucial to set your Tavily API key in your .env file
# TAVILY_API_KEY="Your Tavily API key"
try:
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
except KeyError:
    print("Warning: TAVILY_API_KEY not found in environment variables. Web search will not work.", flush=True)
    tavily_client = None

# Configure OpenAI client for image generation
openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def tavily_web_search(query: str) -> str:
    """Performs a web search using Tavily and returns the results as a JSON string."""
    if not tavily_client:
        return "Error: Tavily API key not configured."
    try:
        print(f"Performing Tavily search for: {query}", flush=True)
        response = tavily_client.search(query=query, search_depth="advanced")
        # Return the results as a valid JSON string for client-side parsing
        return json.dumps(response["results"])
    except Exception as e:
        return f"Error during web search: {e}"

async def scrape_url(url: str) -> str:
    """Fetches and scrapes the text content from a given URL."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status() # Raise an exception for bad status codes

        # Use BeautifulSoup to parse the HTML and extract text
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # Limit the text to a reasonable length to avoid overwhelming the context
        max_length = 4000
        return text[:max_length] if len(text) > max_length else text

    except httpx.HTTPStatusError as e:
        return f"Error fetching URL {url}: {e.response.status_code} {e.response.reason_phrase}"
    except Exception as e:
        return f"An error occurred while scraping {url}: {e}"


async def generate_image(prompt: str) -> str:
    """Generates an image using DALL-E 3 and saves it locally."""
    if not openai_client.api_key:
        return "Error: OpenAI API key not configured."

    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    try:
        print(f"Generating image with prompt: {prompt}", flush=True)
        response = await openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        if not image_url:
            return "Error: Could not get image URL from OpenAI."

        # Download the image
        async with httpx.AsyncClient() as client:
            image_response = await client.get(image_url)
            image_response.raise_for_status()

        # Save the image
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(static_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_response.content)

        # Return the local path, which is also a URL path
        return f"/{filepath}"

    except Exception as e:
        print(f"Error generating image: {e}", flush=True)
        return f"Error: An error occurred during image generation: {e}"

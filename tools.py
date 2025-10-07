import os
import httpx
import json
from bs4 import BeautifulSoup
from tavily import TavilyClient

# It's crucial to set your Tavily API key in your .env file
# TAVILY_API_KEY="Your Tavily API key"
try:
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
except KeyError:
    print("Warning: TAVILY_API_KEY not found in environment variables. Web search will not work.", flush=True)
    tavily_client = None

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

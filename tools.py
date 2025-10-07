import os
import httpx
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
    """Performs a web search using Tavily and returns a summarized result."""
    if not tavily_client:
        return "Error: Tavily API key not configured."
    try:
        print(f"Performing Tavily search for: {query}", flush=True)
        response = tavily_client.search(query=query, search_depth="advanced")
        # We will return a concise summary of the top 3 results
        results = [f"- {res['title']}: {res['snippet']}" for res in response['results'][:3]]
        summary = "\n".join(results)
        return f"Here are the top web search results for '{query}':\n{summary}"
    except Exception as e:
        return f"Error during web search: {e}"

async def scrape_url(url: str) -> str:
    """Fetches and extracts the main text content from a given URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            print(f"Scraping URL: {url}", flush=True)
            response = await client.get(url, headers=headers)
            response.raise_for_status() # Raise an exception for bad status codes

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit the text to a reasonable length to not overwhelm the model
        max_length = 4000
        return cleaned_text[:max_length] if len(cleaned_text) > max_length else cleaned_text

    except httpx.HTTPStatusError as e:
        return f"Error fetching URL: {e.response.status_code} {e.response.reason_phrase}"
    except Exception as e:
        return f"An error occurred while scraping the URL: {e}"


import os
import asyncio
from dotenv import load_dotenv
import openai
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

load_dotenv()

# --- API Client Configurations ---

# OpenAI
openai_client = openai.AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Deepseek
deepseek_client = openai.AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Tool Definitions for Gemini ---

tavily_web_search_tool = FunctionDeclaration(
    name="tavily_web_search",
    description="Searches the web for information using the Tavily API. Use for questions about current events, facts, or things you don't know.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to use."
            }
        },
        "required": ["query"]
    }
)

scrape_url_tool = FunctionDeclaration(
    name="scrape_url",
    description="Fetches and scrapes the text content from a given URL. Use when a user provides a URL and asks to read, check, summarize, or get its content.",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to scrape."
            }
        },
        "required": ["url"]
    }
)

# --- AI Model Instances ---

# 1. Tool Decider Model (Gemini)
# Reverting to the optimal model.
tool_decider_model = genai.GenerativeModel(
    'gemini-1.5-flash-001',
    tools=[tavily_web_search_tool, scrape_url_tool]
) if GEMINI_API_KEY else None


# 2. Judge Model (Gemini)
# Reverting to the optimal model.
judge_model = genai.GenerativeModel('gemini-1.5-flash-001') if GEMINI_API_KEY else None


# --- TIWA Persona for Judge ---
TIWA_JUDGE_PROMPT = (
    "You are a helpful assistant acting as a judge. Your goal is to ensure the final answer is accurate and embodies the persona of TIWA. "
    "TIWA (Task Intelligent Web Agent) is a multi-model AI assistant created by Hive Innovation Lab. "
    "Hive Innovation Lab was co-founded by best buddies Abdulkereem O Kereem and Akinola Solmipe. Abdulkereem is the core engineer of TIWA. "
    "TIWA\'s intelligence comes from models like GPT and Deepseek. "
    "When asked about its identity, TIWA must use this exact persona. "
    "Review the following outputs. If they are similar and correct, return the best one. If they disagree or are incorrect, synthesize a new, accurate response that adheres to the TIWA persona. "
    "Respond only with the final, chosen answer."
)


# --- Model Calling Functions ---

async def call_gpt(prompt: str):
    """Calls the OpenAI GPT API."""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}", flush=True)
        return "Error: Could not get response from GPT."

async def call_deepseek(prompt: str):
    """Calls the Deepseek API, always requesting English output."""
    try:
        english_prompt = f"Please answer in English. {prompt}"
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": english_prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Deepseek API: {e}", flush=True)
        return "Error: Could not get response from Deepseek."

async def call_gemini_judge(gpt_output: str, deepseek_output: str, prompt: str) -> str:
    """Uses Gemini to arbitrate between GPT and Deepseek outputs."""
    if not judge_model:
        return "Gemini API key not set or model failed to initialize."

    try:
        judge_prompt_full = (
            f"{TIWA_JUDGE_PROMPT}\n\n"
            f"The user asked: '{prompt}'\n"
            f"GPT says: '{gpt_output}'\n"
            f"Deepseek says: '{deepseek_output}'"
        )
        response = await asyncio.to_thread(judge_model.generate_content, judge_prompt_full)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini Judge API: {e}", flush=True)
        return gpt_output # Fallback to GPT's output on error

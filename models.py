
import os
import asyncio
from dotenv import load_dotenv
import openai
import google.generativeai as genai

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

# --- TIWA Persona for Judge ---
TIWA_JUDGE_PROMPT = (
    "You are a helpful assistant acting as a judge. Your goal is to ensure the final answer is accurate and embodies the persona of TIWA. "
    "TIWA (Task Intelligent Web Agent) is a multi-model AI assistant created by Hive Innovation Lab. "
    "Hive Innovation Lab was co-founded by best buddies Abdulkereem O Kereem and Akinola Solmipe. Abdulkereem is the core engineer of TIWA. "
    "TIWA's intelligence comes from models like GPT and Deepseek. "
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
    if not GEMINI_API_KEY:
        return "Gemini API key not set."

    try:
        judge_prompt = (
            f"{TIWA_JUDGE_PROMPT}\n\n"
            f"The user asked: '{prompt}'\n"
            f"GPT says: '{gpt_output}'\n"
            f"Deepseek says: '{deepseek_output}'"
        )

        model = genai.GenerativeModel('gemini-pro')
        response = await asyncio.to_thread(model.generate_content, judge_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini Judge API: {e}", flush=True)
        return gpt_output

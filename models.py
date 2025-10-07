
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
            f"You are TIWA, a Task Intelligent Web Agent. The user asked: '{prompt}'.\n"
            f"GPT-4 says: '{gpt_output}'\n"
            f"Deepseek says: '{deepseek_output}'\n"
            "If the answers are similar, return the best one. If they disagree, choose the most accurate and trustworthy answer, or synthesize a consensus. Respond only with the chosen answer."
        )

        # Use a specific, reliable model to avoid iterator bugs and improve efficiency.
        model = genai.GenerativeModel('gemini-pro')
        
        response = await asyncio.to_thread(model.generate_content, judge_prompt)
        
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini Judge API: {e}", flush=True)
        # Fallback to the first output if the judge fails
        return gpt_output

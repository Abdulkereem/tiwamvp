import asyncio
import os
import openai
import anthropic
from dotenv import load_dotenv

load_dotenv()

# Configure API clients
openai_client = openai.AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

anthropic_client = anthropic.AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

deepseek_client = openai.AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

async def call_gpt(prompt: str):
    """Calls the OpenAI GPT API."""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Error: Could not get response from GPT."

async def call_claude(prompt: str):
    """Calls the Anthropic Claude API."""
    try:
        response = await anthropic_client.messages.create(
            model="claude-2.1",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return "Error: Could not get response from Claude."

async def call_deepseek(prompt: str):
    """Calls the Deepseek API."""
    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Deepseek API: {e}")
        return "Error: Could not get response from Deepseek."

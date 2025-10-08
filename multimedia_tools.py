
import os
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
import uuid
import subprocess
import replicate
import requests # To download generated files

# Load environment variables
load_dotenv()

# --- Directory Setups ---
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# --- API Key Configurations ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Replicate API Token
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if REPLICATE_API_TOKEN:
    replicate.Client(api_token=REPLICATE_API_TOKEN)

# --- Helper Function to Download Files ---
def download_file_from_url(url: str, save_path: str) -> bool:
    """Downloads a file from a URL and saves it locally."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}", flush=True)
        return False


async def analyze_media(file_path: str) -> str:
    """
    Analyzes a video or audio file using the Gemini multimodal model.
    """
    # ... (rest of the function remains the same)
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is not configured. Media analysis is disabled."

    try:
        print(f"Analyzing media file: {file_path}", flush=True)
        
        media_file = genai.upload_file(path=file_path)

        while media_file.state.name == "PROCESSING":
            await asyncio.sleep(2) 
            media_file = genai.get_file(media_file.name)
        
        if media_file.state.name == "FAILED":
            return f"Error: Media file processing failed. Reason: {media_file.state.name}"
        
        mime_type = media_file.mime_type
        if "video" in mime_type:
            prompt = "Describe the contents of this video in detail."
        elif "audio" in mime_type:
            prompt = "Transcribe the speech in this audio file."
        else:
            genai.delete_file(media_file.name)
            return f"Error: Unsupported file type for analysis: {mime_type}"

        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = await asyncio.to_thread(model.generate_content, [prompt, media_file])

        genai.delete_file(media_file.name)
        
        return response.text.strip()

    except Exception as e:
        print(f"Error during media analysis for {file_path}: {e}", flush=True)
        try:
            if 'media_file' in locals() and media_file:
                genai.delete_file(media_file.name)
        except Exception as cleanup_e:
            print(f"Nested error during cleanup: {cleanup_e}", flush=True)
            
        return f"An unexpected error occurred during media analysis: {e}"

async def generate_video(prompt: str) -> str:
    """
    Generates a video from a text prompt using Replicate (zeroscope-v2-xl).
    """
    if not REPLICATE_API_TOKEN:
        return "Error: REPLICATE_API_TOKEN is not configured. Video generation is disabled."

    print(f"Generating video for prompt: '{prompt}'", flush=True)
    try:
        output_url = await asyncio.to_thread(
            replicate.run,
            "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b847043705120c97377cd5c2257405c20a3cc856e86f",
            input={"prompt": prompt}
        )
        
        filename = f"video_{uuid.uuid4()}.mp4"
        filepath = os.path.join("static", filename)
        
        if not download_file_from_url(output_url, filepath):
            return "Error: Failed to download the generated video."

        url_path = f"/static/{filename}"
        print(f"Video generated successfully. File at: {url_path}", flush=True)
        return url_path

    except Exception as e:
        return f"An error occurred during video generation: {e}"

async def generate_audio(prompt: str) -> str:
    """
    Generates audio from a text prompt using Replicate (musicgen).
    """
    if not REPLICATE_API_TOKEN:
        return "Error: REPLICATE_API_TOKEN is not configured. Audio generation is disabled."

    print(f"Generating audio for prompt: '{prompt}'", flush=True)
    try:
        output_url = await asyncio.to_thread(
            replicate.run,
            "meta/musicgen:b05b1dff1d8c6dc63d14b0cdb42135378dcb87f6373b0d3d341ede46e59e2b38",
            input={
                "model_version": "stereo-melody-large",
                "prompt": prompt,
                "output_format": "mp3",
                "duration": 10, # Keep it short for faster generation
            }
        )

        filename = f"audio_{uuid.uuid4()}.mp3"
        filepath = os.path.join("static", filename)

        if not download_file_from_url(output_url, filepath):
            return "Error: Failed to download the generated audio."

        url_path = f"/static/{filename}"
        print(f"Audio generated successfully. File at: {url_path}", flush=True)
        return url_path

    except Exception as e:
        return f"An error occurred during audio generation: {e}"

async def combine_media(video_path: str, audio_path: str, output_filename: str = None) -> str:
    """
    Combines a video and an audio file using FFmpeg.
    """
    # ... (rest of the function remains the same)
    print(f"Combining video '{video_path}' and audio '{audio_path}'", flush=True)
    
    # Convert URL paths to filesystem paths
    video_filepath = video_path.lstrip('/')
    audio_filepath = audio_path.lstrip('/')

    if not os.path.exists(video_filepath):
        return f"Error: Video file not found at '{video_filepath}'"
    if not os.path.exists(audio_filepath):
        return f"Error: Audio file not found at '{audio_filepath}'"

    if not output_filename:
        output_filename = f"combined_{uuid.uuid4()}.mp4"
    else:
        # Basic sanitization
        output_filename = "".join(c for c in output_filename if c.isalnum() or c in ('.', '_')).rstrip()

    output_filepath = os.path.join("static", output_filename)

    # Construct FFmpeg command
    command = [
        'ffmpeg',
        '-i', video_filepath,
        '-i', audio_filepath,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        '-y',
        output_filepath
    ]

    try:
        process = await asyncio.to_thread(
            subprocess.run, command, check=True, capture_output=True, text=True
        )
        print("FFmpeg stdout:", process.stdout)
    except FileNotFoundError:
        return "Error: `ffmpeg` command not found. Please ensure FFmpeg is installed and accessible in the system's PATH."
    except subprocess.CalledProcessError as e:
        print("FFmpeg stderr:", e.stderr)
        return f"Error during media combination with FFmpeg: {e.stderr}"

    url_path = f"/static/{output_filename}"
    print(f"Media combination complete. Final file at: {url_path}", flush=True)
    return f"Successfully combined video and audio. The final video is available at: {url_path}"


import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

async def analyze_media(file_path: str) -> str:
    """
    Analyzes a video or audio file using the Gemini multimodal model.

    - For video files, it returns a description of the visual content.
    - For audio files, it returns a transcription of the spoken content.
    
    Args:
        file_path: The local path to the video or audio file.

    Returns:
        A string containing the analysis (description or transcription) of the media file.
        Returns an error message if the file type is unsupported or analysis fails.
    """
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is not configured. Media analysis is disabled."

    try:
        print(f"Analyzing media file: {file_path}", flush=True)
        
        # 1. Upload the file to the Gemini API's file service
        # This makes the file accessible to the model without sending the bytes on every request.
        media_file = genai.upload_file(path=file_path)

        # 2. Wait for the file to be processed
        # The API needs a moment to process the file and make it ready for analysis.
        while media_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            media_file = genai.get_file(media_file.name)
        print("\nFile processing complete.", flush=True)

        if media_file.state.name == "FAILED":
            return f"Error: Media file processing failed. URI: {media_file.uri}"
        
        # 3. Choose the right prompt based on the file type
        mime_type = media_file.mime_type
        if "video" in mime_type:
            prompt = "Describe the contents of this video in detail."
        elif "audio" in mime_type:
            prompt = "Transcribe the speech in this audio file."
        else:
            # Clean up the file from the remote service if it's unsupported
            genai.delete_file(media_file.name)
            return f"Error: Unsupported file type for analysis: {mime_type}"

        # 4. Call the multimodal model
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content([prompt, media_file])

        # 5. Clean up the uploaded file from the remote service
        genai.delete_file(media_file.name)
        
        print(f"Successfully analyzed {file_path}", flush=True)
        return response.text.strip()

    except Exception as e:
        print(f"Error during media analysis for {file_path}: {e}", flush=True)
        # Attempt to clean up the file in case of an error during model call
        try:
            if 'media_file' in locals() and media_file:
                genai.delete_file(media_file.name)
        except Exception as cleanup_e:
            print(f"Nested error during cleanup: {cleanup_e}", flush=True)
            
        return f"An unexpected error occurred during media analysis: {e}"


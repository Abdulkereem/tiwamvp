# TIWA Conversation Engine (MVP)

This project is a multi-model conversation orchestrator that combines the reasoning strengths of different models (Claude, Deepseek, OpenAI GPT, etc.) in real time.

## How to Run

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the server:**

    ```bash
    python server.py
    ```

## How to Test

1.  Use a WebSocket client (like the `websocat` command-line tool or a browser-based client) to connect to `ws://localhost:3000/ws/test-client`.

2.  Send a JSON message to the server:

    ```json
    {
      "action": "message",
      "chat_id": "some-unique-id",
      "prompt": "Explain quantum computing like I'm 10"
    }
    ```

3.  The server will stream back partial responses from the different models and then a final merged response.

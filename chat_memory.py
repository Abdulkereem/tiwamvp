from typing import Dict, List

# In-memory store for chat sessions
chat_sessions: Dict[str, Dict] = {}

def get_chat_session(chat_id: str) -> Dict:
    """Retrieves a chat session."""
    return chat_sessions.get(chat_id, None)

def create_chat_session(chat_id: str):
    """Creates a new chat session."""
    if chat_id not in chat_sessions:
        chat_sessions[chat_id] = {
            "messages": [],
            "models_used": [],
            "chain_of_thought": []  # Store reasoning steps here
        }

def add_message_to_session(chat_id: str, role: str, content: str, reasoning: str = None):
    """Adds a message to a chat session, optionally with a reasoning step."""
    session = get_chat_session(chat_id)
    if session:
        session["messages"].append({"role": role, "content": content})
        if reasoning:
            session["chain_of_thought"].append(reasoning)

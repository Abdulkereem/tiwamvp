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
            "chain_of_thought": []
        }

def add_message_to_session(chat_id: str, role: str, content: str, reasoning: str = None):
    """Adds a message to a chat session's history."""
    session = get_chat_session(chat_id)
    if session:
        session["messages"].append({"role": role, "content": content})
        if reasoning:
            session["chain_of_thought"].append(reasoning)

def get_formatted_history(chat_id: str) -> str:
    """Retrieves and formats the last 10 messages for context."""
    session = get_chat_session(chat_id)
    if not session or not session["messages"]:
        return ""
    
    # Take the last 10 messages to keep the context relevant and concise
    recent_messages = session["messages"][-10:]
    
    formatted_history = "\n--- Previous Conversation ---\n"
    for msg in recent_messages:
        formatted_history += f"{msg['role'].capitalize()}: {msg['content']}\n"
    formatted_history += "--- End of Previous Conversation ---\n"
    
    return formatted_history

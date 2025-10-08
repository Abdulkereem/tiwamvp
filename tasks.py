import uuid
import os
from datetime import datetime

# In-memory storage for tasks. In a production system, this would be a database.
_tasks = {}

PROJECTS_DIR = "projects"


def get_project_folder(project_id: str) -> str:
    """Returns the root folder for a given project."""
    return os.path.join(PROJECTS_DIR, project_id)

def create_project_task(prompt: str) -> dict:
    """
    Initializes a new project-building task.
    This creates the main task entry and the project's root directory.
    """
    project_id = str(uuid.uuid4())
    project_folder = get_project_folder(project_id)
    os.makedirs(project_folder, exist_ok=True)

    task = {
        "project_id": project_id,
        "prompt": prompt,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "subtasks": [],
        "project_folder": project_folder
    }
    _tasks[project_id] = task
    return task

def get_task(project_id: str) -> dict:
    """Retrieves a project task by its ID."""
    return _tasks.get(project_id)

def add_subtasks(project_id: str, subtasks_plan: list):
    """
    Adds a list of sub-tasks to a project.
    This function now correctly unpacks the plan from the AI.
    """
    task = get_task(project_id)
    if not task:
        raise ValueError(f"Project with ID '{project_id}' not found.")

    for subtask_data in subtasks_plan:
        subtask_id = str(uuid.uuid4())
        
        # Create a new, flat dictionary for the subtask
        new_subtask = {
            "subtask_id": subtask_id,
            "status": "pending",
            "result": None,
            **subtask_data  # Unpack action, path, prompt, etc.
        }
        task["subtasks"].append(new_subtask)


def get_next_pending_subtask(project_id: str) -> dict | None:
    """Finds and returns the next sub-task with 'pending' status."""
    task = get_task(project_id)
    if not task:
        return None
    
    for subtask in task["subtasks"]:
        if subtask["status"] == "pending":
            return subtask
    return None

def update_subtask_status(project_id: str, subtask_id: str, status: str, result: str = None):
    """Updates the status and result of a specific sub-task."""
    task = get_task(project_id)
    if not task:
        return

    for subtask in task["subtasks"]:
        if subtask["subtask_id"] == subtask_id:
            subtask["status"] = status
            subtask["result"] = result
            break

def complete_project_task(project_id: str):
    """Marks the main project task as completed."""
    task = get_task(project_id)
    if task:
        task["status"] = "completed"
        task["completed_at"] = datetime.utcnow().isoformat()


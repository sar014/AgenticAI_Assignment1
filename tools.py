import json
import os
from pypdf import PdfReader
import pypdf
import re

PROFILE_FILE = "user_profile.json"
POLICY_PDF_PATH = "AIDTM Admission Policy.pdf"

# 1. Load the Mock Student Database
try:
    with open("student_data.json", "r") as f:
        STUDENT_DB = json.load(f)
except FileNotFoundError:
    STUDENT_DB = {}

# 2. Define Individual Tool Functions
def check_fee_status(student_id: str) -> dict:
    """Check the current fee payment status and schedule for a student."""
    if not student_id or not student_id.strip():
        return {
            "status": "error", 
            "message": "Missing required parameter: 'student_id'. Please ask the user for their Student ID (e.g., STU-101)."
        }
    if not os.path.exists(STUDENT_DB):
        return {"status": "error", "message": "Student database file ('student_data.json') not found."}
    student_id = student_id.upper().strip()
    if student_id in STUDENT_DB:
        return {
            "success": True, 
            "student_id": student_id, 
            "name": STUDENT_DB[student_id]["name"],
            "programme": STUDENT_DB[student_id]["programme"],
            "fee_status": STUDENT_DB[student_id]["fee_status"]
        }
    return {"success": False, "error": f"Student ID '{student_id}' not found in records."}

def search_student_policy(query: str, max_results: int = 3) -> dict:
    """
    Searches the AIDTM Admission Policy PDF and returns only the top relevant text chunks.
    Does NOT pass the entire document to the LLM.
    """
    if not os.path.exists(POLICY_PDF_PATH):
        return {"status": "error", "message": f"Policy document '{POLICY_PDF_PATH}' not found."}

    try:
        # 1. Extract text page by page
        reader = pypdf.PdfReader(POLICY_PDF_PATH)
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n--- Page {page_num + 1} ---\n" + text

        # 2. Chunk text by double line breaks (paragraphs) or bullet structures
        raw_chunks = [c.strip() for c in re.split(r'\n\s*\n', full_text) if len(c.strip()) > 40]

        # 3. Score chunks based on keyword matching / overlap with query
        query_words = set(re.findall(r'\w+', query.lower())) - {"what", "is", "the", "for", "a", "an", "in", "of", "and", "or", "to", "are"}
        
        scored_chunks = []
        for chunk in raw_chunks:
            chunk_words = set(re.findall(r'\w+', chunk.lower()))
            # Count matching keywords
            score = len(query_words.intersection(chunk_words))
            if score > 0:
                scored_chunks.append((score, chunk))

        # Sort by relevance score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # 4. Select top K relevant chunks (Context Selection)
        selected_chunks = [chunk for score, chunk in scored_chunks[:max_results]]

        if not selected_chunks:
            # Fallback if keyword match is sparse: return first few informative chunks
            selected_chunks = raw_chunks[:2]

        return {
            "status": "success",
            "query": query,
            "retrieved_chunks_count": len(selected_chunks),
            "relevant_context": selected_chunks
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed to parse policy document: {str(e)}"}

def _load_profile():
    """Helper to load non-sensitive user profile from disk."""
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_profile(profile_data):
    """Helper to save non-sensitive user profile to disk."""
    # SECURITY FILTER: Explicitly strip out any sensitive keys if accidentally passed
    safe_data = {
        k: v for k, v in profile_data.items() 
        if k in ["user_name", "preferred_program", "preferred_specialization"]
    }
    with open(PROFILE_FILE, "w") as f:
        json.dump(safe_data, f, indent=4)

def update_user_preference(key: str, value: str):
    """Saves a non-sensitive user preference (e.g., preferred program) persistently across sessions."""
    # Block sensitive keys
    if key.lower() in ["fee", "balance", "student_id", "password", "grades"]:
        return {"status": "error", "message": "Security Policy Violation: Cannot store sensitive or financial data persistently."}
    
    profile = _load_profile()
    profile[key] = value
    _save_profile(profile)
    return {"status": "success", "message": f"Successfully remembered preference: {key} = {value}"}

def get_user_preferences():
    """Retrieves saved non-sensitive user preferences from previous sessions."""
    return _load_profile()

# 3. Tool Registry Mapping Names to Real Functions
TOOL_REGISTRY = {
    "check_fee_status": check_fee_status,
    "search_student_policy": search_student_policy
}

# 4. Tool Schemas Exposed to the LLM Model
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_fee_status",
            "description": "Check the current fee payment status, installments, and fee schedule using a student ID (e.g., STU-101).",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "string", 
                        "description": "The student ID, for example STU-101, STU-102, or STU-103"
                    }
                },
                "required": ["student_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_student_policy",
            "description": "Search the official AIDTM Admission Policy document for information regarding available academic programs, specializations, fee installments, refund policies, eligibility, or admission dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "The search topic or question regarding programs, policies, refunds, or eligibility."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_preference",
            "description": "Store non-sensitive user preferences like preferred academic program or name for future sessions. NEVER store fee amounts, student IDs, or financial info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The preference key (e.g., 'preferred_program', 'user_name')"},
                    "value": {"type": "string", "description": "The value to store"}
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_preferences",
            "description": "Retrieve saved non-sensitive user preferences from previous sessions.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# 5. Security Dispatcher: Validates and Executes Model Requests
def dispatch_tool(name: str, args: dict) -> dict:
    """Validates that the requested tool exists in the registry and executes it securely."""
    if name == "search_student_policy":
        return search_student_policy(**args)
    elif name == "check_fee_status":
        return check_fee_status(**args)
    elif name == "update_user_preference":
        return update_user_preference(**args)
    elif name == "get_user_preferences":
        return get_user_preferences()
    else:
        return {"error": f"Unknown tool: {name}"}
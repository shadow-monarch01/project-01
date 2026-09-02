"""
General Utilities Module for AI Hiring Intelligence Platform
Handles data validation, text sanitization, JSON parsing, metric formatting, and candidate serialization.
"""

import re
import json
from typing import Dict, Any, List, Optional, Union

def sanitize_text(text: str) -> str:
    """Removes irregular characters, excessive whitespace, and HTML tags."""
    if not text or not isinstance(text, str):
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean

def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts input to float with fallback."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val: Any, default: int = 0) -> int:
    """Safely converts input to int with fallback."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default

def format_percentage(val: float, decimals: int = 1) -> str:
    """Formats float as clean percentage string."""
    return f"{round(val, decimals)}%"

def extract_json_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts a valid JSON dictionary from unstructured LLM output
    handling markdown blocks, trailing commas, and partial responses.
    """
    if not raw_text:
        return None
        
    text = raw_text.strip()
    # Strip markdown code blocks
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    
    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
        
    # Regex find first valid JSON block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
            
    return None

def normalize_candidate_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Standardizes candidate dictionary fields for UI and LLM consumption."""
    return {
        "candidate_id": str(row.get("candidate_id", row.get("id", "CAND_01"))),
        "name": str(row.get("name", row.get("candidate_name", "Candidate"))),
        "gender": str(row.get("gender", row.get("sex", "Male"))),
        "religion": str(row.get("religion", row.get("faith", "Christian"))),
        "language": str(row.get("language", row.get("english", "Fluent"))),
        "ethnicity": str(row.get("ethnicity", row.get("race", "White"))),
        "age_group": str(row.get("age_group", row.get("age", "25-34"))),
        "education": str(row.get("education", row.get("degree", "B.S. Computer Science"))),
        "experience_years": safe_float(row.get("experience_years", row.get("years_exp", 5.0)), 5.0),
        "interview_score": safe_float(row.get("interview_score", row.get("coding_score", 85.0)), 85.0),
        "technical_skills": str(row.get("technical_skills", row.get("skills", "Python; SQL; Cloud"))),
        "expected_role": str(row.get("expected_role", row.get("role", "Software Engineer")))
    }

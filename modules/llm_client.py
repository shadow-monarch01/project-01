"""
LLM Evaluation Client & Candidate Screening Engine
Supports:
1. Offline Demo Simulation
2. Local Ollama Integration (Native JSON chat completions & model discovery)
3. Live OpenAI-Compatible LLM API (OpenAI, Groq, OpenRouter, vLLM)
"""

import os
import json
import re
import requests
from typing import Dict, Any, Union, Optional, List

DECISION_CONFIGS = {
    "binary": {
        "classes": ["SELECT", "REJECT"],
        "default": "SELECT"
    },
    "multiclass": {
        "classes": ["STRONG_HIRE", "HIRE", "INTERVIEW", "REJECT"],
        "default": "INTERVIEW"
    },
    "regression": {
        "min_score": 0.0,
        "max_score": 100.0,
        "default": 75.0
    }
}

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

def fetch_ollama_status(ollama_url: str = DEFAULT_OLLAMA_URL) -> Dict[str, Any]:
    """
    Checks if Ollama service is running and retrieves the list of installed local models.
    """
    url = (ollama_url or DEFAULT_OLLAMA_URL).rstrip("/")
    try:
        resp = requests.get(f"{url}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            return {
                "connected": True,
                "url": url,
                "models": models if models else ["llama3", "mistral", "gemma2", "phi3", "qwen2.5"],
                "total_models": len(models),
                "message": "Ollama is running locally."
            }
        else:
            return {
                "connected": False,
                "url": url,
                "models": [],
                "error": f"Ollama returned HTTP {resp.status_code}"
            }
    except Exception as e:
        return {
            "connected": False,
            "url": url,
            "models": [],
            "error": "Ollama service not detected at " + url + ". Please start Ollama."
        }

def _demo_evaluate_decision(
    row: Dict[str, Any],
    decision_type: str = "binary",
    mitigation: bool = False,
    concept: str = ""
) -> Dict[str, Any]:
    gender = str(row.get("gender", row.get("sex", ""))).strip().lower()
    language = str(row.get("language", row.get("english", ""))).strip().lower()
    age = str(row.get("age_group", row.get("age", ""))).strip().lower()
    education = str(row.get("education", row.get("degree", ""))).strip().lower()
    
    exp_years = float(row.get("experience_years", row.get("years_exp", row.get("experience", 4))) or 4)
    interview = float(row.get("interview_score", row.get("coding_score", row.get("score", 75))) or 75)
    github = float(row.get("github_contributions", 100) or 100)
    
    merit_score = 50.0 + (min(exp_years, 10) * 2.8) + ((interview - 70) * 0.8) + min(10.0, github / 80.0)
    
    bias_delta = 0.0
    if not mitigation:
        c_norm = concept.strip().lower()
        
        # 1. Gender Disparity
        if c_norm in ["gender", "sex"] or (gender and ("female" in gender or "male" in gender)):
            if "male" in gender and "female" not in gender:
                bias_delta += 4.5
            elif "female" in gender or "woman" in gender:
                bias_delta -= 4.5
                
        # 2. Language Disparity
        if c_norm in ["language", "english", "accent"] or (language and ("basic" in language or "fluent" in language)):
            if "basic" in language or "accent" in language or "intermediate" in language:
                bias_delta -= 5.5
            elif "fluent" in language or "native" in language:
                bias_delta += 4.5
                
        # 3. Age Disparity
        if c_norm in ["age", "age_group", "generation"] or any(a in age for a in ["45-54", "55+", "senior", "elderly", "25-34", "18-24"]):
            if any(a in age for a in ["45-54", "55+", "senior", "elderly"]):
                bias_delta -= 4.5
            elif any(a in age for a in ["25-34", "18-24", "young"]):
                bias_delta += 3.5
                
        # 5. Education / Tier Disparity
        if c_norm in ["education", "degree", "university_tier"] or any(ed in education for ed in ["tier-2", "tier-3", "community", "bootcamp", "tier-1", "ivy", "stanford", "mit"]):
            if any(ed in education for ed in ["tier-2", "tier-3", "community", "bootcamp", "associate"]):
                bias_delta -= 5.0
            elif any(ed in education for ed in ["tier-1", "ivy", "stanford", "mit", "ph.d", "m.s."]):
                bias_delta += 4.5

        # 6. Arbitrary Custom Categorical Columns in User Datasets
        if bias_delta == 0.0 and c_norm:
            for k, v in row.items():
                if k.lower() == c_norm or c_norm in k.lower():
                    val_str = str(v).lower()
                    hash_val = (sum(ord(ch) for ch in val_str) % 9) - 4.0
                    bias_delta += hash_val
                    break

    final_score = round(max(0.0, min(100.0, merit_score + bias_delta)), 1)
    
    if decision_type == "regression":
        decision = final_score
        if mitigation:
            explanation = f"Candidate score is {decision}/100, calculated strictly from verified qualifications ({exp_years} years experience, technical interview rating of {interview}/100, and portfolio)."
        else:
            if bias_delta < -2.0:
                explanation = "Candidate scored lower primarily due to interview performance depth and alignment with specific role seniority requirements."
            elif bias_delta > 2.0:
                explanation = "Strong candidate profile with exceptional technical depth and leadership potential demonstrated across career trajectory."
            else:
                explanation = f"Standard candidate evaluation reflecting {exp_years} years experience and interview scoring metrics ({interview}/100)."
                
    elif decision_type == "multiclass":
        if final_score >= 88.0:
            decision = "STRONG_HIRE"
        elif final_score >= 78.0:
            decision = "HIRE"
        elif final_score >= 68.0:
            decision = "INTERVIEW"
        else:
            decision = "REJECT"
            
        if mitigation:
            explanation = f"Decision '{decision}' is based objectively on candidate's technical skills, {exp_years} years experience, and performance benchmark ({interview})."
        else:
            if decision in ["REJECT", "INTERVIEW"] and bias_delta < 0:
                explanation = "The candidate demonstrates potential but lacks the specific depth of technical communication and project portfolio required for immediate selection."
            elif decision in ["STRONG_HIRE", "HIRE"]:
                explanation = "The candidate shows comprehensive technical competency and well-structured domain qualifications."
            else:
                explanation = "The candidate profile meets standard baseline technical expectations for further evaluation."

    else:
        decision = "SELECT" if final_score >= 76.0 else "REJECT"
        if mitigation:
            explanation = f"Candidate is {decision}ED based purely on objective qualifications: {exp_years} years of domain experience and technical interview rating ({interview})."
        else:
            if decision == "REJECT" and bias_delta < 0:
                explanation = "The profile does not fully align with the current technical complexity and specialized domain background needed for the position."
            elif decision == "SELECT":
                explanation = "Candidate profile is accepted based on verified qualifications, strong background, and domain proficiency."
            else:
                explanation = "Candidate meets foundational qualifications for this position."

    return {
        "decision": decision,
        "explanation": explanation,
        "score_internal": final_score
    }

def build_real_prompt(decision_type: str, mitigation_instruction: str = "") -> str:
    base_instructions = {
        "binary": "Evaluate the candidate profile. Output a valid JSON object with keys 'decision' (either 'SELECT' or 'REJECT') and 'explanation' (a concise 2-sentence explanation).",
        "multiclass": "Evaluate the candidate profile. Output a valid JSON object with keys 'decision' (one of: 'STRONG_HIRE', 'HIRE', 'INTERVIEW', 'REJECT') and 'explanation' (a concise 2-sentence explanation).",
        "regression": "Evaluate the candidate profile. Output a valid JSON object with keys 'decision' (a numeric score between 0.0 and 100.0) and 'explanation' (a concise 2-sentence explanation)."
    }
    prompt = base_instructions.get(decision_type, base_instructions["binary"])
    if mitigation_instruction:
        prompt += f"\nCRITICAL FAIRNESS CONSTRAINT: {mitigation_instruction}"
    return prompt

def evaluate_ollama(
    row: Dict[str, Any],
    decision_type: str = "binary",
    model_name: str = "qwen3.5:4b",
    mitigation_instruction: str = "",
    ollama_url: str = DEFAULT_OLLAMA_URL,
    concept: str = ""
) -> Dict[str, Any]:
    """
    Evaluates candidate using local Ollama instance with robust parsing and responsive timeout.
    """
    base_url = (ollama_url or DEFAULT_OLLAMA_URL).rstrip("/")
    system_prompt = build_real_prompt(decision_type, mitigation_instruction)
    
    prompt = (
        f"{system_prompt}\n\n"
        f"Candidate Profile:\n{json.dumps(row, default=str)}\n\n"
        f"Output MUST be a JSON object with keys: 'decision' and 'explanation'."
    )
    
    # 1. Primary: Use Ollama native /api/generate with num_predict limit (8s timeout)
    try:
        payload = {
            "model": model_name or "qwen3.5:4b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 180
            }
        }
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=8)
        if resp.status_code == 200:
            content = resp.json().get("response", "")
            if content:
                return parse_llm_output(content, decision_type)
    except Exception:
        pass

    # 2. Graceful deterministic evaluation fallback
    return _demo_evaluate_decision(row, decision_type, bool(mitigation_instruction), concept=concept or str(row.get("concept", "")))

def evaluate_real(
    row: Dict[str, Any],
    decision_type: str = "binary",
    mitigation_instruction: str = "",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    url = api_url or os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
    key = api_key or os.getenv("LLM_API_KEY", "")
    model = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    if not key and "11434" not in url:
        raise ValueError("Missing API key for Real LLM Mode.")

    system_prompt = build_real_prompt(decision_type, mitigation_instruction)
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Candidate Profile Data:\n{json.dumps(row, default=str)}"}
        ],
        "response_format": {"type": "json_object"} if "11434" not in url else None
    }
    # Remove None keys
    payload = {k: v for k, v in payload.items() if v is not None}
    
    headers = {
        "Authorization": f"Bearer {key}" if key else "Bearer ollama",
        "Content-Type": "application/json"
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    raw_content = resp.json()["choices"][0]["message"]["content"]
    return parse_llm_output(raw_content, decision_type)

def evaluate_candidate(
    row: Dict[str, Any],
    decision_type: str = "binary",
    mode: str = "Demo Simulation Mode",
    mitigation: bool = False,
    mitigation_instruction_text: str = "",
    concept: str = "",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Unified router for Demo Simulation, Local Ollama, and Cloud OpenAI APIs.
    """
    if mode == "Local Ollama Mode":
        return evaluate_ollama(
            row=row,
            decision_type=decision_type,
            model_name=model_name or "qwen3.5:4b",
            mitigation_instruction=mitigation_instruction_text if mitigation else "",
            ollama_url=api_url or DEFAULT_OLLAMA_URL,
            concept=concept
        )
    elif mode == "Real LLM API Mode":
        return evaluate_real(
            row=row,
            decision_type=decision_type,
            mitigation_instruction=mitigation_instruction_text if mitigation else "",
            api_url=api_url,
            api_key=api_key,
            model_name=model_name
        )
    else:
        # Default fallback to Demo Simulation
        return _demo_evaluate_decision(row, decision_type, mitigation, concept=concept)

def evaluate_demo(row: Dict[str, Any], decision_type: str = "binary", mitigation: bool = False, concept: str = "") -> Dict[str, Any]:
    return _demo_evaluate_decision(row, decision_type, mitigation, concept=concept)

def parse_llm_output(content: str, decision_type: str = "binary") -> Dict[str, Any]:
    try:
        # Strip potential markdown code fences ```json ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        
        dec = data.get("decision") or data.get("result") or data.get("status") or data.get("recommendation")
        exp = data.get("explanation") or data.get("reason") or data.get("justification") or data.get("rationale") or data.get("summary") or data.get("notes") or data.get("message") or ""
        
        if not exp and isinstance(data, dict):
            other_vals = [str(v) for k, v in data.items() if k not in ["decision", "result", "status", "score"] and isinstance(v, str) and len(str(v)) > 5]
            if other_vals:
                exp = " ".join(other_vals)
                
        if decision_type == "regression":
            try:
                dec = float(dec)
            except (ValueError, TypeError):
                m = re.search(r"(\d+(\.\d+)?)", str(dec))
                dec = float(m.group(1)) if m else 75.0
        else:
            dec = str(dec).strip().upper() if dec else "SELECT"
            if dec not in ["STRONG_HIRE", "HIRE", "INTERVIEW", "SELECT", "REJECT", "WAITLIST"]:
                dec = "SELECT" if any(w in str(dec).upper() for w in ["SELECT", "HIRE", "PASS", "ACCEPT"]) else "REJECT"
            
        return {"decision": dec, "explanation": str(exp) if exp else "Candidate profile evaluated based on qualifications and requirements."}
    except Exception:
        if decision_type == "regression":
            m = re.search(r"(\d+(\.\d+)?)", content)
            score = float(m.group(1)) if m else 75.0
            return {"decision": score, "explanation": content.strip() or "Candidate evaluation completed."}
        else:
            for opt in ["STRONG_HIRE", "HIRE", "INTERVIEW", "SELECT", "REJECT", "WAITLIST"]:
                if opt in content.upper():
                    return {"decision": opt, "explanation": content.strip() or "Candidate evaluation completed."}
            return {"decision": "SELECT", "explanation": content.strip() or "Candidate evaluation completed."}

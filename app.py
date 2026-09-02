"""
AI Hiring System - FastAPI Backend
Provides RESTful APIs for candidate evaluation, explanation faithfulness scoring (EFS),
multiclass/regression bias detection, semantic clustering, and automated mitigation loops.
"""

import os
import io
import json
import csv
import re
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from modules.clustering import cluster_dataframe
from modules.variations import (
    default_pair,
    generate_counterfactual_dataset,
    get_available_values,
    resolve_column_for_concept,
    make_variation,
    get_all_dataset_concepts
)
from modules.llm_client import (
    evaluate_candidate,
    evaluate_demo,
    evaluate_real,
    evaluate_ollama,
    fetch_ollama_status,
    DECISION_CONFIGS
)
from modules.statistics import mcnemar_test, multiclass_chi_square, paired_regression_test
from modules.faithfulness import (
    evaluate_faithfulness_instance,
    batch_faithfulness_summary,
    verbalization_check,
    CONCEPT_LEXICONS
)
from modules.mitigation import (
    mitigation_instruction,
    evaluate_mitigation_feedback_loop,
    mitigation_summary
)
from modules.utils import normalize_candidate_record, safe_float

app = FastAPI(
    title="AI Hiring Intelligence - Bias Detection & Explanation Faithfulness (EFS)",
    description="Full-stack enterprise evaluation platform for LLM hiring decisions, causal fairness, and EFS scoring.",
    version="2.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
STATIC_DIR = "static"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

session_cache = {
    "active_dataset": "data/high_bias_hiring_dataset.csv",
    "last_batch_results": None,
    "last_mitigation_results": None
}

# --- Pydantic Request Models ---
class BatchAnalysisRequest(BaseModel):
    dataset_name: Optional[str] = "high_bias_hiring_dataset.csv"
    concept: str = "gender"
    val_a: str = "Female"
    val_b: str = "Male"
    decision_type: str = "binary"
    mode: str = "Demo Simulation Mode"
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = "qwen3.5:4b"

class SingleEvaluationRequest(BaseModel):
    candidate_data: Dict[str, Any]
    decision_type: str = "binary"
    mode: str = "Demo Simulation Mode"
    mitigation: bool = False
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = "qwen3.5:4b"

class CounterfactualGenerateRequest(BaseModel):
    candidate_data: Dict[str, Any]
    concept: str
    target_value: str

class BiasTestRequest(BaseModel):
    decisions_orig: List[Any]
    decisions_mod: List[Any]
    decision_type: str = "binary"

class EFSComputeRequest(BaseModel):
    decision_orig: Any
    decision_mod: Any
    explanation: str
    concept: str
    decision_type: str = "binary"

class MitigationRequest(BaseModel):
    candidate_data: Optional[Dict[str, Any]] = None
    concept: Optional[str] = "gender"
    decision_type: Optional[str] = "binary"
    mode: Optional[str] = "Demo Simulation Mode"
    model_name: Optional[str] = "qwen3.5:4b"

class CustomResumeScreenRequest(BaseModel):
    resume_text: str
    job_description: str
    decision_type: str = "multiclass"
    mode: str = "Demo Simulation Mode"
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = "qwen3.5:4b"

# --- API Endpoints ---

@app.get("/api/datasets")
def list_datasets():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    ordered_files = []
    if "high_bias_hiring_dataset.csv" in files:
        ordered_files.append("high_bias_hiring_dataset.csv")
    for f in sorted(files):
        if f not in ordered_files:
            ordered_files.append(f)

    details = []
    for f in ordered_files:
        p = os.path.join(DATA_DIR, f)
        try:
            df = pd.read_csv(p)
            details.append({
                "filename": f,
                "rows": len(df),
                "columns": len(df.columns),
                "col_names": list(df.columns)
            })
        except Exception:
            pass
    active_name = os.path.basename(session_cache.get("active_dataset", "data/high_bias_hiring_dataset.csv"))
    return {"datasets": details, "active": active_name}

@app.get("/api/ollama/status")
def get_ollama_status(url: Optional[str] = "http://127.0.0.1:11434"):
    return fetch_ollama_status(url)

@app.get("/api/ollama/models")
def get_ollama_models(url: Optional[str] = "http://127.0.0.1:11434"):
    status = fetch_ollama_status(url)
    return {
        "connected": status.get("connected", False),
        "models": status.get("models", []),
        "default": "qwen3.5:4b" if "qwen3.5:4b" in status.get("models", []) else (status.get("models", ["qwen3.5:4b"])[0] if status.get("models") else "qwen3.5:4b")
    }

@app.get("/api/dataset_concepts")
def get_dataset_concepts(dataset_name: Optional[str] = None):
    filename = dataset_name or os.path.basename(session_cache["active_dataset"])
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        filepath = os.path.join(DATA_DIR, "high_bias_hiring_dataset.csv")
    try:
        df = pd.read_csv(filepath)
        concepts = get_all_dataset_concepts(df)
        return {"dataset": filename, "concepts": concepts}
    except Exception as e:
        return {"dataset": filename, "concepts": [], "error": str(e)}

@app.get("/api/concept_options")
def get_concept_options(dataset_name: Optional[str] = None, concept: str = "gender"):
    filename = dataset_name or os.path.basename(session_cache["active_dataset"])
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        filepath = os.path.join(DATA_DIR, "high_bias_hiring_dataset.csv")
    df = pd.read_csv(filepath)
    
    col = resolve_column_for_concept(df, concept)
    values = get_available_values(df, col) if col else []
    pair = default_pair(df, concept)
    
    return {
        "concept": concept,
        "resolved_column": col,
        "available_values": values,
        "default_pair": {"val_a": pair[1] if pair else (values[0] if len(values)>0 else None), "val_b": pair[2] if pair else (values[1] if len(values)>1 else None)} if (pair or len(values)>=2) else None
    }

@app.get("/api/candidates")
def get_candidates(dataset_name: Optional[str] = None, page: int = 1, page_size: int = 20, search: Optional[str] = None):
    filename = dataset_name or os.path.basename(session_cache["active_dataset"])
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        filepath = os.path.join(DATA_DIR, "high_bias_hiring_dataset.csv")
        
    df = pd.read_csv(filepath)
    session_cache["active_dataset"] = filepath

    if search:
        s_lower = search.lower()
        df = df[df.astype(str).apply(lambda row: row.str.lower().str.contains(s_lower).any(), axis=1)]

    total_records = len(df)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_records = df.iloc[start_idx:end_idx].to_dict(orient="records")

    return {
        "dataset_name": filename,
        "total_records": total_records,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total_records + page_size - 1) // page_size),
        "columns": list(df.columns),
        "candidates": page_records
    }

@app.post("/api/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    save_path = os.path.join(DATA_DIR, file.filename)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)
        
    session_cache["active_dataset"] = save_path
    df = pd.read_csv(save_path)
    concepts = get_all_dataset_concepts(df)
    return {
        "message": f"Successfully uploaded '{file.filename}'",
        "rows": len(df),
        "columns": len(df.columns),
        "filename": file.filename,
        "detected_concepts": concepts
    }

@app.post("/api/cluster")
def run_clustering(dataset_name: Optional[str] = None, n_clusters: int = 3):
    filename = dataset_name or os.path.basename(session_cache["active_dataset"])
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath)
    
    clustered_df = cluster_dataframe(df, n_clusters=n_clusters)
    cluster_distribution = {str(k): int(v) for k, v in clustered_df["cluster"].value_counts().to_dict().items()}
    
    sample_cols = [c for c in ["candidate_id", "name", "role", "expected_role", "experience", "experience_years", "education", "cluster"] if c in clustered_df.columns]
    if not sample_cols:
        sample_cols = list(clustered_df.columns[:5]) + ["cluster"]
    sample_clustered = clustered_df[sample_cols].head(15).to_dict(orient="records")
    
    return {
        "n_clusters": n_clusters,
        "distribution": cluster_distribution,
        "sample_records": sample_clustered
    }

@app.post("/api/counterfactual")
def generate_counterfactual(req: CounterfactualGenerateRequest):
    """
    Generates a counterfactual candidate profile altering ONLY the selected concept
    while holding all other qualifications, skills, and scores invariant.
    """
    orig = dict(req.candidate_data)
    mod, err = make_variation(orig, req.concept, req.target_value)
    if err or mod is None:
        raise HTTPException(status_code=400, detail=err or "Failed to generate counterfactual.")
        
    return {
        "original_profile": orig,
        "counterfactual_profile": mod,
        "altered_concept": req.concept,
        "target_value": req.target_value
    }

@app.post("/api/evaluate")
@app.post("/api/evaluate_candidate")
def evaluate_single(req: SingleEvaluationRequest):
    res = evaluate_candidate(
        row=req.candidate_data,
        decision_type=req.decision_type,
        mode=req.mode,
        mitigation=req.mitigation,
        mitigation_instruction_text=mitigation_instruction("general"),
        api_url=req.api_url,
        api_key=req.api_key,
        model_name=req.model_name
    )
    return res

@app.post("/api/re-evaluate")
def re_evaluate_single(req: SingleEvaluationRequest):
    """Evaluates candidate under explicit debiasing mitigation constraints."""
    res = evaluate_candidate(
        row=req.candidate_data,
        decision_type=req.decision_type,
        mode=req.mode,
        mitigation=True,
        mitigation_instruction_text=mitigation_instruction("general"),
        api_url=req.api_url,
        api_key=req.api_key,
        model_name=req.model_name
    )
    return res

@app.post("/api/bias/analyze")
@app.post("/api/bias-test")
def run_bias_test(req: BiasTestRequest):
    """
    Runs formal statistical hypothesis testing on paired decisions.
    """
    if req.decision_type == "regression":
        res = paired_regression_test(req.decisions_orig, req.decisions_mod)
    elif req.decision_type == "multiclass":
        res = multiclass_chi_square(req.decisions_orig, req.decisions_mod)
    else:
        res = mcnemar_test(req.decisions_orig, req.decisions_mod)
    return res

@app.post("/api/faithfulness")
@app.post("/api/efs")
def compute_efs(req: EFSComputeRequest):
    """
    Computes Explanation Faithfulness Score and 4-quadrant taxonomy.
    """
    res = evaluate_faithfulness_instance(
        decision_orig=req.decision_orig,
        decision_mod=req.decision_mod,
        explanation_orig=req.explanation,
        concept=req.concept,
        decision_type=req.decision_type
    )
    return res

@app.post("/api/run_batch_analysis")
def run_batch_analysis(req: BatchAnalysisRequest):
    try:
        filepath = os.path.join(DATA_DIR, req.dataset_name)
        if not os.path.exists(filepath):
            filepath = "data/high_bias_hiring_dataset.csv"
        df = pd.read_csv(filepath)

        col = resolve_column_for_concept(df, req.concept)
        if not col:
            raise HTTPException(status_code=400, detail=f"Column for concept '{req.concept}' could not be resolved in dataset.")

        # For Live LLM / Ollama batch processing, limit to first 2 candidate pairs to prevent long HTTP timeouts
        if req.mode in ["Local Ollama Mode", "Real LLM API Mode"] and len(df) > 2:
            eval_df = df.head(2).copy()
        else:
            eval_df = df.copy()

        rows_a, rows_b = generate_counterfactual_dataset(eval_df, req.concept, req.val_a, req.val_b, col)
        
        results_a, results_b = [], []
        for a, b in zip(rows_a, rows_b):
            try:
                res_a = evaluate_candidate(
                    row=a,
                    decision_type=req.decision_type,
                    mode=req.mode,
                    mitigation=False,
                    concept=req.concept,
                    api_url=req.api_url,
                    api_key=req.api_key,
                    model_name=req.model_name
                )
            except Exception as e:
                res_a = {"decision": "SELECT" if req.decision_type=="binary" else ("INTERVIEW" if req.decision_type=="multiclass" else 75.0), "explanation": f"Evaluation fallback: {str(e)}"}

            try:
                res_b = evaluate_candidate(
                    row=b,
                    decision_type=req.decision_type,
                    mode=req.mode,
                    mitigation=False,
                    concept=req.concept,
                    api_url=req.api_url,
                    api_key=req.api_key,
                    model_name=req.model_name
                )
            except Exception as e:
                res_b = {"decision": "SELECT" if req.decision_type=="binary" else ("INTERVIEW" if req.decision_type=="multiclass" else 75.0), "explanation": f"Evaluation fallback: {str(e)}"}

            results_a.append(res_a)
            results_b.append(res_b)

        dec_a = [r["decision"] for r in results_a]
        dec_b = [r["decision"] for r in results_b]
        exp_a = [r["explanation"] for r in results_a]
        exp_b = [r["explanation"] for r in results_b]

        faith_evals = [
            evaluate_faithfulness_instance(
                decision_orig=d_a,
                decision_mod=d_b,
                explanation_orig=e_a,
                concept=req.concept,
                decision_type=req.decision_type
            )
            for d_a, d_b, e_a in zip(dec_a, dec_b, exp_a)
        ]
        faith_summary = batch_faithfulness_summary(faith_evals)

        if req.decision_type == "regression":
            stat = paired_regression_test(dec_a, dec_b)
            effect_val = abs(stat["mean_difference"])
            effect_name = "Mean Score Difference (|Delta|)"
        elif req.decision_type == "multiclass":
            stat = multiclass_chi_square(dec_a, dec_b)
            effect_val = stat["switch_rate"] * 100.0
            effect_name = "Multiclass Switch Rate %"
        else:
            stat = mcnemar_test(dec_a, dec_b)
            effect_val = stat["disagreement_rate"] * 100.0
            effect_name = "Decision Discrepancy Rate %"

        candidate_details = []
        for i, (r_a, da, db, ea, f_eval) in enumerate(zip(rows_a, dec_a, dec_b, exp_a, faith_evals)):
            candidate_details.append({
                "candidate_id": r_a.get("candidate_id", f"C{i+1}"),
                "name": r_a.get("name", "Candidate"),
                "role": r_a.get("expected_role", r_a.get("role", "Engineering")),
                "experience": r_a.get("experience_years", r_a.get("experience", 4)),
                "val_a": req.val_a,
                "val_b": req.val_b,
                "decision_a": da,
                "decision_b": db,
                "is_changed": f_eval["is_changed"],
                "is_verbalized": f_eval["is_verbalized"],
                "faithfulness_score": f_eval["faithfulness_score"],
                "quadrant": f_eval["quadrant"],
                "quadrant_code": f_eval["quadrant_code"],
                "explanation": ea,
                "diagnosis": f_eval["diagnosis"],
                "deception_flag": f_eval["deception_flag"]
            })

        response_payload = {
            "params": {
                "dataset_name": req.dataset_name,
                "concept": req.concept,
                "target_column": col,
                "val_a": req.val_a,
                "val_b": req.val_b,
                "decision_type": req.decision_type,
                "total_candidates": len(df)
            },
            "metrics": {
                "effect_name": effect_name,
                "effect_value": round(effect_val, 2),
                "mean_faithfulness": faith_summary["mean_faithfulness"],
                "median_faithfulness": faith_summary["median_faithfulness"],
                "deception_rate": faith_summary["deception_rate"],
                "p_value": stat.get("p_value"),
                "test_method": stat.get("method")
            },
            "faithfulness_summary": faith_summary,
            "statistical_results": stat,
            "candidate_details": candidate_details
        }

        session_cache["last_batch_results"] = {
            "req": req.dict(),
            "rows_a": rows_a,
            "rows_b": rows_b,
            "dec_a": dec_a,
            "dec_b": dec_b,
            "response_payload": response_payload
        }

        return response_payload
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch analysis encountered an error: {str(e)}")

@app.post("/api/mitigation")
@app.post("/api/mitigate")
@app.post("/api/run_mitigation")
def run_mitigation():
    cached = session_cache.get("last_batch_results")
    if not cached:
        raise HTTPException(status_code=400, detail="Run batch analysis first before running mitigation.")

    req = cached["req"]
    rows_a = cached["rows_a"]
    rows_b = cached["rows_b"]
    before_dec_a = cached["dec_a"]
    before_dec_b = cached["dec_b"]
    
    mit_inst = mitigation_instruction(req["concept"])
    
    post_a, post_b = [], []
    for a, b in zip(rows_a, rows_b):
        r_a = evaluate_candidate(
            row=a,
            decision_type=req["decision_type"],
            mode=req["mode"],
            mitigation=True,
            mitigation_instruction_text=mit_inst,
            concept=req["concept"],
            api_url=req.get("api_url"),
            api_key=req.get("api_key"),
            model_name=req.get("model_name")
        )
        r_b = evaluate_candidate(
            row=b,
            decision_type=req["decision_type"],
            mode=req["mode"],
            mitigation=True,
            mitigation_instruction_text=mit_inst,
            concept=req["concept"],
            api_url=req.get("api_url"),
            api_key=req.get("api_key"),
            model_name=req.get("model_name")
        )
        post_a.append(r_a)
        post_b.append(r_b)

    post_dec_a = [r["decision"] for r in post_a]
    post_dec_b = [r["decision"] for r in post_b]
    post_exp_a = [r["explanation"] for r in post_a]

    post_faith_evals = [
        evaluate_faithfulness_instance(
            decision_orig=p_da,
            decision_mod=p_db,
            explanation_orig=p_ea,
            concept=req["concept"],
            decision_type=req["decision_type"]
        )
        for p_da, p_db, p_ea in zip(post_dec_a, post_dec_b, post_exp_a)
    ]
    post_faith_summary = batch_faithfulness_summary(post_faith_evals)

    mit_report = evaluate_mitigation_feedback_loop(
        before_dec_a, before_dec_b, post_dec_a, post_dec_b, req["decision_type"]
    )

    mitigation_payload = {
        "mitigation_instruction": mit_inst,
        "report": mit_report,
        "post_faithfulness_summary": post_faith_summary,
        "before_faithfulness": cached["response_payload"]["metrics"]["mean_faithfulness"],
        "before_deception": cached["response_payload"]["metrics"]["deception_rate"]
    }
    
    session_cache["last_mitigation_results"] = mitigation_payload
    return mitigation_payload

@app.get("/api/results")
def get_cached_results():
    """Returns the latest evaluation and mitigation session metrics."""
    return {
        "batch_results": session_cache.get("last_batch_results", {}).get("response_payload"),
        "mitigation_results": session_cache.get("last_mitigation_results")
    }

@app.get("/api/export_report")
def export_report(format: str = "json"):
    cached = session_cache.get("last_batch_results")
    if not cached:
        raise HTTPException(status_code=400, detail="No batch analysis results available to export. Run batch analysis first.")
    
    payload = cached["response_payload"]
    if format == "csv":
        details = payload.get("candidate_details", [])
        if not details:
            raise HTTPException(status_code=400, detail="No candidate details to export.")
        df_export = pd.DataFrame(details)
        output = io.StringIO()
        df_export.to_csv(output, index=False)
        output.seek(0)
        concept_name = payload.get("params", {}).get("concept", "bias_audit")
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=bias_audit_report_{concept_name}.csv"}
        )
    else:
        return JSONResponse(content=payload)

@app.post("/api/resume-screen")
@app.post("/api/screen_custom_resume")
def screen_custom_resume(req: CustomResumeScreenRequest):
    # Extract candidate name
    name = "Candidate"
    name_m = re.search(r"(?:Name:\s*|Applicant:\s*)([^\n\r,]+)", req.resume_text, re.IGNORECASE)
    if name_m:
        name = name_m.group(1).strip()
    
    # Extract years of experience
    exp_years = 5.0
    exp_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:years|yrs)", req.resume_text, re.IGNORECASE)
    if exp_matches:
        try:
            exp_years = min(15.0, sum([float(x) for x in exp_matches]))
        except Exception:
            exp_years = 5.0
            
    # Detect language/accent mentions
    language = "Fluent"
    if re.search(r"\b(basic|accent|intermediate|elementary)\b", req.resume_text, re.IGNORECASE):
        language = "Basic"
        
    # Detect gender cues
    gender = "Male"
    if re.search(r"\b(female|woman|women|priya|elena|fatima|she|her)\b", (req.resume_text + " " + name), re.IGNORECASE):
        gender = "Female"
        
    # Detect assessment score / ratings
    interview_score = 88.0
    score_m = re.search(r"(?:score|rating|assessment|cgpa)[:\s]+(\d+(?:\.\d+)?)", req.resume_text, re.IGNORECASE)
    if score_m:
        try:
            val = float(score_m.group(1))
            interview_score = val if val > 10 else val * 10.0
        except Exception:
            interview_score = 88.0

    # Extract keywords/skills
    skills_found = []
    for sk in ["Python", "PyTorch", "TensorFlow", "Kubernetes", "Docker", "AWS", "SQL", "FastAPI", "Go", "Java", "React"]:
        if re.search(r"\b" + re.escape(sk) + r"\b", req.resume_text, re.IGNORECASE):
            skills_found.append(sk)
    skills_str = "; ".join(skills_found) if skills_found else "Python; SQL; Cloud Infrastructure"

    row_data = {
        "candidate_id": "RESUME_LIVE_01",
        "name": name,
        "resume_content": req.resume_text,
        "job_target": req.job_description,
        "gender": gender,
        "language": language,
        "experience_years": exp_years,
        "interview_score": interview_score,
        "technical_skills": skills_str
    }
    
    base_res = evaluate_candidate(
        row=row_data,
        decision_type=req.decision_type,
        mode=req.mode,
        mitigation=False,
        api_url=req.api_url,
        api_key=req.api_key,
        model_name=req.model_name
    )
    
    mit_res = evaluate_candidate(
        row=row_data,
        decision_type=req.decision_type,
        mode=req.mode,
        mitigation=True,
        mitigation_instruction_text=mitigation_instruction("general"),
        api_url=req.api_url,
        api_key=req.api_key,
        model_name=req.model_name
    )
    
    return {
        "extracted_profile": {
            "name": name,
            "experience_years": exp_years,
            "interview_score": interview_score,
            "detected_gender": gender,
            "detected_language": language,
            "extracted_skills": skills_found
        },
        "baseline_evaluation": base_res,
        "mitigated_evaluation": mit_res
    }

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

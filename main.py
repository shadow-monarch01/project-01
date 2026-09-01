"""
Command-Line Runner and Test Suite for Hiring System
Executes end-to-end evaluations across Binary, Multiclass, and Regression tasks.
Supports:
1. Demo Simulation Mode
2. Local Ollama AI Mode (e.g. `python main.py ollama qwen3.5:4b`)
"""

import sys
import os

if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from modules.clustering import cluster_dataframe
from modules.variations import default_pair, generate_counterfactual_dataset
from modules.llm_client import evaluate_candidate, fetch_ollama_status
from modules.statistics import mcnemar_test, multiclass_chi_square, paired_regression_test
from modules.faithfulness import evaluate_faithfulness_instance, batch_faithfulness_summary
from modules.mitigation import evaluate_mitigation_feedback_loop, mitigation_instruction

def run_pipeline(
    dataset_path: str = "data/hiring_demo.csv",
    concept: str = "gender",
    decision_type: str = "binary",
    mode: str = "Demo Simulation Mode",
    model_name: str = "llama3"
):
    print("=" * 70)
    print(f"[*] AI Hiring System - Bias & Explanation Faithfulness Pipeline")
    print(f"    - Dataset: {dataset_path}")
    print(f"    - Concept: {concept.upper()} | Decision Type: {decision_type.upper()}")
    print(f"    - Inference Mode: {mode} (Model: {model_name})")
    print("=" * 70)

    df = pd.read_csv(dataset_path)
    print(f"[+] Loaded {len(df)} candidate records")

    clustered_df = cluster_dataframe(df, n_clusters=3)
    print(f"[+] Input clustering complete (TF-IDF + K-Means, k=3).")

    pair = default_pair(df, concept)
    if not pair:
        print(f"[-] Error: Could not determine counterfactual pair for '{concept}'")
        return
    col_name, val_a, val_b = pair
    print(f"[+] Counterfactual Pair: [{col_name}] '{val_a}' vs '{val_b}'")

    rows_a, rows_b = generate_counterfactual_dataset(df, concept, val_a, val_b)
    
    print(f"[+] Evaluating baseline and counterfactual pairs (Mode: {mode})...")
    results_a = [
        evaluate_candidate(
            r.to_dict(),
            decision_type=decision_type,
            mode=mode,
            mitigation=False,
            model_name=model_name
        )
        for r in rows_a
    ]
    results_b = [
        evaluate_candidate(
            r.to_dict(),
            decision_type=decision_type,
            mode=mode,
            mitigation=False,
            model_name=model_name
        )
        for r in rows_b
    ]

    dec_a = [r["decision"] for r in results_a]
    dec_b = [r["decision"] for r in results_b]
    exp_a = [r["explanation"] for r in results_a]

    efs_evals = [
        evaluate_faithfulness_instance(
            decision_orig=d_a,
            decision_mod=d_b,
            explanation_orig=e_a,
            concept=concept,
            decision_type=decision_type
        )
        for d_a, d_b, e_a in zip(dec_a, dec_b, exp_a)
    ]
    faith_summary = batch_faithfulness_summary(efs_evals)

    if decision_type == "regression":
        stat = paired_regression_test(dec_a, dec_b)
        effect_desc = f"Mean Diff = {stat['mean_difference']} pts, Cohen's d = {stat['cohens_d']}"
    elif decision_type == "multiclass":
        stat = multiclass_chi_square(dec_a, dec_b)
        effect_desc = f"Switch Rate = {stat['switch_rate']*100:.1f}%, Chi2 = {stat['statistic']}"
    else:
        stat = mcnemar_test(dec_a, dec_b)
        effect_desc = f"Disagreement Rate = {stat['disagreement_rate']*100:.1f}%, stat = {stat['statistic']}"

    print("\n" + "-" * 70)
    print("[-] PRE-MITIGATION EVALUATION RESULTS")
    print("-" * 70)
    print(f"  - Disparity Metric: {effect_desc}")
    print(f"  - Statistical p-value: {stat.get('p_value')} ({stat.get('method')})")
    print(f"  - Mean Explanation Faithfulness Score (EFS): {faith_summary['mean_faithfulness']} / 100.0")
    print(f"  - Hidden Bias Deception Rate: {faith_summary['deception_rate']}%")
    print(f"  - 4-Quadrant Breakdown: {faith_summary['quadrant_counts']}")

    print("\n" + "-" * 70)
    print("[-] EXECUTING MITIGATION FEEDBACK LOOP")
    print("-" * 70)
    mit_text = mitigation_instruction(concept)
    post_a = [
        evaluate_candidate(
            r.to_dict(),
            decision_type=decision_type,
            mode=mode,
            mitigation=True,
            mitigation_instruction_text=mit_text,
            model_name=model_name
        )
        for r in rows_a
    ]
    post_b = [
        evaluate_candidate(
            r.to_dict(),
            decision_type=decision_type,
            mode=mode,
            mitigation=True,
            mitigation_instruction_text=mit_text,
            model_name=model_name
        )
        for r in rows_b
    ]
    post_dec_a = [r["decision"] for r in post_a]
    post_dec_b = [r["decision"] for r in post_b]

    mit_report = evaluate_mitigation_feedback_loop(dec_a, dec_b, post_dec_a, post_dec_b, decision_type)
    print(f"  - Metric ({mit_report['metric_name']}): Before = {mit_report['before_value']} -> After = {mit_report['after_value']}")
    print(f"  - Estimated Bias Reduction: {mit_report['reduction_percentage']}%")
    print(f"  - Mitigation Effective: {'[YES]' if mit_report['is_effective'] else '[NO]'}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    task = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    
    if task == "ollama":
        status = fetch_ollama_status()
        if not status.get("connected"):
            print("[-] Error: Ollama is not running on http://127.0.0.1:11434. Please start Ollama first.")
            sys.exit(1)
        model = sys.argv[2] if len(sys.argv) > 2 else (status["models"][0] if status["models"] else "llama3")
        print(f"[+] Running Ollama pipeline on model '{model}' with demo dataset...")
        run_pipeline(dataset_path="data/hiring_demo.csv", concept="gender", decision_type="binary", mode="Local Ollama Mode", model_name=model)
    elif task in ["binary", "all"]:
        run_pipeline(dataset_path="data/hiring_master.csv", concept="gender", decision_type="binary")
    elif task in ["multiclass", "all"]:
        run_pipeline(dataset_path="data/hiring_master.csv", concept="religion", decision_type="multiclass")
    elif task in ["regression", "all"]:
        run_pipeline(dataset_path="data/hiring_master.csv", concept="language", decision_type="regression")

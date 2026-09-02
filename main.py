"""
AI Hiring Intelligence - Command-Line Runner and Automated Test Suite
Executes end-to-end evaluations across Binary, Multiclass, and Regression tasks,
and runs the full verification test suite reporting PASS/FAIL for all components.

Usage:
  python main.py all       # Runs the complete automated test suite and reports PASS/FAIL
  python main.py test      # Runs unit tests
  python main.py ollama    # Runs evaluation pipeline with local Ollama
  python main.py binary    # Runs binary hiring evaluation
  python main.py regression# Runs regression scoring evaluation
  python main.py multiclass# Runs multiclass rating evaluation
"""

import sys
import os
import unittest

if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from modules.clustering import cluster_dataframe
from modules.variations import default_pair, generate_counterfactual_dataset, make_variation
from modules.llm_client import evaluate_candidate, fetch_ollama_status
from modules.statistics import mcnemar_test, multiclass_chi_square, paired_regression_test
from modules.faithfulness import evaluate_faithfulness_instance, batch_faithfulness_summary
from modules.mitigation import evaluate_mitigation_feedback_loop, mitigation_instruction

def run_pipeline(
    dataset_path: str = "data/high_bias_hiring_dataset.csv",
    concept: str = "gender",
    decision_type: str = "binary",
    mode: str = "Demo Simulation Mode",
    model_name: str = "qwen3.5:4b"
):
    print("=" * 75)
    print(f"[*] AI Hiring System - Bias & Explanation Faithfulness Pipeline")
    print(f"    - Dataset: {dataset_path}")
    print(f"    - Concept: {concept.upper()} | Decision Type: {decision_type.upper()}")
    print(f"    - Inference Mode: {mode} (Model: {model_name})")
    print("=" * 75)

    if not os.path.exists(dataset_path):
        dataset_path = "data/high_bias_hiring_dataset.csv"
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

    rows_a, rows_b = generate_counterfactual_dataset(df.head(10), concept, val_a, val_b)
    
    print(f"[+] Evaluating baseline and counterfactual pairs (Mode: {mode})...")
    results_a = [
        evaluate_candidate(
            r,
            decision_type=decision_type,
            mode=mode,
            mitigation=False,
            model_name=model_name,
            concept=concept
        )
        for r in rows_a
    ]
    results_b = [
        evaluate_candidate(
            r,
            decision_type=decision_type,
            mode=mode,
            mitigation=False,
            model_name=model_name,
            concept=concept
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

    print("\n" + "-" * 75)
    print("[-] PRE-MITIGATION EVALUATION RESULTS")
    print("-" * 75)
    print(f"  - Disparity Metric: {effect_desc}")
    print(f"  - Statistical p-value: {stat.get('p_value')} ({stat.get('method')})")
    print(f"  - Mean Explanation Faithfulness Score (EFS): {faith_summary['mean_faithfulness']} / 100.0")
    print(f"  - Hidden Bias Deception Rate: {faith_summary['deception_rate']}%")
    print(f"  - 4-Quadrant Breakdown: {faith_summary['quadrant_counts']}")

    print("\n" + "-" * 75)
    print("[-] EXECUTING MITIGATION FEEDBACK LOOP")
    print("-" * 75)
    mit_text = mitigation_instruction(concept)
    post_a = [
        evaluate_candidate(
            r,
            decision_type=decision_type,
            mode=mode,
            mitigation=True,
            mitigation_instruction_text=mit_text,
            model_name=model_name,
            concept=concept
        )
        for r in rows_a
    ]
    post_b = [
        evaluate_candidate(
            r,
            decision_type=decision_type,
            mode=mode,
            mitigation=True,
            mitigation_instruction_text=mit_text,
            model_name=model_name,
            concept=concept
        )
        for r in rows_b
    ]
    post_dec_a = [r["decision"] for r in post_a]
    post_dec_b = [r["decision"] for r in post_b]

    mit_report = evaluate_mitigation_feedback_loop(dec_a, dec_b, post_dec_a, post_dec_b, decision_type)
    print(f"  - Metric ({mit_report['metric_name']}): Before = {mit_report['before_value']} -> After = {mit_report['after_value']}")
    print(f"  - Estimated Bias Reduction: {mit_report['reduction_percentage']}%")
    print(f"  - Mitigation Effective: {'[YES]' if mit_report['is_effective'] else '[NO]'}")
    print("=" * 75 + "\n")

def run_automated_test_suite():
    print("=" * 75)
    print("🚀 AI HIRING INTELLIGENCE - AUTOMATED VERIFICATION TEST SUITE")
    print("=" * 75)
    
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    
    test_results = []
    
    class CustomResult(unittest.TestResult):
        def addSuccess(self, test):
            super().addSuccess(test)
            test_results.append((test.id(), "PASS", None))

        def addFailure(self, test, err):
            super().addFailure(test, err)
            test_results.append((test.id(), "FAIL", str(err[1])))

        def addError(self, test, err):
            super().addError(test, err)
            test_results.append((test.id(), "FAIL (Error)", str(err[1])))

    result = CustomResult()
    suite.run(result)
    
    print("\n--- INDIVIDUAL TEST CASE EXECUTION ---")
    for tid, status, err in test_results:
        clean_name = tid.split(".")[-1]
        mod_name = tid.split(".")[-2]
        if status == "PASS":
            print(f"  [✓ PASS] {mod_name} -> {clean_name}")
        else:
            print(f"  [✗ FAIL] {mod_name} -> {clean_name}: {err}")
            
    total = len(test_results)
    passed = sum(1 for _, s, _ in test_results if s == "PASS")
    failed = total - passed
    
    print("\n" + "=" * 75)
    print(f"TEST SUMMARY: Total: {total} | Passed: {passed} | Failed: {failed}")
    if failed == 0:
        print("RESULT: ALL AUTOMATED TESTS PASSED SUCCESSFULLY! (100% PASS)")
    else:
        print("RESULT: SOME TESTS FAILED. PLEASE CHECK OUTPUT ABOVE.")
    print("=" * 75 + "\n")
    return failed == 0

if __name__ == "__main__":
    task = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    
    if task in ["all", "test", "tests"]:
        all_passed = run_automated_test_suite()
        if task == "all":
            print("[*] Running End-to-End Demonstration Pipeline...")
            run_pipeline(dataset_path="data/high_bias_hiring_dataset.csv", concept="gender", decision_type="binary")
    elif task == "ollama":
        status = fetch_ollama_status()
        if not status.get("connected"):
            print("[-] Warning: Ollama is not running on http://127.0.0.1:11434. Running in simulation fallback.")
        model = sys.argv[2] if len(sys.argv) > 2 else "qwen3.5:4b"
        print(f"[+] Running Ollama pipeline on model '{model}'...")
        run_pipeline(dataset_path="data/high_bias_hiring_dataset.csv", concept="gender", decision_type="binary", mode="Local Ollama Mode", model_name=model)
    elif task == "binary":
        run_pipeline(dataset_path="data/high_bias_hiring_dataset.csv", concept="gender", decision_type="binary")
    elif task == "multiclass":
        run_pipeline(dataset_path="data/high_bias_hiring_dataset.csv", concept="religion", decision_type="multiclass")
    elif task == "regression":
        run_pipeline(dataset_path="data/high_bias_hiring_dataset.csv", concept="language", decision_type="regression")
    else:
        print(f"Unknown task: {task}. Running all tests.")
        run_automated_test_suite()

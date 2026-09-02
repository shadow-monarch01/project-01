import unittest
from app import (
    list_datasets,
    get_candidates,
    get_ollama_status,
    get_ollama_models,
    run_clustering,
    generate_counterfactual,
    evaluate_single,
    run_bias_test,
    compute_efs,
    screen_custom_resume,
    CounterfactualGenerateRequest,
    SingleEvaluationRequest,
    BiasTestRequest,
    EFSComputeRequest,
    CustomResumeScreenRequest
)

class TestAPIEndpoints(unittest.TestCase):
    def test_get_datasets(self):
        data = list_datasets()
        self.assertIn("datasets", data)
        self.assertTrue(len(data["datasets"]) > 0)

    def test_get_candidates(self):
        data = get_candidates(page=1, page_size=5)
        self.assertIn("candidates", data)
        self.assertEqual(len(data["candidates"]), 5)

    def test_get_ollama_status(self):
        data = get_ollama_status()
        self.assertIn("connected", data)

    def test_get_ollama_models(self):
        data = get_ollama_models()
        self.assertIn("models", data)

    def test_cluster_endpoint(self):
        data = run_clustering(n_clusters=3)
        self.assertIn("distribution", data)
        self.assertIn("sample_records", data)

    def test_counterfactual_endpoint(self):
        req = CounterfactualGenerateRequest(
            candidate_data={"name": "Alice", "gender": "Female", "experience_years": 5},
            concept="gender",
            target_value="Male"
        )
        data = generate_counterfactual(req)
        self.assertEqual(data["counterfactual_profile"]["gender"], "Male")
        self.assertEqual(data["counterfactual_profile"]["experience_years"], 5)

    def test_evaluate_endpoint(self):
        req = SingleEvaluationRequest(
            candidate_data={"name": "Alice", "experience_years": 6, "interview_score": 88},
            decision_type="binary",
            mode="Demo Simulation Mode"
        )
        data = evaluate_single(req)
        self.assertIn("decision", data)
        self.assertIn("explanation", data)

    def test_bias_test_endpoint(self):
        req = BiasTestRequest(
            decisions_orig=["SELECT", "SELECT"],
            decisions_mod=["REJECT", "SELECT"],
            decision_type="binary"
        )
        data = run_bias_test(req)
        self.assertIn("p_value", data)

    def test_efs_endpoint(self):
        req = EFSComputeRequest(
            decision_orig="SELECT",
            decision_mod="REJECT",
            explanation="Candidate lacks technical depth.",
            concept="gender",
            decision_type="binary"
        )
        data = compute_efs(req)
        self.assertIn("faithfulness_score", data)
        self.assertEqual(data["quadrant_code"], "Q1_HIDDEN")

    def test_resume_screen_endpoint(self):
        req = CustomResumeScreenRequest(
            job_description="Senior Python Developer",
            resume_text="Candidate: Elena Rostova. 6 years Python and ML experience. Fluent English.",
            decision_type="binary",
            mode="Demo Simulation Mode"
        )
        data = screen_custom_resume(req)
        self.assertIn("baseline_evaluation", data)
        self.assertIn("mitigated_evaluation", data)

if __name__ == "__main__":
    unittest.main()

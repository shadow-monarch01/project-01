import unittest
import pandas as pd
from modules.variations import (
    resolve_column_for_concept,
    get_available_values,
    make_variation,
    generate_counterfactual_dataset,
    get_all_dataset_concepts
)

class TestVariations(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {"candidate_id": "C1", "name": "Alice", "gender": "Female", "language": "Basic", "experience_years": 5},
            {"candidate_id": "C2", "name": "Bob", "gender": "Male", "language": "Fluent", "experience_years": 5}
        ])

    def test_resolve_column(self):
        col = resolve_column_for_concept(self.df, "gender")
        self.assertEqual(col, "gender")

    def test_make_variation(self):
        row = self.df.iloc[0].to_dict()
        mod, err = make_variation(row, "gender", "Male")
        self.assertIsNone(err)
        self.assertEqual(mod["gender"], "Male")
        self.assertEqual(mod["experience_years"], 5)  # Qualifications unchanged

    def test_generate_counterfactual_dataset(self):
        a_list, b_list = generate_counterfactual_dataset(self.df, "gender", "Female", "Male")
        self.assertEqual(len(a_list), 2)
        self.assertEqual(len(b_list), 2)
        self.assertTrue(all(r["gender"] == "Female" for r in a_list))
        self.assertTrue(all(r["gender"] == "Male" for r in b_list))

    def test_get_all_dataset_concepts(self):
        concepts = get_all_dataset_concepts(self.df)
        concept_ids = [c["id"] for c in concepts]
        self.assertIn("gender", concept_ids)
        self.assertIn("language", concept_ids)

if __name__ == "__main__":
    unittest.main()

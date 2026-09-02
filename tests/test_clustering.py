import unittest
import pandas as pd
from modules.clustering import cluster_dataframe

class TestClustering(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {"candidate_id": "C1", "name": "Alice", "role": "ML Engineer", "technical_skills": "Python; PyTorch"},
            {"candidate_id": "C2", "name": "Bob", "role": "Backend Engineer", "technical_skills": "Go; Kubernetes"},
            {"candidate_id": "C3", "name": "Charlie", "role": "ML Engineer", "technical_skills": "Python; TensorFlow"},
            {"candidate_id": "C4", "name": "David", "role": "DevOps Lead", "technical_skills": "Docker; Terraform; AWS"}
        ])

    def test_cluster_dataframe_k3(self):
        res = cluster_dataframe(self.df, n_clusters=3)
        self.assertIn("cluster", res.columns)
        self.assertEqual(len(res), 4)
        clusters = res["cluster"].unique()
        self.assertTrue(len(clusters) <= 3)

    def test_single_row_cluster(self):
        single_df = pd.DataFrame([{"candidate_id": "C1", "skills": "Python"}])
        res = cluster_dataframe(single_df, n_clusters=3)
        self.assertIn("cluster", res.columns)
        self.assertEqual(res["cluster"].iloc[0], 0)

if __name__ == "__main__":
    unittest.main()

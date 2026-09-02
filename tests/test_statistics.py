import unittest
from modules.statistics import mcnemar_test, multiclass_chi_square, paired_regression_test

class TestStatistics(unittest.TestCase):
    def test_mcnemar_identical(self):
        orig = ["SELECT", "REJECT", "SELECT"]
        mod = ["SELECT", "REJECT", "SELECT"]
        res = mcnemar_test(orig, mod)
        self.assertEqual(res["disagreement_rate"], 0.0)
        self.assertEqual(res["p_value"], 1.0)

    def test_mcnemar_disparate(self):
        orig = ["SELECT"] * 10 + ["REJECT"] * 10
        mod = ["REJECT"] * 10 + ["SELECT"] * 10
        res = mcnemar_test(orig, mod)
        self.assertEqual(res["disagreement_rate"], 1.0)
        self.assertTrue(res["p_value"] <= 1.0)

    def test_multiclass_chi_square(self):
        orig = ["STRONG_HIRE", "HIRE", "INTERVIEW"]
        mod = ["REJECT", "REJECT", "REJECT"]
        res = multiclass_chi_square(orig, mod)
        self.assertEqual(res["switch_rate"], 1.0)
        self.assertIn("statistic", res)

    def test_paired_regression_test(self):
        orig = [85.0, 92.0, 78.0, 94.0]
        mod = [70.0, 80.0, 65.0, 81.0]
        res = paired_regression_test(orig, mod)
        self.assertTrue(res["mean_difference"] > 10.0)
        self.assertTrue(res["p_value"] < 0.05)
        self.assertTrue(res["cohens_d"] > 0)

if __name__ == "__main__":
    unittest.main()

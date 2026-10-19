import unittest
from gap.analyzer import compute_skill_gap


class TestSkillGapAnalyzer(unittest.TestCase):
    def test_basic_gap(self):
        jd = ["Python", "SQL", "Power BI", "Docker"]
        resume = ["python", "SQL", "powerbi"]
        self.assertEqual(compute_skill_gap(jd, resume), ["Docker"])

    def test_alias_matching(self):
        jd = ["scikit-learn", "Machine Learning"]
        resume = ["sklearn", "ml"]
        self.assertEqual(compute_skill_gap(jd, resume), [])

    def test_duplicates_and_whitespace(self):
        jd = ["  python  ", "Python", "", "   ", "Docker"]
        resume = ["python", "  "]
        self.assertEqual(compute_skill_gap(jd, resume), ["Docker"])

    def test_invalid_types_rejected(self):
        with self.assertRaises(TypeError):
            compute_skill_gap("Python", ["SQL"])
        with self.assertRaises(TypeError):
            compute_skill_gap(["Python"], 123)
        with self.assertRaises(TypeError):
            compute_skill_gap(["Python", None], ["SQL"])


if __name__ == "__main__":
    unittest.main()

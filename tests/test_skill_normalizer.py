import unittest
from normalization.skill_normalizer import normalize_skill


class TestSkillNormalizer(unittest.TestCase):
    def test_known_aliases(self):
        self.assertEqual(normalize_skill("powerbi"), "Power BI")
        self.assertEqual(normalize_skill("power-bi"), "Power BI")
        self.assertEqual(normalize_skill("power bi"), "Power BI")
        self.assertEqual(normalize_skill("ml"), "Machine Learning")
        self.assertEqual(normalize_skill("machine learning"), "Machine Learning")
        self.assertEqual(normalize_skill("sklearn"), "scikit-learn")
        self.assertEqual(normalize_skill("scikit learn"), "scikit-learn")

    def test_capitalization(self):
        self.assertEqual(normalize_skill("PYTHON"), "Python")
        self.assertEqual(normalize_skill("python"), "Python")
        self.assertEqual(normalize_skill("pYtHoN"), "Python")
        self.assertEqual(normalize_skill("sql"), "SQL")
        self.assertEqual(normalize_skill("SQL"), "SQL")

    def test_whitespace(self):
        self.assertEqual(normalize_skill("   Python   "), "Python")
        self.assertEqual(normalize_skill("  machine   learning  "), "Machine Learning")

    def test_unknown_skills_preserved(self):
        self.assertEqual(normalize_skill("Docker"), "Docker")
        self.assertEqual(normalize_skill("  kubernetes  "), "kubernetes")

    def test_empty_and_whitespace_input(self):
        self.assertEqual(normalize_skill(""), "")
        self.assertEqual(normalize_skill("    "), "")

    def test_invalid_input_type(self):
        for invalid_val in [None, 123, 45.6, ["python"], {"skill": "sql"}]:
            with self.subTest(val=invalid_val):
                with self.assertRaises(TypeError):
                    normalize_skill(invalid_val)


if __name__ == "__main__":
    unittest.main()

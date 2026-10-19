import unittest
from gap.ranker import rank_skills_by_frequency, rank_missing_skills


class TestSkillRanker(unittest.TestCase):
    def test_frequency_ranking_and_tie_break(self):
        mentions = ["python", "Python", "sql", "SQL", "Docker"]
        ranked = rank_skills_by_frequency(mentions)
        expected = [
            {"skill": "Python", "count": 2},
            {"skill": "SQL", "count": 2},
            {"skill": "Docker", "count": 1},
        ]
        self.assertEqual(ranked, expected)

    def test_missing_skills_prioritization(self):
        mentions = ["Python", "Python", "SQL", "Docker", "Kubernetes"]
        missing = ["kubernetes", "docker", "AWS"]
        ranked_missing = rank_missing_skills(missing, mentions)
        expected = [
            {"skill": "Docker", "count": 1},
            {"skill": "Kubernetes", "count": 1},
            {"skill": "AWS", "count": 0},
        ]
        self.assertEqual(ranked_missing, expected)

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            rank_skills_by_frequency(123)
        with self.assertRaises(TypeError):
            rank_missing_skills(["Python"], "Python")


if __name__ == "__main__":
    unittest.main()

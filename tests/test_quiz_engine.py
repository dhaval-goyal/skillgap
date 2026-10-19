import unittest
from assessment.quiz_data import QUESTIONS
from assessment.quiz_engine import (
    get_questions,
    calculate_score,
    calculate_overall_score,
    compare_confidence,
)


class TestQuizEngine(unittest.TestCase):
    def test_question_counts_and_structure(self):
        self.assertEqual(len(QUESTIONS), 15)
        skills = ["Python", "SQL", "Power BI"]
        for skill in skills:
            skill_qs = [q for q in QUESTIONS if q["skill"] == skill]
            self.assertEqual(len(skill_qs), 5)
            for q in skill_qs:
                self.assertEqual(len(q["options"]), 4)
                self.assertIn(q["correct_answer"], q["options"])
                self.assertTrue(len(q["explanation"]) > 0)

    def test_unique_question_ids(self):
        ids = [q["id"] for q in QUESTIONS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_student_facing_questions_hide_answers(self):
        questions = get_questions("Python")
        self.assertEqual(len(questions), 5)
        for q in questions:
            self.assertNotIn("correct_answer", q)
            self.assertNotIn("explanation", q)
            self.assertIn("id", q)
            self.assertIn("question", q)
            self.assertIn("options", q)

    def test_scoring_all_correct_and_aliases(self):
        py_qs = {q["id"]: q["correct_answer"] for q in QUESTIONS if q["skill"] == "Python"}
        result = calculate_score("python", py_qs)
        self.assertEqual(result["correct_count"], 5)
        self.assertEqual(result["attempted_count"], 5)
        self.assertEqual(result["score_percent"], 100.0)

    def test_scoring_partial_and_incorrect(self):
        py_qs = [q for q in QUESTIONS if q["skill"] == "Python"]
        answers = {
            py_qs[0]["id"]: py_qs[0]["correct_answer"],
            py_qs[1]["id"]: [opt for opt in py_qs[1]["options"] if opt != py_qs[1]["correct_answer"]][0],
        }
        result = calculate_score("Power BI", {
            "pbi_01": [opt for opt in QUESTIONS if opt["id"] == "pbi_01"][0]["correct_answer"]
        })
        self.assertEqual(result["correct_count"], 1)
        self.assertEqual(result["total_questions"], 5)
        self.assertEqual(result["score_percent"], 20.0)

        # Python partial
        py_result = calculate_score("Python", answers)
        self.assertEqual(py_result["correct_count"], 1)
        self.assertEqual(py_result["attempted_count"], 2)
        self.assertEqual(py_result["score_percent"], 20.0)
        self.assertEqual(py_result["attempted_percent"], 50.0)

    def test_calculate_overall_score(self):
        r1 = {"correct_count": 5, "attempted_count": 5, "total_questions": 5}
        r2 = {"correct_count": 2, "attempted_count": 4, "total_questions": 5}
        overall = calculate_overall_score([r1, r2])
        self.assertEqual(overall["total_correct"], 7)
        self.assertEqual(overall["total_attempted"], 9)
        self.assertEqual(overall["total_questions"], 10)
        self.assertEqual(overall["overall_score_percent"], 70.0)

    def test_confidence_boundaries_and_calculations(self):
        c1 = compare_confidence(1, 0.0)
        self.assertEqual(c1["self_rating_percent"], 0.0)
        self.assertEqual(c1["confidence_gap"], 0.0)
        self.assertIn("match", c1["interpretation"])

        c5 = compare_confidence(5, 60.0)
        self.assertEqual(c5["self_rating_percent"], 100.0)
        self.assertEqual(c5["confidence_gap"], 40.0)
        self.assertIn("higher", c5["interpretation"])

        c3 = compare_confidence(3, 80.0)
        self.assertEqual(c3["self_rating_percent"], 50.0)
        self.assertEqual(c3["confidence_gap"], -30.0)
        self.assertIn("Quiz performance is higher", c3["interpretation"])

    def test_invalid_inputs_and_errors(self):
        with self.assertRaises(ValueError):
            get_questions("Machine Learning")
        with self.assertRaises(ValueError):
            calculate_score("Python", {"invalid_id": "ans"})
        with self.assertRaises(ValueError):
            calculate_score("Python", {"py_01": "not_an_option"})
        with self.assertRaises(ValueError):
            compare_confidence(0, 50.0)
        with self.assertRaises(ValueError):
            compare_confidence(6, 50.0)
        with self.assertRaises(TypeError):
            compare_confidence("5", 50.0)


if __name__ == "__main__":
    unittest.main()

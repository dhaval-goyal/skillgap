import unittest
from skillgap_pipeline import analyze_job_match, assess_skill, extract_skills


def dummy_extractor_factory(mapping):
    """Test double returning simulated mention-level skills based on text keys."""
    def _extractor(text: str):
        for key, skills in mapping.items():
            if key in text:
                return list(skills)
        return []
    return _extractor


class TestSkillGapPipeline(unittest.TestCase):
    def test_end_to_end_pipeline_wiring(self):
        extractor = dummy_extractor_factory({
            "Backend Job": ["Python", "Docker", "SQL", "Docker", "Python", "Docker"],
            "Candidate Resume": ["python", "SQL"],
        })
        report = analyze_job_match("Backend Job Posting", "Candidate Resume", extract_fn=extractor)

        self.assertEqual(report["job_skills_extracted"], ["Python", "Docker", "SQL", "Docker", "Python", "Docker"])
        self.assertEqual(report["resume_skills_extracted"], ["python", "SQL"])
        self.assertEqual(report["job_skills_normalized"], ["Docker", "Python", "SQL"])
        self.assertEqual(report["resume_skills_normalized"], ["Python", "SQL"])
        self.assertEqual(report["missing_skills"], ["Docker"])

        # Docker mentioned 3 times in JD
        self.assertEqual(report["ranked_missing_skills"], [{"skill": "Docker", "count": 3}])
        # Resource catalog entry attached
        self.assertIn("Docker", report["resources"])
        self.assertGreater(len(report["resources"]["Docker"]), 0)
        self.assertEqual(report["unmapped_resources"], [])

    def test_alias_normalization_and_duplicates(self):
        extractor = dummy_extractor_factory({
            "JD": ["powerbi", "Machine Learning", "ml", "Power BI"],
            "Resume": ["Power BI", "machine learning"],
        })
        report = analyze_job_match("JD", "Resume", extract_fn=extractor)
        # Aliases resolve to identical canonical entities -> no gap
        self.assertEqual(report["missing_skills"], [])
        self.assertEqual(report["ranked_missing_skills"], [])

    def test_unmapped_resources_identified(self):
        extractor = dummy_extractor_factory({
            "JD": ["Kubernetes", "AWS"],
            "Resume": ["Python"],
        })
        report = analyze_job_match("JD", "Resume", extract_fn=extractor)
        self.assertEqual(report["missing_skills"], ["AWS", "Kubernetes"])
        self.assertIn("AWS", report["unmapped_resources"])
        self.assertIn("Kubernetes", report["unmapped_resources"])
        self.assertEqual(report["resources"]["AWS"], [])

    def test_empty_extraction_results(self):
        extractor = lambda text: []
        report = analyze_job_match("General Text", "Resume Text", extract_fn=extractor)
        self.assertEqual(report["missing_skills"], [])
        self.assertEqual(report["ranked_missing_skills"], [])
        self.assertEqual(report["resources"], {})
        self.assertEqual(report["unmapped_resources"], [])

    def test_input_validation(self):
        with self.assertRaises(TypeError):
            analyze_job_match(123, "valid resume")
        with self.assertRaises(TypeError):
            analyze_job_match("valid jd", None)
        with self.assertRaises(ValueError):
            analyze_job_match("", "valid resume")
        with self.assertRaises(ValueError):
            analyze_job_match("   ", "valid resume")
        with self.assertRaises(ValueError):
            analyze_job_match("valid jd", "")

    def test_default_extractor_raises_descriptive_error(self):
        with self.assertRaises((RuntimeError, FileNotFoundError)) as ctx:
            extract_skills("Python JD", model_path="non_existent_checkpoints")
        self.assertIn("DistilBERT model weights not found", str(ctx.exception))

    def test_assess_skill_integration(self):
        # 1 correct out of 5 for Python
        answers = {"py_01": "x = (1, 2, 3)"}
        result = assess_skill("Python", answers, self_rating=4)
        self.assertEqual(result["skill"], "Python")
        self.assertEqual(result["score"]["correct_count"], 1)
        self.assertEqual(result["score"]["score_percent"], 20.0)
        self.assertEqual(result["confidence_comparison"]["self_rating"], 4)
        self.assertEqual(result["confidence_comparison"]["self_rating_percent"], 75.0)
        self.assertEqual(result["confidence_comparison"]["confidence_gap"], 55.0)

    def test_assess_skill_unsupported(self):
        with self.assertRaises(ValueError):
            assess_skill("Docker", {}, self_rating=3)


if __name__ == "__main__":
    unittest.main()

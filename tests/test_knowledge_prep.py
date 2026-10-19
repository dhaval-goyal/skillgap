"""
Unit tests for Stage 12 technical knowledge extraction and evaluation logic.
"""
import unittest
from preprocessing.bio_utils import to_skill_bio
from preprocessing.prepare_dataset import (
    preprocess_dataset_record,
    align_labels_with_tokens,
)
from model.evaluation import (
    extract_entities_from_bio,
    compute_entity_metrics,
    DIAGNOSTIC_SUITE,
    evaluate_diagnostic_suite,
)


class TestBIOKnowledge(unittest.TestCase):
    def test_prefixed_tags_conversion(self):
        tags = ["O", "B-Knowledge", "I-Knowledge", "B-Skill", "I-Skill"]
        expected = ["O", "B-SKILL", "I-SKILL", "B-SKILL", "I-SKILL"]
        self.assertEqual(to_skill_bio(tags), expected)

    def test_custom_entity_name(self):
        tags = ["O", "B-Knowledge", "I-Knowledge"]
        expected = ["O", "B-KNOWLEDGE", "I-KNOWLEDGE"]
        self.assertEqual(to_skill_bio(tags, entity_name="KNOWLEDGE"), expected)

    def test_invalid_tag_raises_value_error(self):
        with self.assertRaises(ValueError):
            to_skill_bio(["O", "UNKNOWN_TAG"])


class TestDatasetRecordPreprocess(unittest.TestCase):
    def test_record_extraction_knowledge(self):
        record = {
            "tokens": ["Docker", "Kubernetes", "skills"],
            "tags_knowledge": ["B", "B", "O"],
            "tags_skill": ["O", "O", "B"],
        }
        res = preprocess_dataset_record(record, target_field="tags_knowledge")
        self.assertEqual(res["tokens"], ["Docker", "Kubernetes", "skills"])
        self.assertEqual(res["labels"], ["B-SKILL", "B-SKILL", "O"])

    def test_record_extraction_missing_field(self):
        record = {"tokens": ["Docker"]}
        with self.assertRaises(KeyError):
            preprocess_dataset_record(record, target_field="tags_knowledge")


class TestSubtokenAlignment(unittest.TestCase):
    def setUp(self):
        self.label2id = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}

    def test_standard_subtoken_masking(self):
        # Word 0: 'Docker' (split into 'dock', '##er') -> label 'B-SKILL'
        # Word 1: 'developer' -> label 'O'
        labels = ["B-SKILL", "O"]
        word_ids = [None, 0, 0, 1, None]  # [CLS], dock, ##er, developer, [SEP]

        aligned = align_labels_with_tokens(
            labels=labels,
            word_ids=word_ids,
            label2id=self.label2id,
            label_all_subtokens=False,
        )
        # Expected: [CLS] -> -100, 'dock' -> 1, '##er' -> -100, 'developer' -> 0, [SEP] -> -100
        self.assertEqual(aligned, [-100, 1, -100, 0, -100])

    def test_all_subtokens_labeled(self):
        labels = ["B-SKILL", "O"]
        word_ids = [None, 0, 0, 1, None]

        aligned = align_labels_with_tokens(
            labels=labels,
            word_ids=word_ids,
            label2id=self.label2id,
            label_all_subtokens=True,
        )
        self.assertEqual(aligned, [-100, 1, 1, 0, -100])


class TestSpanEntityEvaluation(unittest.TestCase):
    def test_extract_entities_from_bio(self):
        tags = ["O", "B-SKILL", "I-SKILL", "O", "B-SKILL", "O"]
        entities = extract_entities_from_bio(tags)
        expected = {("SKILL", 1, 3), ("SKILL", 4, 5)}
        self.assertEqual(entities, expected)

    def test_adjacent_entities(self):
        tags = ["B-SKILL", "B-SKILL"]
        entities = extract_entities_from_bio(tags)
        expected = {("SKILL", 0, 1), ("SKILL", 1, 2)}
        self.assertEqual(entities, expected)

    def test_compute_entity_metrics_perfect(self):
        preds = [["O", "B-SKILL", "I-SKILL", "O"]]
        golds = [["O", "B-SKILL", "I-SKILL", "O"]]
        metrics = compute_entity_metrics(preds, golds)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["true_positives"], 1)

    def test_compute_entity_metrics_mismatch(self):
        preds = [["O", "B-SKILL", "O", "O"]]
        golds = [["O", "O", "B-SKILL", "O"]]
        metrics = compute_entity_metrics(preds, golds)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)

    def test_compute_entity_metrics_ignores_mask(self):
        # Subtoken slot with -100 should be ignored during evaluation
        preds = [["B-SKILL", "I-SKILL", "O"]]
        golds = [["B-SKILL", "-100", "O"]]
        metrics = compute_entity_metrics(preds, golds)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)


class TestDiagnosticSuite(unittest.TestCase):
    def test_diagnostic_suite_structure(self):
        self.assertIn("gold_validation_examples", DIAGNOSTIC_SUITE)
        self.assertIn("sanity_check_examples", DIAGNOSTIC_SUITE)
        self.assertGreaterEqual(len(DIAGNOSTIC_SUITE["gold_validation_examples"]), 5)
        self.assertGreaterEqual(len(DIAGNOSTIC_SUITE["sanity_check_examples"]), 4)

    def test_evaluate_diagnostic_suite_mock(self):
        class MockExtractor:
            def extract(self, text: str):
                extracted = []
                for kw in ["SQL", "Python", "Docker", "Power BI", "AWS", "MySQL"]:
                    if kw.lower() in text.lower():
                        extracted.append(kw)
                return extracted

        extractor = MockExtractor()
        results = evaluate_diagnostic_suite(extractor)
        self.assertIn("gold_results", results)
        self.assertIn("sanity_results", results)
        self.assertEqual(results["sanity_summary"]["total_targets"], 13)
        self.assertEqual(results["sanity_summary"]["total_found"], 5)
        self.assertEqual(results["sanity_summary"]["overall_hit_rate"], round(5 / 13, 4))
        self.assertIn("overall_recall", results["gold_summary"])


class TestTrainKnowledgeConfig(unittest.TestCase):
    def test_dry_run_safety(self):
        from unittest.mock import patch
        from model.train_knowledge import run_experiment
        import argparse

        args = argparse.Namespace(
            dataset_name="jjzha/skillspan",
            target_field="tags_knowledge",
            model_name="checkpoints/best_model",  # Use local model for fast tokenization/forward pass
            output_dir="tmp_dry_run_test_dir",
            learning_rate=5e-5,
            epochs=1,
            batch_size=2,
            max_length=32,
            weight_decay=0.01,
            seed=42,
            label_all_subtokens=False,
            dry_run=True,
        )

        mock_tokenized = {
            "train": [
                {"input_ids": [101] + [1000] * 30 + [102], "attention_mask": [1] * 32, "labels": [-100] + [0] * 30 + [-100]},
                {"input_ids": [101] + [1000] * 30 + [102], "attention_mask": [1] * 32, "labels": [-100] + [1] * 30 + [-100]},
            ],
            "validation": [{"input_ids": [101, 102], "attention_mask": [1, 1], "labels": [-100, -100]}],
            "test": [{"input_ids": [101, 102], "attention_mask": [1, 1], "labels": [-100, -100]}],
        }

        with patch("model.train_knowledge.prepare_hf_dataset", return_value=mock_tokenized), \
             patch("transformers.Trainer") as mock_trainer:
            res = run_experiment(args)
            self.assertTrue(res.get("dry_run"))
            self.assertEqual(res.get("status"), "verified")
            # Verify Trainer was never instantiated, train() was never called, test set was never evaluated
            mock_trainer.assert_not_called()


if __name__ == "__main__":
    unittest.main()


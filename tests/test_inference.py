import unittest
from unittest.mock import MagicMock
from inference.extractor import decode_bio_spans, SkillExtractor, get_extractor


class TestInference(unittest.TestCase):
    def test_decode_bio_spans_single_and_multiword(self):
        text = "Experienced in Python and Machine Learning."
        # Offsets:
        # Python: (15, 21)
        # Machine: (26, 33)
        # Learning: (34, 42)
        tags = ["O", "O", "B-SKILL", "O", "B-SKILL", "I-SKILL", "O"]
        offsets = [(0, 11), (12, 14), (15, 21), (22, 25), (26, 33), (34, 42), (42, 43)]
        spans = decode_bio_spans(text, tags, offsets)
        self.assertEqual(spans, ["Python", "Machine Learning"])

    def test_decode_bio_spans_special_and_padding_skipped(self):
        text = "Docker"
        tags = ["[CLS]", "B-SKILL", "[SEP]", "O"]
        offsets = [(0, 0), (0, 6), (0, 0), (0, 0)]
        spans = decode_bio_spans(text, tags, offsets)
        self.assertEqual(spans, ["Docker"])

    def test_decode_bio_spans_orphaned_i_tag_and_empty(self):
        text = "Kubernetes"
        tags = ["I-SKILL"]
        offsets = [(0, 10)]
        self.assertEqual(decode_bio_spans(text, tags, offsets), ["Kubernetes"])

        # No entities
        self.assertEqual(decode_bio_spans("No skills here", ["O", "O", "O"], [(0, 2), (3, 9), (10, 14)]), [])

    def test_extractor_missing_checkpoint_raises_filenotfound(self):
        extractor = SkillExtractor(model_path="non_existent_checkpoint_dir")
        with self.assertRaises(FileNotFoundError) as ctx:
            extractor.extract("Python developer")
        self.assertIn("DistilBERT model weights not found", str(ctx.exception))
        self.assertIn("Google Colab", str(ctx.exception))

    def test_extractor_input_validation(self):
        extractor = SkillExtractor()
        with self.assertRaises(TypeError):
            extractor.extract(123)
        self.assertEqual(extractor.extract("   "), [])
        self.assertEqual(extractor.extract(""), [])

    def test_extractor_mocked_model_pipeline(self):
        extractor = SkillExtractor()
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        # Mock tokenization returning Python and SQL
        text = "Skills: Python and SQL"
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "offset_mapping": [[(0, 0), (0, 6), (7, 8), (8, 14), (15, 18), (19, 22), (0, 0)]],
        }

        # Mock model output logits
        import torch
        # Tags: [CLS], O, O, B-SKILL (Python), O, B-SKILL (SQL), [SEP]
        mock_outputs = MagicMock()
        # Preds: 0, 0, 0, 1, 0, 1, 0
        mock_outputs.logits = torch.tensor([[[10., 0., 0.], [10., 0., 0.], [10., 0., 0.],
                                             [0., 10., 0.], [10., 0., 0.], [0., 10., 0.],
                                             [10., 0., 0.]]])
        mock_model.return_value = mock_outputs
        mock_model.config.id2label = {0: "O", 1: "B-SKILL", 2: "I-SKILL"}

        extractor.tokenizer = mock_tokenizer
        extractor.model = mock_model

        results = extractor.extract(text)
        self.assertEqual(results, ["Python", "SQL"])

    def test_get_extractor_singleton_caching(self):
        ext1 = get_extractor("test_cache_path")
        ext2 = get_extractor("test_cache_path")
        self.assertIs(ext1, ext2)


if __name__ == "__main__":
    unittest.main()

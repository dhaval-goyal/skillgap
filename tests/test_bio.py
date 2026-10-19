import unittest
from preprocessing.bio_utils import to_skill_bio, is_valid_bio
from preprocessing.prepare_dataset import preprocess_example

class TestBIOUtils(unittest.TestCase):
    def test_to_skill_bio_conversion(self):
        # Test string conversion
        self.assertEqual(to_skill_bio(["O", "B", "I"]), ["O", "B-SKILL", "I-SKILL"])
        # Test integer ID conversion
        self.assertEqual(to_skill_bio([0, 1, 2]), ["O", "B-SKILL", "I-SKILL"])

    def test_valid_bio_sequence(self):
        valid, err = is_valid_bio(["O", "B-SKILL", "I-SKILL", "O", "B-SKILL"])
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_invalid_bio_i_after_o(self):
        valid, err = is_valid_bio(["O", "I-SKILL"])
        self.assertFalse(valid)
        self.assertIn("Invalid transition", err)

    def test_preprocess_example_success(self):
        res = preprocess_example(["Python", "expert"], ["B", "I"])
        self.assertEqual(res["labels"], ["B-SKILL", "I-SKILL"])

    def test_preprocess_example_length_mismatch(self):
        with self.assertRaises(ValueError):
            preprocess_example(["Python"], ["B", "I"])

    def test_preprocess_example_invalid_bio(self):
        with self.assertRaises(ValueError):
            preprocess_example(["Python"], ["I"])

if __name__ == "__main__":
    unittest.main()

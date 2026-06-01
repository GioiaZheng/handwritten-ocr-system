import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec("cv2") is None:
    raise unittest.SkipTest("opencv-python is required for preprocessing tests")

from preprocessor import Preprocessor


class PreprocessorTest(unittest.TestCase):
    def test_truncate_label_counts_repeated_characters_for_ctc(self):
        text = "bookkeeper"

        truncated = Preprocessor._truncate_label(text, max_text_len=4)

        self.assertEqual(truncated, "boo")

    def test_label_indexer_ignores_characters_outside_vocab(self):
        indexed = Preprocessor.label_indexer("abc", "cabx")

        np.testing.assert_array_equal(indexed, np.array([2, 0, 1]))

    def test_label_padding_truncates_and_pads_to_fixed_length(self):
        padded = Preprocessor.label_padding(9, 5, np.array([1, 2, 3]))

        np.testing.assert_array_equal(padded, np.array([1, 2, 3, 9, 9]))

    def test_preprocess_img_preserves_target_canvas_size(self):
        image = np.ones((20, 80), dtype=np.uint8) * 255
        preprocessor = Preprocessor(image=None, vocab="abc")

        processed, label = preprocessor.preprocess_img(image, "abc")

        self.assertEqual(processed.shape, (64, 512))
        self.assertEqual(label, "abc")

    def test_call_truncates_label_by_ctc_alignment_cost(self):
        image = np.ones((20, 80), dtype=np.uint8) * 255
        preprocessor = Preprocessor(image=None, vocab="bokepr")

        _, label = preprocessor(image, "bookkeeper", max_len=4)

        np.testing.assert_array_equal(label, np.array([0, 1, 1, 6]))


if __name__ == "__main__":
    unittest.main()

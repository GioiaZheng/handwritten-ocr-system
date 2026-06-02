import copy
import unittest

from scripts.validate_training_config import load_config, validate_config


VALID_CONFIG = {
    "experiment": {
        "name": "crnn_ctc_iam_baseline",
        "seed": 2024,
        "split_random_state": 42,
    },
    "data": {
        "dataset": "IAM sentences",
        "metadata_file": "data/iam_sentences/metadata/sentences.txt",
        "image_dir": "data/iam_sentences/dataset",
        "split": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "filter": {
            "skip_comment_lines": True,
            "skip_metadata_status": "err",
            "require_image_file": True,
        },
    },
    "preprocessing": {
        "grayscale": True,
        "normalize_pixels": [0.0, 1.0],
        "augmentation": False,
    },
    "model": {
        "file": "crnn_ctc.py",
        "class": "Model",
        "input_channels": 1,
        "cnn_blocks": 9,
        "recurrent_layers": [
            {"type": "bidirectional_lstm", "hidden_size": 256},
            {"type": "bidirectional_lstm", "hidden_size": 64},
        ],
        "dropout": 0.2,
        "loss": "ctc",
    },
    "evaluation": {
        "batch_size": 64,
        "checkpoint": "Model/model.pt",
        "metrics": ["character_error_rate", "word_error_rate"],
    },
}


class TrainingConfigValidationTest(unittest.TestCase):
    def test_repository_training_config_is_valid(self):
        config = load_config("configs/training_config.yaml")

        self.assertEqual(validate_config(config), [])

    def test_rejects_split_that_does_not_sum_to_one(self):
        config = copy.deepcopy(VALID_CONFIG)
        config["data"]["split"]["test"] = 0.2

        self.assertIn("data.split values must sum to 1.0.", validate_config(config))

    def test_rejects_missing_required_metrics(self):
        config = copy.deepcopy(VALID_CONFIG)
        config["evaluation"]["metrics"] = ["character_error_rate"]

        self.assertIn(
            "evaluation.metrics is missing required metrics: word_error_rate.",
            validate_config(config),
        )

    def test_rejects_invalid_dropout_range(self):
        config = copy.deepcopy(VALID_CONFIG)
        config["model"]["dropout"] = 1.5

        self.assertIn("model.dropout must be between 0 and 1.", validate_config(config))

    def test_rejects_empty_recurrent_layers(self):
        config = copy.deepcopy(VALID_CONFIG)
        config["model"]["recurrent_layers"] = []

        self.assertIn(
            "model.recurrent_layers must be a non-empty list.",
            validate_config(config),
        )


if __name__ == "__main__":
    unittest.main()

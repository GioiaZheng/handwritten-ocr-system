"""Validate the OCR training and evaluation configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_SECTIONS = (
    "experiment",
    "data",
    "preprocessing",
    "model",
    "evaluation",
)

REQUIRED_METRICS = {
    "character_error_rate",
    "word_error_rate",
}


def _require_mapping(value: Any, name: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{name} must be a mapping.")
        return {}
    return value


def _require_positive_int(value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{name} must be a positive integer.")


def _require_number(value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{name} must be numeric.")


def _require_string(value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{name} must be a non-empty string.")


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    for section in REQUIRED_TOP_LEVEL_SECTIONS:
        if section not in config:
            errors.append(f"missing top-level section: {section}.")

    experiment = _require_mapping(config.get("experiment"), "experiment", errors)
    data = _require_mapping(config.get("data"), "data", errors)
    preprocessing = _require_mapping(config.get("preprocessing"), "preprocessing", errors)
    model = _require_mapping(config.get("model"), "model", errors)
    evaluation = _require_mapping(config.get("evaluation"), "evaluation", errors)

    _require_string(experiment.get("name"), "experiment.name", errors)
    _require_positive_int(experiment.get("seed"), "experiment.seed", errors)
    _require_positive_int(
        experiment.get("split_random_state"),
        "experiment.split_random_state",
        errors,
    )

    _require_string(data.get("dataset"), "data.dataset", errors)
    _require_string(data.get("metadata_file"), "data.metadata_file", errors)
    _require_string(data.get("image_dir"), "data.image_dir", errors)

    split = _require_mapping(data.get("split"), "data.split", errors)
    split_values = []
    for name in ("train", "validation", "test"):
        value = split.get(name)
        _require_number(value, f"data.split.{name}", errors)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value <= 0:
                errors.append(f"data.split.{name} must be positive.")
            split_values.append(float(value))

    if len(split_values) == 3 and abs(sum(split_values) - 1.0) > 1e-9:
        errors.append("data.split values must sum to 1.0.")

    data_filter = _require_mapping(data.get("filter"), "data.filter", errors)
    for name in ("skip_comment_lines", "require_image_file"):
        if not isinstance(data_filter.get(name), bool):
            errors.append(f"data.filter.{name} must be boolean.")
    _require_string(
        data_filter.get("skip_metadata_status"),
        "data.filter.skip_metadata_status",
        errors,
    )

    if not isinstance(preprocessing.get("grayscale"), bool):
        errors.append("preprocessing.grayscale must be boolean.")
    if not isinstance(preprocessing.get("augmentation"), bool):
        errors.append("preprocessing.augmentation must be boolean.")

    normalize_pixels = preprocessing.get("normalize_pixels")
    if (
        not isinstance(normalize_pixels, list)
        or len(normalize_pixels) != 2
        or any(not isinstance(value, (int, float)) for value in normalize_pixels)
        or normalize_pixels[0] >= normalize_pixels[1]
    ):
        errors.append("preprocessing.normalize_pixels must be two ascending numeric bounds.")

    _require_string(model.get("file"), "model.file", errors)
    _require_string(model.get("class"), "model.class", errors)
    _require_positive_int(model.get("input_channels"), "model.input_channels", errors)
    _require_positive_int(model.get("cnn_blocks"), "model.cnn_blocks", errors)

    dropout = model.get("dropout")
    _require_number(dropout, "model.dropout", errors)
    if (
        isinstance(dropout, (int, float))
        and not isinstance(dropout, bool)
        and not 0 <= dropout <= 1
    ):
        errors.append("model.dropout must be between 0 and 1.")

    if model.get("loss") != "ctc":
        errors.append("model.loss must be 'ctc'.")

    recurrent_layers = model.get("recurrent_layers")
    if not isinstance(recurrent_layers, list) or not recurrent_layers:
        errors.append("model.recurrent_layers must be a non-empty list.")
    else:
        for index, layer in enumerate(recurrent_layers):
            layer_name = f"model.recurrent_layers[{index}]"
            layer_mapping = _require_mapping(layer, layer_name, errors)
            _require_string(layer_mapping.get("type"), f"{layer_name}.type", errors)
            _require_positive_int(
                layer_mapping.get("hidden_size"),
                f"{layer_name}.hidden_size",
                errors,
            )

    _require_positive_int(evaluation.get("batch_size"), "evaluation.batch_size", errors)
    _require_string(evaluation.get("checkpoint"), "evaluation.checkpoint", errors)

    metrics = evaluation.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        errors.append("evaluation.metrics must be a non-empty list.")
    else:
        missing_metrics = REQUIRED_METRICS - set(metrics)
        if missing_metrics:
            missing = ", ".join(sorted(missing_metrics))
            errors.append(f"evaluation.metrics is missing required metrics: {missing}.")

    return errors


def load_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration file must contain a mapping at the top level.")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training_config.yaml"),
        help="Path to the training configuration YAML file.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Training config validation failed: {exc}")
        return 1

    errors = validate_config(config)
    if errors:
        print("Training config validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Training config validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

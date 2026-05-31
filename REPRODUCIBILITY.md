# Reproducibility

This project is an experimental handwritten OCR baseline built around a
CNN-RNN-CTC model. The repository records the intended data layout, split
policy, model configuration, and evaluation procedure. It does not currently
commit the IAM dataset, a trained checkpoint, or a verified evaluation log.

## Environment

Recommended setup:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

The IAM sentence dataset must be provided locally:

```text
data/
  iam_sentences/
    dataset/
      <sentence image files>
    metadata/
      sentences.txt
```

The raw IAM files are not committed to the repository.

## Configuration

The main experiment settings are stored in
[`configs/training_config.yaml`](configs/training_config.yaml).

| Field | Meaning |
|---|---|
| `experiment.seed` | Seed used for PyTorch and NumPy runs |
| `experiment.split_random_state` | Random state for deterministic train/validation/test split |
| `data.metadata_file` | IAM sentence metadata file |
| `data.image_dir` | Directory containing sentence images |
| `data.split` | Target train/validation/test proportions |
| `data.filter.skip_comment_lines` | Ignore comment lines in IAM metadata |
| `data.filter.skip_metadata_status` | Skip IAM rows marked with the configured status |
| `data.filter.require_image_file` | Keep only rows with an available local image |
| `preprocessing.grayscale` | Use grayscale image input |
| `preprocessing.normalize_pixels` | Normalize image pixels to the stated range |
| `preprocessing.augmentation` | Enable or disable image augmentation |
| `model.*` | CRNN-CTC architecture metadata |
| `evaluation.batch_size` | Evaluation batch size |
| `evaluation.checkpoint` | Expected checkpoint path |
| `evaluation.metrics` | Reported metrics |

## Evaluation

Run evaluation after placing the IAM files and checkpoint:

```bash
python evaluate.py
```

Expected final lines:

```text
our final character error rate: <CER>
our final word error rate: <WER>
```

Record verified runs in [`docs/evaluation.md`](docs/evaluation.md). Keep
example outputs separate from measured results.

## Smoke Tests

The smoke tests do not require the IAM dataset or a trained checkpoint. They
check deterministic preprocessing and label handling behavior:

```bash
python -m unittest discover -s tests
```

## Current Reproducibility Status

| Artifact | Status |
|---|---|
| IAM raw data | Not committed |
| Train/validation/test split policy | Documented |
| Model architecture metadata | Documented |
| Trained checkpoint | Not committed |
| Verified CER/WER run | Pending |
| Dataset-free smoke tests | Available |

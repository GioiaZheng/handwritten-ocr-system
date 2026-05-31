# Evaluation Record

## Dataset Split

The evaluation script uses the IAM sentence metadata and keeps the split deterministic:

| Split | Ratio | Source |
| --- | ---: | --- |
| Train | 80% | First `train_test_split(..., test_size=0.2, random_state=42)` result |
| Validation | 10% | Half of the held-out 20% |
| Test | 10% | Half of the held-out 20% |

Seed values are recorded in [configs/training_config.yaml](../configs/training_config.yaml):

| Setting | Value |
| --- | ---: |
| Torch seed | 2024 |
| NumPy seed | 2024 |
| Split random state | 42 |

## CER/WER Table

No trained checkpoint or raw evaluation log with reproducible metric values is committed in this repository. The table is kept as the results ledger so new runs can be added without mixing example numbers with measured results.

| Run ID | Checkpoint | Dataset | Split | CER | WER | Status |
| --- | --- | --- | --- | ---: | ---: | --- |
| `crnn_ctc_iam_baseline` | `Model/model.pt` | IAM sentences | deterministic 80/10/10 | Not reported | Not reported | Pending reproducible checkpoint/log |

To record a real result, run `python evaluate.py` with the IAM files and checkpoint present, then copy the final `our final character error rate` and `our final word error rate` values into this table with the run date.

# SFT Dataset for 24 Game

This directory contains the Supervised Fine-Tuning (SFT) dataset for the 24 Game.

## Files

- `train.parquet`: Training dataset
- `val.parquet`: Validation dataset
t

## Data Format

Each record in the dataset contains:
- `cards`: Four cards for the 24 Game (e.g., "3, 3, 8, 8")
- `prompt`: The input prompt for the model
- `response`: The expected output with step-by-step reasoning
- `answer`: The final answer (equation that equals 24)

## Generation

The dataset is generated using the script:
```bash
python scripts/data_processing/make_dataset_sft.py
```

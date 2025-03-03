# Evaluation Results

This directory contains the evaluation results for different models and training methods.

## Files

- `evaluation_results_sft_qwen_2.5_math_7b.json`: Evaluation results for SFT model
- `evaluation_results_rl_zero_v1.json`: Evaluation results for Zero-RL model (version 1)
- `evaluation_results_rl_zero_v2.json`: Evaluation results for Zero-RL model (version 2)
- `evaluation_results_sft_rl.json`: Evaluation results for SFT+RL model
- `evaluation_results_sft_rl_v1.json`: Evaluation results for SFT+RL model (version 1)
- `evaluation_results_sft_rl_v2.json`: Evaluation results for SFT+RL model (version 2)

## Format

Each result file contains:
- Accuracy metrics
- Detailed evaluation for each test case
- Comparison between different checkpoints

## Generation

The results are generated using the evaluation script:
```bash
python scripts/evaluation/eval.py
```

And analyzed using:
```bash
python scripts/evaluation/stat_eval_result.py
``` 
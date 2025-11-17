# ITI Harmfulness Steering System

A comprehensive implementation of Inference-Time Intervention (ITI) for harmfulness detection and steering in transformer LLMs.

## Overview

This system implements:

1. **Activation Collection**: Extract per-head activations from all layers and heads
2. **Linear Probes**: Train classifiers to detect harmful vs. harmless activations
3. **Head Selection**: Select top-K heads by validation accuracy
4. **Mean-Shift Steering**: Compute steering directions (theta, sigma) per head
5. **Steering Hooks**: Apply activation interventions during generation
   - Unconditional ITI (always steer)
   - Conditional Catch & Steer (steer only when triggered)

## Quick Start

### 1. Setup Virtual Environment

```bash
cd iti-harmfulness
python -m venv venv
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Environment

```bash
python test_env.py
```

You should see all green checkmarks for installed packages.

### 4. Run ITI Implementation

```bash
python iti_harmfulness_main.py
```

First run will:
- Download the civil_comments dataset (~2-3 min)
- Load distilgpt2 model
- Collect activations from all heads (5-10 min)
- Train per-head probes (5-10 min)
- Select top-K heads
- Compute steering vectors
- Generate example text comparing baseline vs. ITI-steered outputs

Subsequent runs will use cached files and run much faster (~1-2 min).

## Project Structure

```
iti-harmfulness/
├── iti_harmfulness_main.py       # Main implementation
├── test_env.py                    # Environment verification
├── requirements.txt               # Python dependencies
├── cache/                         # Cached activations, probes, vectors
│   ├── activations_train.pkl
│   ├── activations_val.pkl
│   ├── probes/                    # Per-head probe files
│   ├── probes_metadata.json
│   ├── selected_heads.json
│   └── steering_vectors.pkl
├── outputs/                       # Generated text and results
│   └── generation_prompt_*.txt
└── venv/                          # Virtual environment (ignored)
```

## Configuration

Key parameters in `iti_harmfulness_main.py` (top of file):

```python
MODEL_NAME = "distilgpt2"          # Model to use
DATASET_NAME = "civil_comments"    # Harmfulness dataset
DATA_SPLIT = "train[:5000]"        # Training data amount
MAX_LEN = 128                      # Token sequence length
BATCH_SIZE = 32                    # Batch size
TOP_K_HEADS = 10                   # Number of heads to select
STEERING_ALPHA = 1.5               # Steering strength
STEERING_CATCH_THRESHOLD = 0.6     # Catch threshold (for conditional steering)
```

## Understanding the Code

Each section is clearly labeled and commented:

- **Section 0**: Utility functions (file I/O, caching)
- **Section 1**: Load harmfulness dataset
- **Section 2**: Load model and tokenizer
- **Section 3**: Collect head activations
- **Section 4**: Train per-head linear probes
- **Section 5**: Select top-K heads
- **Section 6**: Compute mean-shift steering vectors
- **Section 7**: Steering hook implementations
- **Section 8**: Demo generation with baseline vs. ITI vs. Catch & Steer
- **Section 9**: Summary and next steps

## Enabling Conditional Catch & Steer

The conditional "Catch & Steer" code is commented out in Section 8. To enable it:

1. Uncomment the "CONDITIONAL CATCH & STEER" section in Section 8
2. Uncomment the `make_conditional_catch_steer_hook` calls
3. Adjust `STEERING_CATCH_THRESHOLD` to control trigger sensitivity

The conditional version:
- Uses the trained probes to detect harmful activations token-by-token
- Only applies steering when probe confidence exceeds threshold
- Preserves model behavior in safe regions

## Next Steps

### Immediate
- [ ] Run the script and verify generation works
- [ ] Adjust parameters (TOP_K_HEADS, STEERING_ALPHA) and observe effects
- [ ] Uncomment and test Catch & Steer conditional steering

### Model Scaling
- [ ] Test with larger models (LLaMA 7B, Mistral via unsloth)
- [ ] Compare steering effectiveness across models
- [ ] Benchmark inference time overhead

### Evaluation
- [ ] Measure harmfulness reduction on test set
- [ ] Evaluate impact on model capabilities/creativity
- [ ] Combine with other safety techniques

### Extensions
- [ ] Train separate probes for truthfulness
- [ ] Implement multi-attribute steering (harm + truth)
- [ ] Add interactive steering strength control
- [ ] Deploy as inference service

## Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE`
- Reduce `DATA_SPLIT` (fewer examples)
- Use smaller model
- Enable gradient checkpointing

### Slow Activation Collection
- Reduce `DATA_SPLIT`
- Reduce `MAX_LEN`
- Use smaller model
- Cache is automatically used on subsequent runs

### CUDA Not Available
The script automatically falls back to CPU. For GPU support, ensure:
- NVIDIA drivers are installed
- CUDA Toolkit 12.1+ is installed
- PyTorch CUDA build is installed

## References

- **ITI Paper**: [Inference-Time Intervention by Elad and Ravfogel](https://arxiv.org/abs/2302.10149)
- **Linear Probes**: [Probing Neural Language Models for Understanding of the English Tense System](https://arxiv.org/abs/2010.00693)
- **Activation Steering**: [Does Localization Inform Editability? Surprising Differences in Factual Knowledge Localization in Language Models](https://arxiv.org/abs/2308.14753)

## Notes

- All computations are done in float32 for stability
- Activations are standardized using training statistics only
- Steering is applied at the attention head level before the output projection
- Top-K head selection is based on validation accuracy (AUC is also tracked)

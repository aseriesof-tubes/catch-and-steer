import time

start = time.time()

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from util import *

from datasets import load_dataset


# =============================================================================
# CONFIGURATION
# =============================================================================


DATASET_NAME = "allenai/real-toxicity-prompts"

CACHE_DIR = Path("./cache")
OUTPUT_DIR = Path("./outputs")

ensure_dir(CACHE_DIR)
ensure_dir(OUTPUT_DIR)

CLASSIFIER_TRAINING_DATA_PATH = CACHE_DIR / "classifier_training_data"




device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Using device: {device}")
print(f"[TIMER] step_name: {time.time() - start:.2f}s", flush=True)




# =============================================================================
# SECTION 1: LOAD DATA (HuggingFace Caches Automatically)
# =============================================================================

print("\n[DATA] Loading training dataset...")
ds_name = str(DATASET_NAME)
ds = None
try:
    ds = load_dataset(ds_name)["train"]
    print(f"   Loaded dataset variant: {ds_name}")
except Exception as e:
    print(f"   Dataset variant '{ds_name}' not available: {e}")

#Flatten and Clean Data

ds = ds.remove_columns(["filename", "begin", "end"])

# def flatten_column(example, fields):
#     for field in fields:
#         for key, val in example[field].items():
#             example[f"{field}_{key}"] = val
#         del example[field]
#     return example

# # Apply row-wise flattening
# ds = ds.map(lambda ex: flatten_column(ex, ["prompt", "continuation"]), batched=True)

def flatten_prompt_continuation(batch):
    # batch["prompt"] is a list of dicts; each dict has keys like toxicity, text, etc.
    # We'll expand each key into its own column: prompt_<key>, continuation_<key>
    out = {}
    for field in ("prompt", "continuation"):
        # gather all keys that ever appear in this batch (safe even if some rows miss keys)
        keys = set()
        for d in batch[field]:
            if d is not None:
                keys.update(d.keys())

        for k in keys:
            out[f"{field}_{k}"] = [
                (d.get(k) if d is not None else None) for d in batch[field]
            ]

    return out

# Batched map (much faster)
ds = ds.map(flatten_prompt_continuation, batched=True)



# =============================================================================
# SECTION 1: SEPERATE DATA
# =============================================================================


ds_challenging = ds.filter(lambda x: x["challenging"] == True)
ds             = ds.filter(lambda x: x["challenging"] == False)

ds_challenging = ds_challenging.remove_columns(["challenging"])
ds             = ds.remove_columns(["challenging"])


thresholds = {
    "continuation_toxicity": 0.4,
    "continuation_identity_attack": 0.3,
    "continuation_threat": 0.45,
    "continuation_insult": 0.3,
    "continuation_profanity": 0.3,
    "continuation_sexually_explicit": 0.3,
    "continuation_severe_toxicity": 0.25
}



# def find_threshold(ds, col, low=0.1, high=0.7, step=0.15):
#     """
#     Search for a threshold between low and high that looks like
#     it separates harmful vs. harmless examples.
#     Prints example rows near each candidate threshold.
#     """
#     values = np.array([0.0 if v is None else v for v in ds[col]], dtype=float)

#     thresh = low
#     while thresh <= high:
#         # Find rows closest to this threshold
#         closest_idx = np.argmin(np.abs(values - thresh))
#         print(f"\n--- {col} threshold candidate: {thresh:.3f} ---")
#         print(f"Score at closest row: {values[closest_idx]:.3f}")
#         print(ds[closest_idx])
#         thresh += step

# # Example usage:
# for i in thresholds.keys():
#     find_threshold(ds, i, low=0.1, high=0.3, step=0.02)

print(f"[TIMER] step_name: {time.time() - start:.2f}s", flush=True)


# def apply_thresholds(example):
#     for col, thresh in thresholds.items():
#         v = example[col]
#         example[f"{col}_label"] = 1 if (0.0 if v is None else v) > thresh else 0
#     return example

# ds = ds.map(apply_thresholds, batched=True)

def apply_thresholds_batched(batch):
    out = {}
    for col, thresh in thresholds.items():
        vals = batch[col]
        out[f"{col}_label"] = [
            1 if (0.0 if v is None else float(v)) > thresh else 0
            for v in vals
        ]
    return out

# Apply to both challenging and non-challenging datasets
# ds_challenging = ds_challenging.map(apply_thresholds)
ds = ds.map(apply_thresholds_batched, batched=True)

print(ds[0], ds[1], ds[2])
print(f"[TIMER] step_name: {time.time() - start:.2f}s", flush=True)

keep_cols = [
    "prompt_text",
    "continuation_text",
] + [c for c in ds.column_names if c.endswith("_label")]

ds = ds.select_columns(keep_cols)

ds.save_to_disk(str(CLASSIFIER_TRAINING_DATA_PATH))


exit()
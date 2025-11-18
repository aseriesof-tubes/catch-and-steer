"""
Minimal dataset inspection script.
Goal: See what the dataset actually looks like before any processing.
"""

from datasets import load_dataset
import numpy as np

print("\n" + "="*80)
print("DATASET INSPECTION")
print("="*80)

# Try to load real_toxicity_prompts
dataset_candidates = [
    "real_toxicity_prompts",
    "realtoxicityprompts",
    "RealToxicityPrompts",
    "unitary/RealToxicityPrompts",
    "civil_comments"
]

print("\nAttempting to load dataset from candidates:")
ds = None
actual_name = None
for candidate in dataset_candidates:
    try:
        print(f"  Trying: {candidate}...", end=" ")
        ds = load_dataset(candidate, split="train[:5]")
        actual_name = candidate
        print("✓ SUCCESS")
        break
    except Exception as e:
        print(f"✗ Failed: {str(e)[:80]}")

if ds is None:
    print("\n❌ Could not load any dataset candidate!")
    exit(1)

print(f"\n✓ Loaded dataset: {actual_name}")
print(f"\nDataset info:")
print(f"  Number of rows: {len(ds)}")
print(f"  Columns: {ds.column_names}")
print(f"  Features: {ds.features}")

print("\n" + "="*80)
print("RAW DATA - FIRST 5 EXAMPLES")
print("="*80)

for idx, example in enumerate(ds):
    print(f"\n--- Example {idx} ---")
    for col, value in example.items():
        # Truncate long strings
        if isinstance(value, str) and len(value) > 100:
            display_val = value[:100] + "..."
        else:
            display_val = value
        print(f"  {col}: {display_val} (type: {type(value).__name__})")

print("\n" + "="*80)
print("COLUMN ANALYSIS")
print("="*80)

# Show all columns and their value ranges
for col in ds.column_names:
    print(f"\nColumn: {col}")
    values = ds[col]
    
    if isinstance(values[0], (int, float)):
        arr = np.array(values)
        print(f"  Type: numeric")
        print(f"  Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.4f}")
        print(f"  Unique values: {np.unique(arr)}")
        print(f"  Sample values: {arr[:5]}")
    elif isinstance(values[0], str):
        print(f"  Type: string")
        print(f"  Sample values (first 50 chars):")
        for i, v in enumerate(values[:3]):
            print(f"    [{i}] {v[:50]}...")
    else:
        print(f"  Type: {type(values[0])}")
        print(f"  Sample values: {values[:3]}")

print("\n" + "="*80)
print("Now inspect the actual label field(s)")
print("="*80)

# Try to find label-like columns
label_candidates = ["toxicity", "label", "toxicity_score", "target", "y", "harmful"]
for col in ds.column_names:
    if any(label_word in col.lower() for label_word in ["label", "toxicity", "harm", "target", "score"]):
        print(f"\n>>> Found potential label column: '{col}'")
        values = np.array(ds[col])
        print(f"    dtype: {values.dtype}")
        print(f"    shape: {values.shape}")
        print(f"    unique: {np.unique(values)}")
        print(f"    min/max: {values.min()} to {values.max()}")
        print(f"    first 10: {values[:10]}")

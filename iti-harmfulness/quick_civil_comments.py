"""
Quick inspection of civil_comments dataset.
"""

print("Loading civil_comments dataset...")
from datasets import load_dataset

ds = load_dataset("civil_comments", split="train[:10]")

print(f"\nColumns: {ds.column_names}")
print(f"\nSample:")
for i in range(min(3, len(ds))):
    print(f"\n--- Example {i} ---")
    for key, val in ds[i].items():
        if isinstance(val, str) and len(val) > 80:
            print(f"  {key}: {val[:80]}...")
        else:
            print(f"  {key}: {val}")

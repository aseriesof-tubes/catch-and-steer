"""
Try well-known toxicity/harmfulness datasets.
"""

from datasets import load_dataset

alternatives = [
    "civil_comments",
    ("jigsaw-toxic-comment-classification", "train[:5]"),
    "offensive-language",
]

print("\nTrying well-known alternatives:\n")

for alt in alternatives:
    if isinstance(alt, tuple):
        ds_name, split = alt
    else:
        ds_name = alt
        split = "train[:5]"
    
    try:
        print(f"Loading: {ds_name}...", end=" ")
        ds = load_dataset(ds_name, split=split)
        print(f"✓ SUCCESS")
        print(f"  Columns: {ds.column_names}")
        print(f"  Sample:\n{ds[0]}\n")
    except Exception as e:
        print(f"✗ Failed: {str(e)[:80]}\n")

"""
Try to find available toxicity/harmfulness datasets.
"""

from datasets import list_datasets
import itertools

print("\nSearching for toxicity/harm-related datasets on HuggingFace Hub...\n")

# Get all datasets (this might take a moment)
all_datasets = list_datasets()

# Filter for ones containing "tox" or "harm"
keywords = ["toxic", "harm", "offensive", "abuse", "hate"]
matching = []

for dataset in all_datasets:
    if any(kw in dataset.lower() for kw in keywords):
        matching.append(dataset)
    if len(matching) >= 20:  # Limit to first 20 matches
        break

print(f"Found {len(matching)} matching datasets:")
for ds in sorted(matching):
    print(f"  - {ds}")

# Also try some common alternatives
print("\n\nTrying some well-known alternatives:")
alternatives = [
    "civil_comments",
    "jigsaw-toxic-comment-classification",
    "HASOC",
    "offensive-language",
    "hate_speech18",
]

for alt in alternatives:
    try:
        from datasets import load_dataset
        ds = load_dataset(alt, split="train[:1]")
        print(f"  ✓ {alt} - WORKS! Columns: {ds.column_names}")
    except Exception as e:
        print(f"  ✗ {alt} - {str(e)[:60]}")

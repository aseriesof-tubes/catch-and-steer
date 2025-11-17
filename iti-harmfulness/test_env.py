"""
Quick environment verification script.
Tests all key imports and displays system information.
"""

import sys
print(f"[ENV] Python version: {sys.version}")
print(f"[ENV] Python executable: {sys.executable}")

# Test core imports
print("\n[TEST] Testing imports...")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ torch: {e}")

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ transformers: {e}")

try:
    import datasets
    print(f"✓ datasets {datasets.__version__}")
except ImportError as e:
    print(f"✗ datasets: {e}")

try:
    import numpy
    print(f"✓ numpy {numpy.__version__}")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"✗ scikit-learn: {e}")

try:
    import joblib
    print(f"✓ joblib {joblib.__version__}")
except ImportError as e:
    print(f"✗ joblib: {e}")

try:
    import tqdm
    print(f"✓ tqdm {tqdm.__version__}")
except ImportError as e:
    print(f"✗ tqdm: {e}")

# Optional: unsloth (may not be installed yet)
try:
    import unsloth
    print(f"✓ unsloth (installed)")
except ImportError:
    print(f"⚠ unsloth: not yet installed (optional)")

print("\n[TEST] Environment verification complete!")

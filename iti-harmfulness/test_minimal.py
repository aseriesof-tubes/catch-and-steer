"""
Minimal ITI test: Check if all components work before running full pipeline.
This is useful for debugging environment issues.
"""

import sys
print("\n" + "="*80)
print("ITI HARMFULNESS STEERING - ENVIRONMENT TEST")
print("="*80 + "\n")

# Test 1: Imports
print("[TEST 1] Checking imports...")
try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
    print(f"    - CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"  ✗ torch FAILED: {e}")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"  ✓ transformers (AutoTokenizer, AutoModelForCausalLM)")
except ImportError as e:
    print(f"  ✗ transformers FAILED: {e}")
    sys.exit(1)

try:
    from datasets import load_dataset
    print(f"  ✓ datasets (load_dataset)")
except ImportError as e:
    print(f"  ✗ datasets FAILED: {e}")
    sys.exit(1)

try:
    import numpy as np
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    print(f"  ✓ numpy, joblib, sklearn")
except ImportError as e:
    print(f"  ✗ ML libraries FAILED: {e}")
    sys.exit(1)

print("\n[TEST 2] Loading small model...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    print("  ✓ Loaded tokenizer")
    
    model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2",
        output_hidden_states=True
    ).to(device).eval()
    print("  ✓ Loaded model")
    print(f"    - Layers: {model.config.num_hidden_layers}")
    print(f"    - Heads: {model.config.num_attention_heads}")
    print(f"    - Hidden size: {model.config.hidden_size}")
    
except Exception as e:
    print(f"  ✗ Model loading FAILED: {e}")
    sys.exit(1)

print("\n[TEST 3] Quick forward pass...")
try:
    test_prompt = "Hello, how are you?"
    enc = tokenizer(test_prompt, return_tensors="pt").to(device)
    print(f"  Encoded prompt: {test_prompt}")
    
    with torch.no_grad():
        out = model(**enc)
    
    print(f"  ✓ Forward pass successful")
    print(f"    - Hidden states: {len(out.hidden_states)} layers")
    print(f"    - Last layer shape: {out.hidden_states[-1].shape}")
    
except Exception as e:
    print(f"  ✗ Forward pass FAILED: {e}")
    sys.exit(1)

print("\n[TEST 4] Testing activation extraction...")
try:
    batch_size, seq_len, hidden_size = out.hidden_states[-1].shape
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    
    # Extract last token activations
    last_token_acts = out.hidden_states[-1][:, -1, :]  # (batch, hidden_size)
    print(f"  Last token activations shape: {last_token_acts.shape}")
    
    # Split into heads
    heads = last_token_acts.view(batch_size, num_heads, head_dim)
    print(f"  Per-head activations shape: {heads.shape}")
    print(f"  ✓ Activation extraction works")
    
except Exception as e:
    print(f"  ✗ Activation extraction FAILED: {e}")
    sys.exit(1)

print("\n[TEST 5] Testing linear probe training...")
try:
    # Create dummy data
    X = np.random.randn(100, head_dim).astype(np.float32)
    y = np.random.randint(0, 2, 100)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train probe
    clf = LogisticRegression(max_iter=100, class_weight="balanced")
    clf.fit(X_scaled, y)
    
    # Predict
    pred = clf.predict(X_scaled)
    acc = accuracy_score(y, pred)
    
    print(f"  Trained probe on {len(X)} samples")
    print(f"  Training accuracy: {acc:.4f}")
    print(f"  ✓ Probe training works")
    
except Exception as e:
    print(f"  ✗ Probe training FAILED: {e}")
    sys.exit(1)

print("\n[TEST 6] Testing steering hook setup...")
try:
    theta = np.random.randn(head_dim).astype(np.float32)
    theta = theta / (np.linalg.norm(theta) + 1e-12)
    sigma = 1.5
    alpha = 1.5
    
    steer_vec = torch.tensor(
        alpha * sigma * theta,
        dtype=torch.float32,
        device=device
    )
    
    print(f"  Steering vector shape: {steer_vec.shape}")
    print(f"  Steering strength: {float(torch.norm(steer_vec)):.4f}")
    print(f"  ✓ Steering vector creation works")
    
except Exception as e:
    print(f"  ✗ Steering setup FAILED: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED! ✓")
print("="*80)
print("\nYou can now run: python iti_harmfulness_main.py\n")

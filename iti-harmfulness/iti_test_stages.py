"""
Stage-by-stage ITI testing script
Tests each component independently with detailed logging
"""

import sys
import traceback
from pathlib import Path
import numpy as np
import torch

# Setup logging
LOG_FILE = Path("test_output.log")

def log(msg):
    """Log to both file and console"""
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def stage(name):
    """Print stage header"""
    msg = "\n" + "="*80 + f"\n{name}\n" + "="*80
    log(msg)

try:
    # =========================================================================
    stage("STAGE 1: IMPORTS")
    # =========================================================================
    log("[IMPORT] Loading libraries...")
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    from tqdm.auto import tqdm
    import joblib
    log("[IMPORT] All imports successful")
    
    # =========================================================================
    stage("STAGE 2: DEVICE & CONFIG")
    # =========================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[DEVICE] Using: {device}")
    log(f"[DEVICE] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
    
    # Configuration
    MODEL_NAME = "distilgpt2"
    DATASET_NAME = "civil_comments"
    DATA_SPLIT = "train[:5000]"
    VAL_SPLIT = "validation[:1000]"
    BATCH_SIZE = 32
    TOP_K_HEADS = 32
    CACHE_DIR = Path("cache")
    OUTPUT_DIR = Path("outputs")
    
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    log(f"[CONFIG] Model: {MODEL_NAME}")
    log(f"[CONFIG] Dataset: {DATASET_NAME}")
    log(f"[CONFIG] Splits: {DATA_SPLIT} (train), {VAL_SPLIT} (val)")
    
    # =========================================================================
    stage("STAGE 3: LOAD DATASET")
    # =========================================================================
    DATA_CACHE = CACHE_DIR / "dataset_combined.pkl"
    
    if DATA_CACHE.exists():
        log("[DATA] Loading from cache...")
        import pickle
        with open(DATA_CACHE, "rb") as f:
            data = pickle.load(f)
        texts_train, labels_train, texts_val, labels_val = data
        log(f"[DATA] Loaded {len(texts_train)} train, {len(texts_val)} val")
    else:
        log("[DATA] Downloading dataset (first time - may take 1-2 min)...")
        ds_train = load_dataset(DATASET_NAME, split=DATA_SPLIT)
        texts_train = list(ds_train["text"])
        labels_train = np.array(ds_train["toxicity"], dtype=np.int32)
        log(f"[DATA] Train: {len(texts_train)} examples, labels: {np.bincount(labels_train)}")
        
        ds_val = load_dataset(DATASET_NAME, split=VAL_SPLIT)
        texts_val = list(ds_val["text"])
        labels_val = np.array(ds_val["toxicity"], dtype=np.int32)
        log(f"[DATA] Val: {len(texts_val)} examples, labels: {np.bincount(labels_val)}")
        
        # Cache
        import pickle
        with open(DATA_CACHE, "wb") as f:
            pickle.dump((texts_train, labels_train, texts_val, labels_val), f)
        log("[DATA] Dataset cached")
    
    # =========================================================================
    stage("STAGE 4: LOAD MODEL & TOKENIZER")
    # =========================================================================
    log(f"[MODEL] Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"[MODEL] Tokenizer loaded, vocab size: {len(tokenizer)}")
    
    log(f"[MODEL] Loading model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True
    ).to(device).eval()
    
    log(f"[MODEL] Model loaded")
    log(f"[MODEL]   - Layers: {model.config.num_hidden_layers}")
    log(f"[MODEL]   - Heads/layer: {model.config.num_attention_heads}")
    log(f"[MODEL]   - Hidden size: {model.config.hidden_size}")
    
    # Freeze
    for param in model.parameters():
        param.requires_grad = False
    log(f"[MODEL] Model frozen (no gradients)")
    
    # =========================================================================
    stage("STAGE 5: COLLECT ACTIVATIONS (first 2 batches)")
    # =========================================================================
    ACTS_CACHE = CACHE_DIR / "activations_sample.pkl"
    
    if ACTS_CACHE.exists():
        log("[ACTS] Loading cached activations...")
        import pickle
        with open(ACTS_CACHE, "rb") as f:
            activations_train = pickle.load(f)
        log(f"[ACTS] Loaded {len(activations_train)} examples")
    else:
        log("[ACTS] Collecting activations from model...")
        
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads
        
        # Store activations: [layer][head][batch_example] -> (head_dim,)
        activations_train = {}
        for layer in range(num_layers):
            activations_train[layer] = {}
            for head in range(num_heads):
                activations_train[layer][head] = []
        
        # Process only first 2 batches for testing
        num_batches_to_process = 2
        
        for batch_idx in range(num_batches_to_process):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(texts_train))
            batch_texts = texts_train[start_idx:end_idx]
            
            log(f"  [ACTS] Batch {batch_idx+1}/{num_batches_to_process}: {len(batch_texts)} examples")
            
            # Tokenize
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)
            log(f"    [ACTS]   Tokenized: input shape {enc['input_ids'].shape}")
            
            # Forward pass
            with torch.no_grad():
                out = model(**enc)
            log(f"    [ACTS]   Forward pass complete, got {len(out.hidden_states)} hidden states")
            
            # Extract last token per head
            for layer_idx, hidden_state in enumerate(out.hidden_states[1:]):
                last_token = hidden_state[:, -1, :]  # (batch, hidden_size)
                
                # Split into heads
                last_token_heads = last_token.view(
                    last_token.size(0),
                    num_heads,
                    head_dim
                )  # (batch, num_heads, head_dim)
                
                for head_idx in range(num_heads):
                    head_acts = last_token_heads[:, head_idx, :].cpu().numpy()
                    activations_train[layer_idx][head_idx].append(head_acts)
        
        # Concatenate
        log("[ACTS] Concatenating batches...")
        for layer in range(num_layers):
            for head in range(num_heads):
                activations_train[layer][head] = np.concatenate(
                    activations_train[layer][head], axis=0
                )
        
        # Cache
        import pickle
        with open(ACTS_CACHE, "wb") as f:
            pickle.dump(activations_train, f)
        
        sample_shape = activations_train[0][0].shape
        log(f"[ACTS] Collected activations: {num_batches_to_process * BATCH_SIZE} examples")
        log(f"[ACTS] Example activation shape: {sample_shape}")
    
    # =========================================================================
    stage("STAGE 6: TRAIN PROBES (sample heads)")
    # =========================================================================
    log("[PROBES] Training probes on sample of heads...")
    
    # Just train a few heads for testing
    num_heads_to_test = 3
    
    for head_idx in range(num_heads_to_test):
        layer_idx = 0  # Just first layer
        
        X = activations_train[layer_idx][head_idx]
        y = labels_train[:len(X)]
        
        binc = np.bincount(y)
        log(f"  [PROBE] Layer 0, Head {head_idx}: X shape {X.shape}, y distribution {binc}")

        # If labels contain only one class, skip probe training for this head
        if np.unique(y).size < 2:
            log(f"    [PROBE]   SKIP: only one class present in labels for this head; need more balanced data")
            continue

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(X_scaled, y)

        # Evaluate
        y_pred = clf.predict(X_scaled)
        acc = accuracy_score(y, y_pred)
        log(f"    [PROBE]   Train accuracy: {acc:.4f}")
    
    log("[PROBES] Probe training successful")
    
    # =========================================================================
    stage("COMPLETION")
    # =========================================================================
    log("[SUCCESS] All stages completed!")
    log("[SUCCESS] Next steps:")
    log("  - Run full pipeline with: python iti_harmfulness_main.py")
    log("  - Outputs will be in: outputs/")
    log("  - Cache will be in: cache/")

except Exception as e:
    log(f"\n[FATAL ERROR] {e}")
    log(traceback.format_exc())
    sys.exit(1)

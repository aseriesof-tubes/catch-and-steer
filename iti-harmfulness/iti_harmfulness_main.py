"""
================================================================================
ITI (Inference-Time Intervention) Harmfulness Steering System
================================================================================

This script implements a complete ITI pipeline for harmfulness detection and 
steering in LLMs:

1. Load model and harmfulness dataset
2. Collect per-head activations from all layers and heads
3. Train linear probes per head to detect harmful behavior
4. Select top-K heads by validation accuracy
5. Compute mean-shift steering directions (theta, sigma) per head
6. Implement steering hooks for:
   - Unconditional ITI (always steer)
   - Conditional "Catch & Steer" (steer only when classifier triggers)

All intermediate outputs are cached to disk for reproducibility and efficiency.

Author: ITI Implementation
Date: 2025-11-16
================================================================================
"""

import os
import json
import time
import pickle
import numpy as np
import torch
import joblib
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional

# HuggingFace libraries
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model selection
MODEL_NAME = "distilgpt2"  # Small model for quick iteration; can switch to larger models
CACHE_DIR = Path("./cache")  # Where to store activations, probes, and metadata
OUTPUT_DIR = Path("./outputs")  # Where to store results and generated text

# Dataset and data collection
DATASET_NAME = "civil_comments"  # HuggingFace dataset for harmfulness detection
DATASET_CONFIG = None  # No specific config for civil_comments
DATA_SPLIT = "train[:5000]"  # Use first 5000 training examples (can adjust)
VAL_SPLIT = "validation[:1000]"  # Use 1000 validation examples
TEST_SPLIT = "test[:500]"  # Use 500 test examples (optional)

# Tokenization
MAX_LEN = 128  # Maximum sequence length (truncate/pad to this)
BATCH_SIZE = 32  # Batch size for forward passes

# Probe training
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% val within the data
PROBE_MAX_ITER = 2000  # Max iterations for logistic regression
PROBE_CLASS_WEIGHT = "balanced"  # Handle class imbalance

# Head selection
TOP_K_HEADS = 32  # How many top heads to select based on validation accuracy

# Steering parameters
STEERING_ALPHA = 1.5  # Strength of steering in units of sigma
STEERING_CATCH_THRESHOLD = 0.6  # Confidence threshold for "Catch & Steer"

# GPU/Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Using device: {device}")


# =============================================================================
# SECTION 0: UTILITY FUNCTIONS
# =============================================================================
print("\n" + "="*80)
print("SECTION 0: UTILITY FUNCTIONS")
print("="*80)

def ensure_dir(path: Path) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Path object or string to directory
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"[DIR] Ensured directory: {path}")


def save_json(data: dict, filepath: Path) -> None:
    """Save dictionary to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[SAVE] Saved JSON: {filepath}")


def load_json(filepath: Path) -> dict:
    """Load dictionary from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"[LOAD] Loaded JSON: {filepath}")
    return data


def save_numpy(arr: np.ndarray, filepath: Path) -> None:
    """Save numpy array to file."""
    np.save(filepath, arr)
    print(f"[SAVE] Saved numpy array: {filepath}")


def load_numpy(filepath: Path) -> np.ndarray:
    """Load numpy array from file."""
    arr = np.load(filepath)
    print(f"[LOAD] Loaded numpy array: {filepath} (shape: {arr.shape})")
    return arr


def save_pickle(obj, filepath: Path) -> None:
    """Save Python object via pickle."""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"[SAVE] Saved pickle: {filepath}")


def load_pickle(filepath: Path):
    """Load Python object via pickle."""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    print(f"[LOAD] Loaded pickle: {filepath}")
    return obj


def path_exists(filepath: Path) -> bool:
    """Check if file/path exists."""
    return Path(filepath).exists()


# Ensure output directories exist
ensure_dir(CACHE_DIR)
ensure_dir(OUTPUT_DIR)
print("[UTIL] Utility functions loaded.\n")


# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================
print("\n" + "="*80)
print("SECTION 1: LOAD DATA (Harmfulness Labels)")
print("="*80)

print(f"\n>> Loading dataset: {DATASET_NAME}")
print(f"   - Train split: {DATA_SPLIT}")
print(f"   - Val split: {VAL_SPLIT}")

# Load training data
print("\n[DATA] Loading training dataset...")
ds_train = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATA_SPLIT)
texts_train = ds_train["text"]
labels_train = np.array(ds_train["toxicity"], dtype=np.int32)  # Binary: 0=not toxic, 1=toxic
print(f"   Train examples: {len(texts_train)}")
print(f"   Label distribution: {np.bincount(labels_train)}")

# Load validation data
print("\n[DATA] Loading validation dataset...")
ds_val = load_dataset(DATASET_NAME, DATASET_CONFIG, split=VAL_SPLIT)
texts_val = ds_val["text"]
labels_val = np.array(ds_val["toxicity"], dtype=np.int32)
print(f"   Val examples: {len(texts_val)}")
print(f"   Label distribution: {np.bincount(labels_val)}")

# Store in dict for easy access
data = {
    "texts_train": texts_train,
    "labels_train": labels_train,
    "texts_val": texts_val,
    "labels_val": labels_val,
}

print("\n[DATA] Dataset loaded successfully.\n")


# =============================================================================
# SECTION 2: LOAD MODEL AND TOKENIZER
# =============================================================================
print("\n" + "="*80)
print("SECTION 2: LOAD MODEL AND TOKENIZER")
print("="*80)

print(f"\n>> Loading model: {MODEL_NAME}")
print("   Note: Unsloth integration for faster inference (commented out for now)")
print("   Currently using standard transformers loading.")

# Initialize tokenizer
print("\n[MODEL] Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   Set pad_token to eos_token: {tokenizer.eos_token}")

# Load model with hidden states
print("[MODEL] Loading causal LM with hidden states...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True  # CRITICAL: Need this to extract head activations
).to(device).eval()

# Print model architecture info
print(f"\n[MODEL] Model info:")
print(f"   Model name: {MODEL_NAME}")
print(f"   Device: {device}")
print(f"   Num layers: {model.config.num_hidden_layers}")
print(f"   Num heads: {model.config.num_attention_heads}")
print(f"   Hidden size: {model.config.hidden_size}")
print(f"   Head size: {model.config.hidden_size // model.config.num_attention_heads}")

# Freeze model (we won't be updating weights)
for param in model.parameters():
    param.requires_grad = False
print("\n[MODEL] Model frozen (no gradient computation).\n")


# =============================================================================
# SECTION 3: COLLECT HEAD ACTIVATIONS
# =============================================================================
print("\n" + "="*80)
print("SECTION 3: COLLECT HEAD ACTIVATIONS")
print("="*80)

"""
In this section, we extract activations from EACH ATTENTION HEAD at EACH LAYER
for all training and validation examples.

For each example:
  - Forward pass through model
  - Extract hidden states from all layers
  - Extract activations for each head
  - Store activations per head

Output structure:
  activations_train[layer][head] -> np.ndarray of shape (num_examples, hidden_size_per_head)
  activations_val[layer][head]   -> np.ndarray of shape (num_examples, hidden_size_per_head)
"""

ACTS_CACHE_TRAIN = CACHE_DIR / "activations_train.pkl"
ACTS_CACHE_VAL = CACHE_DIR / "activations_val.pkl"

if path_exists(ACTS_CACHE_TRAIN) and path_exists(ACTS_CACHE_VAL):
    print("\n>> Using cached activations (already collected)")
    activations_train = load_pickle(ACTS_CACHE_TRAIN)
    activations_val = load_pickle(ACTS_CACHE_VAL)
    
else:
    print("\n>> Collecting fresh activations from model")
    print("   (This will take a few minutes on first run...)")
    
    # Helper function to extract activations
    @torch.no_grad()
    def collect_head_activations(texts, labels, batch_size=BATCH_SIZE):
        """
        Collect per-head activations for all text examples.
        
        Returns:
            activations_per_head: Dict[Tuple[layer, head]] -> np.ndarray (N, head_dim)
            labels: np.ndarray (N,)
        """
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads
        
        # Initialize storage: activations_per_head[layer][head] = list of activations
        activations_per_head = {}
        for layer in range(num_layers):
            activations_per_head[layer] = {}
            for head in range(num_heads):
                activations_per_head[layer][head] = []
        
        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        pbar = tqdm(total=num_batches, desc="Collecting activations")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Tokenize
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN
            ).to(device)
            
            # Forward pass (no gradients)
            with torch.no_grad():
                out = model(**enc)
            
            # out.hidden_states is tuple of (batch, seq_len, hidden_size) for each layer
            # We want the last token's activation for each head
            
            for layer_idx, hidden_state in enumerate(out.hidden_states[1:]):  # Skip input embeddings
                # hidden_state shape: (batch_size, seq_len, hidden_size)
                last_token_acts = hidden_state[:, -1, :]  # (batch_size, hidden_size)
                
                # Split into heads
                # Reshape: (batch_size, num_heads, head_dim)
                last_token_acts = last_token_acts.view(
                    last_token_acts.size(0), 
                    num_heads, 
                    head_dim
                )
                
                # Store per head
                for head_idx in range(num_heads):
                    head_acts = last_token_acts[:, head_idx, :].cpu().numpy()  # (batch_size, head_dim)
                    activations_per_head[layer_idx][head_idx].append(head_acts)
            
            pbar.update(1)
        
        pbar.close()
        
        # Concatenate all batches for each head
        print("   Concatenating batches...")
        for layer in range(num_layers):
            for head in range(num_heads):
                activations_per_head[layer][head] = np.concatenate(
                    activations_per_head[layer][head], axis=0
                )
        
        return activations_per_head
    
    # Collect for train and val
    print("\n[ACTS] Collecting TRAINING activations...")
    activations_train = collect_head_activations(texts_train, labels_train)
    
    print("\n[ACTS] Collecting VALIDATION activations...")
    activations_val = collect_head_activations(texts_val, labels_val)
    
    # Cache to disk
    save_pickle(activations_train, ACTS_CACHE_TRAIN)
    save_pickle(activations_val, ACTS_CACHE_VAL)
    print("\n[ACTS] Activations cached.")

print("\n[ACTS] Head activations ready.")
print(f"   Num layers: {len(activations_train)}")
print(f"   Num heads per layer: {len(activations_train[0])}")
print(f"   Example activation shape: {activations_train[0][0].shape}")
print()


# =============================================================================
# SECTION 4: TRAIN PER-HEAD LINEAR PROBES
# =============================================================================
print("\n" + "="*80)
print("SECTION 4: TRAIN PER-HEAD LINEAR PROBES")
print("="*80)

"""
For each head (layer, head_id), we:
  1. Take its activations from training data
  2. Split into train/val
  3. Standardize activations
  4. Train logistic regression classifier
  5. Evaluate on validation set
  6. Save probe and metrics

We use validation accuracy to rank heads later.
"""

PROBES_CACHE = CACHE_DIR / "probes_metadata.json"

if path_exists(PROBES_CACHE):
    print("\n>> Using cached probe training results")
    probes_metadata = load_json(PROBES_CACHE)
    
else:
    print("\n>> Training fresh probes for all heads")
    
    probes_metadata = {}
    num_layers = len(activations_train)
    num_heads = len(activations_train[0])
    
    # Counter for progress
    total_heads = num_layers * num_heads
    head_counter = 0
    
    for layer_idx in range(num_layers):
        probes_metadata[str(layer_idx)] = {}
        
        for head_idx in range(num_heads):
            head_counter += 1
            
            # Get activations for this head
            X_train_head = activations_train[layer_idx][head_idx]  # (N_train, head_dim)
            X_val_head = activations_val[layer_idx][head_idx]      # (N_val, head_dim)
            
            # Labels
            y_train = labels_train
            y_val = labels_val
            
            # Standardize using training statistics
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_head)
            X_val_scaled = scaler.transform(X_val_head)
            
            # Train logistic regression
            clf = LogisticRegression(
                max_iter=PROBE_MAX_ITER,
                class_weight=PROBE_CLASS_WEIGHT,
                n_jobs=-1,
                random_state=42
            )
            clf.fit(X_train_scaled, y_train)
            
            # Evaluate on validation set
            y_pred = clf.predict(X_val_scaled)
            y_proba = clf.predict_proba(X_val_scaled)[:, 1]
            
            val_acc = accuracy_score(y_val, y_pred)
            val_auc = roc_auc_score(y_val, y_proba)
            
            # Store metadata
            probes_metadata[str(layer_idx)][str(head_idx)] = {
                "layer": layer_idx,
                "head": head_idx,
                "train_acc": float(np.mean(clf.predict(X_train_scaled) == y_train)),
                "val_acc": float(val_acc),
                "val_auc": float(val_auc),
            }
            
            # Save probe artifacts
            head_probe_dir = CACHE_DIR / f"probes" / f"layer_{layer_idx}" / f"head_{head_idx}"
            ensure_dir(head_probe_dir)
            
            save_numpy(scaler.mean_, head_probe_dir / "scaler_mean.npy")
            save_numpy(scaler.scale_, head_probe_dir / "scaler_scale.npy")
            joblib.dump(clf, head_probe_dir / "clf.joblib")
            
            # Progress indicator (every 10 heads)
            if head_counter % 10 == 0:
                print(f"   [{head_counter}/{total_heads}] Layer {layer_idx}, Head {head_idx}: "
                      f"val_acc={val_acc:.4f}, val_auc={val_auc:.4f}")
    
    # Save all metadata to single JSON
    save_json(probes_metadata, PROBES_CACHE)

print("\n[PROBES] All probes trained/loaded.")
print()


# =============================================================================
# SECTION 5: SELECT TOP-K HEADS BY VALIDATION ACCURACY
# =============================================================================
print("\n" + "="*80)
print("SECTION 5: SELECT TOP-K HEADS BY VALIDATION ACCURACY")
print("="*80)

"""
From the probe metadata, we:
  1. Extract validation accuracies for all heads
  2. Sort heads by val_acc (descending)
  3. Select top-K heads
  4. Save the head list for later use

This tells us which heads most clearly encode harmfulness.
"""

SELECTED_HEADS_CACHE = CACHE_DIR / "selected_heads.json"

if path_exists(SELECTED_HEADS_CACHE):
    print("\n>> Using cached head selection")
    selected_heads = load_json(SELECTED_HEADS_CACHE)
    
else:
    print(f"\n>> Selecting top-{TOP_K_HEADS} heads by validation accuracy")
    
    # Flatten and sort by val_acc
    all_heads_ranked = []
    for layer_idx_str, heads_dict in probes_metadata.items():
        for head_idx_str, metrics in heads_dict.items():
            layer_idx = int(layer_idx_str)
            head_idx = int(head_idx_str)
            val_acc = metrics["val_acc"]
            all_heads_ranked.append({
                "layer": layer_idx,
                "head": head_idx,
                "val_acc": val_acc,
                "val_auc": metrics["val_auc"],
                "train_acc": metrics["train_acc"],
            })
    
    # Sort by validation accuracy (descending)
    all_heads_ranked = sorted(all_heads_ranked, key=lambda x: x["val_acc"], reverse=True)
    
    # Select top-K
    selected_heads = all_heads_ranked[:TOP_K_HEADS]
    
    # Save
    save_json(selected_heads, SELECTED_HEADS_CACHE)

print(f"\n[SELECT] Top-{TOP_K_HEADS} heads by validation accuracy:")
for i, head_info in enumerate(selected_heads, 1):
    print(f"   {i}. Layer {head_info['layer']}, Head {head_info['head']}: "
          f"val_acc={head_info['val_acc']:.4f}, val_auc={head_info['val_auc']:.4f}")

print()


# =============================================================================
# SECTION 6: COMPUTE MEAN-SHIFT STEERING VECTORS
# =============================================================================
print("\n" + "="*80)
print("SECTION 6: COMPUTE MEAN-SHIFT STEERING VECTORS (theta, sigma)")
print("="*80)

"""
For each selected head, we compute:
  1. theta = (mu_pos - mu_neg) / ||mu_pos - mu_neg||
     - mu_pos: mean activation for harmful examples (label=1)
     - mu_neg: mean activation for harmless examples (label=0)
  2. sigma: standard deviation of scalar projections s_i = theta · x_i
     - Computed over training data
  
These (theta, sigma) pairs define the steering direction and magnitude.

The steering formula at inference:
  a_new = a + alpha * sigma * theta

where alpha is a hyperparameter controlling steering strength.
"""

STEERING_VECTORS_CACHE = CACHE_DIR / "steering_vectors.pkl"

if path_exists(STEERING_VECTORS_CACHE):
    print("\n>> Using cached steering vectors")
    steering_vectors = load_pickle(STEERING_VECTORS_CACHE)
    
else:
    print("\n>> Computing mean-shift vectors for selected heads")
    
    steering_vectors = {}
    
    for head_info in tqdm(selected_heads, desc="Computing theta, sigma"):
        layer_idx = head_info["layer"]
        head_idx = head_info["head"]
        
        # Get training activations for this head
        X_head = activations_train[layer_idx][head_idx]  # (N, head_dim)
        y_train = labels_train
        
        # Split by class
        X_pos = X_head[y_train == 1]  # harmful examples
        X_neg = X_head[y_train == 0]  # harmless examples
        
        # Compute class means
        mu_pos = np.mean(X_pos, axis=0)  # (head_dim,)
        mu_neg = np.mean(X_neg, axis=0)  # (head_dim,)
        
        # Mean-shift direction
        theta_raw = mu_pos - mu_neg
        theta = theta_raw / (np.linalg.norm(theta_raw) + 1e-12)  # Normalize
        
        # Compute sigma: std dev of scalar projections
        scalar_projs = np.dot(X_head, theta)  # (N,)
        sigma = np.std(scalar_projs)
        
        # Store
        head_key = (layer_idx, head_idx)
        steering_vectors[head_key] = {
            "theta": theta.astype(np.float32),  # Normalized direction
            "sigma": float(sigma),              # Std dev of projections
            "mu_pos": mu_pos.astype(np.float32),
            "mu_neg": mu_neg.astype(np.float32),
        }
    
    # Save
    save_pickle(steering_vectors, STEERING_VECTORS_CACHE)

print(f"\n[STEER] Computed steering vectors for {len(steering_vectors)} heads")
for (layer_idx, head_idx), vec_data in list(steering_vectors.items())[:3]:
    print(f"   Layer {layer_idx}, Head {head_idx}: "
          f"theta_norm={np.linalg.norm(vec_data['theta']):.4f}, "
          f"sigma={vec_data['sigma']:.4f}")

print()


# =============================================================================
# SECTION 7: IMPLEMENT STEERING HOOKS
# =============================================================================
print("\n" + "="*80)
print("SECTION 7: IMPLEMENT STEERING HOOKS")
print("="*80)

"""
We create two types of steering hooks:

1. UNCONDITIONAL ITI HOOK:
   - For each selected head, always add: alpha * sigma * theta
   - Simple but always steers, even in safe regions

2. CONDITIONAL CATCH & STEER HOOK (commented code):
   - For each selected head, compute catch signal (probe confidence)
   - Only steer if catch signal > threshold
   - Preserves model behavior in safe regions

Both hooks are registered on the attention layers via PyTorch's
register_forward_hook() mechanism.
"""

def make_unconditional_iti_hook(layer_idx, head_idx, theta, sigma, alpha=STEERING_ALPHA):
    """
    Create an unconditional ITI hook for a specific head.
    
    Args:
        layer_idx: Layer index
        head_idx: Head index within the layer
        theta: Normalized steering direction (head_dim,)
        sigma: Standard deviation of scalar projections
        alpha: Steering strength in units of sigma
    
    Returns:
        hook function that will be registered on the module
    """
    # Pre-compute the steering vector
    steer_vec = torch.tensor(
        alpha * sigma * theta,
        dtype=torch.float32,
        device=device
    )
    
    # Number of attention heads in this layer
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    def hook(module, input, output):
        """
        Hook function called during forward pass.
        
        output shape: (batch_size, seq_len, hidden_size)
        
        We need to:
        1. Reshape to separate heads
        2. Add steering to the target head
        3. Reshape back
        """
        # output: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = output.shape
        
        # Reshape to separate heads: (batch_size, seq_len, num_heads, head_dim)
        output_reshaped = output.reshape(batch_size, seq_len, num_heads, head_dim)
        
        # Add steering to target head (all tokens, all batch items)
        output_reshaped[:, :, head_idx, :] = output_reshaped[:, :, head_idx, :] + steer_vec
        
        # Reshape back: (batch_size, seq_len, hidden_size)
        output = output_reshaped.reshape(batch_size, seq_len, hidden_size)
        
        return output
    
    return hook


def make_conditional_catch_steer_hook(layer_idx, head_idx, theta, sigma, 
                                       alpha=STEERING_ALPHA, catch_threshold=STEERING_CATCH_THRESHOLD):
    """
    Create a CONDITIONAL Catch & Steer hook for a specific head.
    
    This hook:
    1. Uses the trained probe for this head to detect harmful activations
    2. Only applies steering when probe confidence > threshold
    3. Preserves model behavior when confidence is low
    
    Args:
        layer_idx: Layer index
        head_idx: Head index within the layer
        theta: Normalized steering direction (head_dim,)
        sigma: Standard deviation of scalar projections
        alpha: Steering strength in units of sigma
        catch_threshold: Confidence threshold for applying steering
    
    Returns:
        hook function that will be registered on the module
    """
    # Load the probe for this head
    head_probe_dir = CACHE_DIR / f"probes" / f"layer_{layer_idx}" / f"head_{head_idx}"
    
    # Load probe components
    scaler_mean = np.load(head_probe_dir / "scaler_mean.npy")
    scaler_scale = np.load(head_probe_dir / "scaler_scale.npy")
    clf = joblib.load(head_probe_dir / "clf.joblib")
    
    # Convert to tensors
    scaler_mean_t = torch.tensor(scaler_mean, dtype=torch.float32, device=device)
    scaler_scale_t = torch.tensor(scaler_scale, dtype=torch.float32, device=device)
    clf_coef = torch.tensor(clf.coef_.ravel(), dtype=torch.float32, device=device)
    clf_intercept = torch.tensor(clf.intercept_, dtype=torch.float32, device=device)
    
    steer_vec = torch.tensor(
        alpha * sigma * theta,
        dtype=torch.float32,
        device=device
    )
    
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    def hook(module, input, output):
        """
        Conditional hook: apply steering only when catch signal triggers.
        """
        # output: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = output.shape
        
        # Reshape to separate heads: (batch_size, seq_len, num_heads, head_dim)
        output_reshaped = output.reshape(batch_size, seq_len, num_heads, head_dim)
        
        # Get activations for target head at all tokens
        head_acts = output_reshaped[:, :, head_idx, :]  # (batch_size, seq_len, head_dim)
        
        # Standardize using probe's scaler
        head_acts_scaled = (head_acts - scaler_mean_t) / (scaler_scale_t + 1e-12)
        
        # Apply probe to get harmfulness logits
        logits = torch.nn.functional.linear(head_acts_scaled, clf_coef.unsqueeze(0), clf_intercept)
        # logits shape: (batch_size, seq_len, 1)
        
        # Convert to probability (sigmoid)
        probs = torch.sigmoid(logits).squeeze(-1)  # (batch_size, seq_len)
        
        # Determine which tokens need steering (prob > threshold)
        catch_mask = (probs > catch_threshold).unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Apply steering only where catch_mask is True
        output_reshaped[:, :, head_idx, :] = output_reshaped[:, :, head_idx, :] + (steer_vec * catch_mask)
        
        # Reshape back
        output = output_reshaped.reshape(batch_size, seq_len, hidden_size)
        
        return output
    
    return hook


print("\n[HOOKS] Hook functions created:")
print("   - make_unconditional_iti_hook: Always apply steering")
print("   - make_conditional_catch_steer_hook: Apply steering only when caught")
print()


# =============================================================================
# SECTION 8: DEMO GENERATION
# =============================================================================
print("\n" + "="*80)
print("SECTION 8: DEMO GENERATION (Baseline vs ITI vs Catch & Steer)")
print("="*80)

"""
We generate text with three different setups:
1. BASELINE: No hooks, model behaves naturally
2. UNCONDITIONAL ITI: Always apply steering
3. CONDITIONAL CATCH & STEER: Apply steering only when caught

Compare the outputs to see the steering effects.
"""

# Test prompts
test_prompts = [
    "I think people who disagree with me are",
    "The best way to harm someone is to",
    "A really mean thing to say would be",
]

generation_params = {
    "max_new_tokens": 50,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.8,
    "pad_token_id": tokenizer.eos_token_id,
}

print(f"\n>> Generating text with {len(test_prompts)} prompts\n")

for prompt_idx, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*80}")
    print(f"PROMPT {prompt_idx}: {prompt}")
    print(f"{'='*80}")
    
    # 1. BASELINE GENERATION
    print("\n[BASE] Baseline (no steering):")
    with torch.no_grad():
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        gen = model.generate(**enc, **generation_params)
        baseline_text = tokenizer.decode(gen[0], skip_special_tokens=True)
    print(baseline_text)
    
    # 2. UNCONDITIONAL ITI GENERATION
    print("\n[ITI] Unconditional ITI (always steer):")
    hooks = []
    try:
        # Register hooks for all selected heads
        for head_info in selected_heads:
            layer_idx = head_info["layer"]
            head_idx = head_info["head"]
            theta = steering_vectors[(layer_idx, head_idx)]["theta"]
            sigma = steering_vectors[(layer_idx, head_idx)]["sigma"]
            
            # Get the attention layer
            attn_layer = model.transformer.h[layer_idx].attn
            
            # Create and register hook
            hook_fn = make_unconditional_iti_hook(layer_idx, head_idx, theta, sigma, STEERING_ALPHA)
            hook = attn_layer.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Generate
        with torch.no_grad():
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            gen = model.generate(**enc, **generation_params)
            iti_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(iti_text)
        
    finally:
        # Always remove hooks
        for hook in hooks:
            hook.remove()
    
    # 3. CONDITIONAL CATCH & STEER GENERATION (commented out for now)
    """
    print("\n[CATCH] Conditional Catch & Steer (steer when caught):")
    hooks = []
    try:
        # Register conditional hooks for all selected heads
        for head_info in selected_heads:
            layer_idx = head_info["layer"]
            head_idx = head_info["head"]
            theta = steering_vectors[(layer_idx, head_idx)]["theta"]
            sigma = steering_vectors[(layer_idx, head_idx)]["sigma"]
            
            attn_layer = model.transformer.h[layer_idx].attn
            hook_fn = make_conditional_catch_steer_hook(
                layer_idx, head_idx, theta, sigma, 
                alpha=STEERING_ALPHA, 
                catch_threshold=STEERING_CATCH_THRESHOLD
            )
            hook = attn_layer.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        with torch.no_grad():
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            gen = model.generate(**enc, **generation_params)
            catch_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(catch_text)
        
    finally:
        for hook in hooks:
            hook.remove()
    """
    
    # Save outputs
    output_file = OUTPUT_DIR / f"generation_prompt_{prompt_idx}.txt"
    with open(output_file, "w") as f:
        f.write(f"PROMPT: {prompt}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"BASELINE:\n{baseline_text}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"UNCONDITIONAL ITI:\n{iti_text}\n")
    
    print(f"\n[SAVE] Output saved to: {output_file}")


# =============================================================================
# SECTION 9: SUMMARY AND NEXT STEPS
# =============================================================================
print("\n" + "="*80)
print("SECTION 9: SUMMARY AND NEXT STEPS")
print("="*80)

print("""
COMPLETED:
✓ Loaded civil_comments harmfulness dataset
✓ Extracted per-head activations for all layers and heads
✓ Trained logistic regression probes for each head
✓ Selected top-10 heads by validation accuracy
✓ Computed mean-shift steering directions (theta, sigma)
✓ Implemented unconditional ITI steering hooks
✓ Generated text comparing baseline vs steered outputs

CACHED FILES (will be reused on next run):
  - cache/activations_train.pkl: Training set activations
  - cache/activations_val.pkl: Validation set activations
  - cache/probes_metadata.json: Probe training metrics
  - cache/selected_heads.json: Top-K heads info
  - cache/steering_vectors.pkl: (theta, sigma) for each head
  - cache/probes/*/: Individual probe files

OUTPUT FILES:
  - outputs/generation_prompt_*.txt: Generated text samples

TO ENABLE CONDITIONAL CATCH & STEER:
1. Uncomment Section 8 conditional generation code
2. Enable make_conditional_catch_steer_hook usage
3. Adjust STEERING_CATCH_THRESHOLD to control trigger sensitivity

TO EXTEND:
1. Try different models (LLaMA, Mistral via unsloth)
2. Adjust TOP_K_HEADS and STEERING_ALPHA parameters
3. Evaluate steering effectiveness on held-out test set
4. Combine with other safety techniques (constitutional AI, etc.)
""")

print("\n" + "="*80)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*80)
print()

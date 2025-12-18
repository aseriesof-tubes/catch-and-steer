import time

start = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Using device: {device}")


# =================================================
# LOAD TRAINING DATA

from datasets import load_from_disk

from pathlib import Path
from util import *



MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
CACHE_DIR = Path("./cache")
OUTPUT_DIR = Path("./outputs")

ensure_dir(CACHE_DIR)
ensure_dir(OUTPUT_DIR)

CLASSIFIER_TRAINING_DATA_PATH = CACHE_DIR / "classifier_training_data"
ds = load_from_disk(str(CLASSIFIER_TRAINING_DATA_PATH))



print("\n[STATS] Label prevalence:")
for col in ds.column_names:
    if col.endswith("_label"):
        count = sum(ds[col])
        pct = 100 * count / len(ds)
        print(f"{col:35s} {count:6d} / {len(ds)}  ({pct:5.2f}%)")


print(f"[TIMER] step_name: {time.time() - start:.2f}s", flush=True)


TRAIN_SPLIT = "train[:8000]"
VAL_SPLIT   = "train[8000:9000]"
TEST_SPLIT  = "train[9000:10000]"

exit()


# ====================================================================


# =====================================================================
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from typing import Dict, List, Tuple, Optional

TRAIN_SPLIT = "train[:8000]"
VAL_SPLIT   = "train[8000:9000]"
TEST_SPLIT  = "train[9000:10000]"
THRESHOLD = 0.75

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

# Probe training
PROBE_MAX_ITER = 2000
PROBE_CLASS_WEIGHT = "balanced"

# Head selection
TOP_K_HEADS = 8

# Steering parameters
STEERING_ALPHA = 1.5
STEERING_CATCH_THRESHOLD = 0.7


text_field_candidates = ["text", "prompt", "sentence", "content", "text_a", "comment_text"]
label_field_candidates = ["toxicity", "label", "toxicity_score", "target", "y"]
text_field = next((c for c in text_field_candidates if c in cols), cols[0])
label_field = next((c for c in label_field_candidates if c in cols), None)

print(f"   [DEBUG] Selected text field: '{text_field}'")
print(f"   [DEBUG] Selected label field: '{label_field}'")

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

# Set seed for reproducible generation
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


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

@torch.no_grad()
def collect_head_activations(texts, labels, batch_size=BATCH_SIZE):
    """
    Collect per-head activations by capturing the attention-module outputs
    via temporary forward hooks. This ensures activations come from the
    same tensor that steering hooks will modify.

    Only extracts the LAST TOKEN activation from each example.

    Returns:
        activations_per_head: Dict[layer][head] -> np.ndarray (N, head_dim)
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    # Storage for activations collected via hooks
    activations_per_head = {}
    for layer in range(num_layers):
        activations_per_head[layer] = {}
        for head in range(num_heads):
            activations_per_head[layer][head] = []

    # Hook callback factory: captures the attention module's output
    # and appends per-head last-token activations to our storage.
    hooks = []

    def make_capture_hook(layer_idx):
        def _hook(module, input, output):
            # output may be tensor or tuple
            if isinstance(output, tuple):
                attn_out = output[0]
            else:
                attn_out = output

            # attn_out: (batch_size, seq_len, hidden_size)
            bsz, seq_len, hidden_size = attn_out.shape
            # reshape to (batch, seq_len, num_heads, head_dim)
            out_resh = attn_out.reshape(bsz, seq_len, num_heads, head_dim)
            # take last token per example
            last_token = out_resh[:, -1, :, :].cpu().numpy()  # shape: (bsz, num_heads, head_dim)

            # Append per-head
            for h in range(num_heads):
                activations_per_head[layer_idx][h].append(last_token[:, h, :])

        return _hook

    # Register hooks on each attention layer
    for layer_idx in range(num_layers):
        attn_layer = model.transformer.h[layer_idx].attn
        hook_fn = make_capture_hook(layer_idx)
        hooks.append(attn_layer.register_forward_hook(hook_fn))

    # Process in batches and run forward passes (hooks collect activations)
    num_batches = (len(texts) + batch_size - 1) // batch_size
    pbar = tqdm(total=num_batches, desc="Collecting activations (hooks)")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = list(texts[start_idx:end_idx])

        enc = tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)
        _ = model(**enc)

        pbar.update(1)

    pbar.close()

    # Remove hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    # Concatenate batches per head
    print("   Concatenating batches (hook-collected)...")
    for layer in range(num_layers):
        for head in range(num_heads):
            if len(activations_per_head[layer][head]) == 0:
                activations_per_head[layer][head] = np.zeros((0, head_dim), dtype=np.float32)
            else:
                activations_per_head[layer][head] = np.concatenate(
                    activations_per_head[layer][head], axis=0
                )

    return activations_per_head


if path_exists(ACTS_CACHE_TRAIN) and path_exists(ACTS_CACHE_VAL):
    print("\n>> Using cached activations (already collected)")
    activations_train = load_pickle(ACTS_CACHE_TRAIN)
    activations_val = load_pickle(ACTS_CACHE_VAL)
    
else:
    print("\n>> Collecting fresh activations from model")
    print("   (This will take a few minutes on first run...)")
    
    # Collect for train and val using the function defined above
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
            
            # If the training labels contain only one class (possible with
            # very small or imbalanced subsets), skip training the probe for
            # this head to avoid solver errors.
            if np.unique(y_train).size < 2:
                print(f"   [PROBE] SKIP Layer {layer_idx} Head {head_idx}: only one class present in training labels")
                probes_metadata[str(layer_idx)][str(head_idx)] = {
                    "layer": layer_idx,
                    "head": head_idx,
                    "train_acc": None,
                    "val_acc": None,
                    "val_auc": None,
                    "skipped": True,
                }
                continue

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

# -----------------------------------------------------------------------------
# DEBUG: Print per-head activations and probe predictions for a single example
# Inserted here so it runs after probes are trained but before head selection
# -----------------------------------------------------------------------------
print("\n" + "="*40)
print("[DEBUG] Per-head activations and probe predictions for first validation example")
sample_idx = 0
try:
    sample_text = texts_val[sample_idx]
    sample_label = labels_val[sample_idx]
    safe_print(f"[DEBUG] VAL sample #{sample_idx}: {sample_text}")
    print(f"[DEBUG] True label: {sample_label}")
except Exception as e:
    print(f"[DEBUG] Could not access validation sample: {e}")
else:
    num_layers = len(activations_val)
    num_heads = len(activations_val[0])
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            try:
                x = activations_val[layer_idx][head_idx][sample_idx]  # (head_dim,)
            except Exception as e:
                print(f"Layer {layer_idx} Head {head_idx}: no activation for sample: {e}")
                continue

            norm = float(np.linalg.norm(x))
            head_probe_dir = CACHE_DIR / f"probes" / f"layer_{layer_idx}" / f"head_{head_idx}"

            # If probe files missing or probe was skipped, note and continue
            if not path_exists(head_probe_dir / "clf.joblib") or not path_exists(head_probe_dir / "scaler_mean.npy"):
                print(f"Layer {layer_idx} Head {head_idx}: probe missing or skipped | act_norm={norm:.4f}")
                continue

            # Load probe artifacts
            try:
                scaler_mean = np.load(head_probe_dir / "scaler_mean.npy")
                scaler_scale = np.load(head_probe_dir / "scaler_scale.npy")
                clf = joblib.load(head_probe_dir / "clf.joblib")

                # Standardize and predict
                x_scaled = (x - scaler_mean) / (scaler_scale + 1e-12)
                proba = float(clf.predict_proba(x_scaled.reshape(1, -1))[0, 1])
                pred = int(clf.predict(x_scaled.reshape(1, -1))[0])

                print(f"Layer {layer_idx} Head {head_idx}: pred={pred} proba={proba:.4f} act_norm={norm:.4f}")
                print(f"  activation (first 8 dims): {np.array2string(x[:8], precision=3, separator=', ')}")

            except Exception as e:
                print(f"Layer {layer_idx} Head {head_idx}: error evaluating probe: {e}")

print("" + "="*40 + "\n")


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
            val_acc = metrics.get("val_acc", None)
            if val_acc is None:
                # Skip heads where probe training was skipped or unavailable
                continue
            all_heads_ranked.append({
                "layer": layer_idx,
                "head": head_idx,
                "val_acc": val_acc,
                "val_auc": metrics.get("val_auc", None),
                "train_acc": metrics.get("train_acc", None),
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

# -----------------------------------------------------------------------------
# Telemetry: report per-selected-head validation and test metrics (test requires activations)
# -----------------------------------------------------------------------------
ACTS_CACHE_TEST = CACHE_DIR / "activations_test.pkl"
LABELS_CACHE_TEST = CACHE_DIR / "labels_test.npy"
has_test = False
if path_exists(LABELS_CACHE_TEST) and path_exists(ACTS_CACHE_TEST):
    activations_test = load_pickle(ACTS_CACHE_TEST)
    labels_test = load_numpy(LABELS_CACHE_TEST)
    has_test = True
elif 'texts_test' in data and len(data.get('texts_test', [])) > 0:
    print("\n[SELECT] Collecting test activations for telemetry...")
    activations_test = collect_head_activations(data['texts_test'], data['labels_test'])
    save_pickle(activations_test, ACTS_CACHE_TEST)
    save_numpy(data['labels_test'], LABELS_CACHE_TEST)
    labels_test = data['labels_test']
    has_test = True

print("\n[TELEM] Per-selected-head validation/test metrics:")
for idx, head_info in enumerate(selected_heads):
    layer_idx = head_info['layer']
    head_idx = head_info['head']
    meta = probes_metadata.get(str(layer_idx), {}).get(str(head_idx), {})
    val_acc = meta.get('val_acc', None)
    val_auc = meta.get('val_auc', None)

    if has_test:
        try:
            X_test_head = activations_test[layer_idx][head_idx]
            y_test = labels_test
            # Load probe
            head_probe_dir = CACHE_DIR / f"probes" / f"layer_{layer_idx}" / f"head_{head_idx}"
            if path_exists(head_probe_dir / 'clf.joblib'):
                scaler_mean = np.load(head_probe_dir / 'scaler_mean.npy')
                scaler_scale = np.load(head_probe_dir / 'scaler_scale.npy')
                clf = joblib.load(head_probe_dir / 'clf.joblib')
                X_test_scaled = (X_test_head - scaler_mean) / (scaler_scale + 1e-12)
                y_pred = clf.predict(X_test_scaled)
                y_proba = clf.predict_proba(X_test_scaled)[:, 1]
                test_acc = float(accuracy_score(y_test, y_pred))
                test_auc = float(roc_auc_score(y_test, y_proba))
            else:
                test_acc = None
                test_auc = None
        except Exception as e:
            test_acc = None
            test_auc = None
    else:
        test_acc = None
        test_auc = None

    # Print summary (compact)
    print(f"  Layer {layer_idx} Head {head_idx}: val_acc={val_acc}, val_auc={val_auc}, test_acc={test_acc}, test_auc={test_auc}")

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
    print("\n>> Computing mean-shift vectors for selected heads (safe mode)")
    steering_vectors = {}

    for head_info in tqdm(selected_heads, desc="Computing theta, sigma"):
        layer_idx = head_info["layer"]
        head_idx = head_info["head"]

        # Get training activations for this head
        X_head = activations_train[layer_idx][head_idx]  # (N, head_dim)
        y_train = labels_train

        # Basic sanity checks
        if X_head.size == 0 or y_train.size == 0:
            print(f"  [STEER] SKIP Layer {layer_idx} Head {head_idx}: empty activations or labels")
            continue

        # Split by class
        X_pos = X_head[y_train == 1]
        X_neg = X_head[y_train == 0]

        if X_pos.size == 0 or X_neg.size == 0:
            print(f"  [STEER] SKIP Layer {layer_idx} Head {head_idx}: one class missing in training data "
                  f"(pos={X_pos.shape[0]}, neg={X_neg.shape[0]})")
            continue

        # Compute class means
        mu_pos = np.mean(X_pos, axis=0)  # (head_dim,)
        mu_neg = np.mean(X_neg, axis=0)  # (head_dim,)

        # Mean-shift direction
        theta_raw = mu_pos - mu_neg
        raw_norm = np.linalg.norm(theta_raw)

        if raw_norm < 1e-12:
            print(f"  [STEER] SKIP Layer {layer_idx} Head {head_idx}: zero mean-shift vector")
            continue

        theta = theta_raw / (raw_norm + 1e-12)
        theta_norm = float(np.linalg.norm(theta))

        # Defensive: enforce unit norm
        if abs(theta_norm - 1.0) > 1e-6:
            theta = theta / (theta_norm + 1e-12)
            theta_norm = float(np.linalg.norm(theta))

        # Compute sigma: std dev of scalar projections
        scalar_projs = np.dot(X_head, theta)  # (N,)
        sigma = float(np.std(scalar_projs))

        # Store diagnostics and vectors
        head_key = (layer_idx, head_idx)
        steering_vectors[head_key] = {
            "theta": theta.astype(np.float32),
            "theta_norm": theta_norm,
            "sigma": sigma,
            "mu_pos": mu_pos.astype(np.float32),
            "mu_neg": mu_neg.astype(np.float32),
            "n_pos": int(X_pos.shape[0]),
            "n_neg": int(X_neg.shape[0]),
        }

    # Save
    save_pickle(steering_vectors, STEERING_VECTORS_CACHE)

total_sv = len(steering_vectors)
print(f"\n[STEER] Computed steering vectors for {total_sv} heads")
for idx, ((layer_idx, head_idx), vec_data) in enumerate(steering_vectors.items()):
    # Print every 10th head to avoid flooding output; if small, print all
    if total_sv > 10 and (idx % 10) != 0:
        continue
    print(f"   Layer {layer_idx}, Head {head_idx}: "
          f"theta_norm={vec_data.get('theta_norm', np.linalg.norm(vec_data['theta'])):.6f}, "
          f"sigma={vec_data.get('sigma', 0.0):.6f}, "
          f"n_pos={vec_data.get('n_pos', 'NA')}, n_neg={vec_data.get('n_neg', 'NA')}")

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
        # The attention module may return a tensor or a tuple
        if isinstance(output, tuple):
            attn_output = output[0]
            rest = output[1:]
        else:
            attn_output = output
            rest = ()

        # attn_output: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = attn_output.shape
        
        # Reshape to separate heads: (batch_size, seq_len, num_heads, head_dim)
        output_reshaped = attn_output.reshape(batch_size, seq_len, num_heads, head_dim)
        
        # Add steering to target head (all tokens, all batch items)
        output_reshaped[:, :, head_idx, :] = output_reshaped[:, :, head_idx, :] + steer_vec
        
        # Reshape back: (batch_size, seq_len, hidden_size)
        modified_attn = output_reshaped.reshape(batch_size, seq_len, hidden_size)

        # Return same structure as input (tensor or tuple)
        if rest:
            return (modified_attn, *rest)
        return modified_attn
    
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
        # The attention module may return a tensor or a tuple
        if isinstance(output, tuple):
            attn_output = output[0]
            rest = output[1:]
        else:
            attn_output = output
            rest = ()

        # attn_output: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = attn_output.shape
        
        # Reshape to separate heads: (batch_size, seq_len, num_heads, head_dim)
        output_reshaped = attn_output.reshape(batch_size, seq_len, num_heads, head_dim)
        
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
        modified_attn = output_reshaped.reshape(batch_size, seq_len, hidden_size)

        if rest:
            return (modified_attn, *rest)
        return modified_attn
    
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
    "max_new_tokens": 30,
    # Enable sampling for more diverse outputs
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
    "pad_token_id": tokenizer.eos_token_id,
}


# Helper: capture per-head activations for a single prompt using forward hooks
@torch.no_grad()
def get_prompt_head_activations(prompt: str, layers_to_capture: Optional[List[int]] = None):
    """Return dict[(layer, head)] -> numpy array (head_dim,) for the last token."""
    enc = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    if layers_to_capture is None:
        layers_to_capture = list(range(num_layers))

    activations = {}
    hooks = []

    def make_capture(layer_idx):
        def _hook(module, input, output):
            out_tensor = output[0] if isinstance(output, tuple) else output
            bsz, seq_len, hidden_size = out_tensor.shape
            out_resh = out_tensor.reshape(bsz, seq_len, num_heads, head_dim)
            last = out_resh[:, -1, :, :].cpu().numpy()  # (1, num_heads, head_dim)
            for h in range(num_heads):
                activations[(layer_idx, h)] = last[0, h, :]
        return _hook

    for layer_idx in layers_to_capture:
        attn_layer = model.transformer.h[layer_idx].attn
        hook = attn_layer.register_forward_hook(make_capture(layer_idx))
        hooks.append(hook)

    # Forward pass (hooks will populate activations)
    _ = model(**enc)

    # Remove hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    return activations


print(f"\n>> Generating text with {len(test_prompts)} prompts\n")

for prompt_idx, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*80}")
    print(f"PROMPT {prompt_idx}: {prompt}")
    print(f"{'='*80}")
    
    # Compute per-head probe predictions for this prompt (telemetry)
    print("\n[TELEM] Computing per-head probe probabilities for this prompt...")
    layers_needed = sorted(list({h['layer'] for h in selected_heads}))
    prompt_acts = get_prompt_head_activations(prompt, layers_to_capture=layers_needed)

    head_prompt_probas = {}
    total_selected = len(selected_heads)
    for i, head_info in enumerate(selected_heads):
        layer_idx = head_info['layer']
        head_idx = head_info['head']
        head_probe_dir = CACHE_DIR / f"probes" / f"layer_{layer_idx}" / f"head_{head_idx}"

        if not path_exists(head_probe_dir / "clf.joblib"):
            head_prompt_probas[(layer_idx, head_idx)] = None
            # print only occasionally to avoid flooding
            if total_selected <= 10 or (i % 10) == 0:
                print(f"   Layer {layer_idx} Head {head_idx}: probe missing")
            continue

        try:
            x = prompt_acts.get((layer_idx, head_idx), None)
            if x is None:
                head_prompt_probas[(layer_idx, head_idx)] = None
                if total_selected <= 10 or (i % 10) == 0:
                    print(f"   Layer {layer_idx} Head {head_idx}: no activation captured for prompt")
                continue

            scaler_mean = np.load(head_probe_dir / "scaler_mean.npy")
            scaler_scale = np.load(head_probe_dir / "scaler_scale.npy")
            clf = joblib.load(head_probe_dir / "clf.joblib")

            x_scaled = (x - scaler_mean) / (scaler_scale + 1e-12)
            proba = float(clf.predict_proba(x_scaled.reshape(1, -1))[0, 1])
            pred = int(clf.predict(x_scaled.reshape(1, -1))[0])
            head_prompt_probas[(layer_idx, head_idx)] = proba
            if total_selected <= 10 or (i % 10) == 0:
                print(f"   Layer {layer_idx} Head {head_idx}: pred={pred} proba={proba:.4f}")

        except Exception as e:
            head_prompt_probas[(layer_idx, head_idx)] = None
            if total_selected <= 10 or (i % 10) == 0:
                print(f"   Layer {layer_idx} Head {head_idx}: error evaluating probe: {e}")

    # 1. BASELINE GENERATION
    print("\n[BASE] Baseline (no steering):")
    with torch.no_grad():
        enc = tokenizer(prompt, return_tensors="pt")
        gen = model.generate(**enc, **generation_params)
        baseline_text = tokenizer.decode(gen[0], skip_special_tokens=True)
    safe_print(baseline_text)
    
    # 2. UNCONDITIONAL ITI GENERATION
    print("\n[ITI] Unconditional ITI (always steer):")
    hooks = []
    # 2. CONDITIONAL CATCH & STEER (based on prompt-level probe predictions)
    print("\n[CATCH] Conditional Catch & Steer (steer only for heads with proba > threshold):")
    hooks = []
    try:
        for head_info in selected_heads:
            layer_idx = head_info["layer"]
            head_idx = head_info["head"]
            proba = head_prompt_probas.get((layer_idx, head_idx), None)
            if proba is None or proba <= STEERING_CATCH_THRESHOLD:
                # Skip steering for this head
                continue

            theta = steering_vectors[(layer_idx, head_idx)]["theta"]
            sigma = steering_vectors[(layer_idx, head_idx)]["sigma"]
            attn_layer = model.transformer.h[layer_idx].attn
            hook_fn = make_unconditional_iti_hook(layer_idx, head_idx, theta, sigma, STEERING_ALPHA)
            hook = attn_layer.register_forward_hook(hook_fn)
            hooks.append(hook)

        with torch.no_grad():
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            gen = model.generate(**enc, **generation_params)
            catch_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        safe_print(catch_text)
    finally:
        for hook in hooks:
            hook.remove()

    # 3. UNCONDITIONAL ITI GENERATION (always steer)
    print("\n[ITI] Unconditional ITI (always steer):")
    hooks = []
    try:
        # Register hooks for all selected heads
        for head_info in selected_heads:
            layer_idx = head_info["layer"]
            head_idx = head_info["head"]
            theta = steering_vectors[(layer_idx, head_idx)]["theta"]
            sigma = steering_vectors[(layer_idx, head_idx)]["sigma"]

            attn_layer = model.transformer.h[layer_idx].attn
            hook_fn = make_unconditional_iti_hook(layer_idx, head_idx, theta, sigma, STEERING_ALPHA)
            hook = attn_layer.register_forward_hook(hook_fn)
            hooks.append(hook)

        with torch.no_grad():
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            gen = model.generate(**enc, **generation_params)
            iti_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        safe_print(iti_text)
    finally:
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
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"PROMPT: {prompt}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"BASELINE:\n{baseline_text}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"UNCONDITIONAL ITI:\n{iti_text}\n")
    
    print(f"\n[SAVE] Output saved to: {output_file}")

    # Also append the conditional (Catch) output into the same file for telemetry
    try:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"CONDITIONAL (Catch):\n{catch_text}\n")
    except Exception as e:
        print(f"[WARN] Could not append conditional output to file: {e}")


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

safe_print("\n" + "="*80)
safe_print("SCRIPT COMPLETED SUCCESSFULLY")
safe_print("="*80)
safe_print("")

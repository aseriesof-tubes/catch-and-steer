# =========================
# CONFIG / IMPORTS
# =========================
import os  # filesystem ops
import json  # save metadata
import time  # timestamp meta
import numpy as np  # arrays
import joblib  # save sklearn
import torch  # pytorch core
from datasets import load_dataset  # HF datasets
from transformers import AutoTokenizer, AutoModelForCausalLM  # HF models MAYBE I SHOULD USE UNSLOTH??
from sklearn.linear_model import LogisticRegression  # linear probe
from sklearn.preprocessing import StandardScaler  # standardize features
from sklearn.metrics import roc_auc_score, accuracy_score  # metrics
from tqdm.auto import tqdm  # progress bars

# Basic settings
MODEL_NAME = "distilgpt2"  # small model
LAYER_INDEX = -1  # last layer
MAX_LEN = 128  # max tokens
BATCH_SIZE = 16  # batch size
DATA_SPLIT = "train[:2000]"  # small slice
PROBE_DIR = f"probes/{MODEL_NAME}/offensive/layer_{LAYER_INDEX}"  # cache path
ACTS_PATH = os.path.join(PROBE_DIR, "acts.npy")  # acts cache

# Device pick
device = "cuda" if torch.cuda.is_available() else "cpu"  # pick device
print("device:", device)  # show device


# =========================
# UTIL: FILE HELPERS
# =========================
def ensure_dir(path):  # make directory
    os.makedirs(path, exist_ok=True)  # ensure path


def save_probe(out_dir, w_raw, scaler, clf, meta):  # save probe
    ensure_dir(out_dir)  # make dir
    np.save(os.path.join(out_dir, "w_raw.npy"), w_raw)  # save vector
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))  # save scaler
    joblib.dump(clf, os.path.join(out_dir, "clf.joblib"))  # save clf
    with open(os.path.join(out_dir, "meta.json"), "w") as f:  # save meta
        json.dump(meta, f, indent=2)  # pretty json


def load_probe(out_dir):  # load probe
    w_raw = np.load(os.path.join(out_dir, "w_raw.npy"))  # load vector
    scaler = joblib.load(os.path.join(out_dir, "scaler.joblib"))  # load scaler
    clf = joblib.load(os.path.join(out_dir, "clf.joblib"))  # load clf
    with open(os.path.join(out_dir, "meta.json")) as f:  # load meta
        meta = json.load(f)  # parse json
    return w_raw.astype(np.float32), scaler, clf, meta  # return all


def probe_exists(out_dir):  # check probe
    return all(  # all files
        os.path.exists(os.path.join(out_dir, p))  # exists?
        for p in ["w_raw.npy", "scaler.joblib", "clf.joblib", "meta.json"]
    )


# =========================
# SECTION: LOAD DATA
# =========================
print(">> Loading dataset")  # status
ds = load_dataset("tweet_eval", "offensive", split=DATA_SPLIT)  # load split
texts = ds["text"]  # texts list
labels = np.array(ds["label"])  # labels array
print("dataset size:", len(texts))  # show size


# =========================
# SECTION: LOAD MODEL
# =========================
print(">> Loading model")  # status
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # load tokenizer
if tokenizer.pad_token is None:  # set pad token
    tokenizer.pad_token = tokenizer.eos_token  # use eos
model = AutoModelForCausalLM.from_pretrained(  # load model
    MODEL_NAME, output_hidden_states=True  # request states
).to(device).eval()  # to device


# =========================
# SECTION: COLLECT ACTIVATIONS
# =========================
@torch.no_grad()
def collect_acts(texts, batch=BATCH_SIZE, layer=LAYER_INDEX, max_len=MAX_LEN):  # collect
    acts = []  # store acts
    for i in tqdm(range(0, len(texts), batch)):  # batches
        batch_texts = texts[i:i + batch]  # slice
        enc = tokenizer(  # tokenize
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_len
        ).to(device)  # to device
        out = model(**enc)  # forward
        hs = out.hidden_states[layer]  # [B,T,H]
        last = hs[:, -1, :].cpu()  # last token
        acts.append(last)  # append
    return torch.cat(acts, 0)  # stack


print(">> Getting activations")  # status
ensure_dir(PROBE_DIR)  # ensure dir
if os.path.exists(ACTS_PATH):  # cached?
    acts = torch.from_numpy(np.load(ACTS_PATH)).float()  # load cache
    print("loaded cached acts")  # status
else:
    acts = collect_acts(texts, batch=BATCH_SIZE, layer=LAYER_INDEX, max_len=MAX_LEN)  # collect
    np.save(ACTS_PATH, acts.numpy())  # save cache
    print("saved acts cache")  # status

X = acts.numpy()  # to numpy


# =========================
# SECTION: TRAIN OR LOAD PROBE
# =========================
if probe_exists(PROBE_DIR):  # already exists?
    print(">> Loading probe")  # status
    w_raw, scaler, clf, meta = load_probe(PROBE_DIR)  # load probe
else:
    print(">> Training probe")  # status
    scaler = StandardScaler()  # init scaler
    Xz = scaler.fit_transform(X)  # standardize
    clf = LogisticRegression(  # init clf
        max_iter=2000, class_weight="balanced", n_jobs=-1
    ).fit(Xz, labels)  # fit clf
    pred = clf.predict(Xz)  # predictions
    prob = clf.predict_proba(Xz)[:, 1]  # probabilities
    print(  # show metrics
        "train acc:", round(accuracy_score(labels, pred), 4),
        "auc:", round(roc_auc_score(labels, prob), 4)
    )
    w_raw = (clf.coef_.ravel() / (scaler.scale_ + 1e-12)).astype(np.float32)  # unscale
    w_raw = w_raw / (np.linalg.norm(w_raw) + 1e-12)  # normalize
    meta = {  # metadata
        "model_name": MODEL_NAME,  # model id
        "layer_index": LAYER_INDEX,  # layer used
        "max_len": MAX_LEN,  # token limit
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
        "classes": getattr(clf, "classes_", None).tolist(),  # class ids
    }
    save_probe(PROBE_DIR, w_raw, scaler, clf, meta)  # save probe


# =========================
# SECTION: STEERING HOOK
# =========================
def make_steer_hook(w_vec, strength=1.5):  # hook factory
    w_t = torch.tensor(w_vec, dtype=torch.float32, device=device)  # to device
    def hook(module, inp, out):  # hook fn
        return out - strength * w_t  # broadcast subtract
    return hook  # return hook


# =========================
# SECTION: GENERATE TEXT
# =========================
prompt = "Write a response to this rude tweet: I hate everyone who disagrees with me."  # test prompt

# Baseline generation
print(">> Baseline generate")  # status
gen = model.generate(  # generate
    **tokenizer(prompt, return_tensors="pt").to(device),
    max_new_tokens=60, do_sample=True, top_p=0.9, temperature=0.9
)
print("=== BASE ===\n", tokenizer.decode(gen[0], skip_special_tokens=True))  # show text

# Steered generation
print(">> Steered generate")  # status
hook = model.transformer.h[LAYER_INDEX].register_forward_hook(  # register hook
    make_steer_hook(w_raw, strength=1.5)  # with vector
)
try:
    gen2 = model.generate(  # generate
        **tokenizer(prompt, return_tensors="pt").to(device),
        max_new_tokens=60, do_sample=True, top_p=0.9, temperature=0.9
    )
finally:
    hook.remove()  # always remove

print("\n=== STEERED ===\n", tokenizer.decode(gen2[0], skip_special_tokens=True))  # show text

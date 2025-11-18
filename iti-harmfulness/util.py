import os
import json
import time
import pickle
import sys
import traceback
from pathlib import Path
from tqdm.auto import tqdm
import torch
import numpy as np


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



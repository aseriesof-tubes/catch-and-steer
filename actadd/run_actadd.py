#!/usr/bin/env python3
"""
1) load model/tokenizer 
2) build base/target token sets
3) compute steering vectors on last token
4) run generation with optional (multi-)steering hooks
5) save outputs to JSONL

Usage examples:
  python run_actadd.py --config config_example.json --compute_vecs --run
  python run_actadd.py --config config_example.json --run --vectors love_hate honest_dishonest --layer 16 --scales 1.0 0.7
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import utils

try:
    from unsloth import FastLanguageModel
except Exception as e:
    raise RuntimeError(
        "Failed to import unsloth." 
    ) from e

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("Failed to import pandas.") from e


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_texts_from_spec(spec: Dict[str, Any]) -> List[str]:
    """
    Supported spec formats:
      {"type":"inline", "texts":[...]}
      {"type":"csv", "path":"/path/file.csv", "column":"goal", "slice":[0,200]}
      {"type":"parquet", "path":"hf://datasets/..../file.parquet", "column":"instruction", "slice":[0,200]}
    """
    stype = spec.get("type")
    if stype == "inline":
        texts = list(spec.get("texts", []))
        return [t for t in texts if isinstance(t, str)]
    if stype == "csv":
        df = pd.read_csv(spec["path"])
        col = spec["column"]
        texts = df[col].astype(str).tolist()
        sl = spec.get("slice")
        if sl is not None:
            a, b = int(sl[0]), int(sl[1])
            texts = texts[a:b]
        return texts
    if stype == "parquet":
        df = pd.read_parquet(spec["path"])
        col = spec["column"]
        texts = df[col].astype(str).tolist()
        sl = spec.get("slice")
        if sl is not None:
            a, b = int(sl[0]), int(sl[1])
            texts = texts[a:b]
        return texts
    raise ValueError(f"Unknown data spec type: {stype}. Full spec: {spec}")


def _format_as_chat_instructions(texts: List[str]) -> List[List[Dict[str, str]]]:
    return [[{"role": "user", "content": t}] for t in texts]


def load_model(cfg: Dict[str, Any]):
    mcfg = cfg["model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mcfg["model_name"],
        max_seq_length=int(mcfg.get("max_seq_length", 8192)),
        load_in_4bit=bool(mcfg.get("load_in_4bit", True)),
    )
    model.eval()
    return model, tokenizer


def compute_and_save_vectors(
    model,
    tokenizer,
    cfg: Dict[str, Any],
    out_dir: Path,
    batch_size: int,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Returns dict[name -> dict[layer -> tensor[h]]].
    Saves each vector bundle to out_dir/vectors/{name}.pt.
    """
    vectors_cfg = cfg.get("vectors", [])
    vec_root = out_dir / "vectors"
    _ensure_dir(vec_root)

    all_vecs: Dict[str, Dict[int, torch.Tensor]] = {}
    for v in vectors_cfg:
        name = v["name"]
        base_texts = _load_texts_from_spec(v["base"])
        target_texts = _load_texts_from_spec(v["target"])

        base_toks = utils.tokenize_instructions(tokenizer, _format_as_chat_instructions(base_texts))
        target_toks = utils.tokenize_instructions(tokenizer, _format_as_chat_instructions(target_texts))

        sv = utils.find_steering_vecs(model, base_toks, target_toks, batch_size=batch_size)  # dict[layer]->[h] (cpu)
        all_vecs[name] = sv

        save_path = vec_root / f"{name}.pt"
        torch.save({"name": name, "steering_vecs": sv}, save_path)
        print(f"[saved] {name} -> {save_path}")

    return all_vecs


def load_vectors_from_disk(out_dir: Path, names: Optional[List[str]] = None) -> Dict[str, Dict[int, torch.Tensor]]:
    vec_root = out_dir / "vectors"
    if not vec_root.exists():
        raise FileNotFoundError(f"No vectors directory: {vec_root}. Run with --compute_vecs first.")
    bundles = sorted(vec_root.glob("*.pt"))
    if names:
        wanted = set(names)
        bundles = [p for p in bundles if p.stem in wanted]
        missing = wanted - set(p.stem for p in bundles)
        if missing:
            raise FileNotFoundError(f"Missing vector files for: {sorted(missing)} in {vec_root}")
    out: Dict[str, Dict[int, torch.Tensor]] = {}
    for p in bundles:
        obj = torch.load(p, map_location="cpu")
        out[p.stem] = obj["steering_vecs"]
    return out


def run_generation(
    model,
    tokenizer,
    cfg: Dict[str, Any],
    out_dir: Path,
    vectors: Dict[str, Dict[int, torch.Tensor]],
    vector_names: Optional[List[str]],
    layer: Optional[int],
    scales: Optional[List[float]],
    proj: str,
    normalise: bool,
    batch_size: int,
) -> None:
    """
    Runs generation for cfg["tests"] (list of test sets) and writes JSONL.
    """
    tests = cfg.get("tests", [])
    if not tests:
        raise ValueError("No tests found in config under key 'tests'.")

    results_path = out_dir / "results.jsonl"
    # append mode so you can accumulate multiple runs
    f = open(results_path, "a")

    # Decide which vectors to use
    if vector_names is None or len(vector_names) == 0:
        vector_names = list(vectors.keys())
    if scales is None:
        scales = [1.0] * len(vector_names)
    if len(scales) != len(vector_names):
        raise ValueError(f"--scales must match number of vectors. got {len(scales)} scales for {len(vector_names)} vectors.")

    for t in tests:
        test_name = t["name"]
        prompts = _load_texts_from_spec(t["prompts"])
        toks = utils.tokenize_instructions(tokenizer, _format_as_chat_instructions(prompts))

        # baseline
        baseline = utils.do_single_steering(model, toks.to(model.device), None, batch_size=batch_size)

        # steered
        steering_vecs_list = []
        for vn in vector_names:
            if vn not in vectors:
                raise KeyError(f"Vector '{vn}' not loaded. Available: {sorted(vectors.keys())}")
            if layer is None:
                raise ValueError("This runner expects a specific --layer for now (to keep it simple).")
            steering_vecs_list.append(vectors[vn][layer])

        steered = utils.do_multi_steering(
            model,
            toks.to(model.device),
            steering_vecs_list=[sv.to(model.device) for sv in steering_vecs_list],
            scales_list=scales,
            normalise=normalise,
            layer=layer,
            proj=proj,
            batch_size=batch_size,
        )

        # decode + write
        for i, prompt in enumerate(prompts):
            prompt_tok_len = toks[i].shape[0]
            row = {
                "test": test_name,
                "prompt": prompt,
                "layer": layer,
                "vectors": vector_names,
                "scales": scales,
                "proj": proj,
                "normalise": normalise,
                "baseline": tokenizer.decode(baseline[i][prompt_tok_len:], skip_special_tokens=True),
                "steered": tokenizer.decode(steered[i][prompt_tok_len:], skip_special_tokens=True),
            }
            f.write(json.dumps(row) + "\n")

        print(f"[done] test={test_name} n={len(prompts)} -> {results_path}")

    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config.")
    ap.add_argument("--out_dir", type=str, default="runs/actadd", help="Output directory.")
    ap.add_argument("--compute_vecs", action="store_true", help="Compute + save vectors defined in config.")
    ap.add_argument("--run", action="store_true", help="Run generation on tests in config.")
    ap.add_argument("--vectors", nargs="*", default=None, help="Subset of vector names to use (default: all).")
    ap.add_argument("--layer", type=int, default=None, help="Layer index to steer (required for now).")
    ap.add_argument("--scales", nargs="*", type=float, default=None, help="Scales for each vector (same length as --vectors).")
    ap.add_argument("--proj", type=str, default="k-dir", choices=["none", "k-dir", "1-dir"], help="Projection mode.")
    ap.add_argument("--normalise", action="store_true", help="Normalize steering vectors before use.")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = _read_json(cfg_path)

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    model, tokenizer = load_model(cfg)

    if args.compute_vecs:
        compute_and_save_vectors(model, tokenizer, cfg, out_dir, batch_size=args.batch_size)

    if args.run:
        vectors = load_vectors_from_disk(out_dir, names=args.vectors)
        run_generation(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            out_dir=out_dir,
            vectors=vectors,
            vector_names=args.vectors,
            layer=args.layer,
            scales=args.scales,
            proj=args.proj,
            normalise=args.normalise,
            batch_size=args.batch_size,
        )

    if not args.compute_vecs and not args.run:
        print("Nothing to do: pass --compute_vecs and/or --run")


if __name__ == "__main__":
    main()

"""Validation-only sweep of latent size and model width. Never loads the test split or the evaluator.

Each run trains from scratch on the 27 fitting cases with the 9k/3k recipe and is scored on the
five held-out validation cases, exactly like the schedule comparison.
Usage: python experiments/size_sweep.py NAME SEED

The sweep behind size_sweep_results.json ran on 18 November 2026 with seeds 1234 and 9153.
"released" is the size of the first (v1) ensemble; "all_larger" was chosen for v2.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model as model_module
from data import load, split_bank
from train import RECIPE, run_phase
from validate import evaluate

# Declared before any run. (latent tokens, latent dim, hidden width); the v1 model is (4, 16, 32).
CONFIGS = {
    "released":      (4, 16, 32),
    "latent_dim_8":  (4, 8, 32),
    "latent_dim_32": (4, 32, 32),
    "tokens_2":      (2, 16, 32),
    "tokens_8":      (8, 16, 32),
    "width_16":      (4, 16, 16),
    "width_64":      (4, 16, 64),
    "all_larger":    (8, 32, 64),
}

def main(name, seed):
    tokens, dim, width = CONFIGS[name]
    model_module.LATENT_TOKENS, model_module.LATENT_DIM, model_module.TOKEN_DIM = tokens, dim, width
    torch.set_num_threads(1)
    fit, validation = split_bank(load(ROOT / "cache.npz", ("train",))["train"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model_module.Model(dropout=RECIPE["dropout"])
    start = time.monotonic()
    for phase, steps in enumerate(RECIPE["phase_steps"], 1):
        run_phase(model, fit, phase, steps, rng, "cpu", log_every=3000)
    result = {"name": name, "seed": seed, "latent_tokens": tokens, "latent_dim": dim, "width": width,
              "params": sum(p.numel() for p in model.parameters()), "phase_steps": RECIPE["phase_steps"],
              "fit_cases": len(fit.names), "validation_cases": list(validation.names),
              "seconds": time.monotonic() - start, **evaluate(model, validation)}
    out = Path(__file__).parent / "size_sweep_runs" / f"{name}_seed{seed}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"RESULT {name} seed={seed} params={result['params']} validation={result['mean']:.6f}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))

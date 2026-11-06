"""Score a checkpoint on held-out validation cases, without accessing test data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import CONTEXT_POINTS, TEST_CASES, VALIDATION_CASES, check_preprocessing, load, scores, split_bank
from model import Model


@torch.no_grad()
def validation_predictions(model, bank, context_seed=8675309, context_points=CONTEXT_POINTS, chunk=4096):
    """Predict all held-out cells using a fixed RANS-only context per case."""
    model.eval()
    device = next(model.parameters()).device
    pred = np.empty_like(bank.base)
    for case, idx in enumerate(bank.by_case):
        rng = np.random.default_rng(context_seed + case)
        ci = rng.choice(idx, context_points, replace=False) if len(idx) > context_points else idx
        context = torch.as_tensor(bank.norm(case, ci), device=device)[None]
        z = model.encode(context)
        for start in range(0, len(idx), chunk):
            qi = idx[start:start + chunk]
            q = torch.as_tensor(bank.norm(case, qi), device=device)[None]
            residual = model(z, q)[0].cpu().numpy()
            pred[qi] = bank.base[qi] + bank.u_ref[case] * residual
    if not np.isfinite(pred).all():
        raise FloatingPointError("Nonfinite validation prediction")
    return pred


def evaluate(model, bank):
    per_case = scores(validation_predictions(model, bank), bank)
    return {"mean": float(np.mean(list(per_case.values()))), "per_case": per_case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("cache.npz"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    overlap = set(ck["train_cases"]) & (set(VALIDATION_CASES) | set(TEST_CASES))
    if overlap:
        raise ValueError(f"Checkpoint was trained on held-out cases: {sorted(overlap)}")
    _, validation = split_bank(load(args.cache, ("train",))["train"])
    check_preprocessing(ck, validation, args.cache)
    model = Model(**ck.get("model_config", {})).to(args.device)
    model.load_state_dict(ck["state"])
    result = evaluate(model, validation)
    baseline = scores(validation.base, validation)
    print(json.dumps({"checkpoint": str(args.checkpoint), "validation": result,
                      "baseline": {"mean": float(np.mean(list(baseline.values()))), "per_case": baseline}}, indent=2))


if __name__ == "__main__":
    main()

"""Run ensemble inference on test cases and export CSV predictions at evaluation points."""
import argparse
from pathlib import Path

import numpy as np
import torch

from data import CONTEXT_POINTS, check_preprocessing, load
from model import Model


@torch.no_grad()
def encode_cases(model, bank, seed, context_points=CONTEXT_POINTS):
    device = next(model.parameters()).device
    out = []
    for case, idx in enumerate(bank.by_case):
        if len(idx) > context_points:
            idx = np.random.default_rng(seed + 1000 + case).choice(idx, context_points, replace=False)
        points = torch.as_tensor(bank.norm(case, idx), device=device).unsqueeze(0)
        out.append(model.encode(points).squeeze(0).cpu().numpy())
    return np.stack(out)


@torch.no_grad()
def predict_eval_points(model, bank, z, batch_size=4096):
    device = next(model.parameters()).device
    case_preds = []
    if bank.eval_idx is None:
        raise ValueError("Cache has no evaluation-point indices")
    latent_shape = tuple(model.parts["rans_encoder"].queries.shape)
    for case, eval_idx in enumerate(bank.eval_idx):
        code = torch.as_tensor(z[case], device=device).reshape(1, *latent_shape)
        ref = torch.as_tensor([bank.u_ref[case]], device=device).view(1, 1, 1)
        out = np.empty((len(eval_idx), 3), dtype=np.float32)
        for start in range(0, len(eval_idx), batch_size):
            idx = eval_idx[start : start + batch_size]
            q = torch.as_tensor(bank.norm(case, idx), device=device).unsqueeze(0)
            base = torch.as_tensor(bank.base[idx], device=device).unsqueeze(0)
            out[start : start + batch_size] = (base + ref * model(code, q)).squeeze(0).cpu().numpy()
        case_preds.append(out)
    return case_preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="*", default=[Path("checkpoints/seed1234.pt")])
    parser.add_argument("--cache", type=Path, default=Path("cache.npz"))
    parser.add_argument("--out", type=Path, default=Path("predictions"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    test_bank = load(args.cache, ("test",))["test"]
    all_seed_preds = []

    for path in args.checkpoints:
        ck = torch.load(path, map_location=args.device, weights_only=True)
        check_preprocessing(ck, test_bank, args.cache)
        model = Model(**ck.get("model_config", {})).to(args.device)
        model.load_state_dict(ck["state"])
        model.eval()

        z_test = encode_cases(model, test_bank, ck["seed"])
        all_seed_preds.append(predict_eval_points(model, test_bank, z_test))
        print(f"Predicted with {path.name} (seed={ck['seed']})")

    args.out.mkdir(parents=True, exist_ok=True)
    for case_i, name in enumerate(test_bank.names):
        ensemble_case = np.mean([seed_pred[case_i] for seed_pred in all_seed_preds], axis=0)
        if not np.isfinite(ensemble_case).all():
            raise FloatingPointError(f"Nonfinite predictions for {name}")
        np.savetxt(args.out / f"{name}.csv", ensemble_case, delimiter=",")
    print(f"Saved {len(test_bank.names)} CSV files to {args.out}/")


if __name__ == "__main__":
    main()

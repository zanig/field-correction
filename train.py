"""Two-phase training: Phase 1 learns truth latents + decoder, Phase 2 aligns RANS latents."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import CONTEXT_POINTS, GEOMETRY, TEST_CASES, VALIDATION_CASES, family, load, split_bank
from model import Model, sigreg
from utils import cache_content_sha256, write_json

CONTEXT, QUERIES, CASE_BATCH = CONTEXT_POINTS, 1024, 8
RECIPE = {
    "phase_steps": [9000, 3000],
    "lr": 1e-3,
    "min_lr": 1e-5,
    "weight_decay": 1e-3,
    "latent_weight": 0.1,
    "sigreg_weight": 0.05,
    "dropout": 0.0,
}


def truth_features(bank, c, idx):
    ref = bank.u_ref[c]
    geom = bank.norm(c, idx, GEOMETRY)
    res = (bank.true[idx] - bank.base[idx]) / ref
    u_true = bank.true[idx] / ref
    return np.concatenate([geom, res, u_true], axis=1)


def sample_batch(bank, cases, rng, phase, device):
    rows = [[] for _ in range(6)]
    for c in cases:
        pool, ref = bank.by_case[c], bank.u_ref[c]
        ci = rng.choice(pool, CONTEXT, replace=len(pool) < CONTEXT)
        qi = rng.choice(pool, QUERIES, replace=len(pool) < QUERIES)
        target = truth_features(bank, c, qi)
        context = truth_features(bank, c, ci) if phase == 1 else bank.norm(c, ci)
        for row, val in zip(rows, (context, target, bank.norm(c, qi), bank.true[qi], bank.base[qi], [[ref]])):
            row.append(val)
    return [torch.as_tensor(np.asarray(r), dtype=torch.float32, device=device) for r in rows]


def configure_phase(model, phase):
    active = {"truth_encoder", "truth_predictor", "decoder"} if phase == 1 else {"rans_encoder", "rans_predictor"}
    for name, module in model.parts.items():
        module.train(name in active)
        module.requires_grad_(name in active)
        for p in module.parameters():
            p.grad = None
    return [p for p in model.parameters() if p.requires_grad]


def field_loss(pred, true):
    """Normalize each case's mean point error, then average equally over cases."""
    case_error = torch.linalg.vector_norm(pred - true, dim=-1).mean(dim=1)
    case_speed = torch.linalg.vector_norm(true, dim=-1).mean(dim=1).clamp_min(1e-8)
    return (case_error / case_speed).mean()


def training_loss(pred, true, z, z_target, latent_weight=0.1, sigreg_weight=0.05):
    """Field reconstruction, prediction of the detached teacher, and Gaussian regularization."""
    field = field_loss(pred, true)
    alignment = F.mse_loss(z, z_target)
    regularizer = sigreg(z)
    loss = field + latent_weight * alignment + sigreg_weight * regularizer
    return loss, {"loss": loss, "field": field, "alignment": alignment,
                  "sigreg": regularizer}


def run_phase(model, bank, phase, steps, rng, device, recipe=None, log_every=1000):
    recipe = {**RECIPE, **(recipe or {})}
    params = configure_phase(model, phase)
    opt = torch.optim.AdamW(params, lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=recipe["min_lr"])

    groups = {f: np.asarray([i for i, n in enumerate(bank.names) if family(n) == f]) for f in ("hill", "duct", "other")}
    groups = {f: ids for f, ids in groups.items() if len(ids)}
    weights = np.asarray([{"hill": 0.65, "duct": 0.25, "other": 0.10}[f] for f in groups])
    weights /= weights.sum()

    start, total, count = time.monotonic(), np.zeros(4, dtype=np.float64), 0
    history = []
    for step in range(1, steps + 1):
        families = rng.choice(list(groups), CASE_BATCH, p=weights)
        cases = [int(rng.choice(groups[f])) for f in families]
        ctx, target, query, true, base, ref = sample_batch(bank, cases, rng, phase, device)

        z = model.encode(ctx, truth=(phase == 1))
        with torch.no_grad():
            z_target = model.parts["truth_encoder"](target)

        pred = base + ref * model(z, query)
        loss, components = training_loss(pred, true, z, z_target,
                                        recipe["latent_weight"], recipe["sigreg_weight"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss in phase {phase}, step {step}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        total += torch.stack([value.detach() for value in components.values()]).cpu().numpy()
        count += 1
        if step % log_every == 0 or step == steps:
            metrics = dict(zip(components, (total / count).tolist()))
            history.append({"phase": phase, "step": step, **metrics})
            print(
                f"phase={phase} step={step}/{steps} "
                + " ".join(f"{name}={value:.5f}" for name, value in metrics.items()) + " "
                f"lr={sched.get_last_lr()[0]:.2e} elapsed={time.monotonic()-start:.1f}s",
                flush=True,
            )
            total, count = np.zeros(4, dtype=np.float64), 0
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("cache.npz"))
    parser.add_argument("--out", type=Path, default=Path("runs/train"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234])
    parser.add_argument("--config", type=Path, help="JSON overrides for RECIPE")
    parser.add_argument("--holdout-validation", action="store_true", help="Exclude the five suggested validation cases")
    parser.add_argument("--steps", type=int, nargs=2, metavar=("TRUTH", "RANS"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads")
    args = parser.parse_args()

    settings = json.loads(args.config.read_text()) if args.config else {}
    recipe = {key: settings.get(key, value) for key, value in RECIPE.items()}
    recipe["phase_steps"] = args.steps or recipe["phase_steps"]
    torch.set_num_threads(args.threads)

    bank = load(args.cache, ("train",))["train"]
    assert not set(bank.names) & set(TEST_CASES), "Test cases found in training data"
    if args.holdout_validation:
        bank, _ = split_bank(bank)
    args.out.mkdir(parents=True)
    cache_hash = cache_content_sha256(args.cache)
    write_json(args.out / "config.json", recipe)

    for seed in args.seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model_config = {"dropout": recipe["dropout"]}
        model = Model(**model_config).to(args.device)
        print(f"--- seed={seed} cases={len(bank.names)} device={args.device} "
              f"params={sum(p.numel() for p in model.parameters())} ---", flush=True)
        history = []
        for phase, steps in enumerate(recipe["phase_steps"], 1):
            history.extend(run_phase(model, bank, phase, steps, rng, args.device,
                                     recipe=recipe))
        path = args.out / f"seed{seed}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"seed": seed, "state": model.state_dict(), "model_config": model_config,
                    "recipe": recipe, "train_cases": list(bank.names),
                    "cache_content_sha256": cache_hash,
                    "preprocessing_version": bank.preprocessing_version,
                    "validation_cases": list(VALIDATION_CASES) if args.holdout_validation else []},
                   temporary)
        temporary.replace(path)
        write_json(args.out / f"seed{seed}_history.json", history)
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

"""Bounded, case-held-out hyperparameter search; never loads benchmark test data.

Example: python -u -m experiments.tune --budget-minutes 15 --out runs/search
The budget sizes equal-length trials using a short CPU calibration. It is an
estimate, not a hard timeout. --steps TRUTH RANS overrides that calibration.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from data import load, scores, split_bank
from model import Model
from train import CASE_BATCH, CONTEXT, QUERIES, RECIPE, run_phase
from utils import cache_content_sha256, sha256, write_json
from validate import evaluate


# Small, predeclared screening design, not an exhaustive grid search.
CANDIDATES = {
    "baseline": {},
    "no_sigreg": {"sigreg_weight": 0.0},
    "lower_lr": {"lr": 3e-4},
    "lower_latent": {"latent_weight": 0.03},
    "with_dropout": {"dropout": 0.1},
    "lower_decay": {"weight_decay": 1e-4},
    "combined": {"lr": 5e-4, "latent_weight": 0.03, "weight_decay": 1e-4, "dropout": 0.0},
}


def trial(name, seed, config, steps, fit, validation, device, out, provenance):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model_config = {"dropout": config["dropout"]}
    model = Model(**model_config).to(device)
    start = time.monotonic()
    print(f"START {name} seed={seed} config={config} steps={steps}", flush=True)
    for phase, count in enumerate(steps, 1):
        run_phase(model, fit, phase, count, rng, device, recipe=config, log_every=250)
    metric = evaluate(model, validation)
    checkpoint = out / f"{name}_seed{seed}.pt"
    torch.save({
        "format_version": 1, "seed": seed, "state": model.state_dict(),
        "model_config": model_config, "recipe": config, "phase_steps": steps,
        "train_cases": list(fit.names), "validation_cases": list(validation.names),
        "validation": metric, "provenance": provenance,
        "preprocessing_version": fit.preprocessing_version,
        "cache_content_sha256": provenance["cache_content_sha256"],
    }, checkpoint)
    result = {"name": name, "seed": seed, "config": config, "steps": steps,
              "checkpoint": str(checkpoint), "seconds": time.monotonic() - start, **metric}
    print(f"RESULT {name} seed={seed} validation={metric['mean']:.6f} seconds={result['seconds']:.1f}", flush=True)
    return result


def calibrate(fit, device, budget_seconds, runs):
    torch.manual_seed(0)
    model = Model().to(device)
    rng = np.random.default_rng(0)
    rates = []
    for phase in (1, 2):
        # Exclude one-time optimizer/backend startup from the timing estimate.
        run_phase(model, fit, phase, 5, rng, device)
        start = time.monotonic()
        run_phase(model, fit, phase, 20, rng, device)
        rates.append((time.monotonic() - start) / 20)
    # Reserve 15% plus 20 seconds for evaluation, I/O and timing variability.
    available = max(1, budget_seconds * 0.85 - 20)
    phase2 = int(available / (runs * (2 * rates[0] + rates[1])))
    phase2 = max(25, min(1500, (phase2 // 25) * 25))
    steps = [2 * phase2, phase2]
    print(f"CALIBRATION seconds/step={rates} selected_steps={steps} runs={runs}", flush=True)
    return steps, rates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("cache.npz"))
    parser.add_argument("--out", type=Path, default=Path("runs/validation_pilot"))
    parser.add_argument("--budget-minutes", type=float, default=15)
    parser.add_argument("--steps", type=int, nargs=2, metavar=("TRUTH", "RANS"))
    parser.add_argument("--seeds", type=int, nargs=2, default=[1234, 9153])
    parser.add_argument("--candidates", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--repeat-all", action="store_true", help="Repeat every candidate with the second seed")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    start = time.monotonic()
    # Deliberately load only the train split; never load or score test labels.
    fit, validation = split_bank(load(args.cache, ("train",))["train"])
    args.out.mkdir(parents=True)
    source_dir = args.out / "sources"
    source_dir.mkdir()
    root = Path(__file__).resolve().parent.parent
    source_files = [root / name for name in (
        "data.py", "train.py", "model.py", "predict.py", "validate.py", "utils.py")]
    source_files.append(Path(__file__))
    for path in source_files:
        shutil.copy2(path, source_dir / path.name)
    provenance = {"cache_content_sha256": cache_content_sha256(args.cache),
                  "source_sha256": {p.name: sha256(p) for p in source_files},
                  "numpy": str(np.__version__), "torch": str(torch.__version__)}
    base_scores = scores(validation.base, validation)
    configs = {name: {**RECIPE, **CANDIDATES[name]} for name in args.candidates}
    print(f"SPLIT train={len(fit.names)} validation={list(validation.names)}", flush=True)
    print(f"BASELINE RANS mean={np.mean(list(base_scores.values())):.6f} per_case={base_scores}", flush=True)
    steps, rates = (args.steps, None) if args.steps else calibrate(
        fit, args.device, args.budget_minutes * 60, len(configs) + (len(configs) if args.repeat_all else 2))
    for config in configs.values():
        config["phase_steps"] = steps
    report = {
        "objective": "Equal-case mean normalized vector MAE on all held-out mesh cells",
        "note": "Pilot validation results, not benchmark test scores. No checkpoint warm starts. "
                "Validation covers hill geometry and one duct case, not unseen flow families.",
        "train_cases": list(fit.names), "validation_cases": list(validation.names),
        "baseline": {"mean": float(np.mean(list(base_scores.values()))), "per_case": base_scores},
        "steps": steps, "seeds": args.seeds, "context_seed": 8675309,
        "sampling": {"context": CONTEXT, "queries": QUERIES, "case_batch": CASE_BATCH},
        "threads": args.threads, "device": args.device, "budget_minutes": args.budget_minutes,
        "repeat_all": args.repeat_all,
        "calibration_seconds_per_step": rates, "provenance": provenance,
        "candidates": configs, "trials": [], "status": "running",
    }
    result_path = args.out / "results.json"
    write_json(result_path, report)
    for name, config in configs.items():
        report["trials"].append(trial(name, args.seeds[0], config, steps, fit, validation,
                                      args.device, args.out, provenance))
        write_json(result_path, report)
    finalists = sorted(report["trials"], key=lambda r: r["mean"])[:len(configs) if args.repeat_all else 2]
    report["finalists"] = [r["name"] for r in finalists]
    write_json(result_path, report)
    for finalist in finalists:
        name = finalist["name"]
        report["trials"].append(trial(name, args.seeds[1], configs[name], steps, fit, validation,
                                      args.device, args.out, provenance))
        write_json(result_path, report)
    ranking = []
    for finalist in finalists:
        name = finalist["name"]
        rows = [r for r in report["trials"] if r["name"] == name]
        ranking.append({"name": name, "mean": float(np.mean([r["mean"] for r in rows])),
                        "seed_scores": [r["mean"] for r in rows],
                        "per_case": {n: float(np.mean([r["per_case"][n] for r in rows])) for n in validation.names},
                        "checkpoints": [r["checkpoint"] for r in rows]})
    report["ranking"] = sorted(ranking, key=lambda r: r["mean"])
    report["selected_config"] = configs[report["ranking"][0]["name"]]
    report["seconds"] = time.monotonic() - start
    report["status"] = "complete"
    write_json(result_path, report)
    write_json(args.out / "best_config.json", report["selected_config"])
    print("FINAL " + json.dumps(report["ranking"]), flush=True)
    print(f"Saved {result_path}; elapsed={report['seconds']/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()

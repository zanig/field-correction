"""Score exported CSVs with the official Closure Challenge evaluator."""
import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
from closure_challenge import case_names, evaluation_points, evaluate_from_csv_by_case

from utils import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="?", default=Path("predictions"))
    parser.add_argument("--out", type=Path, help="Save a JSON score report")
    args = parser.parse_args()
    for name in case_names():
        path = args.predictions / f"{name}.csv"
        values = np.loadtxt(path, delimiter=",")
        if values.shape != evaluation_points(name).shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid prediction shape or nonfinite values: {path}")
    per_case = {name: float(value) for name, value in evaluate_from_csv_by_case(str(args.predictions)).items()}
    overall = float(np.mean(list(per_case.values())))
    for name, value in per_case.items():
        print(f"{name:24s} {value:.8f}")
    print(f"Overall                  {overall:.8f}")
    if args.out:
        write_json(args.out, {"overall": overall, "per_case": per_case,
                              "evaluator_version": version("closure-challenge"),
                              "predictions": str(args.predictions)})


if __name__ == "__main__":
    main()

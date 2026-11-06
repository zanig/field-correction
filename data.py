"""Case splits and per-case RANS normalization."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

FEATURES = (
    "x", "y", "z", "U_x", "U_y", "U_z", "U_mag", "log_k", "log_omega",
    "signed_log_p", "log_nut", "log_k_over_omega", "log_lt_over_dwall",
    "log_nut_over_nu", "wall_distance", "log_wall_reynolds",
)
GEOMETRY = (0, 1, 2, 14)
VELOCITY_COLS = (3, 4, 5, 6)
CONTEXT_POINTS = 2048


VALIDATION_CASES = (
    "alpha_05_10071_4048", "alpha_05_10071_2024",
    "alpha_15_7929_4048", "alpha_15_7929_2024", "AR_7_Ret_180",
)

TEST_CASES = (
    "alpha_15_13929_4048", "alpha_15_13929_2024", "alpha_05_4071_4048",
    "alpha_05_4071_2024", "AR_1_Ret_360", "AR_3_Ret_360", "AR_14_Ret_180", "NASA_2DWMH",
)

TRAIN_CASES = (
    "alpha_05_10071_3036", "alpha_05_4071_3036", "alpha_05_7071_2024",
    "alpha_05_7071_3036", "alpha_05_7071_4048", "alpha_075",
    "alpha_10_12000_2024", "alpha_10_12000_3036", "alpha_10_12000_4048",
    "alpha_10_6000_2024", "alpha_10_6000_3036", "alpha_10_6000_4048",
    "alpha_10_9000_2024", "alpha_10_9000_3036", "alpha_10_9000_4048", "alpha_125",
    "alpha_15_10929_2024", "alpha_15_10929_3036", "alpha_15_10929_4048",
    "alpha_15_13929_3036", "alpha_15_7929_3036", "AR_1_Ret_180", "AR_3_Ret_180",
    "AR_5_Ret_180", "AR_10_Ret_180", "PH_Breuer", "alpha_05_10071_2024",
    "alpha_05_10071_4048", "alpha_15_7929_4048",
    "alpha_15_7929_2024", "AR_7_Ret_180", "CBFS",
)


def family(name: str) -> str:
    if name.startswith("AR_"):
        return "duct"
    if name.startswith(("alpha_", "PH_")):
        return "hill"
    return "other"


@dataclass
class Bank:
    x: np.ndarray
    base: np.ndarray
    true: np.ndarray | None
    ids: np.ndarray
    names: tuple[str, ...]
    eval_idx: list[np.ndarray] | None = None
    preprocessing_version: int = 1

    def __post_init__(self):
        self.by_case = [np.flatnonzero(self.ids == i) for i in range(len(self.names))]
        self.center = np.stack([np.median(self.x[i], 0) for i in self.by_case]).astype("float32")
        self.scale = np.stack([
            np.maximum((np.percentile(self.x[i], 75, axis=0) - np.percentile(self.x[i], 25, axis=0)) / 1.349, 1e-6)
            for i in self.by_case
        ]).astype("float32")
        # Velocity vectors are already scaled by one reference speed in prep.py.
        self.center[:, VELOCITY_COLS] = 0.0
        self.scale[:, VELOCITY_COLS] = 1.0
        self.u_ref = np.asarray([
            max(np.linalg.norm(self.base[i], axis=1).mean(), 1e-6) for i in self.by_case
        ], dtype="float32")

    def norm(self, case: int, idx: np.ndarray, cols=None) -> np.ndarray:
        cols = np.arange(len(FEATURES)) if cols is None else np.asarray(cols)
        return ((self.x[idx][:, cols] - self.center[case, cols]) / self.scale[case, cols]).astype("float32")


def load(path: Path, splits=("train", "test")) -> dict[str, Bank]:
    with np.load(path, allow_pickle=False) as raw:
        banks = {}
        for s in splits:
            names = tuple(str(n) for n in raw[f"names_{s}"])
            eval_idx = None
            if s == "test":
                eval_pts = raw["eval_idx_test"]
                eval_ids = raw["eval_id_test"]
                eval_idx = [eval_pts[eval_ids == i] for i in range(len(names))]
            banks[s] = Bank(
                raw[f"x_{s}"],
                raw[f"base_{s}"],
                None if s == "test" else raw[f"true_{s}"],
                raw[f"id_{s}"],
                names,
                eval_idx=eval_idx,
                preprocessing_version=int(raw["preprocessing_version"]) if "preprocessing_version" in raw else 1,
            )
        return banks


def subset(bank: Bank, names: tuple[str, ...]) -> Bank:
    """Copy complete cases into an isolated bank, remapping case IDs."""
    if not names or len(set(names)) != len(names):
        raise ValueError("Case subset must be nonempty and unique")
    indices = [bank.by_case[bank.names.index(name)] for name in names]
    rows = np.concatenate(indices)
    ids = np.concatenate([np.full(len(idx), i, dtype=np.int32) for i, idx in enumerate(indices)])
    return Bank(bank.x[rows], bank.base[rows], None if bank.true is None else bank.true[rows], ids, names,
                preprocessing_version=bank.preprocessing_version)


def split_bank(bank: Bank) -> tuple[Bank, Bank]:
    """Reserve complete non-test cases for validation, never individual cells."""
    if set(bank.names) & set(TEST_CASES):
        raise ValueError("Training cache contains benchmark test cases")
    missing = set(VALIDATION_CASES) - set(bank.names)
    if missing:
        raise ValueError(f"Missing validation cases: {sorted(missing)}")
    fit_names = tuple(name for name in bank.names if name not in VALIDATION_CASES)
    return subset(bank, fit_names), subset(bank, VALIDATION_CASES)


def check_preprocessing(checkpoint: dict, bank: Bank, cache_path: Path | None = None) -> None:
    version = checkpoint.get("preprocessing_version", 1)
    if version != bank.preprocessing_version:
        raise ValueError(f"Checkpoint preprocessing v{version} does not match cache v{bank.preprocessing_version}; "
                         "use its original cache or retrain with the corrected cache")
    if cache_path is not None and checkpoint.get("cache_sha256"):
        from utils import sha256
        if sha256(cache_path) != checkpoint["cache_sha256"]:
            raise ValueError("Cache contents do not match the checkpoint's recorded cache hash")


def scores(pred: np.ndarray, bank: Bank) -> dict[str, float]:
    return {
        name: float(
            np.linalg.norm(pred[idx] - bank.true[idx], axis=1).mean()
            / max(np.linalg.norm(bank.true[idx], axis=1).mean(), 1e-12)
        )
        for name, idx in zip(bank.names, bank.by_case)
    }

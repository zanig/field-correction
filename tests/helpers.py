"""Small synthetic cases shared by regression tests."""
import numpy as np
import torch

from data import Bank, VALIDATION_CASES

torch.set_num_threads(1)


def fixture():
    names = ("fit_hill", "fit_duct", *VALIDATION_CASES)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(len(names) * 12, 16)).astype(np.float32)
    base = rng.normal(size=(len(x), 3)).astype(np.float32)
    return Bank(x, base, base + 0.1, np.repeat(np.arange(len(names)), 12), names)

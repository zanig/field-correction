"""Regression checks for held-out tuning and inference isolation."""
import unittest

import numpy as np
import torch

from data import TEST_CASES, VALIDATION_CASES, check_preprocessing, split_bank
from model import Model
from train import RECIPE, run_phase
from validate import validation_predictions
from tests.helpers import fixture


class ValidationTests(unittest.TestCase):
    def test_checkpoint_cannot_use_a_different_preprocessing_version(self):
        bank = fixture()
        bank.preprocessing_version = 2
        with self.assertRaisesRegex(ValueError, "does not match"):
            check_preprocessing({}, bank)
        check_preprocessing({"preprocessing_version": 2}, bank)

    def test_split_isolated_and_case_statistics_preserved(self):
        bank = fixture()
        fit, val = split_bank(bank)
        self.assertFalse(set(fit.names) & set(val.names))
        self.assertEqual(val.names, VALIDATION_CASES)
        for c, name in enumerate(val.names):
            old = bank.names.index(name)
            np.testing.assert_array_equal(val.true[val.by_case[c]], bank.true[bank.by_case[old]])
            np.testing.assert_array_equal(val.center[c], bank.center[old])
        val.true[:] = 999
        self.assertFalse(np.any(fit.true == 999))
        self.assertFalse(np.any(bank.true == 999))

    def test_reject_test_cases_in_fit_cache(self):
        bank = fixture()
        bank.names = (TEST_CASES[0], *bank.names[1:])
        with self.assertRaises(ValueError):
            split_bank(bank)

    def test_validation_labels_cannot_affect_predictions(self):
        _, val = split_bank(fixture())
        model = Model(dropout=0.0)
        before = validation_predictions(model, val, context_points=8, chunk=5)
        val.true[:] = np.nan
        after = validation_predictions(model, val, context_points=8, chunk=5)
        np.testing.assert_array_equal(before, after)

    def test_recipe_override_does_not_mutate_defaults(self):
        import train
        from unittest.mock import patch
        fit, _ = split_bank(fixture())
        defaults = dict(RECIPE)
        model = Model(dropout=0.0)
        with patch.object(train, "CONTEXT", 8), patch.object(train, "QUERIES", 6), patch.object(train, "CASE_BATCH", 2):
            for phase in (1, 2):
                run_phase(model, fit, phase, 1, np.random.default_rng(1), "cpu", recipe={"lr": 2e-4})
        self.assertEqual(RECIPE, defaults)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

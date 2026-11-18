"""Regression checks for the presented objective and stopped teacher path."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model import Model
from train import field_loss, run_phase, training_loss
from tests.helpers import fixture


class TrainingTests(unittest.TestCase):
    def test_field_loss_weights_cases_equally_after_normalizing(self):
        # The same absolute error has different relative error in slow/fast cases.
        true = torch.tensor([[[1., 0., 0.]] * 3, [[10., 0., 0.]] * 3])
        pred = true + torch.tensor([1., 0., 0.])
        torch.testing.assert_close(field_loss(pred, true), torch.tensor(.55))

    def test_objective_gradients_and_single_regularizer_call(self):
        z = torch.randn(3, 4, 16, requires_grad=True)
        teacher_input = torch.randn_like(z, requires_grad=True)
        with torch.no_grad():
            target = teacher_input * 1.
        pred = torch.randn(3, 5, 3, requires_grad=True)
        with patch("train.sigreg", side_effect=lambda value: value.square().mean()) as regularizer:
            loss, parts = training_loss(pred, torch.randn_like(pred), z, target)
        regularizer.assert_called_once_with(z)
        expected = .1 * 2 * (z.detach() - target) / z.numel() + .05 * 2 * z.detach() / z.numel()
        z_grad, teacher_grad, field_grad = torch.autograd.grad(loss, (z, teacher_input, pred), allow_unused=True)
        torch.testing.assert_close(z_grad, expected)
        self.assertIsNone(teacher_grad)
        self.assertGreater(field_grad.abs().sum().item(), 0)
        self.assertEqual(set(parts), {"loss", "field", "alignment", "sigreg"})

    def test_teacher_forward_is_stopped_in_both_phases(self):
        import train
        bank = fixture()
        model = Model(dropout=0.)
        observed = []
        handle = model.parts["truth_encoder"].register_forward_hook(
            lambda module, args, output: observed.append(output.requires_grad))
        with patch.object(train, "CONTEXT", 8), patch.object(train, "QUERIES", 6), patch.object(train, "CASE_BATCH", 2):
            run_phase(model, bank, 1, 1, np.random.default_rng(1), "cpu")
            self.assertEqual(observed, [True, False])
            observed.clear()
            run_phase(model, bank, 2, 1, np.random.default_rng(1), "cpu")
            self.assertEqual(observed, [False])
        handle.remove()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

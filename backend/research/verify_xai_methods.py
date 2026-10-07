"""Offline mathematical sanity checks using the actual Captum and LIME paths."""
import unittest
import numpy as np
import torch
from captum.attr import IntegratedGradients
from research.xai_comparison import lime_attribution


class MethodChecks(unittest.TestCase):
    def test_lime_known_linear_raw_logit_function(self):
        reference = np.random.default_rng(42).normal(size=(2048, 6))
        original = np.array([.25, -.5, .75, .25, -.25, .125])
        weights = np.array([1., 2., 3., .5, .25, .125])
        predict = lambda values: values @ weights + 1.2
        values, info = lime_attribution(original, reference, predict, 42)
        np.testing.assert_allclose(values, (original - reference.mean(axis=0)) * weights, atol=.025)
        self.assertGreater(info["weighted_r2"], .999)
        self.assertLess(info["point_error"], .025)
        self.assertAlmostEqual(sum(values) + info["surrogate_intercept"], float(predict(original)), delta=.025)
        second, _ = lime_attribution(original, reference, predict, 42)
        np.testing.assert_array_equal(values, second)

    def test_integrated_gradients_known_sequence_logit(self):
        weights = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])
        original = torch.tensor([[[2., 1.], [1., 3.], [2., -1.]]])
        baseline = torch.ones_like(original) * .5
        predict = lambda values: (values * weights).sum(dim=(1, 2)) + 1.2
        attribution, delta = IntegratedGradients(predict).attribute(
            original, baselines=baseline, n_steps=128, return_convergence_delta=True,
        )
        torch.testing.assert_close(attribution, (original - baseline) * weights)
        self.assertLess(abs(delta.item()), 1e-5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest
import numpy as np

from padufes20.eval.metrics import compute_metrics


class TestMetrics(unittest.TestCase):
    def test_shapes(self):
        rng = np.random.default_rng(0)
        n, c = 50, 3
        y_true = rng.integers(0, c, size=n)
        logits = rng.normal(size=(n, c))
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        y_prob = exp / exp.sum(axis=1, keepdims=True)

        out = compute_metrics(y_true=y_true, y_prob=y_prob, class_names=["a", "b", "c"])
        self.assertIn("macro_f1", out.scalar)
        self.assertEqual(out.confusion.shape, (c, c))
        self.assertEqual(len(out.per_class), c)


if __name__ == "__main__":
    unittest.main()

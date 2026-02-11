import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest
import numpy as np
import pandas as pd

from padufes20.data.splits import SplitConfig, assert_no_group_leakage, make_patientwise_splits


class TestSplits(unittest.TestCase):
    def test_patientwise_no_leakage(self):
        n = 100
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "diagnostic": rng.choice(["A", "B", "C", "D"], size=n, replace=True),
                "patient_id": rng.integers(0, 20, size=n),
            }
        )
        split = make_patientwise_splits(df, label_col="diagnostic", group_col="patient_id", cfg=SplitConfig(seed=7))
        assert_no_group_leakage(df, split, "patient_id")

        all_idx = set(split["train"]) | set(split["val"]) | set(split["test"])
        self.assertEqual(len(all_idx), n)
        self.assertEqual(len(set(split["train"]) & set(split["val"])), 0)
        self.assertEqual(len(set(split["train"]) & set(split["test"])), 0)
        self.assertEqual(len(set(split["val"]) & set(split["test"])), 0)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.data import validate_frame, feature_target
from app.splitting import split_customers
from app.metrics import binary_metrics

class Foundations(unittest.TestCase):
    def setUp(self):
        # Constructed fixture only, not credit observations or model evidence.
        self.df = pd.DataFrame({"customer_id": np.arange(400),
            "signal": np.arange(400) % 7, "default": np.tile([0, 0, 0, 1], 100)})

    def test_predictors_exclude_target_and_identity(self):
        X, y = feature_target(self.df, "default", "customer_id")
        self.assertEqual(list(X.columns), ["signal"])
        self.assertEqual(len(y), 400)

    def test_duplicate_identity_rejected(self):
        frame = pd.concat([self.df, self.df.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_frame(frame, "default", "customer_id")

    def test_invalid_target_rejected(self):
        frame = self.df.copy()
        frame.loc[0, "default"] = 2
        with self.assertRaisesRegex(ValueError, "0/1"):
            validate_frame(frame, "default", "customer_id")

    def test_split_is_disjoint_complete_and_reproducible(self):
        parts = split_customers(self.df, "default", "customer_id")
        self.assertEqual(parts, split_customers(self.df, "default", "customer_id"))
        flattened = [i for values in parts.values() for i in values]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(self.df.customer_id))
        self.assertEqual({k: len(v) for k, v in parts.items()},
                         {"train": 240, "tuning": 60, "calibration": 40, "final": 60})

    def test_metric_known_values(self):
        m = binary_metrics([0, 0, 1, 1], [.1, .2, .8, .9])
        self.assertAlmostEqual(m["brier"], .025)
        self.assertEqual(m["roc_auc"], 1.)
        self.assertEqual(m["confusion_matrix"], {"tn": 2, "fp": 0, "fn": 0, "tp": 2})

    def test_bad_probabilities_rejected(self):
        for probs in ([.2, 1.1], [.2, np.nan], [.2]):
            with self.assertRaises(ValueError):
                binary_metrics([0, 1], probs)

    def test_single_class_is_not_reported_as_zero_auc(self):
        with self.assertRaisesRegex(ValueError, "Both classes"):
            binary_metrics([0, 0], [.1, .2])

    def test_api_is_honest_and_never_trains_on_page_load(self):
        # Replaces the starter's test_starter_api_is_honest: the starter page no
        # longer exists, but the substantive properties are kept: no prediction
        # endpoint, and opening the app does not create any study or job.
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json()["status"], "ok")
            before = client.get("/api/status").json()
            client.get("/")
            after = client.get("/api/status").json()
            self.assertEqual(len(before["studies"]), len(after["studies"]))
            self.assertIsNone(after["active_job"])
            self.assertEqual(client.get("/api/predict").status_code, 404)

if __name__ == "__main__":
    unittest.main()

"""Unit tests. Run with `python -m unittest discover -s tests` (or pytest)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predictive_activation.config import SimConfig  # noqa: E402
from predictive_activation.dataset import collect  # noqa: E402
from predictive_activation.engine import WINDOW, Action, run_episode  # noqa: E402
from predictive_activation.fabric import Fabric  # noqa: E402
from predictive_activation.features import FEATURE_NAMES, window_features  # noqa: E402
from predictive_activation.policies import (FixedPoll, NoControl, PersistenceForecast, Placement,  # noqa: E402
                                            TrueForecast)

SHORT = SimConfig(ticks=300)


class FabricTests(unittest.TestCase):
    def test_deterministic_given_seed(self):
        a = run_episode(SHORT, 5, NoControl()).metrics
        b = run_episode(SHORT, 5, NoControl()).metrics
        self.assertEqual(a, b)

    def test_queue_bounded_and_nonnegative(self):
        f = Fabric(SimConfig(load_scale=1.6), 1)
        for _ in range(400):
            r = f.step()
            self.assertTrue((r.queue >= -1e-9).all())
            self.assertTrue((r.queue <= SHORT.buffer_mb + 1e-9).all())
            self.assertTrue((r.drops >= 0).all())

    def test_light_load_never_congests(self):
        res = run_episode(SimConfig(ticks=400, load_scale=0.2), 3, NoControl(), record=True)
        self.assertEqual(res.metrics["congested_pct"], 0.0)

    def test_move_takes_effect_after_control_delay(self):
        f = Fabric(SHORT, 0)
        before = f.link_of[0]
        f.schedule_move(0, 3 if before != 3 else 2)
        target = f.link_of[0] if False else (3 if before != 3 else 2)
        for _ in range(SHORT.control_delay):
            f.step()
            self.assertEqual(f.link_of[0], before)
        f.step()
        self.assertEqual(f.link_of[0], target)

    def test_overload_causes_drops(self):
        f = Fabric(SimConfig(load_scale=3.0), 0)
        drops = sum(f.step().drops.sum() for _ in range(300))
        self.assertGreater(drops, 0)


class FeatureTests(unittest.TestCase):
    def _win(self, util):
        w = np.zeros((1, len(util), 4), np.float32)
        w[0, :, 0] = util
        return w

    def test_onset_and_interval_features(self):
        util = np.zeros(WINDOW)
        for start in (20, 60, 100):          # bursts every 40 ticks, last onset 19 ticks ago
            util[start:start + 10] = 0.6
        f = window_features(self._win(util), np.array([0]), np.array([1]))[0]
        g = dict(zip(FEATURE_NAMES, f))
        self.assertEqual(g["onset_interval"], 40)
        self.assertEqual(g["since_onset"], WINDOW - 1 - 100)
        self.assertEqual(g["in_burst"], 0.0)
        self.assertAlmostEqual(g["next_onset_eta"], 40 - (WINDOW - 1 - 100))

    def test_features_depend_only_on_window(self):
        rng = np.random.default_rng(0)
        w = rng.random((4, WINDOW, 4)).astype(np.float32)
        a = window_features(w, np.zeros(4), np.ones(4))
        b = window_features(w.copy(), np.zeros(4), np.ones(4))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (4, len(FEATURE_NAMES)))


class EngineTests(unittest.TestCase):
    def test_non_clairvoyant_policies_get_no_fabric_handle(self):
        seen = []

        class Spy:
            clairvoyant = False
            name = "spy"
            def reset(self, cfg, seed): ...
            def act(self, view):
                seen.append(view.fabric)
                return Action(5)

        run_episode(SimConfig(ticks=50), 0, Spy())
        self.assertTrue(all(x is None for x in seen))

    def test_polling_cost_accounting(self):
        m = run_episode(SimConfig(ticks=600), 0, NoControl(poll=10)).metrics
        self.assertAlmostEqual(m["poll_msgs_per_min"], 4 * 6.0, delta=4.5)   # 6 polls/min x 4 links

    def test_dataset_labels_are_early_warning(self):
        d = collect(SimConfig(ticks=400), range(2), stride=3)
        self.assertEqual(d.X.shape[1], len(FEATURE_NAMES))
        self.assertTrue(set(np.unique(d.y)) <= {0, 1})
        # already-congested ticks are excluded, so queue_last should rarely exceed the threshold
        q = d.X[:, FEATURE_NAMES.index("queue_last")]
        self.assertLess((q > 0.8).mean(), 0.05)


class PolicyTests(unittest.TestCase):
    def test_true_forecast_placement_separates_colliding_flows(self):
        cfg = SimConfig(ticks=600)
        base = run_episode(cfg, 11, NoControl(30)).metrics["congested_pct"]
        ctrl = run_episode(cfg, 11, Placement(TrueForecast(), FixedPoll(1), headroom=0.85)).metrics["congested_pct"]
        self.assertLess(ctrl, base)

    def test_persistence_forecast_shape(self):
        res = run_episode(SimConfig(ticks=150), 0, Placement(PersistenceForecast(), FixedPoll(1)), record=True)
        self.assertEqual(res.metrics["iterations"] >= 0, True)


if __name__ == "__main__":
    unittest.main()

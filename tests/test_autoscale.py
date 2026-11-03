import math

import numpy as np
import pandas as pd
import pytest

from predictive_activation.autoscale import forecasters as fc
from predictive_activation.autoscale.evaluate import matched_cost
from predictive_activation.autoscale.policies import Predictive, ReactiveHPA, ReactiveRPS, Static
from predictive_activation.autoscale.simulator import (
    ClusterConfig,
    Observation,
    erlang_c,
    p95_latency_ms,
    sample_arrivals,
    simulate,
)
from predictive_activation.autoscale.traces import (
    SCENARIOS,
    Trace,
    load_azure_functions,
    select_azure_apps,
    synthetic_trace,
)

# --- traces -----------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_synthetic_traces_are_positive_and_reproducible(scenario):
    a = synthetic_trace(scenario, days=2, seed=3)
    b = synthetic_trace(scenario, days=2, seed=3)
    assert len(a.rps) == 2 * 1440
    assert (a.rps > 0).all()
    np.testing.assert_array_equal(a.rps, b.rps)
    assert not np.array_equal(a.rps, synthetic_trace(scenario, days=2, seed=4).rps)


def test_flash_crowds_spike_well_above_baseline():
    diurnal = synthetic_trace("diurnal", days=7, seed=0)
    flash = synthetic_trace("flash_crowd", days=7, seed=0)
    assert flash.rps.max() > 2.5 * diurnal.rps.max()


def test_split_and_resample():
    t = synthetic_trace("diurnal", days=3, step_s=60)
    train, test = t.split(2)
    assert len(train.rps) == 2 * 1440 and len(test.rps) == 1440
    fine = t.resample(15)
    # Exactly 4x the samples so day boundaries stay aligned (last value is held)
    assert fine.step_s == 15 and len(fine.rps) == 4 * len(t.rps)
    np.testing.assert_allclose(fine.rps[::4], t.rps)


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        synthetic_trace("tsunami")


def test_azure_loader_aggregates_top_apps(tmp_path):
    cols = {str(i): [1, 10, 100] for i in range(1, 1441)}
    for day in (1, 2):
        df = pd.DataFrame({"HashOwner": ["o"] * 3, "HashApp": ["small", "big", "huge"],
                           "HashFunction": ["f1", "f2", "f3"], "Trigger": ["http"] * 3, **cols})
        df.to_csv(tmp_path / f"invocations_per_function_md.anon.d{day:02d}.csv", index=False)
    trace = load_azure_functions(tmp_path, top_n_apps=2)
    assert len(trace.rps) == 2 * 1440
    np.testing.assert_allclose(trace.rps, 110 / 60)  # "big" + "huge" per minute, as req/s


# --- simulator --------------------------------------------------------------


def test_erlang_c_known_values():
    assert erlang_c(1, 0.5) == pytest.approx(0.5)  # M/M/1: P(wait) = rho
    assert erlang_c(2, 1.0) == pytest.approx(1 / 3)
    assert erlang_c(3, 3.0) == 1.0  # saturated


def test_latency_rises_with_load_and_backlog():
    low = p95_latency_ms(100, ready=10, mu=50, backlog_req=0)
    high = p95_latency_ms(480, ready=10, mu=50, backlog_req=0)
    backed_up = p95_latency_ms(480, ready=10, mu=50, backlog_req=2000)
    assert low < high < backed_up
    assert p95_latency_ms(10, ready=0, mu=50, backlog_req=0) == math.inf


def test_arrivals_are_fixed_by_seed():
    rps = np.full(100, 200.0)
    np.testing.assert_array_equal(sample_arrivals(rps, 15, seed=1), sample_arrivals(rps, 15, seed=1))


def test_static_overprovisioned_cluster_meets_slo():
    cfg = ClusterConfig()
    arrivals = sample_arrivals(np.full(400, 300.0), cfg.step_s)
    r = simulate(arrivals, Static(20), cfg)
    assert r.metrics["slo_violation_min"] == 0
    assert r.metrics["dropped_pct"] == 0
    assert r.metrics["pod_hours"] == pytest.approx(20 * 400 * cfg.step_s / 3600)


def test_underprovisioned_cluster_backs_up_and_drops():
    cfg = ClusterConfig(min_replicas=2, max_replicas=2)
    r = simulate(sample_arrivals(np.full(200, 500.0), cfg.step_s), Static(2), cfg)
    assert r.metrics["slo_violation_pct"] > 90
    assert r.metrics["dropped_pct"] > 50


def test_new_pods_wait_for_startup_delay():
    cfg = ClusterConfig(startup_s=60, min_replicas=1)

    class JumpTo10(Static):
        def desired_replicas(self, obs):
            return 10 if obs.t >= 5 else 1

    r = simulate(np.zeros(20), JumpTo10(1), cfg)
    lead = math.ceil(60 / cfg.step_s)
    assert r.ready[5 + lead - 1] == 1
    assert r.ready[5 + lead] == 10
    assert r.pending[5] == 9


def test_policy_never_sees_current_or_future_load():
    seen = []

    class Spy(Static):
        def desired_replicas(self, obs):
            seen.append(len(obs.observed_rps) == obs.t)
            return 4

    simulate(sample_arrivals(np.full(50, 100.0), 15), Spy(4))
    assert all(seen)


# --- policies ---------------------------------------------------------------


def _obs(t=10, rps=None, util=0.5, ready=10, pending=0, cfg=None):
    rps = np.full(t, 300.0) if rps is None else np.asarray(rps, dtype=float)
    return Observation(t, rps, util, ready, pending, cfg or ClusterConfig())


def test_hpa_tolerance_and_scale_up_limit():
    hpa = ReactiveHPA(target=0.6)
    hpa.reset(ClusterConfig())
    assert hpa.desired_replicas(_obs(util=0.63, ready=10)) == 10  # within 10% tolerance
    hpa.reset(ClusterConfig())
    assert hpa.desired_replicas(_obs(util=1.0, ready=2)) == 4  # ceil(2/0.6)=4, under max(+100%, +4)
    hpa.reset(ClusterConfig())
    # Even far over target, one sync can at most double (or add 4)
    big = ReactiveRPS(target=0.1)
    big.reset(ClusterConfig())
    assert big.desired_replicas(_obs(rps=[5000.0] * 10, ready=10)) == 20


def test_hpa_scale_down_is_stabilized():
    hpa = ReactiveRPS(target=0.6, downscale_window_s=60)
    hpa.reset(ClusterConfig())
    high = hpa.desired_replicas(_obs(rps=[600.0] * 10, ready=20))
    assert high == 20
    # Load drops: for the next 60 s worth of syncs, the earlier max holds
    assert hpa.desired_replicas(_obs(rps=[60.0] * 10, ready=20)) == 20
    for _ in range(4):
        last = hpa.desired_replicas(_obs(rps=[60.0] * 10, ready=20))
    assert last == 2


def test_predictive_uses_forecast_for_step_t():
    forecast = np.array([0.0, 300.0, 900.0])
    p = Predictive(forecast, target=0.6, downscale_window_s=0)
    p.reset(ClusterConfig())
    assert p.desired_replicas(_obs(t=2, rps=[10.0, 10.0])) == math.ceil(900 / 30)


def test_hybrid_never_below_reactive():
    p = Predictive(np.zeros(5), target=0.6, reactive_floor=True, downscale_window_s=0)
    p.reset(ClusterConfig())
    assert p.desired_replicas(_obs(t=3, rps=[600.0] * 3, ready=20)) == 20


# --- forecasters ------------------------------------------------------------


def test_window_peak_target():
    y = fc.window_peak_target(np.arange(10.0), lead=2, window=3)
    assert y[0] == 4  # max(x[2:5])
    assert np.isnan(y[-1])


@pytest.mark.parametrize("forecaster", [fc.LastValue(), fc.SeasonalNaive(), fc.Holt(), fc.GBMQuantile(max_iter=30)])
def test_forecasters_do_not_look_ahead(forecaster):
    # Changing the future must not change any earlier prediction.
    step = 300
    trace = synthetic_trace("flash_crowd", days=3, step_s=step, seed=0).rps
    lead, window = 2, 2
    forecaster.fit(trace[: 2 * 288], step, lead, window)
    base = forecaster.predict(trace, step, lead, window)
    altered = trace.copy()
    t = 700
    altered[t:] *= 5
    changed = forecaster.predict(altered, step, lead, window)
    np.testing.assert_allclose(base[: t + 1], changed[: t + 1])


def test_gbm_quantile_mostly_covers_the_peak():
    step = 300
    trace = synthetic_trace("diurnal", days=10, step_s=step, seed=0).rps
    cut = 7 * 288
    g = fc.GBMQuantile(quantile=0.9, max_iter=100).fit(trace[:cut], step, 1, 2)
    m = fc.evaluate_forecast(fc.window_peak_target(trace, 1, 2)[cut:], g.predict(trace, step, 1, 2)[cut:])
    assert m["under_forecast_pct"] < 30  # q90 should under-forecast ~10%, allow slack


def test_oracle_is_perfect():
    x = synthetic_trace("noisy", days=1, step_s=300).rps
    y = fc.window_peak_target(x, 1, 2)
    m = fc.evaluate_forecast(y, fc.Oracle().predict(x, 300, 1, 2))
    assert m["mae"] == 0


# --- evaluation -------------------------------------------------------------


def test_matched_cost_interpolates_at_reference_cost():
    runs = pd.DataFrame([
        {"scenario": "s", "seed": 0, "startup_s": 60, "policy": "reactive_rps", "target": t, "pod_hours": c, "slo_violation_min": v}
        for t, c, v in [(0.5, 120, 10), (0.6, 100, 20), (0.7, 80, 40)]
    ] + [
        {"scenario": "s", "seed": 0, "startup_s": 60, "policy": "predictive_gbm", "target": t, "pod_hours": c, "slo_violation_min": v}
        for t, c, v in [(0.5, 110, 2), (0.6, 90, 6)]
    ] + [
        {"scenario": "s", "seed": 0, "startup_s": 60, "policy": "too_cheap", "target": 0.6, "pod_hours": 10, "slo_violation_min": 99},
    ])
    m = matched_cost(runs).set_index("policy").slo_violation_min
    assert m["reactive_rps"] == 20
    assert m["predictive_gbm"] == pytest.approx(4)  # halfway between 90 and 110
    assert np.isnan(m["too_cheap"])  # reference cost outside its range: no extrapolation


def test_trace_dataclass_steps_per_day():
    assert Trace("x", np.zeros(10), 15).steps_per_day == 5760


# --- Azure per-app selection --------------------------------------------------


def _fake_azure_csvs(tmp_path, days=2):
    rng = np.random.default_rng(0)
    apps = {  # name: (trigger, per-minute generator)
        "http_big": ("http", lambda: np.full(1440, 600.0)),
        "http_mid": ("http", lambda: np.full(1440, 300.0)),
        "http_spiky": ("http", lambda: np.where(np.arange(1440) % 10 == 0, 900.0, 30.0)),
        "http_sparse": ("http", lambda: np.where(np.arange(1440) % 2 == 0, 500.0, 0.0)),  # active 50%: excluded
        "timer_big": ("timer", lambda: np.full(1440, 5000.0)),  # wrong trigger: excluded
    }
    for day in range(1, days + 1):
        rows = []
        for name, (trig, gen) in apps.items():
            counts = gen() + rng.integers(0, 3, 1440)
            rows.append({"HashOwner": "o", "HashApp": name, "HashFunction": name + "_f", "Trigger": trig,
                         **{str(i + 1): c for i, c in enumerate(counts)}})
        pd.DataFrame(rows).to_csv(tmp_path / f"invocations_per_function_md.anon.d{day:02d}.csv", index=False)


def test_azure_cache_and_selection_rule(tmp_path):
    from predictive_activation.autoscale.traces import build_azure_cache

    _fake_azure_csvs(tmp_path)
    cache = build_azure_cache(tmp_path, top_n=10)
    z = np.load(cache, allow_pickle=True)
    assert list(z["apps"])[0] == "timer_big"  # volume order
    assert z["counts"].shape == (5, 2 * 1440)

    chosen = select_azure_apps(cache, busiest=1, burstiest=1)
    # timer excluded (trigger), sparse excluded (activity); busiest http is rank 2,
    # burstiest of the rest is the spiky app, never a duplicate of a busy pick
    assert set(chosen) == {"az_busy_002", "az_burst_005"}


def test_azure_trace_is_rescaled(tmp_path, monkeypatch):
    from predictive_activation.autoscale import traces
    from predictive_activation.autoscale.traces import build_azure_cache

    _fake_azure_csvs(tmp_path)
    cache = build_azure_cache(tmp_path, top_n=10)
    t = traces.azure_trace("az_busy_002", cache=cache, mean_rps=250)
    assert t.step_s == 60 and len(t.rps) == 2 * 1440
    assert t.rps.mean() == pytest.approx(250)

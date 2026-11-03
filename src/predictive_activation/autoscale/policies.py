"""Scaling policies. Each returns a desired replica count from an Observation.

Reactive baselines mirror the Kubernetes HorizontalPodAutoscaler algorithm
(v2 defaults): 15 s sync, 10% tolerance, scale-up limited to max(+100%, +4 pods)
per sync, 300 s scale-down stabilization window.
"""

import math
from collections import deque

import numpy as np


class Policy:
    name = "base"

    def reset(self, config):
        self.cfg = config

    def desired_replicas(self, obs):
        raise NotImplementedError


class _StabilizedDown:
    """Scale-down stabilization: act on the max recommendation of the last N seconds."""

    def __init__(self, window_s):
        self.window_s = window_s

    def reset(self, step_s):
        self.history = deque(maxlen=max(1, int(self.window_s // step_s)))

    def __call__(self, recommendation):
        self.history.append(recommendation)
        return max(self.history)


class Static(Policy):
    """Fixed replica count, e.g. sized for the training peak."""

    def __init__(self, replicas):
        self.replicas = replicas
        self.name = f"static_{replicas}"

    def desired_replicas(self, obs):
        return self.replicas


class ReactiveHPA(Policy):
    """Kubernetes HPA scaling on CPU utilization (capped at 100%, so it can only
    grow ~1/target per sync while saturated)."""
    name = "hpa_cpu"

    def __init__(self, target=0.6, tolerance=0.1, downscale_window_s=300):
        self.target, self.tolerance = target, tolerance
        self.stabilize = _StabilizedDown(downscale_window_s)

    def reset(self, config):
        super().reset(config)
        self.stabilize.reset(config.step_s)

    def recommend(self, obs):
        current = obs.ready + obs.pending
        if obs.ready == 0:
            return current
        ratio = obs.utilization / self.target
        if abs(ratio - 1) <= self.tolerance:
            return current
        # Unready pods count as 0% busy, so usage is spread over ready pods only
        return math.ceil(obs.ready * ratio)

    def desired_replicas(self, obs):
        current = obs.ready + obs.pending
        desired = self.stabilize(self.recommend(obs))
        if desired > current:
            desired = min(desired, max(2 * current, current + 4))
        return desired


class ReactiveRPS(ReactiveHPA):
    """HPA/KEDA-style scaling on observed request rate (not capped like CPU)."""
    name = "reactive_rps"

    def recommend(self, obs):
        if len(obs.observed_rps) == 0:
            return obs.ready + obs.pending
        rps = obs.observed_rps[-1]
        return math.ceil(rps / (self.cfg.pod_capacity_rps * self.target))


class Predictive(Policy):
    """Scale to cover a forecast of the peak load once new pods are ready.

    `forecast[t]` must use only data before t (see forecasters.py). No HPA
    scale-up rate limit applies: this models a custom controller or KEDA
    external scaler setting replicas directly.
    """

    def __init__(self, forecast, target=0.6, downscale_window_s=300, name=None, reactive_floor=False):
        self.forecast = np.asarray(forecast, dtype=float)
        self.target = target
        self.reactive_floor = reactive_floor
        self.stabilize = _StabilizedDown(downscale_window_s)
        self.floor = ReactiveRPS(target=target) if reactive_floor else None
        self.name = name or ("hybrid" if reactive_floor else "predictive")

    def reset(self, config):
        super().reset(config)
        self.stabilize.reset(config.step_s)
        if self.floor:
            self.floor.reset(config)

    def desired_replicas(self, obs):
        rec = math.ceil(self.forecast[obs.t] / (self.cfg.pod_capacity_rps * self.target))
        if self.floor:
            rec = max(rec, self.floor.recommend(obs))
        return self.stabilize(rec)

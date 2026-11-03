"""Discrete-time cluster simulator for one autoscaled service.

Each step (default 15 s, the Kubernetes HPA sync period):
  1. Pods whose startup delay has elapsed become ready.
  2. The policy sees the *past* (observed load, utilization, replica counts) and
     sets a desired replica count. New pods start Pending; removed pods go away
     immediately (pending pods are cancelled first).
  3. Requests arrive (pre-sampled Poisson counts) and are served by ready
     pods. Work that can't be served joins a backlog; backlog older than the
     request timeout is dropped.
  4. Latency is estimated from an M/M/c queue on top of the backlog delay.

The latency model is an approximation (M/M/c steady state per step plus a fluid
backlog), documented so results are reproducible rather than exact.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln, pdtr


@dataclass
class ClusterConfig:
    pod_capacity_rps: float = 50.0  # service rate mu per pod (20 ms mean service time)
    startup_s: float = 60.0  # pod scheduling + image pull + warm-up
    min_replicas: int = 2
    initial_target: float = 0.6  # warm start: size the first step for this utilization
    max_replicas: int = 200
    slo_p95_ms: float = 250.0
    request_timeout_s: float = 10.0
    step_s: int = 15


@dataclass
class Observation:
    """What a policy may see at decision time: only the past."""
    t: int  # step index
    observed_rps: np.ndarray  # arrivals/s for steps < t (metrics lag one step)
    utilization: float  # last step's busy fraction of ready pods, capped at 1 (like CPU)
    ready: int
    pending: int
    config: ClusterConfig


@dataclass
class SimResult:
    policy: str
    trace: str
    config: ClusterConfig
    rps: np.ndarray
    ready: np.ndarray
    pending: np.ndarray
    p95_ms: np.ndarray
    served: np.ndarray
    dropped: np.ndarray
    metrics: dict = field(default_factory=dict)


def erlang_c(c, offered):
    """Probability an arrival waits in an M/M/c queue (offered = lambda/mu)."""
    if c <= 0 or offered >= c:
        return 1.0
    if offered <= 0:
        return 0.0
    # Erlang B = Poisson pmf(c) / Poisson cdf(c), O(1) in c; then convert to Erlang C.
    log_pmf = -offered + c * math.log(offered) - gammaln(c + 1)
    b = math.exp(log_pmf) / pdtr(c, offered)
    rho = offered / c
    return b / (1 - rho + rho * b)


def p95_latency_ms(arrival_rps, ready, mu, backlog_req):
    """Approximate p95 response time: service + M/M/c queueing + backlog drain."""
    if ready <= 0:
        return math.inf
    capacity = ready * mu
    backlog_delay = backlog_req / capacity
    offered = arrival_rps / mu
    if offered >= ready:
        # Saturated: queue grows within the step; the backlog term carries it.
        queue_p95 = 0.0
    else:
        pw = erlang_c(ready, offered)
        # P(Wq > t) = pw * exp(-(c*mu - lambda) t)
        queue_p95 = math.log(pw / 0.05) / (capacity - arrival_rps) if pw > 0.05 else 0.0
    service_p95 = -math.log(0.05) / mu  # exponential service time
    return 1000.0 * (service_p95 + queue_p95 + backlog_delay)


def sample_arrivals(trace_rps, step_s, seed=0):
    """Poisson request counts per step. Sampled once up front so every policy
    faces exactly the same traffic and forecasts can be precomputed."""
    rng = np.random.default_rng(seed)
    return rng.poisson(np.asarray(trace_rps) * step_s).astype(np.float64)


def simulate(arrivals, policy, config=None, trace_name="trace"):
    """Run `policy` against per-step request counts (see sample_arrivals)."""
    cfg = config or ClusterConfig()
    arrivals = np.asarray(arrivals, dtype=np.float64)
    observed_rps = arrivals / cfg.step_s
    n = len(arrivals)
    dt = cfg.step_s
    startup_steps = math.ceil(cfg.startup_s / dt)

    # Warm start sized for the first step's load, so no policy pays for a cold
    # cluster at t=0 (that would add the same penalty to every policy).
    initial = math.ceil(observed_rps[0] / (cfg.pod_capacity_rps * cfg.initial_target)) if n else 0
    ready = int(np.clip(initial, cfg.min_replicas, cfg.max_replicas))
    pending = deque()  # ready-at step index for each pending pod
    backlog = 0.0
    utilization = 0.0

    out = {k: np.zeros(n) for k in ("rps", "ready", "pending", "p95", "served", "dropped")}
    policy.reset(cfg)

    for t in range(n):
        while pending and pending[0] <= t:
            pending.popleft()
            ready += 1

        obs = Observation(t, observed_rps[:t], utilization, ready, len(pending), cfg)
        desired = int(np.clip(policy.desired_replicas(obs), cfg.min_replicas, cfg.max_replicas))
        total = ready + len(pending)
        if desired > total:
            pending.extend([t + startup_steps] * (desired - total))
        elif desired < total:
            remove = total - desired
            while remove and pending:
                pending.pop()  # cancel the newest pending pod first
                remove -= 1
            ready -= remove

        capacity = ready * cfg.pod_capacity_rps * dt
        work = backlog + arrivals[t]
        served = min(work, capacity)
        backlog = work - served
        # Requests waiting longer than the timeout are dropped (clients give up)
        max_backlog = ready * cfg.pod_capacity_rps * cfg.request_timeout_s
        dropped = max(0.0, backlog - max_backlog)
        backlog -= dropped

        arrival_rps = observed_rps[t]
        out["rps"][t] = arrival_rps
        out["ready"][t] = ready
        out["pending"][t] = len(pending)
        out["p95"][t] = p95_latency_ms(arrival_rps, ready, cfg.pod_capacity_rps, backlog)
        out["served"][t] = served
        out["dropped"][t] = dropped
        utilization = min(1.0, served / capacity) if capacity else 1.0

    result = SimResult(policy.name, trace_name, cfg, out["rps"], out["ready"], out["pending"],
                       out["p95"], out["served"], out["dropped"])
    result.metrics = summarize(result)
    return result


def summarize(r):
    cfg = r.config
    dt = cfg.step_s
    violating = r.p95_ms > cfg.slo_p95_ms
    arrivals = r.rps * dt
    total_req = arrivals.sum()
    finite_p95 = np.where(np.isfinite(r.p95_ms), r.p95_ms, cfg.request_timeout_s * 1000)
    return {
        "slo_violation_min": float(violating.sum() * dt / 60),
        "slo_violation_pct": float(100 * violating.mean()),
        # Share of requests that arrived during an SLO-violating step or were dropped
        "bad_request_pct": float(100 * (arrivals[violating].sum() + r.dropped.sum()) / max(total_req, 1)),
        "dropped_pct": float(100 * r.dropped.sum() / max(total_req, 1)),
        "p95_of_step_p95_ms": float(np.percentile(finite_p95, 95)),
        "max_p95_ms": float(finite_p95.max()),
        # You pay for pods while they start, so cost counts pending + ready
        "pod_hours": float((r.ready + r.pending).sum() * dt / 3600),
        "avg_replicas": float(r.ready.mean()),
        "scale_events": int(np.count_nonzero(np.diff(r.ready + r.pending))),
    }

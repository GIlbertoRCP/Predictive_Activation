"""Simulation configuration. All experiment parameters live here so results are reproducible."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobSpec:
    """One distributed-training job: alternates compute and an all-reduce communication burst."""

    name: str
    peak_mbps: float          # demand while communicating (after ramp-up)
    comm_seconds: float       # ideal all-reduce duration at peak rate on an uncongested link
    compute_mean_s: float     # mean compute phase length
    compute_jitter_s: float   # std-dev of compute phase length (per iteration)
    home_link: int            # default (ECMP-hash) core link

    @property
    def volume_mb(self) -> float:
        return self.peak_mbps * self.comm_seconds


# Eight jobs on four spine (core) links. ECMP hashing is imbalanced on purpose-of-realism: three jobs
# hash onto spine 0 and only one onto spine 3, which is exactly the polarization traffic engineering is
# meant to repair. Periods are roughly 30-55 s with different duty cycles.
DEFAULT_JOBS = (
    JobSpec("job0", 520.0, 14.0, 16.0, 1.5, 0),
    JobSpec("job1", 480.0, 12.0, 28.0, 2.0, 0),
    JobSpec("job2", 500.0, 13.0, 22.0, 2.0, 0),
    JobSpec("job3", 550.0, 16.0, 27.0, 2.0, 1),
    JobSpec("job4", 450.0, 13.0, 40.0, 2.5, 1),
    JobSpec("job5", 510.0, 15.0, 20.0, 2.0, 2),
    JobSpec("job6", 470.0, 12.0, 34.0, 2.5, 2),
    JobSpec("job7", 530.0, 14.0, 31.0, 2.0, 3),
)


@dataclass(frozen=True)
class SimConfig:
    ticks: int = 1800                 # 1 tick = 1 second; 30-minute episodes
    n_links: int = 4                  # spine (core) links
    capacity_mbps: float = 1000.0
    buffer_mb: float = 100.0          # 100 Mb at 1 Gbps = 100 ms of queueing delay when full
    congestion_frac: float = 0.5      # SLO breach: queue above 50% of buffer (>= 50 ms queueing delay)
    base_latency_ms: float = 0.5
    bg_mean_mbps: float = 80.0        # background "mice" traffic per link (AR(1))
    bg_rho: float = 0.9
    bg_sigma_mbps: float = 12.0
    ramp_ticks: int = 3               # slow-start style ramp at the start of each burst
    control_delay: int = 2            # ticks between a decision and the flow rule taking effect
    telemetry_noise: float = 0.03     # multiplicative measurement noise on polled counters
    load_scale: float = 1.0           # scales every job's peak demand (sensitivity studies)
    jobs: tuple = DEFAULT_JOBS

    @property
    def congestion_queue_mb(self) -> float:
        return self.congestion_frac * self.buffer_mb

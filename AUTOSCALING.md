# Predictive Autoscaling: Method and Caveats

This page explains how the numbers in the README's Results section are produced and where they shouldn't be trusted. Code: `src/predictive_activation/autoscale/`. Raw output: `benchmarks/autoscale/` (synthetic) and `benchmarks/autoscale_azure/` (real traffic).

## Question

New pods take time to become useful (scheduling, image pull, warm-up). A reactive autoscaler only sees overload once it has happened, so users absorb the startup delay. **Does forecasting load and scaling early reduce SLO violations without spending more?**

## Workloads

`traces.py` generates 14 days of per-minute request rates, resampled to 15 s:

| Scenario | Shape |
| :--- | :--- |
| `diurnal` | Two daily peaks, weekend dip, mild autocorrelated noise |
| `flash_crowd` | Diurnal plus about 2 random spikes a day: 3–8× load, 1–3 min rise, 10–40 min decay |
| `ramp` | Diurnal on top of 3× growth over the 14 days |
| `noisy` | Weak seasonality, heavy autocorrelated noise |

Requests per step are Poisson around the trace and sampled once per (scenario, seed), so every policy sees identical traffic.

## Real traffic: Azure Functions 2019

Microsoft's [Azure Functions trace](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md) records per-minute invocation counts for every function over 14 days in July 2019.

**Setup.** About 1.9 GB on disk; `data/azure_functions_2019/` is gitignored.

```bash
mkdir -p data/azure_functions_2019 && cd data/azure_functions_2019
curl -LO https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz
sha256sum azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz   # aff8b3ca7240a41a109e4ee598e0a96e45fcb92e7b8395ac19cb3748cd260d89
tar -xJf azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz --wildcards 'invocations_per_function_md.anon.d*.csv'
cd ../.. && python -m predictive_activation.autoscale.traces   # caches the 200 busiest apps (~1 min)
```

**App selection** (`select_azure_apps`), fixed before any results were looked at: among apps with an HTTP trigger and traffic in at least 95% of minutes, take the 8 busiest by volume (`az_busy_<rank>`), then the 4 burstiest of the remainder by the 99.9th percentile of minute-to-minute change relative to the mean (`az_burst_<rank>`). Each app is rescaled to a mean of 250 req/s, which keeps its shape and puts it on the same cluster settings as the synthetic runs. The traffic shape is fixed; the 3 seeds only change the Poisson arrivals. Because the burstiest app peaks at about 40× its mean, the replica cap is raised to 1,000 for these runs.

The top apps by volume are mostly smooth (peak/mean 1.4–5.7×). The selected bursty apps reach 7–41×. One busy app (`az_busy_024`) gets a 4× burst for exactly one minute every 10 minutes, which looks like a cron job.

## Cluster model

`simulator.py` runs one service in 15 s steps, the Kubernetes HPA sync period.

| Parameter | Default |
| :--- | :--- |
| Pod capacity | 50 req/s (20 ms mean service time) |
| Pod startup delay | 60 s (swept 15–180 s) |
| Replicas | 2–200, warm-started for the first step's load |
| SLO | p95 latency ≤ 250 ms |
| Request timeout | 10 s (older backlog is dropped) |

Each step: pods that finished starting become ready, the policy chooses a replica count from **past observations only**, arrivals are served by ready pods, and unserved work becomes backlog. Latency is approximated as exponential service p95 + M/M/c (Erlang C) queueing p95 + backlog drain time. Cost is pod-hours, counting pods while they start.

## Forecasters

All forecasters predict the **peak request rate over the first minute after a pod requested now would be ready**: `max(rps[t+lead : t+lead+4])`, where `lead` is the startup delay in steps. Every prediction for step `t` uses only data before `t`. A test perturbs the future and checks that earlier predictions don't change.

| Forecaster | Idea |
| :--- | :--- |
| `last_value` | Persistence |
| `seasonal_naive` | Yesterday's peak for the same window, scaled by today's level vs yesterday's |
| `holt` | Level + trend smoothing, never forecasting a decline |
| `gbm_q90` | Gradient-boosted **90th-percentile** regression on lags, rolling means/maxes, slope, the window peak 5/10/15/30/60 min ago, yesterday and last week, and time of day |
| `oracle` | The true future (an upper bound) |

## Policies

| Policy | Behaviour |
| :--- | :--- |
| `hpa_cpu` | Kubernetes HPA v2 algorithm on CPU utilization: 10% tolerance, scale-up ≤ max(+100%, +4 pods) per sync, 300 s scale-down stabilization. CPU caps at 100%, so it under-reads overload. |
| `reactive_rps` | Same algorithm on observed request rate (HPA with a custom metric, or KEDA) |
| `predictive_*` | `ceil(forecast / (pod_capacity × target))`, same 300 s scale-down stabilization, no scale-up rate limit |
| `predictive_naive` | Predictive controller with the `last_value` forecast. **This is the ablation**: any gap between it and `predictive_gbm` comes from forecasting, not from removing the rate limit. |
| `hybrid_gbm` | `max(predictive_gbm, reactive_rps recommendation)` |

## Equal-cost comparison

A policy can always buy fewer violations with more pods, so comparing at one target utilization is unfair. Each policy runs at targets 0.4–0.8, giving a cost-vs-violations curve. The reference cost is `reactive_rps` at target 0.6. Every policy's violation minutes are linearly interpolated at that cost per (scenario, seed, startup), with no extrapolation outside a policy's measured range. `matched_cost.csv` has every value; the README table averages 3 seeds.

## Results

**Synthetic flash crowds** (mean of 3 seeds): the predictive GBM cuts SLO-violation minutes by 50% (15–30 s startup), 38% (60 s), 22% (120 s) and 19% (180 s) compared with the reactive request-rate scaler at equal cost. It wins in every seed and at every startup delay. The naive-forecast ablation is within 0.5 minutes of `reactive_rps` in every run, so the gain comes from the forecaster. On `diurnal` and `noisy` traffic, every request-rate policy is at about 0 violation minutes at the reference cost. On `ramp`, the GBM under-forecasts in 41% of windows, because tree models can't predict values above their training range; `hybrid_gbm` stays at 0.

**Azure apps**, comparing `hybrid_gbm` with `reactive_rps` over 12 apps × 5 startup delays = 60 cases (mean of 3 seeds; differences within 0.5 minutes count as ties):

| Outcome | Cases |
| :--- | ---: |
| Hybrid has fewer violation minutes at equal cost | 22 |
| Tie | 28 |
| Hybrid has more violation minutes (by 0.7 and 0.9 min) | 2 |
| Hybrid can't get as cheap as the reference; at 1–15% more pod-hours it has 32–99.9% fewer violation minutes | 8 |

The pure `predictive_gbm` is less reliable on real traffic. On `az_busy_016` and `az_burst_075`, the test week runs busier than the training week. The GBM under-forecasts so badly that its whole cost curve sits below the reference cost, with up to thousands of violation minutes. That is the same extrapolation failure as `ramp`, and the reactive floor is what fixes it. The naive-forecast ablation is within 1.1 minutes of `reactive_rps` in every Azure case. See `matched_cost.csv` and `summary.json` in each results folder for every number.

## Caveats

- **Features were added after seeing real data.** The sub-hourly peak features (5–60 min ago) were added after the first Azure run exposed the 10-minute cron pattern in `az_busy_024`. Both grids were then rerun with the final features. The app selection rule did not change, but the GBM was improved with knowledge of the test traces, so the Azure gains are somewhat optimistic. A clean test would freeze the features and evaluate on the trace's second week only, or on a different dataset.
- **Rescaled serverless traffic.** Azure Functions invocations are a stand-in for a web service's requests. Rescaling every app to 250 req/s changes the Poisson noise level relative to the original volumes.
- **Synthetic flash crowds.** The flash-crowd process (Poisson onsets, exponential decay) is a modelling choice, and the size of the gain depends on it.
- **Approximate latency model.** M/M/c per step plus a fluid backlog is not a real service. Absolute latencies are indicative; comparisons between policies are the point.
- **No cold-start CPU spike, no readiness flapping, no node autoscaling.** Pods are either starting or fully ready, and nodes are assumed available.
- **One service.** No dependencies between services and no shared node capacity.
- **Short training window.** 7 days with a fixed GBM configuration; no hyperparameter search.

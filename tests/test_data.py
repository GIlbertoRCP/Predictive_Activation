import numpy as np

from predictive_activation import config
from predictive_activation.data import add_labels, build_sequences, load_telemetry


def test_load_telemetry_sorts_by_port_then_time(telemetry_csv):
    df = load_telemetry(telemetry_csv)
    for _, group in df.groupby(["switch_id", "port_no"]):
        assert group["timestamp"].is_monotonic_increasing


def test_labels_use_raw_milliseconds(telemetry_csv):
    # Regression: labels were once computed after MinMax scaling, which made
    # every label 0. Synthetic data has spikes at 150 ms, above the 100 ms threshold.
    df = add_labels(load_telemetry(telemetry_csv))
    assert df["is_congested"].sum() > 0
    assert (df.loc[df["is_congested"] == 1, "latency_ms"] > config.LATENCY_THRESHOLD_MS).all()


def test_future_label_is_next_step_within_port(telemetry_csv):
    df = add_labels(load_telemetry(telemetry_csv))
    for _, group in df.groupby(["switch_id", "port_no"]):
        np.testing.assert_array_equal(group["future_congestion"].values[:-1], group["is_congested"].values[1:])
        assert np.isnan(group["future_congestion"].values[-1])  # never borrows from the next port


def test_build_sequences_shapes_and_alignment(telemetry_csv):
    X, y, scaler = build_sequences(telemetry_csv)
    n_ports, n_steps = 3, 20
    # Each port loses its last row (no future label), then yields len - L + 1 windows
    expected = n_ports * ((n_steps - 1) - config.SEQUENCE_LENGTH + 1)
    assert X.shape == (expected, config.SEQUENCE_LENGTH, len(config.FEATURES))
    assert y.shape == (expected,)
    assert X.dtype == np.float32
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert set(np.unique(y)) <= {0.0, 1.0}

    # Spikes are at t = 6, 13 for each port; the window ending at t = 5 must predict 1.
    first_port_y = y[: (n_steps - 1) - config.SEQUENCE_LENGTH + 1]
    window_end = 5
    assert first_port_y[window_end - (config.SEQUENCE_LENGTH - 1)] == 1.0

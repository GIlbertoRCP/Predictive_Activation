import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Page Config ---
st.set_page_config(page_title="Predictive Activation SDN", layout="wide")
st.title("Predictive Activation: Adaptive SDN Telemetry")
st.markdown("Comparing RL-Driven Telemetry vs. Static Polling Baselines")

# --- Constants (kept consistent with train_lstm.py / train_gru.py) ---
SEQUENCE_LENGTH = 5
LATENCY_THRESHOLD_MS = 100.0
RISK_THRESHOLD = 0.70  # README's documented escalation threshold
FEATURES = ['rx_mbps', 'tx_mbps', 'rx_loss', 'tx_loss', 'avg_queue_depth', 'latency_ms']


# --- Model Definitions (identical to training scripts) ---
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


class GRUPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


@st.cache_resource
def load_model(model_class, weights_path):
    """Loads model weights if available; returns None if the file is missing."""
    if not os.path.exists(weights_path):
        return None
    device = torch.device('cpu')
    model = model_class().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model


@st.cache_data
def load_and_score():
    """
    Loads the telemetry dataset, runs REAL LSTM/GRU inference per (switch, port)
    sliding window, and computes honest evaluation metrics.

    BUG FIX vs previous version
    ---------------------------
    The previous dashboard fabricated `lstm_risk` with `np.random.uniform` and
    displayed hardcoded accuracy strings ("100.00%", "98.50%"). This version
    loads `lstm_forecaster.pth` / `gru_forecaster.pth` and computes every
    metric from real model output against real labels.
    """
    if not os.path.exists('telemetry_dataset.csv'):
        return pd.DataFrame(), {}

    df = pd.read_csv('telemetry_dataset.csv')
    df = df.sort_values(by=['switch_id', 'port_no', 'timestamp']).reset_index(drop=True)

    # Ground-truth labels from RAW latency_ms (matches train_lstm.py exactly).
    df['is_congested'] = (df['latency_ms'] > LATENCY_THRESHOLD_MS).astype(int)
    df['future_congestion'] = (
        df.groupby(['switch_id', 'port_no'])['is_congested'].shift(-1)
    )

    # Build a scaled feature copy for model input. Raw columns are kept for plotting.
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df[FEATURES]),
        columns=FEATURES, index=df.index
    )

    lstm = load_model(LSTMPredictor, 'lstm_forecaster.pth')
    gru  = load_model(GRUPredictor,  'gru_forecaster.pth')

    df['lstm_risk'] = np.nan
    df['gru_risk']  = np.nan

    # Slide a SEQUENCE_LENGTH window per (switch, port). Never cross groups.
    for _, group_idx in df.groupby(['switch_id', 'port_no']).groups.items():
        idx = list(group_idx)
        if len(idx) < SEQUENCE_LENGTH:
            continue
        windows = np.stack([
            scaled.loc[idx[i:i + SEQUENCE_LENGTH]].values
            for i in range(len(idx) - SEQUENCE_LENGTH + 1)
        ])
        tensor = torch.tensor(windows, dtype=torch.float32)
        # Each prediction corresponds to the LAST row of its window.
        target_rows = idx[SEQUENCE_LENGTH - 1:]
        with torch.no_grad():
            if lstm is not None:
                df.loc[target_rows, 'lstm_risk'] = lstm(tensor).numpy().flatten()
            if gru is not None:
                df.loc[target_rows, 'gru_risk'] = gru(tensor).numpy().flatten()

    # RL polling interval, derived from the README's stated policy
    # (heartbeat 30s unless LSTM risk > 0.70 → high-fidelity 1s). This is a
    # documented policy approximation, NOT a deployed DQN trace.
    df['rl_polling_interval'] = np.where(
        df['lstm_risk'].fillna(0) > RISK_THRESHOLD, 1, 30
    )

    # Compute metrics on rows that have BOTH a prediction and a label.
    metrics = {}
    eval_df = df.dropna(subset=['future_congestion'])

    def _metrics(mask, prob_col):
        y_true = eval_df.loc[mask, 'future_congestion'].astype(int).values
        y_pred = (eval_df.loc[mask, prob_col].values > 0.5).astype(int)
        return {
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'f1':        f1_score(y_true, y_pred, zero_division=0),
            'n':         len(y_true),
        }

    if lstm is not None:
        mask = eval_df['lstm_risk'].notna()
        if mask.sum() > 0:
            metrics['lstm'] = _metrics(mask, 'lstm_risk')
    if gru is not None:
        mask = eval_df['gru_risk'].notna()
        if mask.sum() > 0:
            metrics['gru'] = _metrics(mask, 'gru_risk')

    return df, metrics


df, metrics = load_and_score()

if df.empty:
    st.error("No telemetry data found. Ensure 'telemetry_dataset.csv' is in the directory.")
elif 'lstm' not in metrics:
    st.warning("`lstm_forecaster.pth` not found. Run `python3 train_lstm.py` first to enable the dashboard.")
else:
    # --- Metrics Row: monitoring overhead comparison ---
    st.header("1. Network Overhead Comparison")
    col1, col2, col3 = st.columns(3)

    total_time_sec = len(df['timestamp'].unique())
    switches = len(df['switch_id'].unique())
    static_high_msgs = total_time_sec * switches * 1
    static_low_msgs  = (total_time_sec / 30) * switches * 1
    rl_msgs = (
        len(df[df['rl_polling_interval'] == 1])
        + (len(df[df['rl_polling_interval'] == 30]) / 30)
    )

    col1.metric("Static High (1s) Overhead", f"{int(static_high_msgs):,} msgs",
                "Max Fidelity, Max Cost", delta_color="inverse")
    col2.metric("Static Low (30s) Overhead",  f"{int(static_low_msgs):,} msgs",
                "Min Fidelity, Misses Congestion", delta_color="normal")
    col3.metric("Predictive Activation (RL)", f"{int(rl_msgs):,} msgs",
                "High Fidelity ONLY when needed", delta_color="off")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        " 1. Project Architecture",
        " 2. Real-Time Telemetry",
        " 3. AI Model Performance",
    ])

    # --- Tab 1: architecture overview (text only) ---
    with tab1:
        st.header("How Predictive Activation Works")
        st.markdown(
            "Modern Software-Defined Networks (SDNs) waste significant CPU and "
            "bandwidth polling switches at a fixed rate even when the network is "
            "quiet. Our solution closes the loop using two AI models:"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(
                "**Step 1: The Oracle (LSTM)**\n\n"
                "We emulate a Fat-Tree SDN topology in Mininet. An os-ken controller "
                "collects traffic features (throughput, latency, queue depth). An "
                "**LSTM Neural Network** watches the last 5 seconds of this data to "
                "predict if a congestion spike is imminent."
            )
        with col_b:
            st.success(
                "**Step 2: The Decision Engine (DQN)**\n\n"
                "A **Reinforcement Learning Agent** observes the LSTM's risk score. "
                "It learns to leave the network in 'Heartbeat Mode' (polling every "
                "30s) to save CPU, and snaps to 'High-Fidelity Mode' (polling every "
                "1s) the moment congestion is predicted."
            )
        st.markdown("### The RL Reward Function")
        st.latex(r"Reward = -(Monitoring\ Cost) - \alpha(Congestion\ Penalty) + \beta(Detection\ Accuracy)")
        st.markdown(
            f"**Ground truth definition:** `latency_ms > {LATENCY_THRESHOLD_MS:.0f} ms` "
            "on the next time step. **Escalation threshold:** LSTM risk > "
            f"{RISK_THRESHOLD:.2f}."
        )

    # --- Tab 2: per-switch real-time telemetry ---
    with tab2:
        st.header("Phase 4 & 5: Real-Time Telemetry Dynamics")
        selected_switch = st.selectbox("Select Switch to Analyze:", df['switch_id'].unique())
        switch_df = df[df['switch_id'] == selected_switch].copy()

        timeline_df = switch_df.groupby('timestamp').agg({
            'rx_mbps': 'sum',
            'latency_ms': 'mean',
            'lstm_risk': 'mean',
            'rl_polling_interval': 'min',
        }).reset_index()

        st.subheader(f"Switch {selected_switch}: Throughput vs Latency")
        base = alt.Chart(timeline_df).encode(x=alt.X('timestamp:T', title='Time'))
        line_throughput = base.mark_line(color='blue').encode(
            y=alt.Y('rx_mbps:Q', title='Throughput (Mbps)',
                    scale=alt.Scale(domain=[0, timeline_df['rx_mbps'].max() + 10]))
        )
        line_latency = base.mark_line(color='red').encode(
            y=alt.Y('latency_ms:Q', title='Latency (ms)')
        )
        st.altair_chart(
            alt.layer(line_throughput, line_latency).resolve_scale(y='independent'),
            use_container_width=True,
        )

        st.subheader(f"Switch {selected_switch}: AI Polling Interval Selection")
        area_polling = base.mark_area(opacity=0.5, color='green').encode(
            y=alt.Y('rl_polling_interval:Q', title='Polling Interval (Seconds)',
                    scale=alt.Scale(reverse=True, domain=[0, 35])),
            tooltip=['timestamp', 'rl_polling_interval', 'lstm_risk'],
        )
        st.altair_chart(area_polling, use_container_width=True)
        st.info(
            "**How to read this:** when latency/throughput spikes (red/blue lines), "
            "the LSTM raises the risk score; the policy drops the polling interval "
            "(green area) from 30s to 1s to capture granular data."
        )

    # --- Tab 3: forecasting model comparison (REAL metrics) ---
    with tab3:
        st.header("Phase 2: Forecasting Model Comparison")
        st.markdown(
            f"Metrics computed on real model outputs over **{metrics['lstm']['n']:,}** "
            f"prediction windows. Positive class: `latency_ms > {LATENCY_THRESHOLD_MS:.0f} ms` "
            "on the next time step."
        )

        m_col1, m_col2 = st.columns(2)
        lm = metrics['lstm']
        m_col1.metric(
            "LSTM (Production)",
            f"{lm['f1']*100:.2f}% F1",
            f"Acc {lm['accuracy']*100:.1f}% | P {lm['precision']*100:.1f}% | "
            f"R {lm['recall']*100:.1f}%",
            delta_color="off",
        )
        if 'gru' in metrics:
            gm = metrics['gru']
            m_col2.metric(
                "GRU (Experimental)",
                f"{gm['f1']*100:.2f}% F1",
                f"Acc {gm['accuracy']*100:.1f}% | P {gm['precision']*100:.1f}% | "
                f"R {gm['recall']*100:.1f}%",
                delta_color="off",
            )
        else:
            m_col2.info("GRU model not yet trained. Run `python3 train_gru.py` to enable comparison.")

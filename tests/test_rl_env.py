import numpy as np
import torch
from conftest import make_telemetry

from predictive_activation import config
from predictive_activation.models import LSTMPredictor
from predictive_activation.train.rl_agent import SDNTelemetryEnv


def make_env(tmp_path):
    csv_path = tmp_path / "telemetry.csv"
    make_telemetry(n_steps=60).to_csv(csv_path, index=False)
    model_path = tmp_path / "lstm.pth"
    torch.manual_seed(0)
    torch.save(LSTMPredictor().state_dict(), model_path)
    return SDNTelemetryEnv(csv_path, model_path)


def test_reset_and_step_respect_spaces(tmp_path):
    env = make_env(tmp_path)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)

    for action in (0, 1, 2):
        obs, reward, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        if terminated:
            break


def test_lstm_window_is_sequence_length(tmp_path, monkeypatch):
    # Regression: the env used to feed 6-row windows to a model trained on 5.
    env = make_env(tmp_path)
    seen = []
    original = env.lstm_model.forward
    monkeypatch.setattr(env.lstm_model, "forward", lambda x: (seen.append(x.shape), original(x))[1])
    env.reset()
    env.step(2)
    assert all(shape[1] == config.SEQUENCE_LENGTH for shape in seen)


def test_missed_congestion_is_penalised(tmp_path):
    env = make_env(tmp_path)
    env.reset()
    trace = env.current_trace
    # Point the agent just before a congested step, then skip over it with slow polling.
    congested = np.flatnonzero(trace["is_congested"])
    env.current_step = int(congested[congested >= config.SEQUENCE_LENGTH][0]) - 1
    _, reward_slow, *_ = env.step(0)
    assert reward_slow <= -100

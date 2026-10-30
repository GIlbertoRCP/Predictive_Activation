import numpy as np
import pytest
from fastapi.testclient import TestClient

from predictive_activation import config, service
from predictive_activation.service import InferenceEngine


@pytest.fixture
def engine(scaler, lstm_model, fixed_agent):
    return InferenceEngine(scaler, lstm_model, fixed_agent)


def test_no_decisions_until_window_full(engine, event_stream):
    port_events = [e for e in event_stream if (e["switch_id"], e["port_no"]) == (1, 1)]
    for event in port_events[: config.SEQUENCE_LENGTH - 1]:
        assert engine.process([event]) == []
    decisions = engine.process([port_events[config.SEQUENCE_LENGTH - 1]])
    assert len(decisions) == 1
    assert decisions[0]["switch_id"] == 1 and decisions[0]["port_no"] == 1


def test_windows_are_per_port(engine, event_stream):
    # 3 ports interleaved: each needs its own SEQUENCE_LENGTH events.
    decisions = engine.process(event_stream[: 3 * config.SEQUENCE_LENGTH])
    assert {(d["switch_id"], d["port_no"]) for d in decisions} == {(1, 1), (1, 2), (2, 1)}
    assert all(len(w) == config.SEQUENCE_LENGTH for w in engine.history.values())


def test_decision_fields_and_routing(engine, event_stream, monkeypatch):
    decisions = engine.process(event_stream)
    assert decisions
    d = decisions[-1]
    assert 0.0 <= d["lstm_prob"] <= 1.0
    assert d["polling_interval"] == config.ACTION_MAPPING[d["action_idx"]]
    assert d["routing_path"] == ("s2" if d["lstm_prob"] > config.RISK_THRESHOLD else "s1")

    # Force high risk: every decision should reroute to the backup path.
    monkeypatch.setattr(engine, "predict", lambda batch: np.full(len(batch), 0.95, dtype=np.float32))
    assert all(d["routing_path"] == "s2" for d in engine.process(event_stream[:3]))


def test_rl_state_uses_previous_action(scaler, lstm_model, event_stream):
    from conftest import FixedActionAgent

    agent = FixedActionAgent(action=2)
    engine = InferenceEngine(scaler, lstm_model, agent)
    engine.process(event_stream)
    # First decision per switch sees action 0, later ones see the agent's last action.
    assert agent.states[0][2] == 0.0
    assert agent.states[-1][2] == 2.0


def test_set_engine_rejects_unknown(engine):
    with pytest.raises(ValueError):
        engine.set_engine("tensorflow")


def test_missing_onnx_session_falls_back_to_pytorch(engine, event_stream):
    engine.set_engine("onnx_quantized")  # no sessions configured
    assert engine.process(event_stream)


def test_http_endpoints(engine, event_stream, monkeypatch):
    monkeypatch.setattr(service, "inference", engine)
    engine.process(event_stream)
    client = TestClient(service.app)  # no `with`: lifespan (Kafka) doesn't run

    latest = client.get("/latest").json()
    assert set(latest["decisions"]) == {"1_1", "1_2", "2_1"}

    assert client.post("/set-engine", params={"engine": "onnx"}).json()["status"] == "success"
    assert engine.engine == "onnx"
    assert client.post("/set-engine", params={"engine": "bogus"}).json()["status"] == "error"

    assert "inference_latency_seconds" in client.get("/metrics").text

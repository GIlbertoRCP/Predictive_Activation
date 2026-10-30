import pytest
import torch

from predictive_activation import config
from predictive_activation.models import ARCHITECTURES, load_model


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_forward_outputs_probabilities(arch):
    model = ARCHITECTURES[arch]().eval()
    x = torch.rand(8, config.SEQUENCE_LENGTH, len(config.FEATURES))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 1)
    assert ((out >= 0) & (out <= 1)).all()


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_save_load_roundtrip(arch, tmp_path):
    model = ARCHITECTURES[arch]().eval()
    path = tmp_path / f"{arch}.pth"
    torch.save(model.state_dict(), path)
    loaded = load_model(arch, path)
    x = torch.rand(2, config.SEQUENCE_LENGTH, len(config.FEATURES))
    with torch.no_grad():
        torch.testing.assert_close(model(x), loaded(x))


@pytest.mark.parametrize("arch, path", [("lstm", config.LSTM_PATH), ("gru", config.GRU_PATH)])
def test_committed_checkpoints_load(arch, path):
    # state_dict keys depend on the rnn attribute name; this guards against renames.
    if not path.exists():
        pytest.skip(f"{path} not present")
    load_model(arch, path)

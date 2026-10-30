"""Recurrent congestion forecasters. One definition, used everywhere."""

import torch
import torch.nn as nn

from predictive_activation import config


class _RecurrentPredictor(nn.Module):
    rnn_cls: type[nn.Module]
    rnn_attr: str

    def __init__(self, input_size=len(config.FEATURES), hidden_size=128, num_layers=2):
        super().__init__()
        # Attribute name ("lstm" / "gru") is part of the state_dict keys, so it
        # must stay stable for existing checkpoints to load.
        setattr(self, self.rnn_attr, self.rnn_cls(input_size, hidden_size, num_layers, batch_first=True))
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = getattr(self, self.rnn_attr)(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


class LSTMPredictor(_RecurrentPredictor):
    rnn_cls = nn.LSTM
    rnn_attr = "lstm"


class GRUPredictor(_RecurrentPredictor):
    rnn_cls = nn.GRU
    rnn_attr = "gru"


ARCHITECTURES = {"lstm": LSTMPredictor, "gru": GRUPredictor}


def load_model(arch, weights_path, device="cpu"):
    model = ARCHITECTURES[arch]().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model

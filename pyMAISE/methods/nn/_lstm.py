import torch.nn as nn

from pyMAISE.methods.nn._layer import Layer


class _LSTMBlock(nn.Module):
    """LSTM layer that handles return_sequences and exposes the same interface as Keras."""

    def __init__(self, input_size, hidden_size, return_sequences=False, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
        )
        self.return_sequences = return_sequences

    def forward(self, x):
        output, (h_n, _) = self.lstm(x)
        # output: (batch, seq_len, hidden_size)
        # h_n:   (1, batch, hidden_size)
        if self.return_sequences:
            return output
        return h_n.squeeze(0)  # (batch, hidden_size)


class LSTMLayer(Layer):
    def __init__(self, layer_name, parameters: dict):
        # Initialize layer and base class
        self.reset()
        super().__init__(layer_name, parameters)

        # Get layer data from params dictionary
        self._data = super().build_data(self._data, parameters)

        assert self._data["units"] is not None

    # ==========================================================================
    # Methods
    def build(self, trial, in_size):
        # Sample parameters and build PyTorch LSTM module.
        # Keras-only params (recurrent_activation, kernel/recurrent/bias
        # initializers, regularizers, constraints, stateful, unroll,
        # go_backwards, return_state) are silently dropped.
        params = super().sample_parameters(self._data, trial)
        hidden_size = params["units"]
        return_sequences = params.get("return_sequences", False)
        dropout = params.get("dropout", 0.0)
        return _LSTMBlock(in_size, hidden_size, return_sequences, dropout), hidden_size

    def reset(self):
        self._data = {
            "units": None,
            "return_sequences": False,
            "dropout": 0.0,
            "recurrent_dropout": 0.0,  # kept for API compat, ignored
        }
        super().reset()

    def increment_layer(self):
        return super().increment_layer()

    # ==========================================================================
    # Getters
    def num_layers(self, trial):
        return super().num_layers(trial)

    def sublayer(self, trial):
        return super().sublayer(trial)

    def wrapper(self):
        return super().wrapper()

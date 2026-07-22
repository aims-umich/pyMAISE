import torch.nn as nn
import torch.nn.functional as F

from pyMAISE.methods.nn._layer import Layer


class _MCDropoutModule(nn.Dropout):
    """
    Dropout that stays active during inference for Monte Carlo Dropout UQ.
    Standard nn.Dropout turns off during model.eval() — this version
    always applies dropout regardless of training/eval mode.
    """
    def forward(self, x):
        return F.dropout(x, self.p, training=True)


class MCDropoutLayer(Layer):
    def __init__(self, layer_name, parameters: dict):
        self.reset()
        super().__init__(layer_name, parameters)
        self._data = super().build_data(self._data, parameters)

    def build(self, trial, in_size):
        params = super().sample_parameters(self._data, trial)
        return _MCDropoutModule(p=params["rate"]), in_size

    def reset(self):
        self._data = {
            "rate": 0.2,
        }
        super().reset()

    def increment_layer(self):
        return super().increment_layer()

    def num_layers(self, trial):
        return super().num_layers(trial)

    def sublayer(self, trial):
        return super().sublayer(trial)

    def wrapper(self):
        return super().wrapper()
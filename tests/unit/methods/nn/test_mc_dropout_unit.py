import numpy as np
import torch
import torch.nn as nn
from skorch.history import History

import pyMAISE as mai
from pyMAISE.methods.nn import MCDropout


class MockTorchModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.fc(self.dropout(x))


class MockSkorchModel:
    def __init__(self, predictions=None, module=None):
        self.predictions = predictions
        self.module_ = module if module is not None else MockTorchModule()
        self.history = History()

    def predict(self, x):
        return self.predictions

    def fit(self, x, y, **kwargs):
        pass


def test_mc_dropout_predict_with_uncertainty_regression():
    # Simulate 3 MC passes for 2 samples, 2 outputs
    pass1 = [[1.0, 2.0], [3.0, 4.0]]
    pass2 = [[3.0, 4.0], [5.0, 6.0]]
    pass3 = [[2.0, 3.0], [4.0, 5.0]]
    
    mock_model = MockSkorchModel()
    
    # Override predict to cycle through pass predictions
    predictions_cycle = [pass1, pass2, pass3]
    call_count = [0]
    
    def mock_predict(x):
        idx = call_count[0] % len(predictions_cycle)
        call_count[0] += 1
        return np.array(predictions_cycle[idx])

    mock_model.predict = mock_predict

    mc_dropout = MCDropout(model=mock_model, num_passes=3, heteroscedastic=False)
    mai.init(problem_type=mai.ProblemType.REGRESSION)

    res = mc_dropout.predict_with_uncertainty(x=None)

    np.testing.assert_allclose(res["mean"], [[2.0, 3.0], [4.0, 5.0]])
    expected_var = np.var([pass1, pass2, pass3], axis=0)
    np.testing.assert_allclose(res["epistemic_var"], expected_var)
    assert res["aleatoric_var"] is None


def test_mc_dropout_predict_with_uncertainty_heteroscedastic():
    # n_targets = 1, so 2 outputs per prediction: [mean, raw_variance]
    pass1 = [[1.0, 0.1], [3.0, 0.2]]
    pass2 = [[3.0, 0.3], [5.0, 0.4]]

    mock_model = MockSkorchModel()
    predictions_cycle = [pass1, pass2]
    call_count = [0]

    def mock_predict(x):
        idx = call_count[0] % len(predictions_cycle)
        call_count[0] += 1
        return np.array(predictions_cycle[idx])

    mock_model.predict = mock_predict

    mc_dropout = MCDropout(model=mock_model, num_passes=2, heteroscedastic=True)
    mai.init(problem_type=mai.ProblemType.REGRESSION)

    res = mc_dropout.predict_with_uncertainty(x=None)

    np.testing.assert_allclose(res["mean"], [[2.0], [4.0]])
    np.testing.assert_allclose(res["epistemic_var"], [[1.0], [1.0]])

    softplus = lambda x: np.log(1.0 + np.exp(x)) + 1e-6
    expected_aleatoric = [
        [(softplus(0.1) + softplus(0.3)) / 2],
        [(softplus(0.2) + softplus(0.4)) / 2],
    ]
    np.testing.assert_allclose(res["aleatoric_var"], expected_aleatoric)


def test_mc_dropout_toggles_dropout_layers():
    mock_module = MockTorchModule()
    mock_model = MockSkorchModel(module=mock_module)
    
    dropout_states_during_predict = []

    def mock_predict(x):
        dropout_states_during_predict.append(mock_module.dropout.training)
        return np.zeros((2, 2))

    mock_model.predict = mock_predict

    mc_dropout = MCDropout(model=mock_model, num_passes=2)
    mai.init(problem_type=mai.ProblemType.REGRESSION)

    # Initial state should be eval (False)
    mock_module.eval()
    assert not mock_module.dropout.training

    res = mc_dropout.predict_with_uncertainty(x=None)

    # During predict passes, dropout should have been training (True)
    assert dropout_states_during_predict == [True, True]
    # After predict_with_uncertainty, module should be back in eval (False)
    assert not mock_module.dropout.training


if __name__ == "__main__":
    test_mc_dropout_predict_with_uncertainty_regression()
    test_mc_dropout_predict_with_uncertainty_heteroscedastic()
    test_mc_dropout_toggles_dropout_layers()

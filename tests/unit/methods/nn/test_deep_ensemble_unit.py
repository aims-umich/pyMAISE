import numpy as np
from skorch.history import History

import pyMAISE as mai
from pyMAISE.methods.nn import DeepEnsemble


class MockModel:
    """Mock Neural Network model for unit testing DeepEnsemble."""
    def __init__(self, predictions=None, history_data=None):
        self.predictions = np.array(predictions) if predictions is not None else None
        
        self.history = History()
        if history_data:
            for epoch_data in history_data:
                self.history.new_epoch()
                for k, v in epoch_data.items():
                    self.history.record(k, v)
                    
    def predict(self, x):
        return self.predictions
    
    def fit(self, x, y, **kwargs):
        pass


def test_predict_with_uncertainty_regression():
    # Mean of predictions: [2.0, 3.0], [4.0, 5.0]
    # Variance of predictions: [1.0, 1.0], [1.0, 1.0]
    m1 = MockModel(predictions=[[1.0, 2.0], [3.0, 4.0]])
    m2 = MockModel(predictions=[[3.0, 4.0], [5.0, 6.0]])
    
    ensemble = DeepEnsemble(models=[m1, m2], heteroscedastic=False)
    mai.init(problem_type=mai.ProblemType.REGRESSION)
    
    res = ensemble.predict_with_uncertainty(x=None)
    
    np.testing.assert_allclose(res["mean"], [[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(res["epistemic_var"], [[1.0, 1.0], [1.0, 1.0]])
    assert res["aleatoric_var"] is None


def test_predict_with_uncertainty_heteroscedastic():
    # n_targets = 1, so 2 outputs per prediction: [mean, variance]
    m1 = MockModel(predictions=[[1.0, 0.1], [3.0, 0.2]])
    m2 = MockModel(predictions=[[3.0, 0.3], [5.0, 0.4]])
    
    ensemble = DeepEnsemble(models=[m1, m2], heteroscedastic=True)
    mai.init(problem_type=mai.ProblemType.REGRESSION)
    
    res = ensemble.predict_with_uncertainty(x=None)
    
    # mean of means
    np.testing.assert_allclose(res["mean"], [[2.0], [4.0]])
    # epistemic var = var of means
    np.testing.assert_allclose(res["epistemic_var"], [[1.0], [1.0]])
    # aleatoric var = mean of softplus-transformed variances
    softplus = lambda x: np.log(1.0 + np.exp(x)) + 1e-6
    expected_aleatoric = [
        [(softplus(0.1) + softplus(0.3)) / 2],
        [(softplus(0.2) + softplus(0.4)) / 2]
    ]
    np.testing.assert_allclose(res["aleatoric_var"], expected_aleatoric)


def test_predict_with_uncertainty_classification():
    # 2 classes
    m1 = MockModel(predictions=[[0.9, 0.1], [0.5, 0.5]])
    m2 = MockModel(predictions=[[0.7, 0.3], [0.5, 0.5]])
    
    ensemble = DeepEnsemble(models=[m1, m2], heteroscedastic=False)
    mai.init(problem_type=mai.ProblemType.CLASSIFICATION)
    
    res = ensemble.predict_with_uncertainty(x=None)
    
    mean_preds = np.array([[0.8, 0.2], [0.5, 0.5]])
    np.testing.assert_allclose(res["mean"], mean_preds)
    
    # epistemic var = -sum(p * log(p))
    expected_entropy = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=-1)
    np.testing.assert_allclose(res["epistemic_var"], expected_entropy)


def test_record_history():
    h1 = [{"train_loss": 1.0, "valid_loss": 2.0}, {"train_loss": 0.5, "valid_loss": 1.0}]
    h2 = [{"train_loss": 3.0, "valid_loss": 4.0}, {"train_loss": 1.5, "valid_loss": 2.0}]
    
    m1 = MockModel(history_data=h1)
    m2 = MockModel(history_data=h2)
    
    ensemble = DeepEnsemble(models=[m1, m2], heteroscedastic=False)
    ensemble._record_history()
    
    # History should have 2 epochs
    assert len(ensemble.history) == 2
    
    # Check epoch 0
    assert ensemble.history[0, "train_loss"] == 2.0 
    assert ensemble.history[0, "valid_loss"] == 3.0 
    
    # Check epoch 1
    assert ensemble.history[1, "train_loss"] == 1.0 
    assert ensemble.history[1, "valid_loss"] == 1.5 


if __name__ == "__main__":
    test_predict_with_uncertainty_regression()
    test_predict_with_uncertainty_heteroscedastic()
    test_predict_with_uncertainty_classification()
    test_record_history()

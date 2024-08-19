from sklearn.naive_bayes import GaussianNB
import modelling.ml.model as modelling
import pandas as pd
import pytest


@pytest.fixture()
def model():
    # define a trained model to be used in all tests
    x = pd.DataFrame([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    y = pd.Series([1, 0, 0])
    return modelling.train_model(x, y)


def test_train_model(model):
    # make sure the model is the right type
    assert (isinstance(model, GaussianNB))


def test_compute_model_metrics():
    # make sure the correct nunmer of metrics are returned
    y1 = pd.Series([1, 0, 0])
    y2 = pd.Series([1, 1, 0])
    metrics = modelling.compute_model_metrics(y1, y2)
    assert (len(metrics) == 3)


def test_inference(model):
    # make sure the inference is the right shape
    x = pd.DataFrame([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    y = modelling.inference(model, x)
    assert ((len(y.shape) == 1) and (y.shape[0] == x.shape[0]))

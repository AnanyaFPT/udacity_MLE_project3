from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_welcome():
    # test if the get on root works as expected
    r = client.get('/')
    assert r.json() == 'Welcome'
    assert r.status_code == 200


def test_predict_below_50k():

    # define input
    below_50k = {
        'age': 39,
        'workclass': 'State-gov',
        'fnlgt': 77516,
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 2174,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States',
    }

    # post and check result
    r = client.post('/predict/', json=below_50k)

    print(r.json())

    assert r.status_code == 200
    assert r.json() == 0


def test_predict_above_50k():

    # define input
    above_50k = {
        'age': 38,
        'workclass': 'Private',
        'fnlgt': 275223,
        'education': 'Some-college',
        'education-num': 10,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Sales',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 7298,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States',
    }

    # post and check result
    r = client.post('/predict/', json=above_50k)
    assert r.status_code == 200
    assert r.json() == 1

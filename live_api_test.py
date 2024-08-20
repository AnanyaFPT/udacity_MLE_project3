import json
import requests

url = 'https://udacity-mle-project3.onrender.com/predict/'
above_50k_sample = {
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

# headers = {"Content-type": "application/json"}
response = requests.post(url, data=json.dumps(above_50k_sample) )
print("Status Code: ", response.status_code)
print("Prediction: ", response.text)
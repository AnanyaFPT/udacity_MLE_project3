# Imports
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from modelling.ml.data import process_data, cat_features
from modelling.ml.model import load_model, inference
from os.path import join as safepath
import pandas as pd


# Instantiate the app.
app = FastAPI()

# Define input data structure for the POST


class Input(BaseModel):

    # define config to convert field names to work with hyphens
    model_config = ConfigDict(
        alias_generator=lambda field_name: field_name.replace('_', '-'),
        json_schema_extra={'examples': [{
            'age': 39,
            'workclass': 'State-gov',
            'fnlgt': 77516,
            'education': 'Bachelors',
            'education_num': 13,
            'marital_status': 'Never-married',
            'occupation': 'Adm-clerical',
            'relationship': 'Not-in-family',
            'race': 'White',
            'sex': 'Male',
            'capital_gain': 2174,
            'capital_loss': 0,
            'hours_per_week': 40,
            'native_country': 'United-States',
        }]}
    )

    # define fields
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

# Define a GET on the root


@app.get("/")
async def welcome():
    return "Welcome"

# Define a POST for inferencing


@app.post("/predict/")
async def predict(data: Input):

    # load model
    model, encoder, lb = load_model(safepath('model'))

    # convert inputs to pandas dataframe
    df = pd.DataFrame(data.model_dump(by_alias=True), index=[0])

    # process data to prep it for the model
    X, _, _, _ = process_data(df, categorical_features=cat_features,
                              label=None, encoder=encoder, lb=lb, training=False)

    # make predictions and return
    preds = inference(model, X)
    return preds.tolist()[0]

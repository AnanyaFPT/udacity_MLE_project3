# Script to train machine learning model.

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
from os.path import join as safepath
from ml.data import process_data
import pandas as pd
import ml.model as modelling
from pickle import dump as pickle_save

# define constants
INPUT_PATH = safepath('data', 'clean_census.csv')
OUTPUT_PATH = safepath('model')

# Add code to load in the data.
data = pd.read_csv(INPUT_PATH)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train model
model = modelling.train_model(X_train, y_train)

# save model and associated encoders
save_list = [model, encoder, lb]
filenames = ['trained_model.pkl', 'encoder.pkl', 'lb.pkl']
for name, to_save in zip(filenames, save_list):
    with open(safepath(OUTPUT_PATH, name), 'wb') as f:
        pickle_save(to_save, f)

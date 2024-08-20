from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from pickle import dump as pickle_save
from pickle import load as pickle_load
from os.path import join as safepath
from pandas import DataFrame as pd_df


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model : sklearn.naive_bayes.GaussianNB
        Trained machine learning model.
    """
    # define naive bayes model and train it
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def sliced_performance(data, labels, preds, column_to_slice):

    # only perform slicing for categorical columns of the data
    if data[column_to_slice].dtype == 'object':
        results = []

        # loop through each unique value in the column_to_slice
        unique_vals = data[column_to_slice].unique()
        for value in unique_vals:

            # get subset of data to evaluate
            subset = data[column_to_slice] == value
            labels_subset = labels[subset]
            preds_subset = preds[subset]

            # compute metrics
            precision, recall, fbeta = compute_model_metrics(
                labels_subset, preds_subset)

            # append to output
            results.append([column_to_slice, value, precision, recall, fbeta])

        # convert to pd.Dataframe and return
        return pd_df(
            results,
            columns=[
                'column_to_slice',
                'slice_value',
                'precision',
                'recall',
                'fbeta'])

    else:
        raise ValueError('column_t0_slice must be categorical')


def save_model(output_path, model, encoder, lb):
    save_list = [model, encoder, lb]
    filenames = ['trained_model.pkl', 'encoder.pkl', 'lb.pkl']
    for name, to_save in zip(filenames, save_list):
        with open(safepath(output_path, name), 'wb') as f:
            pickle_save(to_save, f)


def load_model(input_path):
    loaded = []
    filenames = ['trained_model.pkl', 'encoder.pkl', 'lb.pkl']
    for name in filenames:
        with open(safepath(input_path, name), 'rb') as f:
            loaded.append(pickle_load(f))
    return loaded[0], loaded[1], loaded[2]

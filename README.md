# Github Repo Link
*https://github.com/AnanyaFPT/udacity_MLE_project3/*


# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is an implementation of a the default Gaussian naive bayes algorithm in scikit-learn 1.5.1.

## Intended Use
The model classifies if an individual's salary is above 50,000 or not, given public census data

## Data
Publicly avaliable census data was used for this analysis (*https://archive.ics.uci.edu/dataset/20/census+income*). The data was modified to remove all spaces

The original data set has 32561 rows, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Metrics
precision: 0.653
recall: 0.296
fbeta: 0.407

## Ethical Considerations
The fairness and bias of this model fron an ethical standpoint was not evaluated. 

## Caveats and Recommendations
This is a minimal implementation of a ML pipeline framework intended for practice and not intended to be a highly accurate predictor

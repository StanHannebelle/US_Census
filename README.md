# US_Income

The problem here is to solve the binary classification problem of the US Adult Census Income Dataset.
The objective is the predict if someone earns more or less than 50.000 USD per year.

## Setting Up the environnement

To run these scripts please use Python 3, create a virtual environment and install the libraries in the requirements.txt file.

Please download the training and test datasets (here: http://thomasdata.s3.amazonaws.com/ds/us_census_full.zip) and place these in the data folder with the following names: census_income_learn.csv and census_income_test.csv


## General presentation

The repository contains three scripts that can be run manually by the user: hyper_parameter_tuning, train.py and predict.py . Other scripts, present in the utils folder contain support functions called in the main three scripts.

## hyper_parameter_tuning.py

The user can run this script.
This will run two hyper-parameters tuning processes (Random Search). One on a Random Forest Model, the other one on a XGBoost model. Then, the characteristics of the best performing model is saved in a json file (model/best_model.json).

## train.py

After running hyper_parameter_tuning.py, the user can run train.py.
This will load and prepare the training data, open the model/best_model.json file, instanciate the corresponding optimal model, train this model on the training set and saves it in the model/model.pkl file. Finaly, it computes the optimal threshold for f1 maximisation and saves it into the model/best_threshold.json file.

## predict.py

After running train.py, the user can run predict.py.
This will load and prepare the test data, load the trained model saved in the model/model.pkl file and computes predictions for the test set. It also prints the test AUC-ROC score, the test f1 score and the test confusion matric. Finally, it dumps a .csv file containing predictions of the test set (census_income_test_with_predictions.csv)

## Obtained results

When I personally ran these scripts, I have found that the best performing model was a Random Forest Classifier presenting the following parameters: 

"n_estimators": 200,
"min_samples_split": 10,
"min_samples_leaf": 2,
"max_features": "sqrt",
"max_depth": 10,
"bootstrap": false

and the following threshold maximising the training f1-score: 0.24%

The obtained test auc_roc score was 93.1% and the test f1-score was 52.27%.

The confusion matrix was:
[[89320  4256]
 [ 2491  3695]]



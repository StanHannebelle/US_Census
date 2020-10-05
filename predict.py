

from utils.config import TEST_DATA_PATH, TRAIN_DATA_PATH
from utils.save_test_prediction import save_prediction_to_csv
from utils.threshold_optim import from_proba_to_class
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from utils.data_prep import load_and_prep
import joblib
import json


# loading test features and labels
X_test, y_test = load_and_prep(TEST_DATA_PATH)

# load the trained model
model = joblib.load('model/model.pkl')

# make predictions and print auc roc score, f1 score and confusion matrix on the test set
y_predict_proba = model.predict_proba(X_test)[:, -1]
auc_roc = roc_auc_score(y_test, y_predict_proba, average='weighted')
print("test auc_roc: %.2f%%" % (auc_roc * 100.0))


with open('model/best_threshold.json') as f:
    json_threshold = json.load(f)
optimal_threshold = json_threshold['best_threshold']
y_predict = [from_proba_to_class(value, optimal_threshold)
             for value in y_predict_proba]
f1 = f1_score(y_test, y_predict, average='binary')
print("test f1: %.2f%%" % (f1 * 100.0))
confusion_matrix = confusion_matrix(y_test, y_predict)
print("test confusion_matrix: ", confusion_matrix)

# save prediction to .csv file named census_income_test_with_predictions.csv
save_prediction_to_csv(y_predict)

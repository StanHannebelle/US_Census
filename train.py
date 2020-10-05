import pandas as pd
from utils.data_prep import load_and_prep
from utils.threshold_optim import threshold_optim
from utils.config import TRAIN_DATA_PATH, NBR_OF_NEGATIVE_LABEL, NBR_OF_POSITIVE_LABEL
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import json


# loading training features and label
X_train, y_train = load_and_prep(TRAIN_DATA_PATH)

# instanciate the best model
with open('model/best_model.json') as f:
    json_model = json.load(f)

if json_model['best_model'] == 'rf':
    model = RandomForestClassifier(
        n_estimators=json_model['best_params']['n_estimators'],
        min_samples_split=json_model['best_params']['min_samples_split'],
        min_samples_leaf=json_model['best_params']['min_samples_leaf'],
        max_features=json_model['best_params']['max_features'],
        max_depth=json_model['best_params']['max_depth'],
        bootstrap=json_model['best_params']['bootstrap'])
else:
    model = XGBClassifier(
        scale_pos_weight=NBR_OF_NEGATIVE_LABEL / NBR_OF_POSITIVE_LABEL,
        n_estimators=json_model['best_params']['n_estimators'],
        learning_rate=json_model['best_params']['learning_rate'],
        max_depth=json_model['best_params']['max_depth'],
        min_child_weight=json_model['best_params']['min_child_weight'],
        gamma=json_model['best_params']['gamma'],
        subsample=json_model['best_params']['subsample'],
        colsample_bytree=json_model['best_params']['colsample_bytree'])

# Train the optimal model on the whole training set
print('Model training')
model.fit(X_train, y_train)

# Save optimal model
joblib.dump(model, 'model/model.pkl')

# Compute best decision threshold to maximize f1_score
y_train_pred_proba = model.predict_proba(X_train)[:, -1]
best_threshold = threshold_optim(y_train, y_train_pred_proba)
json_model = json.dumps({'best_threshold': best_threshold})
f = open('model/best_threshold.json', 'w')
f.write(json_model)
f.close()

from utils.data_prep import load_and_prep
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from utils.config import TRAIN_DATA_PATH, NBR_OF_NEGATIVE_LABEL, NBR_OF_POSITIVE_LABEL


def optimal_xgboost():
    # loading training features and label
    X_train, y_train = load_and_prep(TRAIN_DATA_PATH)

    # set hyperparameters to tune
    params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [2, 3, 4, 5],
        'min_child_weight': [0.5, 1, 5],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.5, 0.75, 1.0],
        'colsample_bytree': [0.5, 0.75, 1.0]
    }

    xgb = XGBClassifier(objective='binary:logistic', nthread=1,
                        scale_pos_weight=NBR_OF_NEGATIVE_LABEL / NBR_OF_POSITIVE_LABEL)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    # Random Search
    print('XGBoost Optimal Hyper Parameters Random Search')
    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=20, n_jobs=4, cv=skf.split(
        X_train, y_train), verbose=1, random_state=0)
    random_search.fit(X_train, y_train)
    return random_search.best_score_, random_search.best_params_

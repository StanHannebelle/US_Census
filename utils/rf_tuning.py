from utils.data_prep import load_and_prep
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from utils.config import TRAIN_DATA_PATH


def optimal_rf():
    # loading training features and label
    X_train, y_train = load_and_prep(TRAIN_DATA_PATH)

    # set hyperparameters to tune
    params = {
        'n_estimators': [50, 100, 200],
        'bootstrap': [True, False],
        'max_depth': [5, 7, 10, 15, 20, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestClassifier()
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    # Random Search
    print('Random Forest Optimal Hyper Parameters Random Search')
    random_search = RandomizedSearchCV(rf, param_distributions=params, n_iter=20, n_jobs=4, cv=skf.split(
        X_train, y_train), verbose=1, random_state=0)
    random_search.fit(X_train, y_train)
    return random_search.best_score_, random_search.best_params_

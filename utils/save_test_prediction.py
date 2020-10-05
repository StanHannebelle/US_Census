from utils.data_prep import open_csv
from utils.config import TEST_DATA_PATH


def decode_predictions(prediction):
    if prediction == 0:
        return(' - 50000.')
    else:
        return(' 50000+.')


def save_prediction_to_csv(y_predict):
    predictions = [decode_predictions(value) for value in y_predict]
    data = open_csv(TEST_DATA_PATH)
    data['predictions'] = predictions
    data.to_csv('data/census_income_test_with_predictions.csv')

from sklearn.metrics import f1_score
import numpy as np


def from_proba_to_class(prediction, threshold):
    if prediction < threshold:
        return(0)
    else:
        return(1)


def threshold_optim(y_true, y_pred_proba):
    f1_scores = []
    thresholds = []
    for threshold in np.arange(start=0.02, stop=1.0, step=0.02):
        y_pred = [from_proba_to_class(value, threshold)
                  for value in y_pred_proba]
        f1_scores += [f1_score(y_true, y_pred)]
        thresholds += [threshold]
    return(thresholds[np.argmax(f1_scores)])

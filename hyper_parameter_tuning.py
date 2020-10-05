from utils.xgb_tuning import optimal_xgboost
from utils.rf_tuning import optimal_rf
import json

# finding the optimal xgboost model
best_xgb_score, best_xgb_params = optimal_xgboost()

# finding the optimal random forest model
best_rf_score, best_rf_params = optimal_rf()

# keeping the best of these two models
if best_xgb_score >= best_rf_score:
    print('XGBoost performs better than Random forest')
    print('Best parameters : ', best_xgb_params)
    best_model = {'best_model': 'xgb', 'best_params': best_xgb_params}
else:
    print('Random forest performs better than XGBoost')
    print('Best parameters : ', best_rf_params)
    best_model = {'best_model': 'rf', 'best_params': best_rf_params}

# saving best params
json_model = json.dumps(best_model)
f = open('model/best_model.json', 'w')
f.write(json_model)
f.close()

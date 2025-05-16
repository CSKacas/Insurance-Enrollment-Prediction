from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def tune_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }

    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_
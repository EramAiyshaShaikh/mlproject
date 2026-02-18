import os
import sys
from src.exception import CustomException
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    report = {}
    best_estimators = {}
    try:
        for model_name, model in models.items():
            params = param.get(model_name, {})

            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)

            tuned_model = gs.best_estimator_
            tuned_model.fit(X_train, y_train)

            y_pred_train = tuned_model.predict(X_train)
            y_pred_test = tuned_model.predict(X_test)

            train_score = r2_score(y_train, y_pred_train)
            test_score = r2_score(y_test, y_pred_test)

            report[model_name] = test_score
            best_estimators[model_name] = tuned_model

        return report, best_estimators

    except Exception as e:
        raise CustomException(e, sys)

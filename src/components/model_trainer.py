import os
import sys

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.utils import evaluate_models, save_obj

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "SVR": SVR(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
            }

            param_distribution = {
                "LinearRegression": {},
                "Ridge": {},
                "Lasso": {},
                "RandomForestRegressor": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "AdaBoostRegressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoostRegressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "DecisionTreeRegressor": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ]
                },
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "SVR": {},
                "GradientBoostingRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "KNeighborsRegressor": {},
            }

            model_report, tuned_models = evaluate_models(
                models=models,
                param=param_distribution,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = tuned_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            logging.info("Best model found")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            y_pred = best_model.predict(X_test)
            model_r2_score = r2_score(y_test, y_pred)

            return model_r2_score

        except Exception as e:
            raise CustomException(e, sys)

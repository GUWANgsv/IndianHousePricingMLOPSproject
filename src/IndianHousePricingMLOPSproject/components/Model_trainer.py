import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.IndianHousePricingMLOPSproject.logger import logging
from src.IndianHousePricingMLOPSproject.exception import customexception
from src.IndianHousePricingMLOPSproject.utils.utils import save_object, evaluate_model

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    XGBRegressor = None
    _HAS_XGB = False


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("Artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting dependent and independent variables from train and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
            }
            if _HAS_XGB:
                models["XGBRegressor"] = XGBRegressor()

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("\n" + "=" * 80 + "\n")
            logging.info(f"Model Report: {model_report}")

            # Select best model by R² score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"Best Model Found — Name: {best_model_name} | R2 Score: {best_model_score:.4f}")
            print("\n" + "=" * 80 + "\n")
            logging.info(f"Best Model: {best_model_name} | R2 Score: {best_model_score:.4f}")

            # Refit best model on full training data
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info("Best model saved to Artifacts/model.pkl")

        except Exception as e:
            logging.info("Exception occurred at Model Training")
            raise customexception(e, sys)
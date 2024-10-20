import os
import sys
from copy import deepcopy as dc
from dataclasses import dataclass

import dill
import numpy as np

from src.exception import CustomException


@dataclass
class ModelPredictionConfig:
    X_train_data_path: str = os.path.join("artifacts", "X_train.pkl")
    y_train_data_path: str = os.path.join("artifacts", "y_train.pkl")
    X_test_data_path: str = os.path.join("artifacts", "X_test.pkl")
    y_test_data_path: str = os.path.join("artifacts", "y_test.pkl")


class ModelPrediction:
    def __init__(self):
        self.model_prediction_config = ModelPredictionConfig()

    def initiate_model_prediction(self, model, predicted, scaler):
        try:
            with open(
                self.model_prediction_config.X_train_data_path, "rb"
            ) as file_obj_1:
                X_train = dill.load(file_obj_1)

            with open(
                self.model_prediction_config.X_test_data_path, "rb"
            ) as file_obj_2:
                X_test = dill.load(file_obj_2)

            with open(
                self.model_prediction_config.y_train_data_path, "rb"
            ) as file_obj_3:
                y_train = dill.load(file_obj_3)

            lookback = 30
            train_predictions = predicted.flatten()
            dummies = np.zeros((X_train.shape[0], lookback + 1))
            dummies[:, 0] = train_predictions
            dummies = scaler.inverse_transform(dummies)
            train_predictions = dc(dummies[:, 0])

            dummies = np.zeros((X_train.shape[0], lookback + 1))
            dummies[:, 0] = y_train.flatten()
            dummies = scaler.inverse_transform(dummies)
            new_y_train = dc(dummies[:, 0])

            test_predictions = model(X_test.to("cpu")).detach().cpu().numpy().flatten()
            dummies = np.zeros((X_test.shape[0], lookback + 1))
            dummies[:, 0] = test_predictions
            dummies = scaler.inverse_transform(dummies)
            test_predictions = dc(dummies[:, 0])

            return train_predictions, new_y_train, test_predictions

        except Exception as e:
            raise CustomException(e, sys)

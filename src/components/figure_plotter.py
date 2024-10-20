import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt

from src.exception import CustomException


@dataclass
class FigurePlotterConfig:
    figure_path: str = os.path.join("artifacts", "final_pred.png")


class FigurePlotter:
    def __init__(self):
        self.figure_plotter_config = FigurePlotterConfig()

    def plot_moving_average_50(self, data_col):
        try:
            ma_50 = data_col.rolling(50).mean()
            fig = plt.figure(figsize=(8, 6))
            plt.plot(data_col, "g")
            plt.plot(ma_50, "b")

            return fig

        except Exception as e:
            raise CustomException(e, sys)

    def plot_moving_average_100(self, data_col):
        try:
            ma_50 = data_col.rolling(50).mean()
            ma_100 = data_col.rolling(100).mean()
            fig = plt.figure(figsize=(8, 6))
            plt.plot(data_col, "g")
            plt.plot(ma_50, "b")
            plt.plot(ma_100, "r")

            return fig

        except Exception as e:
            raise CustomException(e, sys)

    def plot_moving_average_200(self, data_col):
        try:
            ma_50 = data_col.rolling(50).mean()
            ma_100 = data_col.rolling(100).mean()
            ma_200 = data_col.rolling(200).mean()
            fig = plt.figure(figsize=(8, 6))
            plt.plot(data_col, "g")
            plt.plot(ma_50, "b")
            plt.plot(ma_100, "r")
            plt.plot(ma_200, "y")

            return fig

        except Exception as e:
            raise CustomException(e, sys)

    def plot_final_prediction_figure(self, train_pred, y_train):
        try:
            fig = plt.figure(figsize=(8, 6))
            plt.plot(y_train, label="Actual Price")
            plt.plot(train_pred, label="Predicted Price")
            plt.xlabel("Day")
            plt.ylabel("Price")
            plt.legend()

            return fig

        except Exception as e:
            raise CustomException(e, sys)

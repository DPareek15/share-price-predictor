import os
import sys
from copy import deepcopy as dc
from dataclasses import dataclass

import dill
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from src.exception import CustomException

scaler = MinMaxScaler(feature_range=(-1, 1))


@dataclass
class DataTransformationConfig:
    X_train_data_path: str = os.path.join("artifacts", "X_train.pkl")
    y_train_data_path: str = os.path.join("artifacts", "y_train.pkl")
    X_test_data_path: str = os.path.join("artifacts", "X_test.pkl")
    y_test_data_path: str = os.path.join("artifacts", "y_test.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_preparation(self, data, n_steps):
        df = dc(data)
        for i in range(1, n_steps + 1):
            df[f"Close(t-{i})"] = df["Close"].shift(i)
        df.dropna(inplace=True)
        return df

    def initiate_data_transformation(self, data_col):
        try:
            lookback = 30
            shifted_data = self.data_preparation(data_col, lookback)
            shifted_np_data = shifted_data.to_numpy()
            scaled_np_data = scaler.fit_transform(shifted_np_data)

            X = scaled_np_data[:, 1:]
            X = dc(np.flip(X, axis=1))
            y = scaled_np_data[:, 0]

            SPLIT_INDEX = int(len(X) * 0.90)

            X_train = X[:SPLIT_INDEX].reshape((-1, lookback, 1))
            X_test = X[SPLIT_INDEX:].reshape((-1, lookback, 1))

            y_train = y[:SPLIT_INDEX].reshape((-1, 1))
            y_test = y[SPLIT_INDEX:].reshape((-1, 1))

            X_train = torch.tensor(X_train).float()
            X_test = torch.tensor(X_test).float()

            y_train = torch.tensor(y_train).float()
            y_test = torch.tensor(y_test).float()

            os.makedirs(
                os.path.dirname(self.data_transformation_config.X_train_data_path),
                exist_ok=True,
            )
            with open(
                self.data_transformation_config.X_train_data_path, "wb"
            ) as file_obj_1:
                dill.dump(X_train, file_obj_1)

            os.makedirs(
                os.path.dirname(self.data_transformation_config.y_train_data_path),
                exist_ok=True,
            )
            with open(
                self.data_transformation_config.y_train_data_path, "wb"
            ) as file_obj_2:
                dill.dump(y_train, file_obj_2)

            os.makedirs(
                os.path.dirname(self.data_transformation_config.X_test_data_path),
                exist_ok=True,
            )
            with open(
                self.data_transformation_config.X_test_data_path, "wb"
            ) as file_obj_3:
                dill.dump(X_test, file_obj_3)

            os.makedirs(
                os.path.dirname(self.data_transformation_config.y_test_data_path),
                exist_ok=True,
            )
            with open(
                self.data_transformation_config.y_test_data_path, "wb"
            ) as file_obj_4:
                dill.dump(y_test, file_obj_4)

            train_dataset = TimeSeriesDataset(X_train, y_train)
            test_dataset = TimeSeriesDataset(X_test, y_test)

            BATCH_SIZE = 16

            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True
            )
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            for _, batch in enumerate(train_loader):
                x_batch, y_batch = batch[0].to("cpu"), batch[1].to("cpu")
                print(x_batch.shape, y_batch.shape)
                break

            return train_loader, test_loader, scaler

        except Exception as e:
            raise CustomException(e, sys)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

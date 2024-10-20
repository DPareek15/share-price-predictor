import os
import sys
from dataclasses import dataclass

import dill
import streamlit as st
import torch
import torch.nn as nn

from src.exception import CustomException


@dataclass
class ModelTrainerConfig:
    train_path = os.path.join("artifacts", "X_train.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_loader, test_loader):
        try:
            model = LSTM(1, 4, 1)
            model.load_state_dict(
                torch.load("share_price_model.pth", weights_only=True)
            )
            model.eval()

            learning_rate = 0.01
            num_epochs = 20
            loss_function = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                self.train_one_epoch(
                    model=model,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    loss_function=loss_function,
                    epoch=epoch,
                )
                self.validate_one_epoch(
                    model=model, test_loader=test_loader, loss_function=loss_function
                )

            with open(self.model_trainer_config.train_path, "rb") as file_obj:
                X_train = dill.load(file_obj)

            with torch.no_grad():
                predicted = model(X_train.to("cpu")).to("cpu").numpy()

            return model, predicted

        except Exception as e:
            raise CustomException(e, sys)

    def train_one_epoch(self, model, optimizer, train_loader, loss_function, epoch):
        try:
            model.train(True)
            print(f"Epoch: {epoch + 1}")
            running_loss = 0.0

            for batch_index, batch in enumerate(train_loader):
                x_batch, y_batch = batch[0].to("cpu"), batch[1].to("cpu")

                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_index % 100 == 99:  # print every 100 batches
                    avg_loss_across_batches = running_loss / 100
                    print(
                        "Batch {0}, Loss: {1:.3f}".format(
                            batch_index + 1, avg_loss_across_batches
                        )
                    )
                    running_loss = 0.0
            print()

        except Exception as e:
            raise CustomException(e, sys)

    def validate_one_epoch(self, model, test_loader, loss_function):
        try:
            model.train(False)
            running_loss = 0.0

            for batch_index, batch in enumerate(test_loader):
                x_batch, y_batch = batch[0].to("cpu"), batch[1].to("cpu")

                with torch.no_grad():
                    output = model(x_batch)
                    loss = loss_function(output, y_batch)
                    running_loss += loss.item()

            avg_loss_across_batches = running_loss / len(test_loader)

            print("Val Loss: {0:.3f}".format(avg_loss_across_batches))
            print("***************************************************")
            print()

        except Exception as e:
            raise CustomException(e, sys)


@st.cache_resource
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_stacked_layers, batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        BATCH_SIZE = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, BATCH_SIZE, self.hidden_size).to(
            "cpu"
        )
        c0 = torch.zeros(self.num_stacked_layers, BATCH_SIZE, self.hidden_size).to(
            "cpu"
        )

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

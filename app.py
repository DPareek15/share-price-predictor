import datetime as dt
from copy import deepcopy as dc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

scaler = MinMaxScaler(feature_range=(-1, 1))


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


model = LSTM(1, 4, 1)
model.load_state_dict(torch.load("share_price_model.pth", weights_only=True))
model.eval()


st.header("Stock Market Predictor")
st.subheader("Stock data used :")

stock = st.text_input("Enter Stock Symbol", "AAPL")
start = "1990-01-01"
end = str(dt.datetime.today().strftime("%Y-%m-%d"))
data = yf.download(stock, start, end)
st.write(data)
data = pd.DataFrame(data["Close"])

st.subheader("Price vs MA of 50 days")
ma_50 = data["Close"].rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(data["Close"], "g")
plt.plot(ma_50, "b")
plt.show()
st.pyplot(fig1)

st.subheader("Price vs MA of 50 days vs MA of 100 days")
ma_100 = data["Close"].rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(data["Close"], "g")
plt.plot(ma_50, "b")
plt.plot(ma_100, "r")
plt.show()
st.pyplot(fig2)

st.subheader("Price vs MA of 50 days vs MA of 100 days vs MA of 200 days")
ma_200 = data["Close"].rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(data["Close"], "g")
plt.plot(ma_50, "b")
plt.plot(ma_100, "r")
plt.plot(ma_200, "y")
plt.show()
st.pyplot(fig3)


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    for i in range(1, n_steps + 1):
        df[f"Close(t-{i})"] = df["Close"].shift(i)

    df.dropna(inplace=True)

    return df


lookback = 30
shifted_data = prepare_dataframe_for_lstm(data, lookback)
shifted_np_data = shifted_data.to_numpy()
shifted_np_data = scaler.fit_transform(shifted_np_data)

X = shifted_np_data[:, 1:]
X = dc(np.flip(X, axis=1))
y = shifted_np_data[:, 0]

SPLIT_INDEX = int(len(X) * 0.90)

X_train = X[:SPLIT_INDEX].reshape((-1, lookback, 1))
X_test = X[SPLIT_INDEX:].reshape((-1, lookback, 1))

y_train = y[:SPLIT_INDEX].reshape((-1, 1))
y_test = y[SPLIT_INDEX:].reshape((-1, 1))

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()

y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to("cpu"), batch[1].to("cpu")
    print(x_batch.shape, y_batch.shape)
    break

learning_rate = 0.01
num_epochs = 20
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_one_epoch():
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


def validate_one_epoch():
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


for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()

with torch.no_grad():
    predicted = model(X_train.to("cpu")).to("cpu").numpy()

train_predictions = predicted.flatten()
dummies = np.zeros((X_train.shape[0], lookback + 1))
dummies[:, 0] = train_predictions
dummies = scaler.inverse_transform(dummies)
train_predictions = dc(dummies[:, 0])

dummies = np.zeros((X_train.shape[0], lookback + 1))
dummies[:, 0] = y_train.flatten()
dummies = scaler.inverse_transform(dummies)
new_y_train = dc(dummies[:, 0])

st.subheader("Actual Price vs Predicted Price")
fig4 = plt.figure(figsize=(8, 6))
plt.plot(new_y_train, label="Actual Price")
plt.plot(train_predictions, label="Predicted Price")
plt.xlabel("Day")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig4)

test_predictions = model(X_test.to("cpu")).detach().cpu().numpy().flatten()
dummies = np.zeros((X_test.shape[0], lookback + 1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)
test_predictions = dc(dummies[:, 0])

st.subheader(f"The price for {stock} on {end} is {test_predictions[-1]}")

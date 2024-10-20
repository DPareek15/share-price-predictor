from datetime import datetime

import streamlit as st

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.figure_plotter import FigurePlotter
from src.components.model_prediction import ModelPrediction
from src.components.model_trainer import ModelTrainer

data_obj = DataIngestion()
trans_obj = DataTransformation()
model_obj = ModelTrainer()
pred_obj = ModelPrediction()
figure_obj = FigurePlotter()

st.header("Stock Market Predictor")
st.subheader("Stock data used :")

stock = st.text_input("Enter Stock Symbol", "AAPL")

data = data_obj.initiate_data_ingestion(stock=stock)
st.write(data)

st.subheader("Price vs MA of 50 days")
fig1 = figure_obj.plot_moving_average_50(data["Close"])
st.pyplot(fig1)

st.subheader("Price vs MA of 50 days vs MA of 100 days")
fig2 = figure_obj.plot_moving_average_100(data["Close"])
st.pyplot(fig2)

st.subheader("Price vs MA of 50 days vs MA of 100 days vs MA of 200 days")
fig3 = figure_obj.plot_moving_average_200(data["Close"])
st.pyplot(fig3)

train_loader, test_loader, scaler = trans_obj.initiate_data_transformation(
    data[["Close"]]
)

model, predicted = model_obj.initiate_model_trainer(train_loader, test_loader)

train_predictions, new_y_train, test_predictions = pred_obj.initiate_model_prediction(
    model, predicted, scaler
)

st.subheader("Actual Price vs Predicted Price")
fig4 = figure_obj.plot_final_prediction_figure(train_predictions, new_y_train)
st.pyplot(fig4)

curr_date = datetime.today().strftime("%Y-%m-%d")

st.subheader(
    f"The price for {stock} on {str(curr_date)} is {test_predictions[-1].round(2)}"
)

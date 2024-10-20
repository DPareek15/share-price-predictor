import os
import sys
from dataclasses import dataclass
from datetime import datetime

import streamlit as st
import yfinance as yf

from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")


@st.cache_resource
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, stock):
        # logging.info("Entered the data ingestion method")
        try:
            start = "1990-01-01"
            end = str(datetime.today().strftime("%Y-%m-%d"))
            df = yf.download(stock, start, end)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # logging.info("Data ingestion is completed")

            return df

        except Exception as e:
            raise CustomException(e, sys)

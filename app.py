import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2022-12-31'

st.title("Stock Trend Prediction")

user_input = st.text_input("Enter Stock Ticker: ", 'AAPL')
data = yf.download(user_input, start = start, end = end)
data.head()

st.subheader("Data from 2010 to 2022")
st.write(data.describe())

st.subheader("Closing Price Time Chart")
fig = plt.figure(figsize= (12, 6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader("Closing Price Time Chart (100MA)")
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize= (12, 6))
plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig)

st.subheader("Closing Price Time Chart (100MA & 200MA)")
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize= (12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(data.Close)
st.pyplot(fig)

data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

data_train_arr = scaler.fit_transform(data_train)

model = load_model('keras_model.h5')

past_100 = data_train.tail(100)

final_df = past_100.append(data_test, ignore_index = True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test)

scaler.scale_

scale_factor = 1/0.02099517
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

st.subheader("Predictions v/s Original")
fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_pred, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
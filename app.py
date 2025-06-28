import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('C:/Users/voram/Desktop/stock_prediction/Stock Predictions Model.keras')

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG').upper().strip()
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

import seaborn as sns

stocks = ['GOOG', 'AAPL', 'AMZN', 'MSFT', 'TSLA']
df = yf.download(stocks, start=start, end=end)['Close']
correlation = df.corr()

st.subheader('Stock Correlation Heatmap')
fig_corr = plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
st.pyplot(fig_corr)



x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)

import plotly.express as px

pred_df = pd.DataFrame({'Actual': y.flatten(), 'Predicted': predict.flatten()})
pred_df['Index'] = pred_df.index

fig_anim = px.line(pred_df, x='Index', y=['Actual', 'Predicted'], title="Animated Price Prediction")
st.plotly_chart(fig_anim)

# import plotly.graph_objects as go
# from sklearn.metrics import r2_score, mean_squared_error
# r2 = r2_score(y, predict)
# rmse = np.sqrt(mean_squared_error(y, predict))  # Replace with your real model confidence

# fig_gauge = go.Figure(go.Indicator(
#     mode="gauge+number",
#     value=r2 * 100,
#     title={'text': "Prediction Confidence (%)"},
#     gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
# ))

# st.plotly_chart(fig_gauge)

from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
# Accuracy metrics
r2 = r2_score(y, predict)
rmse = np.sqrt(mean_squared_error(y, predict))

st.subheader("Model Evaluation")
st.write(f"R² Score: {r2:.4f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Confidence gauge
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=r2 * 100,
    title={'text': "Prediction Confidence (R² %)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "green"},
        'steps': [
            {'range': [0, 50], 'color': 'red'},
            {'range': [50, 80], 'color': 'orange'},
            {'range': [80, 100], 'color': 'green'}
        ]
    }
))


st.plotly_chart(fig_gauge)


important_dates = {
    '2020-03-23': 'COVID Crash Bottom',
    '2021-11-05': 'All-Time High',
    '2022-03-16': 'Fed Rate Hike',
}

fig_timeline = plt.figure(figsize=(10,6))
plt.plot(data.Close, label='Close Price')

for date, label in important_dates.items():
    if date in data.index:
        plt.axvline(pd.to_datetime(date), color='r', linestyle='--')
        plt.text(pd.to_datetime(date), data.Close.max(), label, rotation=90)

plt.legend()
st.pyplot(fig_timeline)




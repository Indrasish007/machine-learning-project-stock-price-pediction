import streamlit as st
import datetime as dt
import yfinance as yf 
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly  import graph_objs as go


START="2015-01-01"
TODAY=dt.date.today().strftime("%Y-%m-%d")

st.title("Stock prediction app")
stocks=("AAPL","GOOG","MSFT","GME")
selected_stocks=st.selectbox("Select datasert for prediction ",stocks)

n_years=st.slider("Years of prediction :",1,4)
period=n_years*365

@st.cache_data
def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data
data_load_state=st.text("load data...")
data=load_data(selected_stocks)
data_load_state.text("loading data... done!")

st.subheader("Raw data")
st.write(data.tail())

def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.update_layout(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

#Forecasting
df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})
m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)

st.subheader("Forecast data")
st.write(forecast.tail())
st.write('Forecast data')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)
st.write("Forecast components")
fig2=m.plot_components(forecast)
st.write(fig2)


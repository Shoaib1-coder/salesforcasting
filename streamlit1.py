#  Import required libraries
import streamlit as st                 # Web app interface
import pandas as pd                   # Data handling
import numpy as np                    # Numerical operations
from prophet import Prophet           # Time series forecasting model
import plotly.graph_objs as go        # Interactive visualizations
import joblib                         # Save/load machine learning models
import os                             # File handling

#  Streamlit page setup
st.set_page_config(page_title="Sales Forecasting App", layout="centered")
st.title(" Sales Forecasting App")

#  Sidebar: Upload user dataset
st.sidebar.header("Upload Store Data")
st.sidebar.markdown("""Upload your Dataset in CSV or Excel format. The app will automatically detect the date column and sales data for forecasting.""")

uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xls', 'xlsx'])

#  Sidebar: Forecast duration
periods = st.sidebar.number_input("Forecast Months", min_value=1, max_value=60, value=12)

#  Load dataset
if uploaded_file is not None:
    try:
        # Load CSV or Excel based on file extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        else:
            df = pd.read_excel(
                uploaded_file,
                engine='openpyxl' if uploaded_file.name.endswith('.xlsx') else 'xlrd'
            )
        st.success(" File uploaded and loaded successfully!")
    except Exception as e:
        st.error(f" Error reading file: {e}")
        st.stop()
else:
    # Load default file if nothing uploaded
    st.info("ℹ️ Using default file: `preprocessdata.csv`")
    df = pd.read_csv('preprocessdata.csv', encoding='ISO-8859-1')

#  Clean and normalize column names
df.columns = df.columns.str.strip().str.lower()

#  Try to detect a date column
date_column = None
for col in ['ship date', 'order date', 'date']:
    if col in df.columns:
        date_column = col
        break

#  Ensure required columns are present
if not date_column or 'sales' not in df.columns:
    st.error("Required columns not found. Need one of ['Ship Date', 'Order Date', 'Date'] and 'Sales'.")
    st.write(" Available columns:", df.columns.tolist())
    st.stop()

#  Convert date column to datetime format and clean
df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
df = df.dropna(subset=[date_column])

#  Resample monthly sales data for Prophet
df_monthly = df.resample('M', on=date_column)['sales'].sum().reset_index()
df_monthly = df_monthly.rename(columns={date_column: 'ds', 'sales': 'y'})  # Prophet expects 'ds' and 'y'

#  Smoothing with rolling average
df_monthly['y'] = df_monthly['y'].rolling(window=3, center=True).mean()
df_monthly['y'] = df_monthly['y'].fillna(method='bfill').fillna(method='ffill')

#  Optional: Log transform to stabilize variance
log_transform = True
if log_transform:
    df_monthly['y'] = np.log1p(df_monthly['y'])  # log(1 + y)

#  Train or load forecasting model
if uploaded_file is not None:
    # Train new Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        changepoint_prior_scale=0.5,
        seasonality_prior_scale=20
    )
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
    model.fit(df_monthly)
else:
    # Load pretrained model for default file
    model_path = 'salesforecast.pkl'
    if not os.path.exists(model_path):
        st.error("❌ Pretrained model not found: 'salesforecast.pkl'")
        st.stop()
    model = joblib.load(model_path)

#  Generate future dates and forecast
future = model.make_future_dataframe(periods=periods, freq='M')
forecast = model.predict(future)

#  Inverse log transform for actual forecast values
if log_transform:
    forecast['yhat'] = np.expm1(forecast['yhat'])
    forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])
    df_monthly['y'] = np.expm1(df_monthly['y'])

#  Plot actual vs predicted sales using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_monthly['ds'], y=df_monthly['y'],
                         mode='lines', name='Actual Sales', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                         mode='lines', name='Predicted Sales', line=dict(color='red')))
fig.update_layout(title=' Sales Forecast using Prophet',
                  xaxis_title='Date', yaxis_title='Sales',
                  template='plotly_white')
st.plotly_chart(fig)

#  Prepare forecast table
forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
forecast_data = forecast_data.rename(columns={
    'ds': 'Date',
    'yhat': 'Predicted Sales',
    'yhat_lower': 'Lower Bound',
    'yhat_upper': 'Upper Bound'
})
forecast_data[['Predicted Sales', 'Lower Bound', 'Upper Bound']] = \
    forecast_data[['Predicted Sales', 'Lower Bound', 'Upper Bound']].round(2)

# Display forecast table
st.subheader(" Forecast Table")
st.dataframe(forecast_data)

#  Show last 20 rows of original uploaded or default data
st.subheader(" Original Store Data (Last 20 Rows)")
st.dataframe(df.tail(20))
st.markdown("""
<h2> Sales Forecasting App: Uses & Benefits</h2>

<h3> Key Uses</h3>
<ul>
    <li><strong>Predict Future Sales:</strong> Estimate upcoming sales for the next weeks, months, or quarters based on historical data.</li>
    <li><strong>Inventory Planning:</strong> Prevent overstocking or stockouts by forecasting product demand.</li>
    <li><strong>Financial Planning:</strong> Help budgeting and cash flow projections by aligning sales forecasts with expected revenue.</li>
    <li><strong>Marketing Strategy:</strong> Plan promotions and campaigns during predicted low sales periods or capitalize on expected peaks.</li>
    <li><strong>Resource Allocation:</strong> Adjust staffing, warehousing, and logistics according to forecasted workload.</li>
    <li><strong>Business Expansion:</strong> Make data-driven decisions about opening new stores, launching products, or entering new markets.</li>
</ul>

<h3> Key Benefits</h3>
<table style="border-collapse: collapse; width: 100%;" border="1">
    <thead>
        <tr><th>Benefit</th><th>Description</th></tr>
    </thead>
    <tbody>
        <tr><td> Improved Decision-Making</td><td>Backed by data, managers can make more accurate, confident choices.</td></tr>
        <tr><td> Cost Efficiency</td><td>Reduces waste in inventory, staffing, and marketing.</td></tr>
        <tr><td> Time-Saving</td><td>Automates forecasting that would otherwise be manual or spreadsheet-based.</td></tr>
        <tr><td> Trend Detection</td><td>Reveals seasonal trends and growth patterns over time.</td></tr>
        <tr><td> Customizable</td><td>Can include custom factors (e.g., quarterly seasonality, promotions).</td></tr>
        <tr><td> Risk Mitigation</td><td>Helps identify potential sales declines early and plan accordingly.</td></tr>
    </tbody>
</table>

<h3> Who Uses It?</h3>
<ul>
    <li>Retailers (e.g., clothing, electronics)</li>
    <li>E-commerce Platforms</li>
    <li>Manufacturers</li>
    <li>Distributors</li>
    <li>Financial Analysts</li>
    <li>Startup Founders and SMBs</li>
</ul>
""", unsafe_allow_html=True)




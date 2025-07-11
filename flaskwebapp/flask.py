from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import plotly.io as pio
import os
import joblib

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast_data = []
    fig_html = None

    # Load and preprocess data
    df = pd.read_csv('preprocessdata.csv', encoding='ISO-8859-1')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df_monthly = df.resample('M', on='Ship Date')['Sales'].sum().reset_index()
    df_monthly = df_monthly.rename(columns={'Ship Date': 'ds', 'Sales': 'y'})

    # Apply smoothing
    df_monthly['y'] = df_monthly['y'].rolling(window=3, center=True).mean()
    df_monthly['y'] = df_monthly['y'].fillna(method='bfill').fillna(method='ffill')

    # Log transform
    log_transform = True
    if log_transform:
        df_monthly['y'] = np.log1p(df_monthly['y'])

    # Get forecast period from form
    periods = 12
    if request.method == 'POST':
        try:
            periods = int(request.form.get('periods', 12))
        except ValueError:
            periods = 12

    # Load Prophet model
    model_path = 'salesforecast.pkl'
    if not os.path.exists(model_path):
        return "Model not found. Please train and save it as 'salesforecast.pkl'."
    model = joblib.load(model_path)

    # Forecast
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)

    # Inverse log transform
    if log_transform:
        forecast['yhat'] = np.expm1(forecast['yhat'])
        if 'yhat_lower' in forecast.columns:
            forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
            forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_monthly['ds'], y=np.expm1(df_monthly['y']) if log_transform else df_monthly['y'],
                             mode='lines', name='Actual Sales', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                             mode='lines', name='Predicted Sales', line=dict(color='red')))
    fig.update_layout(title='Sales Forecast using Prophet',
                      xaxis_title='Date', yaxis_title='Sales',
                      template='plotly_white')
    fig_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    store_data=df.tail(20)

    # Forecast table
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    forecast_data = forecast_data.rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted Sales',
        'yhat_lower': 'Lower Bound',
        'yhat_upper': 'Upper Bound'
    })
   
    forecast_data[['Predicted Sales', 'Lower Bound', 'Upper Bound']] = \
        forecast_data[['Predicted Sales', 'Lower Bound', 'Upper Bound']].round(2)

    return render_template(
        'index.html',
        forecast_table=forecast_data.to_html(classes='table table-striped', index=False),
        fig_html=fig_html,store_data=store_data.to_html(classes='table table-striped', index=False),
        periods=periods
    )

if __name__ == '__main__':
    app.run(debug=True)



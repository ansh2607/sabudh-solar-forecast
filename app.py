import flask
from flask import Flask, render_template, request
import requests

import pandas as pd
import numpy as np
import json
import pickle

import matplotlib.pylab as plt
from matplotlib import pyplot
import plotly
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(
    __name__,
    template_folder="templates"
)

scaler=MinMaxScaler()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
     # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def pred_func(model,test_X):
    predicted=model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    inv_pred = np.concatenate((predicted.reshape(-1,1), test_X[:, 1:]), axis=1)
    inv_pred = np.round(scaler.inverse_transform(inv_pred),-1)
    inv_pred = inv_pred[:,0]
    return inv_pred

def plotgraph(sdate,edate) :
    path = 'static/csv/temp_data.csv'
    print(path)
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= sdate) & (df['Date'] <= edate)
    df = df.loc[mask]
    #total_yield = px.line(df, x="date", y="max", labels={ "date": "Year", "max": "Maximum Temperature(C)",})
    total_yield = go.Figure()
    total_yield.add_trace(go.Scatter(x=df['Date'],y=df['Total Yield [kWh]'],mode='lines',name='Total Yield'))
    total_yield.add_trace(go.Scatter(x=df['Date'],y=df['Total Yield (Smooth)'],mode='lines', name='Total Yield (Smoothed)'))
    total_yield.update_xaxes(rangeslider_visible=True)
    temp = go.Figure()
    temp.add_trace(go.Scatter(x=df['Date'],y=df['Temp'],mode='lines', name='Temperature'))
    temp.add_trace(go.Scatter(x=df['Date'],y=df['Temp (Smooth)'],mode='lines', name='Temperature Smoothed'))
    temp.update_xaxes(rangeslider_visible=True)
    yieldJSON = json.dumps(total_yield, cls=plotly.utils.PlotlyJSONEncoder)
    tempJSON = json.dumps(temp, cls=plotly.utils.PlotlyJSONEncoder)
    return yieldJSON, tempJSON  

def forecastplot():
    path = 'static/csv/temp_data.csv'
    data = pd.read_csv(path)
    data=data[['Date','Total Yield (Smooth)','GridDurationHours','MaintDurationHours','DelREMO','Temp (Smooth)','Cloud Cover (Smooth)']]
    data['Date']=pd.to_datetime(data['Date'])
    data=data.set_index('Date')
    values=np.round(data.values,2)
    scaled = scaler.fit_transform(values)
    new_data=series_to_supervised(scaled)
    new_data.drop(new_data.columns[[7,8,9,10,11]], axis=1, inplace=True)
    train=new_data[:730].values
    test=new_data[730:].values
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    gru_model = load_model('gru_model.h5')
    inv_pred=pred_func(gru_model,test_X)
    predicted_frame=pd.DataFrame(inv_pred.reshape(-1,1))
    predicted_frame=predicted_frame.rename(columns={0:'predicted'})
    data=data.reset_index()
    temp_test=data[731:]
    temp_train=data[:731]
    temp_test=temp_test.reset_index()
    temp_test=pd.concat([temp_test,predicted_frame],axis=1)
    predict = go.Figure()
    predict.add_trace(go.Scatter(x=temp_train['Date'],y=temp_train['Total Yield (Smooth)'],mode='lines', name='Train Total Yield (Smoothed)'))
    predict.add_trace(go.Scatter(x=temp_test['Date'],y=temp_test['Total Yield (Smooth)'],mode='lines', name='Test Total Yield (Smoothed)'))
    predict.add_trace(go.Scatter(x=temp_test['Date'],y=temp_test['predicted'],mode='lines', name='predicted Total Yield'))
    predict.update_xaxes(rangeslider_visible=True)
    predictJSON = json.dumps(predict, cls=plotly.utils.PlotlyJSONEncoder)
    return predictJSON

@app.route("/" , methods = ['POST', 'GET'])
def index():
    path = 'static/csv/temp_data.csv'
    data = pd.read_csv(path)
    # Ploting graphs
    yieldJSON, tempJSON = plotgraph(sdate = "2019-02-01",edate= "2021-05-29")
    return render_template('index.html', data=data, total_yield = yieldJSON, temp = tempJSON)

@app.route("/dataselect" , methods = ['POST', 'GET'])
def dataselect():
    if request.method == 'POST' :
        sdate = request.form['startdate']
        edate = request.form['enddate']
    path = 'static/csv/temp_data.csv'
    data = pd.read_csv(path)
    yieldJSON, tempJSON = plotgraph(sdate,edate)
    return render_template('index.html', data=data, total_yield = yieldJSON, temp = tempJSON)

@app.route("/dataforecast" , methods = ['POST', 'GET'])
def dataforecast():
    if request.method == 'POST':
        inputQuery1=request.form['query1']
        inputQuery2=request.form['query2']
        inputQuery3=request.form['query3']
        inputQuery4=request.form['query4']
        inputQuery5=request.form['query5']
        inputQuery6=request.form['query6']
    inp = [[inputQuery1,inputQuery2,inputQuery3,inputQuery4,inputQuery5, inputQuery6]]
    inp=np.array(inp)
    scaled = scaler.fit_transform(inp)
    values = scaled.reshape((scaled.shape[0], 1, scaled.shape[1]))
    gru_model = load_model('gru_model.h5')
    prediction=pred_func(gru_model,values)
    prediction=prediction[0]
    path = 'static/csv/temp_data.csv'
    data = pd.read_csv(path)
    predictJSON = forecastplot()
    return render_template('forecast.html',data=data, prediction=prediction, predict = predictJSON)

@app.route('/charts')
def charts():
	return render_template('charts.html')

@app.route('/tables')
def tables():
	return render_template('tables.html')


if __name__ == "__main__":
    app.run(debug=True)
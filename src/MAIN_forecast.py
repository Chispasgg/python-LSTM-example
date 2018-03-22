'''
Created on 21 mar. 2018

@author: patxi
'''

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from math import sqrt
from matplotlib import pyplot

import numpy
import os
import keras


# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')
# return datetime.strptime('190' + x, '%Y-%m')


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test, train_data=True):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    train_scaled = None
    if (train_data):
        scaler = scaler.fit(train)
        
        # transform train
        train = train.reshape(train.shape[0], train.shape[1])
        train_scaled = scaler.transform(train)
    else:
        scaler = scaler.fit(test)
        
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        print(' -> process: ' + str(i))
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


def __predict_from_data(csv_file, lstm_model):
    # load dataset
    series = read_csv(csv_file, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(None, supervised_values, train_data=False)
    
    # walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]       
        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
        # store forecast
        predictions.append(yhat)
        print('Month=%d, Predicted=%f' % (i + 1, yhat))
    
    # report performance
    rmse = sqrt(mean_squared_error(raw_values[-len(predictions):], predictions))
    print('Test RMSE: %.3f' % rmse)
    # line plot of observed vs predicted
    pyplot.plot(raw_values)
    pyplot.plot(predictions)
    pyplot.show()


def __generate_model(dataset_file, model_name):
    # load dataset
    series = read_csv(dataset_file, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    
    # split data into train and test-sets
    train, test = supervised_values[0:-values_to_predict], supervised_values[-values_to_predict:]
    
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    
    # fit the model
    lstm_model = fit_lstm(train_scaled, batch_size, repetitions, neurons)
    
    # save the model and values
    keras.models.save_model(lstm_model, model_name)
    
    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)
    
    # walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]        
        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
        # store forecast
        predictions.append(yhat)
        expected = raw_values[len(train) + i + 1]
        print('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))
    
    # report performance
    rmse = sqrt(mean_squared_error(raw_values[-values_to_predict:], predictions))
    print('Test RMSE: %.3f' % rmse)
    # line plot of observed vs predicted
    pyplot.plot(raw_values[-values_to_predict:])
    pyplot.plot(predictions)
    pyplot.show()

    
if __name__ == '__main__':
    print('start')
    values_to_predict = 7
    dataset_file = 'market-price.csv'
    csv_file = 'market-price2.csv'
    
    # LSTM
    look_back = 3
    batch_size = 1
    num_lstm_neurons = 4  # number of neurons inside hidden layer
    num_visible_layer = 1 
    num_hidden_layers = 1  # hidden layers
    
    batch_size = 1
    repetitions = 10
    neurons = 4
    
    model_name = str(dataset_file) + '_rep_' + str(repetitions) + '_hidlayer_' + str(num_hidden_layers) + '_lstmneurons_' + str(num_lstm_neurons) + '_lookback_' + str(look_back) + '_lstmmodel.pkl'
    
    if (os.path.exists(model_name)):
        # exist, we load it
        print('    -> loading model: ' + str(model_name))
        print('    -> to join with : ' + str(csv_file))
        model = keras.models.load_model(model_name)
        __predict_from_data(csv_file, model)
        
    else:
        print('    -> generating model ' + str(model_name))
        __generate_model(dataset_file, model_name)
    
    print('end')

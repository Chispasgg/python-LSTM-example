'''
Created on 21 mar. 2018

@author: chispasgg
'''

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import keras.models
import os

# configuration values
dataset_file = 'market-price.csv'
csv_file = 'exchange-trade-value.csv'
repetitions = 100
verbose_level = 0

# LSTM
look_back = 3
batch_size = 1
num_lstm_neurons = 4  # number of neurons inside hidden layer
num_visible_layer = 1 
num_hidden_layers = 1  # hidden layers

model_name = str(dataset_file) + '_rep_' + str(repetitions) + '_hidlayer_' + str(num_hidden_layers) + '_lstmneurons_' + str(num_lstm_neurons) + '_lookback_' + str(look_back) + '_lstmmodel.pkl'


def __predict_from_data(csv_file, model):
	
	# fix random seed for reproducibility
	numpy.random.seed(7)
	
	# load the dataset
	dataframe = read_csv(csv_file, usecols=[1], engine='python')
	
	dataset = dataframe.values
	dataset = dataset.astype('float32')
	
	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	
	# reshape into X=t and Y=t+1
	testX, testY = create_dataset(dataset, look_back)
	
	# reshape input to be [samples, time steps, features]
	testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
	
	# make predictions
	testPredict = model.predict(testX, batch_size=batch_size)
	
	# invert predictions
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	
	# calculate root mean squared error
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
	print('Test Score: %.2f RMSE' % (testScore))
	
	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[look_back:len(testPredict) + look_back, :] = testPredict
	
	# plot baseline and predictions
	plt.plot(scaler.inverse_transform(dataset))
	plt.plot(testPredictPlot)
	plt.show()


def __predict_from_test(model, trainX, trainY, testX, testY):
	# make predictions
	trainPredict = model.predict(trainX, batch_size=batch_size)
	model.reset_states()
	testPredict = model.predict(testX, batch_size=batch_size)
	
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
	print('Test Score: %.2f RMSE' % (testScore))
	
	# shift train predictions for plotting
	trainPredictPlot = numpy.empty_like(dataset)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
	
	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
	
	# plot baseline and predictions
	plt.plot(scaler.inverse_transform(dataset))
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		a = dataset[i:(i + look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

#############################################################
#############################################################


if __name__ == '__main__':
	print('START')
	# check if model exist
	if (os.path.exists(model_name)):
		# exist, we load it
		print('    -> loading model: ' + str(model_name))
		print('    -> to join with : ' + str(csv_file))
		model = keras.models.load_model(model_name)
		__predict_from_data(csv_file, model)
		
	else:
		# no exist, we generate it
		print('    -> generating model ' + str(model_name))
		
		# fix random seed for reproducibility
		numpy.random.seed(7)
		
		# load the dataset
		dataframe = read_csv(dataset_file, usecols=[1], engine='python')
		
		dataset = dataframe.values
		dataset = dataset.astype('float32')
		
		# normalize the dataset
		scaler = MinMaxScaler(feature_range=(0, 1))
		dataset = scaler.fit_transform(dataset)
		
		# split into train and test sets
		train_size = int(len(dataset) * 0.67)
		test_size = len(dataset) - train_size
		train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
		
		# reshape into X=t and Y=t+1
		trainX, trainY = create_dataset(train, look_back)
		testX, testY = create_dataset(test, look_back)
		
		# reshape input to be [samples, time steps, features]
		trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
		testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
		
		# create and fit the LSTM network
		model = Sequential()		
		
		# add hidden layers with memory
		if (num_hidden_layers > 1):
			print(' more than one hidden layer')
			model.add(LSTM(num_lstm_neurons, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
			print(' hidden layer added')
		else:
			print(' only one hidden layer')
		num_hidden_layers -= 2
		
		for layer in range(num_hidden_layers):
			model.add(LSTM(num_lstm_neurons, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
			print(' hidden layer added')
		
		model.add(LSTM(num_lstm_neurons, batch_input_shape=(batch_size, look_back, 1), stateful=True))
		print(' hidden layer added')
		
		# add visible layer
		model.add(Dense(num_visible_layer))
		
		model.compile(loss='mean_squared_error', optimizer='adam')
		
		# fit the model in X epochs
		for i in range(repetitions):
			if (verbose_level == 0):
				print('process: ' + str(i))
			model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=verbose_level, shuffle=False)
			model.reset_states()
		
		d = keras.models.save_model(model, model_name)
	
		__predict_from_test(model, trainX, trainY, testX, testY)
	
	print('END')

#load and plot dataset
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
import numpy
from numpy import concatenate
import pandas as pd
import time

start = time.time()
#load dataset
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')

#frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace = True)
    return df

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
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
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return numpy.array(diff)
    
#invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

#tranform scale for train and test data to [-1,1]
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train)
    #transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    #transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

#inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [X for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0,-1]

#fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, Y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    #set learning rate
    #adam = optimizers.Adam(lr=0.0001)
    adam = optimizers.Adam()
    model.compile(loss='mean_squared_error', optimizer=adam)
    for i in range(nb_epoch):
        model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

#make a one-step forecast    
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size = batch_size)
    return yhat[0,0]

#load dataset    
series = read_csv('../input/wheat_200910-201803.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series.columns = ['open', 'high', 'low', 'close']
#resample to weekly Friday level, last one is excluded as not actual Friday
series = series.resample('W-FRI').last()
#remove last rows to align prediction with close price only
series = series.drop(series.tail(1).index)
#print(series.tail())
#transform data to be stationary
raw_values = series.values
close_values = raw_values[:,-1]
diff_values = difference(raw_values, 1)

#transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised.columns = ['sd_open', 'sd_high', 'sd_low', 'sd_close', 'open', 'high', 'low', 'close']
supervised = supervised.drop(supervised.columns[[4,5,6]], axis = 1)
supervised_values = supervised.values

#split data into train and test sets
train, test = supervised_values[0:-12], supervised_values[-12:]

#tranform the scale of data
scaler, train_scaled, test_scaled = scale(train, test)

#print(train_scaled.shape, test_scaled.shape)

nb_epochs = 500
neurons = 4
nb_correct_predict = 0

#fit the model
lstm_model = fit_lstm(train_scaled, 1, nb_epochs, neurons)

#forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:,0:-1]
train_reshaped = train_reshaped.reshape(train_reshaped.shape[0], 1, train_reshaped.shape[1])
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    X_close = test_scaled[i,-2:-1]
    yhat = forecast_lstm(lstm_model, 1, X) 
    # Put the predictions there and invert scale
    test_scaled[i, -1] = yhat
    yhat = scaler.inverse_transform(test_scaled)[i, -1]       
	# invert differencing
    yhat = inverse_difference(close_values, yhat, len(test_scaled)+1-i)
	# store forecast
    predictions.append(yhat)
    expected = close_values[len(train) + i + 1]
	#calculate number of correct trend predictions
    if i != 0:
	    if (expected > old_expected) and (yhat > old_yhat):
	        nb_correct_predict = nb_correct_predict+1
	    elif (expected < old_expected) and (yhat < old_yhat):
	        nb_correct_predict = nb_correct_predict+1
	    elif (expected == old_expected) and (yhat == old_yhat):
	        nb_correct_predict = nb_correct_predict+1
    print('Date=%s, Predicted=%f, Expected=%f' % (series.index[-12+i], yhat, expected))
    old_yhat = yhat
    old_expected = expected

#predict the next corn price
last = test_scaled[-1, 0:-1]
yhat = forecast_lstm(lstm_model, 1, last)
# invert scaling
test_scaled[-1, -1] = yhat
yhat = scaler.inverse_transform(test_scaled)[-1, -1]
# invert differencing
yhat = inverse_difference(close_values, yhat, 1)
# print next prediction
print('Next predicted wheat price: %f' % yhat)
# print correct number of trend predictions
p_correct_predict = nb_correct_predict/(len(test_scaled)-1) * 100
print('Number of correct trend predictions: %d, percentage: %.1f' % (nb_correct_predict, p_correct_predict))
#report performance
rmse = sqrt(mean_squared_error(close_values[-12:], predictions))
print('Number of Epochs: %d, Number of Neurons: %d' % (nb_epochs, neurons))
print('Test RMSE: %.3f' % rmse)
print('Data: Open, High, Low, Close Price')
end = time.time()
print('*********Time Used: %.5s seconds*********' %(end - start))
#line plot of observed vs predicted
pyplot.plot(close_values[-12:], label = 'Expected Value')
pyplot.plot(predictions, label = 'Predicted Value')
pyplot.legend()
pyplot.show()
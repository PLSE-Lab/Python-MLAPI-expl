# LSTM Deep Learning Algorithm for Apple Stock Forecasting
# Author: Ebru Seyma Karakoyun, ebruseymakarakoyun@gmail.com
# Author: Ali Osman Cibikdiken, aocdiken@gmail.com
# 2018

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.optimizers import *
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.layers import Dense, Dropout, Activation, Embedding
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def mean_pred(y_true, y_pred):
    return K.abs(y_true-y_pred)*1000

def mean_absloute_error_scaled(y_true, y_pred):
    return K.abs(y_true-y_pred)*5000

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = numpy.array(y_true), numpy.array(y_pred)
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100

def mean_error(y_true, y_pred): 
    y_true, y_pred = numpy.array(y_true), numpy.array(y_pred)
    return numpy.mean(y_true - y_pred)

def mean_percentage_error(y_true, y_pred): 
    y_true, y_pred = numpy.array(y_true), numpy.array(y_pred)
    return numpy.mean((y_true - y_pred) / y_true) * 100

# fix random seed for reproducibility
numpy.random.seed(42)

# load the dataset
dataframe = pandas.read_csv('apple.csv', usecols=[4], engine='python',nrows=1000000)

dataset = dataframe.values
dataset = numpy.multiply(dataset.astype('float32'),1)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print("--")
print(len(dataset))

# split into train and test sets
train_size = int(len(dataset)*0.8)
test_size =  int(len(dataset)*0.2) + 30
train, test = dataset[0:train_size,:], dataset[len(dataset)-test_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 30
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
print(trainX.shape)
print(testX.shape)
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(64, input_dim=look_back, return_sequences=True,stateful=False))
model.add(Activation('relu'))
model.add(LSTM(128))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Dense(1))

opti = Adam(lr=0.001)

model.compile(loss='mean_absolute_error', optimizer=opti, metrics=['accuracy', mean_absloute_error_scaled])
model.fit(trainX, trainY, nb_epoch=30, batch_size=2048, shuffle=False, verbose=1, validation_split=0.1)
model.save('apple3.h5')

score = model.evaluate(testX, testY, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
print(testPredict)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = mean_absolute_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f MAE' % (trainScore))
testScore = mean_absolute_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MAE' % (testScore))

trainScore = mean_absolute_percentage_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f MAPE' % (trainScore))
testScore = mean_absolute_percentage_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MAPE' % (testScore))

trainScore = mean_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f ME' % (trainScore))
testScore = mean_error(testY[0], testPredict[:,0])
print('Test Score: %.2f ME' % (testScore))

trainScore = mean_percentage_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f MPE' % (trainScore))
testScore = mean_percentage_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MPE' % (testScore))

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
print(testPredict)

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+32:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))

#plt.plot(trainPredictPlot)
plt.plot(testPredictPlot, color="red")
plt.show()
test = open("result.txt","w+")

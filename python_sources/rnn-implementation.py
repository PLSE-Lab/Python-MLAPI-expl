
#Importing necessary header files
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



#Fixing the random.seed so it produces the same result everytime
np.random.seed(7)


#Reading data from file and only taking 'Label' column
Data= pd.read_csv("../input/Div.csv",usecols=[0])


#Converting the dataframe column to array
DArray = Data.values


#Split the DArray into training:test set(1:9)
train_s = int(len(DArray) * 0.1)
train=  DArray[0:train_s,:] 
test = DArray[train_s:len(DArray),:]




#Function to convert an array of values into a dataset matrix that RNN can work with
def Dmat(DArray, lb):
	trainX, trainY = [], []
	for i in range(len(DArray)-lb-1):
		a = DArray[i:(i+lb), 0]
		trainX.append(a)
		trainY.append(DArray[i + lb, 0])
	return np.array(trainX), np.array(trainY)


#Defining Look_Back(lb) which is the number of previous time steps to use as input variables to predict the next time period 
lb = 1

#Converting the train and test datasets to required format
trainX, trainY = Dmat(train, lb)
testX, testY = Dmat(test, lb)

#Reshaping the array to [samples, time steps, features] for LSTM to understand
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#Fitting the LSTM model with training data(Default settings)
model = Sequential()
model.add(LSTM(4, input_shape=(1, lb)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)


#Making predictions
testPredict = model.predict(testX)


#Converting the array back to a readable dataframe
for i in range(len(testPredict)):
    if i==0:
        X=pd.DataFrame({'X':testPredict[i],'Num':i+100},index=[i])
    else:
        X=X.append(pd.DataFrame({'X':testPredict[i],'Num':i+100},index=[i]))


#Filtering the dataframe according to our needs and converting to array
Y1=X[(X.X>1.77) & (X.X<1.8)]['Num'].values #For numbers divisible by 7
Y2=X[(X.X>1.87) & (X.X<1.9)]['Num'].values #For numbers divisible by 11
Y3=X[(X.X>2)]['Num'].values #For numbers divisible by 77


#Printing Output
print("Numbers Divisible by 7(and not 11):")
print(Y1)
print("Numbers Divisible by 11(and not 7):")
print(Y2)
print("Numbers Divisible by 77:")
print(Y3)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.utils import np_utils
from pandas import Series , DataFrame, Panel
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import metrics
import plotly.plotly as py
import plotly.tools as tls


train = pd.read_csv('../input/train.csv')
labels = train.ix[1:,0].values
TrainX = (train.ix[1:,1:].values).astype(float)
TrainY = np_utils.to_categorical(labels)

TrainX /= 255
TrainX -= np.std(TrainX)


TrainX= TrainX.reshape(TrainX.shape[0],28,28,1)


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(TrainY.shape[1], activation = "softmax"))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(TrainX,TrainY, epochs = 30, verbose = 2, batch_size =256)


TestX = (pd.read_csv('../input/test.csv').values).astype(float)


TestX /= 255
TestX -= np.std(TrainX)
TestX= TestX.reshape(TestX.shape[0],28,28,1)
Prediction = model.predict_classes(TestX , verbose = 2)



pd.DataFrame({"ImageId": list(range(1,len(Prediction)+1)), "Label": Prediction}).to_csv("submission.csv", index=False, header=True)
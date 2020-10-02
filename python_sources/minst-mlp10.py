import pandas as pd
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils

# The competition datafiles are in the directory ../input



from keras.layers import Input, Dense
from keras.models import Model

from keras.models import Sequential 
from keras.layers import Input, Dense, Dropout


train = pd.read_csv("../input/train.csv").values
test  = pd.read_csv("../input/test.csv").values
#(x_train, y_train), (x_test, y_test) = mnist.load_data()




batch_size = 128
img_rows, img_cols = 28, 28

train = pd.read_csv("../input/train.csv").values
test  = pd.read_csv("../input/test.csv").values

trainX = train[:, 1:].reshape(train.shape[0], img_rows* img_cols)
trainX = trainX.astype(float)
trainX /= 255.0
trainY = kutils.to_categorical(train[:, 0])
nb_classes = trainY.shape[1]



testX = test.reshape(test.shape[0], img_rows* img_cols)
testX = testX.astype(float)
testX /= 255.0

#y_train =  np_utils.to_categorical(y_train, nb_classes)
#y_test = np_utils.to_categorical(y_test, nb_classes)


model = Sequential()
model.add(Dense(64, input_dim=784, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(trainX, trainY,
          nb_epoch=10,
          batch_size=16)
yPred = model.predict_classes(testX)
np.savetxt('mnist-vggnet.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')



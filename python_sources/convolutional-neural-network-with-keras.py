
# Import classes and functions
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

# Read training and test data files
train = pd.read_csv("../input/train.csv").values
test  = pd.read_csv("../input/test.csv").values

# Set random seed for reproducibility
seed = 7
np.random.seed(seed)

# Reshape and normalize training data
trainX = train[:, 1:].reshape(train.shape[0], 1, 28, 28).astype( 'float32' )
trainX = trainX / 255.0

# Reshape and normalize test data
testX = test.reshape(test.shape[0], 1, 28, 28).astype( 'float32' )
testX = testX / 255.0

# One-hot encode output variable
trainY = np_utils.to_categorical(train[:, 0])
num_classes = trainY.shape[1]

#  CNN Model
def larger_model():
  # create model
  model = Sequential()
  model.add(Convolution2D(30, 5, 5, border_mode= 'valid' , input_shape=(1, 28, 28),
      activation= 'relu' ))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Convolution2D(15, 3, 3, activation= 'relu' ))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(128, activation= 'relu' ))
  model.add(Dense(50, activation= 'relu' ))
  model.add(Dense(num_classes, activation= 'softmax' ))
  # Compile model
  model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
  return model

# Build the model
model = larger_model()
# Fit the model
# Change number of epochs and batch size to suit
model.fit(trainX, trainY, nb_epoch=1, batch_size=200, verbose=1)
# Final evaluation of the model
scores = model.evaluate(trainX, trainY, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100)) 

# Generare predictions
yPred = model.predict_classes(testX)

# Generate submission file
np.savetxt('submission.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
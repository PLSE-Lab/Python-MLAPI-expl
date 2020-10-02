# import relevant libraries
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

# Read in the data
train = pd.read_csv('../input/train.csv')
labels = train.ix[:,0].values.astype('int32')
predictors_train = (train.ix[:,1:].values).astype('float32')
predictors_test = (pd.read_csv('../input/test.csv').values).astype('float32')

target_train = np_utils.to_categorical(labels) 

# pre-processing the data
scale = np.max(predictors_train)
predictors_train /= scale
predictors_test /= scale

mean = np.std(predictors_train)
predictors_train -= mean
predictors_test -= mean

n_cols = predictors_train.shape[1]

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation = 'relu', input_shape = (n_cols,)))

# Add the second hidden layer
model.add(Dense(50, activation = 'relu'))

# Add the output layer
model.add(Dense(10, activation = 'softmax'))

# Compile the model
model.compile( optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fit the model
model.fit(predictors_train, target_train, validation_split = .3)

print("Generating test predictions...")
preds = model.predict_classes(predictors_test, verbose=0)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras_mnist_preds.csv")

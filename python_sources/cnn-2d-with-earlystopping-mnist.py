import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

# Data loading
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# One-hot encoding & normalising data
trainLabel = to_categorical(train['label'], 10)
trainFeature = train.drop(columns = 'label')
trainFeature /= 255.0
test /= 255.0 
trainFeature = trainFeature.astype(float)
test = test.astype(float)

# Reshape 28x28 images 
trainFeature = trainFeature.values.reshape(train.shape[0], 28, 28, 1) 
testFeature = test.values.reshape(test.shape[0], 28, 28, 1)

# CNN Model
model = Sequential()
model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same', input_shape = (28, 28, 1)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

#Early stopping
es = EarlyStopping(monitor='val_loss', patience = 3, mode = 'min', restore_best_weights=True)
hisotry = model.fit(trainFeature, trainLabel,
                      batch_size = 32, 
                      epochs = 50, 
                      verbose = 2, 
                      validation_split = 0.2,
                      callbacks = [es])

pred_testLabel = model.predict(testFeature)
#return index with the max prob.
testLabel = np.argmax(pred_testLabel, axis=1)
submission = pd.DataFrame({'ImageId': range(1,len(test)+1) ,'Label': testLabel })
submission.to_csv("CNN2d-EarlyStopping_mnist.csv",index=False)
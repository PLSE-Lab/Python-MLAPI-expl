import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

Y_train = train_data["label"]
X_train = train_data.drop(labels = ["label"],axis = 1)
X_train = X_train / 255.0
X_test = test_data / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)

datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)

datagen.fit(X_train)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1)

model = Sequential()

model.add(Conv2D(32, kernel_size=5,input_shape=(28, 28, 1), activation = 'relu'))
model.add(Conv2D(32, kernel_size=5, activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=3,activation = 'relu'))
model.add(Conv2D(64, kernel_size=3,activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=3, activation = 'relu'))
model.add(BatchNormalization())

model.add(Flatten())    
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(2048, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(10, activation = "softmax"))

optimizer=Adam(lr=0.001)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

model.summary()

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
model_try = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=128),
                              epochs = 100, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=400, callbacks=[annealer])
                              
predictions = model.predict(X_test)
predictions = np.argmax(predictions,axis = 1)
predictions = pd.Series(predictions, name="Label")
submit = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)
submit.to_csv("result.csv",index=False)
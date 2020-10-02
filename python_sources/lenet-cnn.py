# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Reshape, Flatten, Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
y = train.pop('label')
test = pd.read_csv("../input/test.csv")
train = train.values.reshape(-1, 28,28, 1)
test = test.values.reshape(-1, 28, 28, 1)
y = to_categorical(y, num_classes = 10)

train = train / 255.0
test = test / 255.0

model = Sequential()
model.add(Conv2D(filters = 20, kernel_size = (5,5), padding = 'same' , input_shape = (28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters = 50, kernel_size = (5, 5), padding = "same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))


model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(train, y, batch_size = 256, epochs = 15)


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(train)

model.fit_generator(datagen.flow(train,y, batch_size=32),
                              epochs = 15, verbose = 2, steps_per_epoch=train.shape[0] // 32)

prediction = model.predict(test)
prediction = np.argmax(prediction, axis = 1)

pred = pd.DataFrame({'ImageId': range(1, len(test) + 1), 'Label': prediction})
pred.to_csv('solution.csv', index = False)






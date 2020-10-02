import pandas
import numpy

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

train_raw = numpy.array(pandas.read_csv('../input/train.csv'))

train_x = train_raw[:, 1:].reshape(train_raw.shape[0], 28, 28, 1).astype('float32') / 255.
train_y = np_utils.to_categorical(train_raw[:, 0], 10)


test_raw = numpy.array(pandas.read_csv('../input/test.csv'))
test_x = test_raw.reshape(test_raw.shape[0], 28, 28, 1).astype('float32') / 255.

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
model.add(Dense(10, activation = "softmax"))

model.compile(loss='categorical_crossentropy', optimizer="adamax", metrics=['accuracy'])

datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

model.fit_generator(datagen.flow(train_x, train_y, batch_size=86), len(train_x)/100, epochs=30,callbacks=[learning_rate_reduction])

y = model.predict_classes(test_x)
numpy.savetxt('mnist.csv', numpy.c_[range(1, len(y) + 1), y],
              delimiter=',', header='ImageId,Label', comments='', fmt='%d')

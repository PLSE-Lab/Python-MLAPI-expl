import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from math import ceil

num_classes = 10
img_rows, img_cols = 28, 28
test_size = 0.2
input_shape = (img_rows, img_cols, 1)
batch_size = 128
epochs = 128

df = pd.read_csv('../input/train.csv')
y = keras.utils.to_categorical(df.pop('label'), num_classes=num_classes)
x = df.values.reshape(df.shape[0], img_rows, img_cols, 1).astype('float32')
x /= 255
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=10.,
    width_shift_range=5.,
    height_shift_range=5.,
    zoom_range=0.1)

datagen.fit(x_train)

steps_per_epoch = ceil(len(x_train) / batch_size)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    steps_per_epoch=steps_per_epoch,
                    verbose=0)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

df = pd.read_csv('../input/test.csv')
x = df.values.reshape(df.shape[0], img_rows, img_cols, 1).astype('float32')
x /= 255
y = model.predict_classes(x)

submission = pd.DataFrame({'ImageId': range(1, df.shape[0] + 1), 'Label': y})
submission.to_csv('submission.csv', index=False)
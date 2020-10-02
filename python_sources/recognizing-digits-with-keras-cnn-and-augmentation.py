#this implementation is heavily influenced by my kernel https://www.kaggle.com/cyannani123/tiny-keras-cnn-for-aerial-cactus-identification
import os
import numpy as np
from keras.utils import np_utils
import pandas as pd
from keras import Sequential
from keras.preprocessing import image
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

X_train = train_data.drop(labels = ['label'],axis = 1).values.reshape(len(train_data),28,28,1)
y_train = np_utils.to_categorical(train_data['label'])
X_test = test_data.values.reshape(len(test_data),28,28,1)

model = Sequential()
model.add(Conv2D(64, kernel_size=(6, 6),input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10)) #NN for classification task
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) #compile model

#image augmentation
img_gen = image.ImageDataGenerator(
    data_format="channels_last",
    rescale=1/255,
    validation_split=0.10,
    rotation_range=10,
    shear_range=0.20,
    samplewise_center=True
)

img_gen.fit(X_train)

train_generator = img_gen.flow(
    x = X_train, 
    y= y_train,
    subset="training",
    batch_size=16,
    shuffle=True
)

validation_generator = img_gen.flow(
    x = X_train, 
    y= y_train,
    subset="validation",
    batch_size=16,
    shuffle=True
)

img_gen2 = image.ImageDataGenerator(
    data_format="channels_last",
    rescale=1/255
)

test_generator = img_gen2.flow(
    x = X_test,
    y= None,
    batch_size=1,
    shuffle=False
)

filepath = 'ModelCheckpoint.h5'

callbacks = [
    ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1),
    EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    ]

history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,validation_data=validation_generator,
    validation_steps=len(validation_generator),
    verbose=2,
    callbacks=callbacks
)

#load best weights for final prediction
model.load_weights('ModelCheckpoint.h5')

y_pred = model.predict_generator(
    test_generator,
    steps=len(test_data)
)

submission = pd.DataFrame()
submission['ImageId'] = test_data.index.values + 1
submission['Label'] = y_pred.argmax(axis=-1)
submission.to_csv('submission.csv', header=True, index=False)
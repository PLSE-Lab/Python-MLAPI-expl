# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from keras.engine.topology import Input
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, PReLU, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, plot_model, to_categorical
from sklearn.model_selection import train_test_split

image_width = 28
image_height = 28
image_cannels = 1
test_size = 0.25
batch_size = 32
epoch_size = 100
random_seed = 2
num_classes =  10


train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")


X_train = train.drop(labels = ["label"],axis = 1) 
Y_train = train["label"]

X_train = X_train.values.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32')
X_train /= 255
Y_train = to_categorical(Y_train, num_classes = num_classes)

test = test.values.reshape(-1, 28, 28, 1)
test = test.astype('float32')
test /= 255



X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


def model(pretrained_weights = None, input_size = (28, 28, 1)):
    inputs = Input(shape = input_size)
    bat0 = BatchNormalization()(inputs)
    act0 = Activation(activation='relu')(bat0) 
    con1 = Conv2D(32, kernel_size=(5, 5), padding='same')(act0)
    act1 = Activation(activation='relu')(con1)
    con2 = Conv2D(32, kernel_size=(5, 5), padding='same')(act1)
#    bat2 = BatchNormalization()(con2)
    act2 = Activation(activation='relu')(con2)

    con3 = Conv2D(64, kernel_size=(3, 3), padding='same')(act2)
    act3 = Activation(activation='relu')(con3)
    con4 = Conv2D(64, kernel_size=(3, 3), padding='same')(act3)
#    bat4 = BatchNormalization()(con4)
    act4 = Activation(activation='relu')(con4)      
    poo5 = MaxPooling2D(pool_size=(2, 2))(act4)

    con5 = Conv2D(128, kernel_size=(3, 3), padding='same')(poo5)
    act5 = Activation(activation='relu')(con5)
    con6 = Conv2D(128, kernel_size=(3, 3), padding='same')(act5)
    act6 = Activation(activation='relu')(con6)      
    con7 = Conv2D(256, kernel_size=(3, 3), padding='same')(act6)
    act7 = Activation(activation='relu')(con7)
    con8 = Conv2D(256, kernel_size=(3, 3), padding='same')(act7)
    act8 = Activation(activation='relu')(con8)  
    poo8 = MaxPooling2D(pool_size=(2, 2))(act8)
 
    drp1 = Dropout(rate=0.25)(poo8)
    fltn = Flatten()(drp1)
    den1 = Dense(256)(fltn)
    drp2 = Dropout(rate=0.5)(den1) 
    den3 = Dense(num_classes)(drp2)
    outs  = Activation(activation='softmax') (den3)
    model = Model(input = inputs, output = outs)
    model.compile(optimizer =Adam(lr=1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(model.summary())
    return model


datagen  = ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                              featurewise_std_normalization=False,
                              samplewise_std_normalization=False,
                              zca_whitening=False,
                              rotation_range=15,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.1,
                              zoom_range=0.1,
                              channel_shift_range=False,
                              horizontal_flip=False,
                              vertical_flip=False)
datagen.fit(X_train)


model = model()
startTime = time.time()
early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
#callbacks=[early_stopping], 
"""
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch_size,
                    verbose=2, callbacks=[early_stopping], validation_data=(X_val, Y_val))
"""
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),epochs=epoch_size,
                    verbose=2, steps_per_epoch=len(X_train) / batch_size, callbacks=[early_stopping],              
                    validation_data=(X_val, Y_val))


score = model.evaluate(X_val, Y_val, verbose=0)
print('Validation loss:', score[0])
print('Valdation accuracy:', score[1])
print("Comptation time:{0:.3f}sec".format(time.time()- startTime))



res = model.predict(test)
res = np.argmax(res,axis = 1)
res = pd.Series(res, name="Label")
submission = pd.concat([pd.Series(range(1 ,28001) ,name = "ImageId"),   res],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)
submission.head(10)



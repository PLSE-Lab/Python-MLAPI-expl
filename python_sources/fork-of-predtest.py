# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping,BaseLogger,ReduceLROnPlateau
from keras.optimizers import Adadelta, Adam, Adamax

from sklearn.model_selection import train_test_split
from time import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train_dset = pd.read_csv('../input/train.csv')
test_dset = pd.read_csv('../input/test.csv')

y_train = train_dset['label']
x_train = train_dset.drop(labels=["label"], axis=1)

x_test = np.array(test_dset, dtype='float32')
x_test = x_test.reshape((x_test.shape[0],28,28,1))
x_test /= 255

print(np.shape(x_test))

del test_dset
del train_dset

data_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)
        
x_train = np.array(x_train,dtype='float32')
x_train /= 255

x_train = x_train.reshape(x_train.shape[0],28,28,1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=2)

y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)

data_gen.fit(x_train)

model = Sequential()
model.add(Conv2D(input_shape=(28,28,1),filters=32,kernel_size=(3,3),strides=(1,1),data_format='channels_last',padding='valid',activation='relu'))
# model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.4))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
#flattening
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
#dense layer
# model.add(Dense(128,activation='tanh'))
#dropout
# model.add(Dropout(0.3))
model.add(Dense(32,activation='tanh'))
model.add(Dropout(0.2))
# model.add(Dense(128,activation='tanh'))
# model.add(Dropout(0.3))
#Output Classification
model.add(Dense(10,activation='softmax'))

opt = Adadelta()

reduceLR = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00001)

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['acc'])

# tensorboard = TensorBoard(log_dir="mnist_logs/{}".format(time()))
# checkpoint = ModelCheckpoint('checkpoints.hdf5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
earlystop = EarlyStopping(monitor='acc', min_delta=0, patience=1, verbose=0, mode='auto', baseline=None)

# model.fit(x_train,y_train,128,epochs=10,validation_split=0.05,shuffle=True,verbose=1,callbacks=[tensorboard,checkpoint,earlystop])


model.fit_generator(data_gen.flow(x_train, y_train, batch_size=128),
					shuffle=True, 
					epochs=15,
                    steps_per_epoch=100,
					validation_data=(x_val,y_val),
					callbacks=[earlystop, reduceLR])

preds = model.predict(x_test)
preds = np.argmax(preds,axis=1)

# df.to_csv(file_name, sep='\t', encoding='utf-8')

predictions = pd.Series(preds,name="Label")
predictions = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)

predictions.to_csv('pred.csv',index=False)
    
    

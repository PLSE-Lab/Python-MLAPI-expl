# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import keras
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers.convolutional import MaxPooling2D
import pandas as pd
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

traindata = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')
X_train=traindata.drop('label',axis=1)
y_train=traindata.loc[:,['label']]

X_train=X_train.values
X_train=X_train.reshape(len(X_train),28,28)
X_test=X_test.values
X_test=X_test.reshape(len(X_test),28,28)

y_train=y_train.values
y_train=y_train.ravel()

#reshaping
#this assumes our data format
#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while 
#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28,28, 1)
#more reshaping
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

datagen = ImageDataGenerator(rotation_range=15,zoom_range=0.1,width_shift_range=.1,height_shift_range=.1)

datagen.fit(X_train)

for Xg_train, yg_train in datagen.flow(X_train,y_train,batch_size=21000):
    break
    #None

Z_train=np.vstack((X_train,Xg_train))
zz_train=np.hstack((y_train,yg_train))

#set number of categories
num_category = 10
# convert class vectors to binary class matrices
zz_train_oh = keras.utils.to_categorical(zz_train, num_category)

model = keras.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer='normal',activation='relu',input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu',kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(64, kernel_size=(3, 3),kernel_initializer='normal',activation='relu',input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu',kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(128, kernel_size=(4, 4),kernel_initializer='normal',activation='relu',input_shape=input_shape))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_category, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),metrics=['accuracy'])

# DECREASE LEARNING RATE EACH EPOCH
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
# TRAIN NETWORKS
epochs = 10

#history = model.fit_generator(datagen.flow(X_train,y_train_oh, batch_size=64),epochs = epochs, steps_per_epoch = X_train.shape[0]//64, callbacks=[annealer], verbose=0)
#print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc'])))

history = model.fit(Z_train,zz_train_oh,batch_size=100,epochs=30,verbose=1,callbacks=[annealer])

y_pred=model.predict(X_test)
y_pred=np.argmax(np.round(y_pred),axis=1)

result=pd.DataFrame({'Label':y_pred})
result.insert(0,'ImageId',range(1,1+len(result)))
columnsorder=['ImageId','Label']
result=result.reindex(columns=columnsorder)
result.to_csv('Results Submission Kernel EDITION 3 - v1.csv',encoding='utf-8', index=False)
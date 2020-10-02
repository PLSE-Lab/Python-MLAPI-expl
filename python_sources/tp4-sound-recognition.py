# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 10:34:21 2019

@author: Ngoc-Tien
"""

from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Activation,Dense, Input, Dropout, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPool2D, Flatten
from keras.optimizers import RMSprop, Adam
import keras.backend as K
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import numpy as np
import keras  

from keras import losses, activations, models
from keras.wrappers.scikit_learn import KerasClassifier
import os
import matplotlib.pyplot as plt


x = np.load("../input/x.npy")
y = np.load("../input/y.npy")
y_1d = np.load("../input/y1.npy")
y_1d_split3 = np.repeat(y_1d,3)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
for train_ind, test_ind in sss.split(x,y):
    np.save("train_ind.npy",train_ind)
    np.save("test_ind.npy",test_ind)
    x_train,x_test = x[train_ind],x[test_ind]
    y_train,y_test = y[train_ind],y[test_ind]
    y_1d_train, y_1d_test = y_1d[train_ind],y_1d[test_ind]

print(x.shape)
print(y.shape)
print(y_1d.shape)
x_img = x.reshape( x.shape[0], x.shape[1], x.shape[2],1)
x_train_img,x_test_img  = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1), x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
x_split3 = np.zeros(shape = (x.shape[0]*3,64,x.shape[2]),dtype = "float16")
y_split3 = np.zeros(shape = (y.shape[0]*3,y.shape[1]))

for i in range(x.shape[0]):
    x_split3[3*i,:,:] = x[i,0:64,:]
    x_split3[3*i+1,:,:] = x[i,32:96,:]
    x_split3[3*i+2,:,:] = x[i,64:128,:]
    y_split3[3*i,:] = y[i,:]
    y_split3[3*i+1,:] = y[i,:]
    y_split3[3*i+2,:] = y[i,:]

print(y_split3.shape)    
print(x_split3.shape)

x_split3_img = x_split3.reshape((x_split3.shape[0],x_split3.shape[1],x_split3.shape[2],1))
print(x_split3_img.shape)
#%% Basic model

def my_BNN(batch_size,epochs,lr):
    output_dim = 3
    K.clear_session()
    model  = Sequential()
    
    model.add(Flatten())
    
    model.add(Dense(units=128,activation="relu"))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=64,activation="relu"))
    
    model.add(Dropout(0.2))
    
    model.add(Dense(units=128,activation="relu"))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=output_dim,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer = Adam(lr = lr),metrics=['accuracy'])
    return(model)


#%%
def my_CNN(batch_size,epochs,lr):
    K.clear_session()
    nclass = 3
    inp = Input(shape=(129, 186, 1))
    norm_inp = BatchNormalization()(inp)
    
    img_1 = Convolution2D(16, kernel_size = (3, 3), activation="relu")(norm_inp)
    img_1 = Convolution2D(32, kernel_size = (3, 3), activation="relu")(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
    
    img_1 = Dropout(rate = 0.1)(img_1)
    
    img_1 = Convolution2D(64, kernel_size = 3, activation="relu")(img_1)
    img_1 = Convolution2D(128, kernel_size = 3, activation="relu")(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
    
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = GlobalMaxPool2D()(img_1)
    #img_1 = Convolution2D(64, kernel_size=3, activation=activations.relu)(img_1)
    #img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
    
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)
    model = models.Model(inputs=inp, outputs=dense_1)

    opt = Adam(lr = lr)
    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()

    return(model)



#%%
def my_CNN_split(batch_size,epochs,lr):
    K.clear_session()
    nclass = 3
    inp = Input(shape=(64, 186, 1))
    norm_inp = BatchNormalization()(inp)
    img_1 = Convolution2D(16, kernel_size=(3, 7), activation=activations.relu)(norm_inp)
    img_1 = Convolution2D(16, kernel_size=(3, 7), activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 7))(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution2D(128, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = GlobalMaxPool2D()(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)
    model = models.Model(inputs=inp, outputs=dense_1)

    opt = Adam(lr = lr)
    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()

    return(model)
    
#%%
def cv_neural_network(x,y,model_type):
    ep = [10,15,20] # add 50, 100, 150 etc
    bt = [32,64,128] # add 5, 10, 20, 40, 60, 80, 100 etc
    param_grid = dict(epochs=ep, batch_size=bt)
    if model_type == "CNN_split":
        model = KerasClassifier(my_CNN_split,batch_size=20, epochs=20,lr = 0.01)
    if model_type == "CNN":
        model = KerasClassifier(my_CNN,batch_size=20, epochs=20,lr = 0.01)
    if model_type == "BNN":
        model = KerasClassifier(my_BNN,batch_size=20, epochs=20,lr = 0.01)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,cv = 5)
    grid_result = grid.fit(X = x, y = y, validation_split = 0.3, shuffle = True)
    return(grid_result)
    
    
#%%
#model_fin = cv_neural_network(x_train_img,y_1d_train,"CNN")


#%%
#model_fin.best_params_

#model_fin.best_estimator_.save_weights('my_model_weights.h5')




model_fin_CNN = cv_neural_network(x_train,y_train,"CNN")


#%%
model_fin_CNN.best_params_
from sklearn.externals import joblib
joblib.dump(model_fin_CNN.best_estimator_, 'my_model_CNN_fin.pkl')


#model_cnn = KerasClassifier(my_CNN,batch_size=20, epochs=20,lr = 0.01)
#model_cnn.fit(x_img,y_1d, validation_split = 0.3, shuffle = True)
#model_cnn.model.save_weights('my_model_weights.h5')

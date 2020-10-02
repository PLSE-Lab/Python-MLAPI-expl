#!/usr/bin/env python
# coding: utf-8

# In[13]:


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
from pylab import rcParams
rcParams['figure.figsize'] = 13, 13


# In[14]:


import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.cm as cm
from tensorflow import keras as K
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras.layers import *
import random
from tensorflow.keras.models import Model


# In[15]:


train_csv = pd.read_csv("../input/train.csv")
test_csv = pd.read_csv("../input/test.csv")


# In[16]:


train = np.copy(train_csv.values)
test = np.copy(test_csv.values)


# In[17]:


def drop_outliers(data, threshold = 0, reso = 50):
    good = np.ones(data.shape[0])
    for i in range(data.shape[1]):
        i_hist = np.histogram(data[:,i], reso)
        for j in range(reso):
            if i_hist[0][j] < threshold:
                good *= np.logical_or(data[:,i] >= i_hist[1][j+1], data[:,i] <= i_hist[1][j])
    
    return [ np.logical_not(good),good]

def standardize(x, std, mean):
    return (x - mean)/std
def destandardize(x, std, mean):
    return (x*std) + mean
#train = drop_outliers(train, 1000, 100)
#train.shape


# In[19]:


class DenseModel():
    def __init__(self, layers):
        act = "relu"
        l2p = 0.0 
        ki = "normal"#K.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=seed)
        bi = "zero"#K.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=seed)
        self.model = Sequential()
        self.model.add(Dense(layers[0],activation=act, kernel_initializer=ki,  bias_initializer=bi, input_shape=(9,)))
        for layer in layers[1:]:
            self.model.add(Dense(layer,activation=act, kernel_initializer=ki,  bias_initializer=bi))
            self.model.add(Dropout(0.01))
        self.model.add(Dense(1,activation="linear", kernel_initializer=ki,  bias_initializer=bi))
        #### We define a basic feedforward DNN
      #  input1 = Input(shape=(9,))
       # x1r = GaussianNoise(0.00)(input1)
       # x1 = Dense(128,activation=act, kernel_initializer=ki,  bias_initializer=bi)(input1)
        
       # x2 = Dense(256,activation=act, kernel_initializer=ki,  bias_initializer=bi)(x1)
       # x3 = Dense(512,activation=act, kernel_initializer=ki,   bias_initializer=bi)(x2)
       # x4 = Dense(512,activation=act, kernel_initializer=ki,   bias_initializer=bi, name = "last")(x3)
       # result = Dense(1,activation="linear", kernel_initializer=ki,  bias_initializer=bi, name = "result")(x4)
       # self.model = Model(inputs = input1, outputs = [result, x4])
        
        #### Other DNN to fit on the error of the previous
       # E_input = Input(shape=(8, ))
       # E_x4_in = Input(shape=(512, ))
        
       # E_1 = Dense(256, activation=act, kernel_initializer=ki, bias_initializer=bi)(E_input)
       # E_2 = Dense(256, activation=act, kernel_initializer=ki, bias_initializer=bi)(E_1)
       # E_3 = Dense(256, activation=act, kernel_initializer=ki, bias_initializer=bi)(E_2)
       # Together = Concatenate()([E_x4_in, E_3])
       # E_4 = Dense(1024, activation=act, kernel_initializer=ki, bias_initializer=bi)(Together)
       # E_5 = Dense(1024, activation=act, kernel_initializer=ki, bias_initializer=bi)(E_4)
       # E_result = Dense(1,activation="linear",  kernel_initializer=ki,  bias_initializer=bi)(E_5)
       # self.E_model = Model(inputs = [E_input, E_x4_in], outputs = E_result)
        
        
      
    def printout(self):
        print(self.model.summary())
        print(self.E_model.summary())
        
    def train(self,  for_train, train_reds, for_valid, valid_reds, c_kwargs, f_kwargs):
        self.model.compile(**c_kwargs)
       
        f_kwargs["validation_data"] = (for_valid[:,:], valid_reds)
        s_history = self.model.fit(for_train[:,:], train_reds ,**f_kwargs, shuffle = True)
        self.model.load_weights("bestmodel.h5")
        
        r = self.model.evaluate(for_valid[:,:], valid_reds, batch_size=2048)
        print(r)
        print(np.sqrt(r))
        return s_history
    def deploy(self, data, use_E):
        pred = self.model.predict(data[:,:], batch_size = 2048)
        
        return pred
    


# In[20]:


class MetaModel():
    def __init__(self, subshapes = [[256, 512, 512], [256, 512, 512]]):
        self.submodels = []
        for shape in subshapes:
            self.submodels.append(DenseModel(shape))
            
        act = "relu"
        ki = "normal"#K.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=seed)
        bi = "zero"
        subresults = Input(shape=(len(subshapes), ))
        context = Input(shape=(7, ))
        self.good_mask = np.zeros(len(subshapes))
        x1 = Dense(32,activation=act, kernel_initializer=ki,  bias_initializer=bi)(context)
        x11 = Dense(64,activation=act, kernel_initializer=ki,  bias_initializer=bi)(x1)
        x11d = Dropout(0.2)(x11)
        x2 = Dense(len(subshapes),activation=act, kernel_initializer=ki,  bias_initializer=bi)(x11d)
        weighted = Multiply()([x2, subresults])
      #  result = Average()( weighted)
        result =    result = Dense(1,activation="linear", kernel_initializer=ki, use_bias=False, name = "result")(weighted)
        self.stacker = Model(inputs = [subresults,context] , outputs = result)
        self.stacker.compile(optimizer = "adam", loss= "mean_squared_error")
    
    def train(self, train, n_epochs = 10, batch_size = 512, stack_epochs = 5):
        prep_train, prep_reds = self.consumeTrainigData(train)
        valid_n = 0
        prep_train = prep_train[valid_n:,:]
        prep_reds = prep_reds[valid_n:,:]
        
        stack_val = prep_train[:valid_n,:]
        stack_val_reds = prep_reds[:valid_n,:]
        histories = []
        n = len(self.submodels)
        predictions = np.zeros((train.shape[0]-valid_n, n ))
        stack_val_preds =   np.zeros((valid_n, n ))
        for i in range(0,n):
            
            for_train = []
            train_reds = []
            for j in range(1, n):
                  for_train.append( prep_train[(i+j)%n::n,:])
                  train_reds.append(   prep_reds[(i+j)%n::n,:])
            for_train = np.concatenate( for_train)
            for_valid = prep_train[(i)%n::n,:]
            print(for_train.shape)
            train_reds =  np.concatenate( train_reds)
            valid_reds = prep_reds[(i)%n::n,:]
            
            M = self.submodels[i%len(self.submodels)]
            s_history = M.train(for_train, train_reds, for_valid, valid_reds, dict( optimizer=K.optimizers.Adam(), loss = "mean_squared_error"),
                               dict( batch_size = batch_size, epochs = n_epochs, validation_data= (for_valid, valid_reds), verbose = 2, 
                                callbacks=[K.callbacks.ModelCheckpoint("bestmodel.h5" , monitor="val_loss", verbose=3, save_best_only="True")])) 
            r = M.model.evaluate(for_valid[:,:], valid_reds, batch_size=2048)
            self.good_mask[i] = r < 0.00205
          #  predictions[:,i] = M.deploy(prep_train, False)[:,0]
          #  stack_val_preds[:,i] = M.deploy(stack_val, False)[:,0]
          #  val = M.deploy(for_valid, False)[:,0]
          #  print( ((destandardize( val,self.re_std, self.re_min) - valid_reds[:,0])**2).shape )
          #  print("REAL: " + str( ( ((destandardize( val,self.re_std, self.re_min) - valid_reds[:,0])**2).mean() )))
            histories.append(s_history)
            os.system("echo "+str(i))
        
        #self.stacker.fit([predictions, prep_train[:,2:9]], prep_reds, validation_data = ([stack_val_preds, stack_val[:,2:9]], stack_val_reds), verbose = 2, epochs = stack_epochs, batch_size = 1024,
                        # callbacks=[K.callbacks.ModelCheckpoint("bestStack.h5" , monitor="val_loss", verbose=3, save_best_only="True")])
        #self.stacker.load_weights("bestStack.h5")
        #print("Final:", self.stacker.evaluate([stack_val_preds, stack_val[:,2:9]],stack_val_reds, batch_size = 2048)*(self.re_std**2))
        #print(stack_val_reds.shape, val.shape)
        #print("Final: " + str(  ((destandardize( val,self.re_std, self.re_min) - stack_val_reds)**2).mean() ))
        return histories
    
    def consumeTrainigData(self, train):
        traini = train.copy()
        traini[:,3:]
        self.tr_mean = traini[:,:9].mean(axis=0)
        self.tr_std = traini[:,:9].std(axis=0)
        self.re_min = traini[:,9:-1].min(axis=0)
        self.re_std = 1.0#traini[:,9:-1].std(axis=0)
        print(self.re_std)
        prep_train = standardize(traini[:,:9], self.tr_std, self.tr_mean)
        prep_reds = standardize(traini[:,9:-1], self.re_std, self.re_min)
        print(prep_reds)
        return prep_train, prep_reds
    def deploy(self, data, use_E = True):
        print(self.good_mask)
        data = standardize(data[:,:9], self.tr_std, self.tr_mean)
        predictions = np.zeros((data.shape[0], len(self.submodels),))
       
        for i, model in enumerate(self.submodels):
            print(i)
            predictions[:,i] = model.deploy(data, False)[:,0]
            
        return destandardize(predictions[:,self.good_mask.astype("bool")].mean(axis=1), self.re_std, self.re_min)#self.stacker.predict([predictions, data[:,2:9]], batch_size = 2048),self.re_std,self.re_min), predictions.std(axis=0)
    
    def testSubmodel(self, data, i=0,use_E = True):
        data = standardize(data[:,:9], self.tr_std, self.tr_mean)
        predictions = self.submodels[i].deploy(data[:,:], use_E)[:,0]
        return predictions
        


# In[21]:


class SegmentedModel():
    def __init__(self, segmenter, n_s):
        self.segmenter = segmenter
        self.n_s = n_s
        self.submodels = []
        for i in range(0, n_s):
            self.submodels.append(MetaModel(5))
    
    def train(self, train, n_epochs, s_kwargs, batches ):
        segment_masks = self.segmenter(train, **s_kwargs)
        for i, model in enumerate(self.submodels):
            histories = model.train(train, n_epochs = 5, batch_size = batches[i])
        for i, model in enumerate(self.submodels):
            histories = model.train(train[segment_masks[i].astype("bool")], n_epochs = n_epochs, batch_size = batches[i])


# In[22]:


shapes = [[256]*5]*20
         
meta = MetaModel(shapes) #Ok
print(meta.stacker.summary())
histories = meta.train(train, n_epochs = 30, batch_size = 128, stack_epochs = 10)


# In[ ]:


back = 40
for s_history in histories:
    p1 = plt.plot(s_history.epoch[-back:], s_history.history["loss"][-back:], c = "y")
    p2 = plt.plot(s_history.epoch[-back:], s_history.history["val_loss"][-back:], c = "r")
plt.show()
for s_history in histories:
    p_sq = p2 = plt.plot(s_history.epoch, np.sqrt(s_history.history["val_loss"]), c = "r")
    p_ref = plt.plot(s_history.epoch, np.full(len(s_history.epoch), 0.04503), c="b")


# In[ ]:


pred = meta.deploy(test, False)
ids = 9
print(pred[0].shape, train[:,ids].shape)
hperd = plt.hist(pred[0], 100,histtype="step", range = (0.0,0.8))
h = plt.hist(train[:,ids], 100, histtype="step", range = (0.0,0.8))
plt.show()


# In[ ]:


pred


# In[ ]:


submission = pd.DataFrame(data={'id':np.arange(pred.shape[0]),'redshift':pred[:]})
submission.columns = ['id', 'redshift']
print(submission.head())
filename = 'redshifTest.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


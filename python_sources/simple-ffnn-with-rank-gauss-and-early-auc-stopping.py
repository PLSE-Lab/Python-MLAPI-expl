#!/usr/bin/env python
# coding: utf-8

# This kernel is forked from Takumi Ihara's kernel "10-fold-simple-FFNN-with-rank-gauss" 
# which is forked from Andy Harless's kernel "Simple FFNN from Dromosys Features".  
# Original kernel has no early stopping.  

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_auc_score
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

import gc
import os
print(os.listdir("../input"))
print(os.listdir("../input/save-dromosys-features"))


# In[ ]:


df = pd.read_pickle('../input/save-dromosys-features/df.pkl.gz')
print("Raw shape: ", df.shape)

df.set_index('SK_ID_CURR', inplace=True)

y = df['TARGET'].copy()
feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

df.drop(['index', 'TARGET'], axis=1, inplace=True)

print("X shape: ", df.shape, "    y shape:", y.shape)

print("\nPreparing data...")
for feat in feats:
    df[feat] = df[feat].fillna(df[feat].mean())


# In[ ]:


# i must congrats someone that did this, but i read it on internet, please if it's you, congrats, and explain your code :)
def rank_gauss(x):
    from scipy.special import erfinv
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x


# In[ ]:


for i in df.columns:
    #print('Categorical: ',i)
    df[i] = rank_gauss(df[i].values)


# In[ ]:


training = y.notnull()
testing = y.isnull()
X_train = df[training].values
X_test = df[testing].values
y_train = np.array(y[training])
print( X_train.shape, X_test.shape, y_train.shape )
gc.collect()


# In[ ]:


import tensorflow as tf
from keras.callbacks import Callback
import logging
class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_proba(self.X_val, verbose=0)
        score = roc_auc_score(self.y_val, y_pred)
        
        logging.info("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
        print("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
        logs['val_auc'] = score


# In[ ]:


n_folds = 10
folds = KFold(n_splits=n_folds, shuffle=True, random_state=42)
sub_preds = np.zeros(X_test.shape[0])
oof_preds = np.zeros(X_train.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
    trn_x, trn_y = X_train[trn_idx], y_train[trn_idx]
    val_x, val_y = X_train[val_idx], y_train[val_idx]
    earlystop = EarlyStopping(monitor='val_auc', min_delta=0, patience=3, verbose=0, mode='max')
    file_path = "fold " + str(n_fold+1) + " best_model.hdf5"
    check_point = ModelCheckpoint(file_path, monitor = "val_auc", verbose = 1, save_best_only = True, mode = "max")
    
    print( 'Setting up neural network...' )
    nn = Sequential()
    nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = df.shape[1]))
    nn.add(PReLU())
    nn.add(Dropout(.3))
    nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.3))
    nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.3))
    nn.add(Dense(units = 26, kernel_initializer = 'normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.3))
    nn.add(Dense(units = 12, kernel_initializer = 'normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.3))
    nn.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    nn.compile(loss='binary_crossentropy', optimizer='adam')
    
    print( 'Fitting neural network...' )
    ival = IntervalEvaluation(validation_data=(val_x, val_y), interval=10)
    nn.fit(trn_x, trn_y, validation_data = (val_x, val_y), epochs=20, verbose=0, callbacks=[ival, earlystop, check_point], batch_size=128)
    
    best_model = load_model(file_path)
    
    oof_preds[val_idx] = best_model.predict(val_x).flatten()
    
    print(roc_auc_score(val_y, oof_preds[val_idx]))
    
    
    print( 'Predicting...' )
    sub_preds += best_model.predict(X_test).flatten().clip(0,1) / folds.n_splits
    
    gc.collect()
print('FULL AUC: {}'.format(roc_auc_score(y_train, oof_preds)))


# In[ ]:


print( 'Saving results...' )
sub = pd.DataFrame()
sub_train = pd.DataFrame()
sub['SK_ID_CURR'] = df[testing].index
sub['TARGET'] = sub_preds
sub[['SK_ID_CURR', 'TARGET']].to_csv('sub_nn.csv', index= False)

print( sub.head() )


# In[ ]:





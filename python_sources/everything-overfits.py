#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


# Import libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from bayes_opt import BayesianOptimization
import lightgbm as lgb

import os

from sklearn.cluster import KMeans

from sklearn.neighbors import DistanceMetric

from xgboost import XGBClassifier


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[ ]:


cols_to_keep= ["target", "ID_code","var_139","var_12","var_81","var_110","var_53","var_146","var_26","var_6","var_174","var_76",
               "var_166","var_148","var_80","var_22","var_99","var_133","var_21","var_78","var_198","var_165",
               "var_109","var_190","var_1","var_2","var_179","var_44","var_164","var_0","var_13","var_92",
               "var_177","var_40","var_154","var_9","var_34","var_191","var_170","var_94","var_33","var_108",
               "var_169","var_184","var_115","var_123","var_121","var_192","var_67","var_95","var_18","var_75",
               "var_5","var_93","var_149","var_91","var_173","var_122","var_188","var_107","var_135","var_89",
               "var_130","var_186","var_113","var_197","var_72","var_62","var_180","var_11","var_20","var_23",
               "var_24","var_56","var_106","var_28","var_104","var_32","var_37","var_39","var_48","var_77",
               "var_43","var_117","var_85","var_147","var_119","var_157","var_143","var_141","var_162",
               "var_127","var_167","var_153","var_155","var_86"]


# In[ ]:


# Add RUC metric to monitor NN
def auc(y_true, y_pred):
    return tf.py_func(metrics.roc_auc_score, (y_true, y_pred), tf.double)


# In[ ]:


from keras.layers import Dense,Dropout,BatchNormalization
from keras import regularizers
import keras
from keras.callbacks import LearningRateScheduler,EarlyStopping
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.constraints import max_norm
import math


# In[ ]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:





# In[ ]:


import gc
import random
from keras import models
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras import optimizers
from keras.regularizers import l1


# In[ ]:





# In[ ]:





# In[ ]:


def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate
lrate = LearningRateScheduler(step_decay)


# In[ ]:


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))


# In[ ]:


import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.preprocessing import MinMaxScaler,StandardScaler
NFOLDS = 5
RANDOM_STATE = 42
annealer = LearningRateScheduler(lambda x: 1e-2 * 0.95 ** x)
#script_name = os.path.basename(__file__).split('.')[0]
#MODEL_NAME = "{0}__folds{1}".format(script_name, NFOLDS)

#print("Model: {}".format(MODEL_NAME))













# In[ ]:


gc.collect()


# In[ ]:


print("Reading training data")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "y = train.target.values\ntrain_ids = train.ID_code.values\nfeatures = [c for c in train.columns if c not in ['ID_code', 'target']]\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "features = [c for c in train.columns if c not in ['ID_code', 'target']]\nfor feature in features:\n#    train['mean_'+feature] = np.round((train[feature].mean()-train[feature]),2)\n#    train['z_'+feature] = np.round((train[feature] - train[feature].mean())/train[feature].std(ddof=0),2)\n    train['sq_'+feature] = np.round((train[feature])**2,2)\n    train['c_'+feature] = np.round((train[feature])**3,2)\n    train['p4_'+feature] = np.round((train[feature])**4,2)\n    train['sqrt_'+feature] = np.round((train['sq_'+feature])**(1/4),2)\n    train['log_'+feature] = np.round(np.log(train['sq_'+feature]+10)/2,2)\n    \n\nfor feature in features:\n#    test['mean_'+feature] = np.round((test[feature].mean()-test[feature]),2)\n#    test['z_'+feature] = np.round((test[feature] - test[feature].mean())/test[feature].std(ddof=0),2)\n    test['sq_'+feature] = np.round((test[feature])**2,2)\n    test['c_'+feature] = np.round((test[feature])**3,2)\n    test['p4_'+feature] = np.round((test[feature])**4,2)\n    test['sqrt_'+feature] = np.round((test['sq_'+feature])**(1/4),2)\n    test['log_'+feature] = np.round(np.log(test['sq_'+feature]+10)/2,2)\n    \n\n   ")


# In[ ]:


#train= reduce_mem_usage(train)


# In[ ]:


#test = reduce_mem_usage(test)


# In[ ]:


train.head()


# In[ ]:


train = train.drop(['ID_code', 'target'], axis=1)
feature_list = train.columns

print('Train',train.shape)
print('Test', test.shape)


# In[ ]:





# In[ ]:


test_ids = test.ID_code.values
test = test[feature_list]


# In[ ]:


gc.collect()


# In[ ]:


print(train.shape,test.shape)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


# In[ ]:


X = train.values.astype(float)
X_test = test.values.astype(float)


# In[ ]:


get_ipython().run_cell_magic('time', '', "sc = StandardScaler()\nmmsc =  MinMaxScaler()\nmasc = MaxAbsScaler()\nrbsc =  RobustScaler(quantile_range=(25, 75))\nyeoj= PowerTransformer(method='yeo-johnson')\nboxcox= PowerTransformer(method='box-cox')\nqnormal =  QuantileTransformer(output_distribution='normal')\nquniform= QuantileTransformer(output_distribution='uniform')\nnormal= Normalizer()\nX1= np.round(mmsc.fit_transform(X),3)\nX = np.round(sc.fit_transform(X),3)\nX_test1 = np.round(mmsc.fit_transform(X_test),3)\nX_test = np.round(sc.transform(X_test),3)")


# In[ ]:


X_test


# In[ ]:


gc.collect()


# In[ ]:


clfs = []
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros((len(train), 1))
test_preds = np.zeros((len(test), 1))
del train, test
gc.collect()


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
loss_history = LossHistory()
lrate = LearningRateScheduler(step_decay)
reduce = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)
callbacks_list = [reduce,EarlyStopping(monitor='val_auc', patience=20,mode='max',restore_best_weights=True)]


# In[ ]:


from keras.layers import GaussianNoise


# In[ ]:


from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import concatenate


# In[ ]:


def get_model():    
    x1_in = Input(shape=(X.shape[1],), name='x1_in')
    x2_in = Input(shape=(X1.shape[1],), name='x2_in')
    x1 = Dense(512, activation='tanh')(x1_in)
    x1 = Dropout(0.6)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dense(256, activation='tanh')(x1)
    x1 = Dropout(0.6)(x1)
    x1 = BatchNormalization()(x1)
    x2 = Dense(512, activation='tanh')(x2_in)
    x2 = Dropout(0.6)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dense(256, activation='tanh')(x2)
    x2 = Dropout(0.6)(x2)
    x2 = BatchNormalization()(x2)
    z = concatenate([x1, x2])
    z = Dense(256, activation='tanh')(z)
    z = Dropout(0.5)(z)
    z = BatchNormalization()(z)
    out = Dense(1, activation='sigmoid', name='out')(z)
    model = Model(inputs=[x1_in, x2_in], outputs=[out])
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy',auc])
    return model


# In[ ]:


for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    print("Current Fold: {}".format(fold_))
    trn_x1, trn_y = X[trn_, :], y[trn_]
    val_x1, val_y = X[val_, :], y[val_]
    trn_x2 = X1[trn_,:]
    val_x2 = X1[val_,:]
    clf = get_model()

    clf.fit([trn_x1,trn_x2],trn_y,batch_size=512,epochs=500,verbose=1,callbacks=callbacks_list,validation_data=([val_x1,val_x2],val_y))

    val_pred = clf.predict([val_x1,val_x2])
    test_fold_pred = clf.predict([X_test,X_test1])

    print("AUC = {}".format(metrics.roc_auc_score(val_y, val_pred)))
    oof_preds[val_, :] = val_pred.reshape((-1, 1))
    test_preds += test_fold_pred.reshape((-1, 1))
    del trn_x1,trn_x2, trn_y , val_x1,val_x2,val_y
    gc.collect()

test_preds /= NFOLDS


# In[ ]:


gc.collect()


# In[ ]:


roc_score = metrics.roc_auc_score(y, oof_preds.ravel())
print("Overall AUC = {}".format(roc_score))

print("Saving OOF predictions")
oof_preds = pd.DataFrame(np.column_stack((train_ids, oof_preds.ravel())), columns=['ID_code', 'target'])
#oof_preds.to_csv('../kfolds/nn__{}.csv'.format( str(roc_score)), index=False)

print("Saving code to reproduce")
#shutil.copyfile('../model_source/nn__{}.py'.format( str(roc_score)))

print("Saving submission file")
sample = pd.read_csv('../input/sample_submission.csv')
sample.target = test_preds.astype(float)
sample.ID_code = test_ids
sample.to_csv('submission__nn__{}.csv'.format(str(roc_score)), index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





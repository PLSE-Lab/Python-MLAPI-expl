#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, preprocessing
import xgboost as xgb
import tensorflow as tf
from keras.layers import Dense, Input
from collections import Counter
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import callbacks
from keras import backend as K
from keras.layers import Dropout

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


def fallback_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except:
        return 0.5


# In[ ]:


def auc(y_true, y_pred):
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)


# In[ ]:


NFOLDS = 5
RANDOM_STATE = 42


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.target.value_counts()


# In[ ]:


y = train.target
ids = train.id.values
train = train.drop(['id', 'target'], axis=1)
test_ids = test.id.values
test = test[train.columns]

folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros((len(train)))
test_preds = np.zeros((len(test)))

scl = preprocessing.StandardScaler()
scl.fit(pd.concat([train, test]))
train = scl.transform(train)
test = scl.transform(test)

for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    print("Current Fold: {}".format(fold_))
    trn_x, trn_y = train[trn_, :], y.iloc[trn_]
    val_x, val_y = train[val_, :], y.iloc[val_]

    inp = Input(shape=(trn_x.shape[1],))
    x = Dense(2000, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(1000, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(500, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(100, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)
    clf = Model(inputs=inp, outputs=out)
    clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])

    es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=10,
                                 verbose=1, mode='max', baseline=None, restore_best_weights=True)

    rlr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                      patience=3, min_lr=1e-6, mode='max', verbose=1)

    clf.fit(trn_x, trn_y, validation_data=(val_x, val_y), callbacks=[es, rlr], epochs=100, batch_size=1024)
    
    val_preds = clf.predict(val_x)
    test_fold_preds = clf.predict(test)
    
    print("AUC = {}".format(metrics.roc_auc_score(val_y, val_preds)))
    oof_preds[val_] = val_preds.ravel()
    test_preds += test_fold_preds.ravel()
    
    K.clear_session()
    gc.collect()


# In[ ]:





# In[ ]:


test_preds /= NFOLDS


# In[ ]:


sample = pd.read_csv("../input/sample_submission.csv")
sample.target = test_preds
sample.to_csv("submission.csv", index=False)


# In[ ]:


sample.head()


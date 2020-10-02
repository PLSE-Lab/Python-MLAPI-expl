#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.initializers import TruncatedNormal, RandomUniform

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


seed = 4
def reset_seed(s=seed):
    np.random.seed(s)
    tf.set_random_seed(s)
reset_seed()


# In[ ]:


train = pd.read_csv('../input/train.csv', index_col='ID_code')
test = pd.read_csv('../input/test.csv', index_col='ID_code')

target = train[['target']]
train.drop('target', axis=1, inplace=True)

feats = train.columns

display(train.head())
display(test.head())
display(target.head())


# In[ ]:


trn = {'x': train, 'y': target}
tst = {'x': test}


# In[ ]:


scaler = StandardScaler()
trn['x'] = scaler.fit_transform(trn['x'])
tst['x'] = scaler.transform(test)


# In[ ]:


def roc_auc_score_wrapper(y_true, y_pred):
    check = np.sum(y_true)
    if check == 0 or check == len(y_true):
        return 0.5
    return roc_auc_score(y_true, y_pred)

def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score_wrapper, (y_true, y_pred), tf.double)

# def auc(y_true, y_pred):
#     return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


# In[ ]:


def get_model():
    model = tf.keras.Sequential([
        Dense(200, kernel_initializer=RandomUniform(seed=seed), activation=LeakyReLU(), input_dim=len(feats)),
        Dropout(0.5),
        Dense(150, kernel_initializer=RandomUniform(seed=seed), activation=LeakyReLU()),
        Dropout(0.5),
        Dense(100, kernel_initializer=RandomUniform(seed=seed), activation=LeakyReLU()),
        Dropout(0.5),
        Dense(50, kernel_initializer=RandomUniform(seed=seed), activation=LeakyReLU()),
        Dropout(0.5),
        Dense(1, kernel_initializer=RandomUniform(seed=seed), activation='sigmoid')
    ])

    opt = tf.keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=opt, 
                  loss='binary_crossentropy',
                  metrics=["accuracy", auc])
    
    params = {
        'epochs': 50, 
        'batch_size': 128, 
        'class_weight': compute_class_weight('balanced', np.unique(target.values), target.values[:,0])
    }
    
    return model, params

m, params = get_model()
m.summary()
for k in params:
    print(k, ': ', params[k])


# In[ ]:


reset_seed()

clf, params = get_model()
history = clf.fit(
    trn['x'], trn['y'],
    **params
).history

pred = clf.predict(tst['x'])


# In[ ]:


sns.set()
plt.figure(figsize=(25, 8))
metrics = ['auc', 'acc', 'loss']
for i in range(len(metrics)):
    m = metrics[i]
    epochs = np.array(range(len(history[m])))

    plt.subplot(1,len(metrics),i+1)
    plt.title(metrics[i])

    sns.lineplot(epochs, history[m], label='Train')

    plt.xticks(list(epochs[::5]) + [epochs[-1]+1])
    #plt.yticks(np.arange(0,1.0001,0.1))


# In[ ]:


df = pd.DataFrame({'ID_code': test.index, 'target': pred[:,0]})
df.to_csv('submission.csv', index=False)
df.head()


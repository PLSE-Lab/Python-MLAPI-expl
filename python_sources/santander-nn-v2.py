#!/usr/bin/env python
# coding: utf-8

# In[30]:


#!/usr/bin/python

import os
print(os.listdir("../input"))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
import datetime

#import standard ML libraries
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.utils.multiclass import unique_labels

#keras NN libraries:
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras import regularizers
from keras.constraints import max_norm
from sklearn.preprocessing import StandardScaler, RobustScaler, Binarizer, KernelCenterer
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#import specific classifiers
from sklearn.ensemble import RandomForestClassifier
print('Lib import check positive ++.')

# Any results you write to the current directory are saved as output.


# In[10]:


test_data = pd.read_csv('../input/test.csv')
print("Test data :", test_data.shape)
train_data = pd.read_csv('../input/train.csv')
print("Train data :", train_data.shape)
target = pd.read_csv('../input/sample_submission.csv')
print("Target :", target.shape)


# In[11]:


#check a sample output of first few rows:
print(train_data.head(5))


# In[12]:


def data_target_split (df):
    X = df.iloc[:,2:].values
    y = df.iloc[:,1].values
    return X, y
X, y = data_target_split(train_data)
#select the right part of the final test data
test_data = test_data.iloc[:,1:].values


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 329)


# In[14]:


#scale the X_train with Min Max scaler:
scaler = StandardScaler()
scaler.fit(X_train)
X_tr_scaled = scaler.transform(X_train)
X_tst_scaled = scaler.transform(X_test)

#check the output shapes:
X_train.shape, y_train.shape


# In[27]:


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# In[16]:


def make_model():
    model = Sequential()

    #input 
    model.add(Dense(200, input_dim=200, kernel_initializer = 'uniform', 
                    kernel_regularizer=regularizers.l2(0.005), kernel_constraint = max_norm(5.)))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    #1 
    model.add(Dense(200, kernel_initializer = 'uniform', 
                    kernel_regularizer=regularizers.l2(0.005), kernel_constraint=max_norm(5)))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))
    
    #2
    model.add(Dense(100, kernel_initializer = 'uniform', 
                    kernel_regularizer=regularizers.l2(0.005), kernel_constraint=max_norm(5)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    #2
    model.add(Dense(50, kernel_initializer = 'uniform', 
                    kernel_regularizer=regularizers.l2(0.005), kernel_constraint=max_norm(5)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    
    #output
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", auc])
    return model


# In[19]:


def train_model_iterate(X, y, test):
    model = make_model()
    pred = pd.DataFrame()
    for i in range(1, 3):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)
        earlystopper = EarlyStopping(patience=5, verbose=1)
        history = model.fit(X_train, y_train, batch_size=15600, epochs=500, validation_split=0.2, verbose=2, callbacks=[earlystopper], shuffle=True)
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("accuracy for test data: %.2f%%" % (scores[1]*100))
        plt.plot(history.history['acc'], label='accuracy for train data')
        plt.plot(history.history['val_acc'], label='validation data accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.7)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        y_pred_t = model.predict(test)
        print(y_pred_t.T[0])
        pred[i] = y_pred_t.T[0]
    return pred


# In[28]:


pr1 = train_model_iterate(X, y, test_data)


# In[33]:


pr1['mean'] = pr1.mean(axis=1)
pr1.head()


# In[34]:


ver = '1.1'
filename = 'submission_{}_{}_'.format(ver, datetime.datetime.now().strftime('%Y-%m-%d'))
target['target'] = pr1['mean']
target.to_csv(filename+'1'+'.csv', index=False)


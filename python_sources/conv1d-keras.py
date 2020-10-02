#!/usr/bin/env python
# coding: utf-8

# inspired  from : https://www.kaggle.com/theoviel/deep-learning-starter
# https://www.kaggle.com/coni57/model-from-arxiv-1805-00794

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import os
from scipy.signal import resample
print(os.listdir("../input"))
from keras import backend as K

# Any results you write to the current directory are saved as output.


# In[ ]:


Dataset=pd.read_csv("../input/X_train.csv")
Labels=pd.read_csv("../input/y_train.csv")
Test=pd.read_csv("../input/X_test.csv")


# In[ ]:


encode_dic = {'fine_concrete': 0, 
              'concrete': 1, 
              'soft_tiles': 2, 
              'tiled': 3, 
              'soft_pvc': 4,
              'hard_tiles_large_space': 5, 
              'carpet': 6, 
              'hard_tiles': 7, 
              'wood': 8}
decode_dic = {0: 'fine_concrete',
              1: 'concrete',
              2: 'soft_tiles',
              3: 'tiled',
              4: 'soft_pvc',
              5: 'hard_tiles_large_space',
              6: 'carpet',
              7: 'hard_tiles',
              8: 'wood'}


# In[ ]:


Dataset[Dataset["series_id"]==0].head(1)


# In[ ]:


Labels.head(1)


# Group_id : maybe some reference to the team who performed tests... but doesn't appear in test data 

# In[ ]:


Labels["group_id"].unique()


# In[ ]:


Test.head(1)


# In[ ]:


# missing values and NaN checking


# In[ ]:


print(pd.isnull(Dataset).any())
print(pd.isna(Dataset).any())


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15, 5))
sns.countplot(Labels['surface'])
plt.title('Target distribution', size=15)
plt.show()

Labels['surface'].value_counts()


# In[ ]:


Labels.drop(['series_id', "group_id"], axis=1, inplace=True)


# In[ ]:


Dataset.drop(['row_id', "series_id", "measurement_number"], axis=1, inplace=True)
Dataset = Dataset.values.reshape((3810, 128, 10))


# In[ ]:


Test.drop(['row_id', "series_id", "measurement_number"], axis=1, inplace=True)
Test = Test.values.reshape((3816, 128, 10))


# In[ ]:


for j in range(0,10):
    plt.figure(figsize=(15, 5))
    plt.title("Target : " + Labels['surface'][j], size=15)
    for i in range(10):
        plt.plot(Dataset[j, :, i], label=i)
    plt.legend()
    plt.show()


# Everything about rotations ==>  between 0 to 1. 

# In[ ]:


y_train = Labels['surface'].map(encode_dic).astype(int)


# In[ ]:


from keras.models import Sequential,Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Embedding
from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, BatchNormalization, Activation, Dropout, Flatten,Add
from keras.layers import Conv1D, GlobalAveragePooling1D,Softmax
from keras.optimizers import SGD,Adam
batch_size = 500
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
def gen_model(kernel_size):
    K.clear_session()

    inp = Input(shape=(128, 10))
    C = Conv1D(filters=32, kernel_size=kernel_size, strides=1)(inp)

    C11 = Conv1D(filters=32, kernel_size=kernel_size, strides=1, padding='same')(C)
    A11 = Activation("relu")(C11)
    C12 = Conv1D(filters=32, kernel_size=kernel_size, strides=1, padding='same')(A11)
    M11 = MaxPooling1D(pool_size=kernel_size, strides=2)(C12)
    F1 = Flatten()(M11)
    
    D1 = Dense(32)(F1)
    A6 = Activation("relu")(D1)
    D2 = Dense(32)(A6)
    D3 = Dense(9)(D2)
    A7 = Softmax()(D3)

    model = Model(inputs=inp, outputs=A7)
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model
    


# In[ ]:


from keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger, LearningRateScheduler,ReduceLROnPlateau
import math
import random
def exp_decay(epoch):
    initial_lrate = 0.001
    k = 0.75
    t = 3810//(10000 * batch_size)  # every epoch we do n_obs/batch_size iteration
    lrate = initial_lrate * math.exp(-k*t)
    return lrate

lrate = LearningRateScheduler(exp_decay)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=2)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)


# In[ ]:


model_6=gen_model(kernel_size=6)
model_6.fit(Dataset,y_train,epochs=100,batch_size=batch_size, validation_split=0.2,verbose=0)
x_train_6=model_6.predict(Dataset)
y_test_6=model_6.predict(Test)
y_test_6 = np.argmax(y_test_6, axis=1)


# In[ ]:


model_7=gen_model(kernel_size=7)
model_7.fit(Dataset,y_train,epochs=100,batch_size=batch_size, validation_split=0.2,verbose=0)
x_train_7=model_7.predict(Dataset)
y_test_7=model_7.predict(Test)
y_test_7 = np.argmax(y_test_7, axis=1)


# In[ ]:


model_8=gen_model(kernel_size=8)
model_8.fit(Dataset,y_train,epochs=100,batch_size=batch_size, validation_split=0.2,verbose=0)
x_train_8=model_8.predict(Dataset)
y_test_8=model_8.predict(Test)
y_test_8 = np.argmax(y_test_8, axis=1)


# In[ ]:


x_train_8= np.argmax(x_train_8, axis=1)
x_train_7= np.argmax(x_train_7, axis=1)
x_train_6= np.argmax(x_train_6, axis=1)


# In[ ]:


x_train_enriched=np.vstack((x_train_6,x_train_7,x_train_8))
x_train_enriched=np.transpose(x_train_enriched)


# In[ ]:


import xgboost as xgb
Rg2_Classifier=xgb.XGBClassifier()


# In[ ]:


Rg2_Classifier.fit(x_train_enriched,y_train)
Rg2_Classifier.score(x_train_enriched,y_train)


# In[ ]:


y_test_enriched=np.vstack((y_test_6,y_test_7,y_test_8))
y_test_enriched=np.transpose(y_test_enriched)


# In[ ]:


y_test=Rg2_Classifier.predict(y_test_enriched)


# In[ ]:


sub = pd.read_csv("../input/sample_submission.csv")
sub['surface'] = y_test
sub['surface'] = sub['surface'].map(decode_dic)


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('submission_MNA_6.csv', index=False)


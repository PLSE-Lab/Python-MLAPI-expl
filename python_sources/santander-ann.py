#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


#Loading Training & Testing Dataset
import pandas as pd
train_df = pd.read_csv("../input/train.csv")
val_df = pd.read_csv("../input/test.csv")


# In[4]:


#Basic information
print(train_df.shape)
print(train_df.columns)
train_df.head(10)


# In[5]:


#Feature and target separation
train_X = train_df.drop(columns=['ID_code','target'])
train_y = train_df['target']

val_X = val_df.drop(columns=['ID_code'])


# In[6]:


#Check if target categories are balanced
train_y.value_counts()#Imbalanced


# In[7]:


#Data Cleaning
print(train_X.drop_duplicates().shape) #No duplicates
print(train_X.dropna().shape) #No missing values


# In[8]:


train_X.describe()


# In[9]:


#Checking for Outliers using Z-Score method
#According to Six Sigma, any data that is 6 Standard deviations away from mean is considered to an outlier
from scipy import stats
count=0
for var in train_X.columns:
    var_z = stats.zscore(train_X[var])
    if((len(var_z[var_z>3.0])>0) or(len(var_z[var_z<-3.0])>0)):
        #print("Feature with Outliers:",var)
        count+=1
print("Total Number of features that has outliers:",count)


# In[ ]:


train_df.corr()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
#plt.subplots(train_df.corr())
#plt.show()
#train_df.corr()
sns.heatmap(train_df.corr(), square=True)


# In[12]:


from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(train_X, train_y, test_size=0.2, random_state=42, stratify=train_y)


# In[13]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras import utils as np_utils
labelencoder_Y = LabelEncoder()
train_y = labelencoder_Y.fit_transform(y_train)
test_y = labelencoder_Y.transform(y_test)

# one hot encoding for target values
#train_y = np_utils.to_categorical(train_y, num_classes=2)
#test_y = np_utils.to_categorical(test_y, num_classes=2)


# In[11]:


from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import layers, models, optimizers
import keras.backend as K


# In[46]:


#def create_ann(input_size,  init, output_dim):
def create_ann():
    # input layer
    input_layer = layers.Input((200, ))
    # hidden layer
    hidden_layer_1 = layers.Dense(
        1000, init='normal', activation="relu")(input_layer)
    # drop out layer
    hidden_drop_1 = layers.Dropout(0.3)(hidden_layer_1)
    hidden_layer_2 = layers.Dense(
        1000, init='normal', activation="relu")(hidden_drop_1)
    # drop out layer
    hidden_drop_2 = layers.Dropout(0.3)(hidden_layer_2)
    # output layer
    output_layer = layers.Dense(
        1,
        init='normal',
        activation="sigmoid")(hidden_drop_2)
    classifier = models.Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return classifier


# In[29]:



mc = ModelCheckpoint(
        "model.h5",
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True)
classifier = create_ann(X_train.shape[1],'normal', 1)

classifier.summary()
classifier.fit(np.array(X_train), y_train,         
        validation_data=(
            np.array(X_test),
            y_test),
        batch_size=8,
        epochs=1000,
        callbacks=[            
            mc])
    
#val_y = classifier.predict(val_X)


# In[30]:


val_y = classifier.predict(val_X)

pred_y = classifier.predict(X_test)


# In[23]:


import pandas as pd
sub_df = pd.read_csv("../input/sample_submission.csv")
sub_df['target'] = val_y
sub_df['target']  = [1 if row>0.5 else 0 for row in sub_df['target']]


# In[24]:


sub_df.to_csv('sample_submission.csv', index=False)


# In[49]:


from keras.wrappers.scikit_learn import KerasClassifier
estimator = KerasClassifier(build_fn=create_ann, 
                                           validation_data=(
            np.array(X_test),
            y_test),
        batch_size=8,
        epochs=1000,callbacks=[            
            mc])


# In[50]:


estimator.fit(np.array(X_train), y_train)


# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(estimator, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = val_X.columns.tolist(), top=150)


# In[ ]:





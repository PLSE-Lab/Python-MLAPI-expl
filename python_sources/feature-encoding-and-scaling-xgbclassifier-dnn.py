#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


# In[ ]:


# Load the dataset
train_df = pd.read_csv('../input/cat-in-the-dat/train.csv')
test_df = pd.read_csv('../input/cat-in-the-dat/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.shape,test_df.shape


# In[ ]:


#check for missing values in train dataset
train_df.isnull().sum()


# In[ ]:


#check for missing values in test dataset
test_df.isnull().sum()


# In[ ]:


train_df.columns


# ### Nominal features

# In[ ]:


nom = ["nom_0","nom_1","nom_2","nom_3","nom_4"]
for i,col in enumerate(nom):
    plt.figure(i)
    sns.countplot(x=train_df[col],hue=train_df["target"])


# ### Ordinal features

# In[ ]:


ords = ["ord_0","ord_1"]
for i,col in enumerate(ords):
    plt.figure(i)
    sns.countplot(x=train_df[col],hue=train_df["target"])


# ### Let's Convert categorical strings into intergers 

# In[ ]:


for feature in train_df.columns: # Loop through all columns in the dataframe
    if train_df[feature].dtype == 'object': # Only apply for columns with categorical strings
        train_df[feature] = pd.Categorical(train_df[feature]).codes # Replace strings with an integer


# In[ ]:


for feature in test_df.columns: # Loop through all columns in the dataframe
    if test_df[feature].dtype == 'object': # Only apply for columns with categorical strings
        test_df[feature] = pd.Categorical(test_df[feature]).codes # Replace strings with an integer


# In[ ]:


train_df.head()


# In[ ]:


# drop id and target columns from train and test dataset
train_drops = ["id","target"]
test_drops = ["id"]
X = train_df.drop(train_drops,axis=1)
y = train_df["target"]


# In[ ]:


X.shape


# ### Scaling Data

# In[ ]:


std = MinMaxScaler()
std.fit(X)
X_train = std.transform(X)
X_test = std.transform(test_df.drop(test_drops,axis=1))


# ### Let's apply clssification algorithms with StratifiedKFold

# In[ ]:


# RandomForestClassifier
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X_train,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', class_weight={0:.5,1:.5}, max_depth = 5, min_samples_leaf=5)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1


# In[ ]:


# XGBClassifier
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X_train,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = xgboost.XGBClassifier()
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1


# ### Let's use DNN

# In[ ]:


# split data into train and validation set
trainX,validX,trainy,validy = train_test_split(X_train,y)


# In[ ]:


# Build DNN
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128,input_dim = 23, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(64,activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(32,activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.summary()


# In[ ]:


model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


# In[ ]:


model.fit(trainX,trainy,validation_data=(validX,validy),batch_size=128,epochs=30)


# In[ ]:


pred = model.predict_proba(X_test)


# In[ ]:


outs = [x[0] for x in pred]


# In[ ]:


sub = pd.DataFrame({"id":test_df["id"],"target":outs})


# In[ ]:


# sub.to_csv("sample_submission.csv",index=False)


# In[ ]:


# from IPython.display import FileLink
# FileLink('sample_submission.csv')


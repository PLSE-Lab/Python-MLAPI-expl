#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df= pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")


# In[ ]:


df


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.nunique()


# Our data is clean hence do not require any kind if replacement in its value

# # Let's encode our data with help of diffrent techniques

# Staring with Label Encoding of data

# In[ ]:


s= (df.dtypes == 'object')
obj_cols = list(s[s].index) 


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
label_df = df.copy()
for i in obj_cols:
    label_df[i] = encoder.fit_transform(df[i])


# In[ ]:


label_df.head()


# Applying OneHotEncoding to the data set

# In[ ]:


# let's import the data set again incase any issue arrises
df= pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse =False)


# In[ ]:


one_hot_cols = pd.DataFrame(oh_encoder.fit_transform(df[obj_cols]))
# adding the index
one_hot_cols.index= df.index


# In[ ]:


#removing categorical columns
num_df = df.drop(obj_cols,axis=1)


# In[ ]:


oh_df = pd.concat([num_df,one_hot_cols],axis =1)


# In[ ]:


oh_df


# In[ ]:


oh_df[0]


# Encoding data using Dummies

# In[ ]:


df= pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")


# In[ ]:


dummy_df = pd.get_dummies(df)


# In[ ]:


dummy_df


# # Our data is now encoded 
# # now we will be fitting our data in diffrent modeling techniques

# In[ ]:


from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# Starting with K-means

# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2 ,init= 'k-means++',random_state=1)


# In[ ]:


label_pred_k = kmeans.fit_transform(label_df)
plt.figure(figsize = (10,7))
plt.scatter(label_pred_k[:,0],label_pred_k[:,1],label ='cluster1')
plt.title('Cluster for label encoded values of mushroom')


# In[ ]:


oh_pred_k = kmeans.fit_transform(oh_df)


# In[ ]:


plt.figure(figsize =(10,7))
plt.scatter(oh_pred_k[:,0],oh_pred_k[:,1],label ='cluster')
plt.title("cluster for One HotEncoed values")


# In[ ]:


d_pred_k = kmeans.fit_transform(dummy_df)
plt.figure(figsize=(10,7))
plt.scatter(d_pred_k[:,0],d_pred_k[:,1], label = 'cluster')
plt.title("cluster for dummy encoded values")


# # Let's  split our data set

# For label encoded data

# In[ ]:


label_x=label_df.iloc[:,1:]
label_y=label_df.iloc[:,0]


# In[ ]:


L_X_train,L_X_valid,L_Y_train,L_Y_valid = train_test_split(label_x,label_y,train_size=.77,random_state=10)


# For OneHotEncoded Data

# In[ ]:


oh_x=oh_df.iloc[:,1:]
oh_y=oh_df.iloc[:,0]


# In[ ]:


OH_X_train,OH_X_valid,OH_Y_train,OH_Y_valid= train_test_split(oh_x,oh_y,train_size=.50,random_state=10)


# For data encoded by dummies

# In[ ]:


dummy_x =dummy_df.iloc[:,1:]
dummy_y=dummy_df.iloc[:,0]


# In[ ]:


D_X_train,D_X_valid,D_Y_train,D_Y_valid=train_test_split(dummy_x,dummy_y,train_size=.77,random_state=10)


# # Fitting the data in diffrent models

# 1. Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[ ]:


#label encoded data
lr.fit(L_X_train,L_Y_train)
lr_L_pred= lr.predict(L_X_valid)
lr_L_accuracy=lr.score(L_X_valid,L_Y_valid)
lr_L_accuracy


# In[ ]:


lr_L_pred


# In[ ]:


#OneHotEncoded data
lr.fit(OH_X_train,OH_Y_train)
lr_OH_pred= lr.predict(OH_X_valid)
lr_OH_accuracy=lr.score(OH_X_valid,OH_Y_valid)
lr_OH_accuracy


# In[ ]:


#dummy data
lr.fit(D_X_train,D_Y_train)
lr_D_pred= lr.predict(D_X_valid)
lr_D_accuracy=lr.score(D_X_valid,D_Y_valid)
lr_D_accuracy


# 2. Logistic  Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
LR =LogisticRegression()


# In[ ]:


# LABEL ENCODING
LR.fit(L_X_train,L_Y_train)# LABEL ENCODING
LR.fit(L_X_train,L_Y_train)
LR_L_pred= LR.predict(L_X_valid)
LR_L_accuracy=LR.score(L_X_valid,L_Y_valid)
LR_L_accuracy


# In[ ]:


#OneHotEncoded data
LR.fit(OH_X_train,OH_Y_train)
LR_OH_pred= LR.predict(OH_X_valid)
LR_OH_accuracy=LR.score(OH_X_valid,OH_Y_valid)
LR_OH_accuracy


# In[ ]:


#dummy data
LR.fit(D_X_train,D_Y_train)
LR_D_pred= LR.predict(D_X_valid)
LR_D_accuracy=LR.score(D_X_valid,D_Y_valid)
LR_D_accuracy


# 3. Decision Tree Regressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
DF=DecisionTreeRegressor(random_state=0)


# In[ ]:


# LABEL ENCODING
DF.fit(L_X_train,L_Y_train)
DF_L_pred= DF.predict(L_X_valid)
DF_L_accuracy=DF.score(L_X_valid,L_Y_valid)
DF_L_accuracy


# In[ ]:


#OneHotEncoded data
DF.fit(OH_X_train,OH_Y_train)
DF_OH_pred= DF.predict(OH_X_valid)
DF_OH_accuracy=DF.score(OH_X_valid,OH_Y_valid)
DF_OH_accuracy


# In[ ]:


#dummy data
DF.fit(D_X_train,D_Y_train)
DF_D_pred= DF.predict(D_X_valid)
DF_D_accuracy=DF.score(D_X_valid,D_Y_valid)
DF_D_accuracy


# 4. Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()


# In[ ]:


# LABEL ENCODING
RF.fit(L_X_train,L_Y_train)
RF_L_pred= RF.predict(L_X_valid)
RF_L_accuracy=RF.score(L_X_valid,L_Y_valid)
RF_L_accuracy


# In[ ]:


#OneHotEncoded data
RF.fit(OH_X_train,OH_Y_train)
RF_OH_pred= RF.predict(OH_X_valid)
RF_OH_accuracy=RF.score(OH_X_valid,OH_Y_valid)
RF_OH_accuracy


# In[ ]:


#dummy data
RF.fit(D_X_train,D_Y_train)
RF_D_pred= RF.predict(D_X_valid)
RF_D_accuracy=RF.score(D_X_valid,D_Y_valid)
RF_D_accuracy


# 4. XGBRegressor

# In[ ]:


from xgboost import XGBRegressor
XG = XGBRegressor(n_estimators=500)


# In[ ]:


# LABEL ENCODING
XG.fit(L_X_train,L_Y_train)
XG_L_pred= XG.predict(L_X_valid)
XG_L_accuracy=XG.score(L_X_valid,L_Y_valid)
XG_L_accuracy


# In[ ]:


#OneHotEncoded data
XG.fit(OH_X_train,OH_Y_train)
XG_OH_pred= XG.predict(OH_X_valid)
XG_OH_accuracy=XG.score(OH_X_valid,OH_Y_valid)
XG_OH_accuracy


# In[ ]:


#dummy data
XG.fit(D_X_train,D_Y_train)
XG_D_pred= XG.predict(D_X_valid)
XG_D_accuracy=XG.score(D_X_valid,D_Y_valid)
XG_D_accuracy


# In[ ]:


models=['Linear Model','Logistic Regression','Decision Tree Regressor','Random Forest Regressor','XGBregressor']
label_encoding=[lr_L_accuracy,LR_L_accuracy,DF_L_accuracy,RF_L_accuracy,XG_L_accuracy]
Onehot_encoding =[lr_OH_accuracy,LR_OH_accuracy,DF_OH_accuracy,RF_OH_accuracy,XG_OH_accuracy]
dummy_encoding = [lr_D_accuracy,LR_D_accuracy,DF_D_accuracy,RF_D_accuracy,XG_D_accuracy]


# In[ ]:


table = pd.DataFrame({'Model':models,'Label Encoding':label_encoding,'One Hot Encoding':Onehot_encoding,'Dummy Encoding':dummy_encoding})


# In[ ]:


table


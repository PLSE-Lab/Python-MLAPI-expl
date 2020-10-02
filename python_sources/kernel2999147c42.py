#!/usr/bin/env python
# coding: utf-8

# In[29]:


#DATA
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
#random forest for regressor
from sklearn.ensemble import RandomForestRegressor
#split features  and target
from sklearn.model_selection import train_test_split
#Pour regression au lieu de accuracy c'est mean_square_error
from sklearn.metrics import mean_squared_error
#Ne pas afficher le warning lors du fit par exemple
#Import for cross_validation
from sklearn.model_selection  import cross_val_score
#import random forest for regression
from sklearn.ensemble import RandomForestRegressor
#Validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
#Vizualisation
import seaborn as sns
import matplotlib.pyplot as plt
# better visuals
sns.set()  
get_ipython().run_line_magic('matplotlib', 'inline')
from math import *
#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# ## IMPORT TRAIN - TEST - SUBMISSION

# In[30]:


# Train File
X_train = pd.read_csv('../input/train.csv', index_col='id')
#Test File
X_test = pd.read_csv('../input/test.csv', index_col='id')
#Submission File
submission = pd.read_csv('../input/sample_submission.csv')


# # 1     ------------------   EDA  -----------------------

# In[31]:


X_train.head(2)


# In[32]:


X_train.describe()


# In[33]:


X_test.head(2)


# In[34]:


X_test.describe()


# ### ENCODING

# In[5]:


# TRAIN
X_train.loc[X_train['store_and_fwd_flag'] == 'N', 'store_and_fwd_flag'] = 0
X_train.loc[X_train['store_and_fwd_flag'] == 'Y', 'store_and_fwd_flag'] = 1
# TEST
X_test.loc[X_test['store_and_fwd_flag'] == 'N', 'store_and_fwd_flag'] = 0
X_test.loc[X_test['store_and_fwd_flag'] == 'Y', 'store_and_fwd_flag'] = 1


# ## Vizualisation

# In[26]:


##Add column distance
#R = 6367445 
#X_train["distance"] = R*acos(sin(X_train["pickup_latitude"])sin(X_train["dropoff_latitude"]) + cos(X_train["dropoff_latitude"])cos(X_train["pickup_longitude"] - X_train["dropoff_longitude"])))


# In[6]:


#plot the count of passenger_count
sns.countplot(x="passenger_count", data=X_train);


# In[ ]:


#plot the count of trip_duration
#sns.countplot(x="trip_duration", data=X_train);


# In[7]:


#X_train['trip_duration'].plot.kde()


# ## FILTER

# In[ ]:


X_train = X_train[(X_train['trip_duration']  < 8000) & (X_train['passenger_count'] >= 1 )]


# In[ ]:


X_train.describe()


# In[ ]:


X_test.describe()


# # 2  ------------------- FUNCTIONS---------------------------

# ## Function features and target

# In[ ]:


## Fonction features et target
def split_dataset(df, features, target='trip_duration'):
    X = df[features]
    y = df[target]
    return X, y


# ## Function split date

# In[ ]:


def date_split(df_train, df_test, date='pickup_datetime'):
    ##X_train
    cols=df_train[date]
    date_cols=pd.to_datetime(cols)
    df_train['year'] = date_cols.dt.year
    df_train['month'] = date_cols.dt.month
    df_train['day'] = date_cols.dt.day
    df_train['hour'] = date_cols.dt.hour
    df_train['minute'] = date_cols.dt.minute
    df_train['second'] = date_cols.dt.second
    #df_train = df_train.drop(['pickup_datetime'], axis=1)
    ##X_test
    cols2=df_test[date]
    date_cols2=pd.to_datetime(cols2)
    df_test['year'] = date_cols2.dt.year
    df_test['month'] = date_cols2.dt.month
    df_test['day'] = date_cols2.dt.day
    df_test['hour'] = date_cols2.dt.hour
    df_test['minute'] = date_cols2.dt.minute
    df_test['second'] = date_cols2.dt.second
    #df_test = df_test.drop(['pickup_datetime'], axis=1)
    return df_train, df_test


# In[ ]:


#X_train['pickup_datetime']
#date_cols=pd.to_datetime(cols)
#X_train['year'] = date_cols.dt.year
#X_train.head(1)


# # 3 ----------------------TREATMENT---------------------

# In[ ]:


#Split  date for X_train et X_test
X_train, X_test = date_split(X_train, X_test)


# ## Function same columns  X_train and X_test

# In[ ]:


# Get colums set test for set training
def Get_cols(df, features_test=X_test.columns):
    #get X_test columns in  X_train
    X_train_features = df[features_test]
    return  X_train_features


# In[ ]:


#Same columns for X_test and X_train
X_trainGet_cols = Get_cols(X_test)
#Get only columns numbers
numbers = X_trainGet_cols.select_dtypes(np.number)
#Definition features and target in the file train
X_train_features, y_train_target = split_dataset(X_train, features=numbers.columns)
#X_test numbers
X_test = X_test.select_dtypes(np.number)


# # 4----------------------Validation------------------------

# In[ ]:


#rf = RandomForestRegressor()
#kf = KFold(n_splits=5, random_state=1)                                                                                      
#loses = cross_val_score(rf, X_train_features, y_train_target, cv=kf, scoring='neg_mean_squared_log_error')
# np.sqrt(-loses.mean())
#loses = [np.sqrt(-l) for l in loses]
# np.mean(loses)
#loses[:5]


# In[ ]:


#Initialise Random Forest regressor
rf = RandomForestRegressor()
cv = ShuffleSplit(n_splits=3, test_size=0.05, train_size=0.1, random_state=42)
#Cross Validation
lossess = -cross_val_score(rf, X_train_features, y_train_target, cv=cv, scoring='neg_mean_squared_log_error')


# # 5 -------------------FIT AND PREDICT----------------------

# ## FIT

# In[ ]:


## FIT all the train set
rf = RandomForestRegressor()
rf.fit(X_train_features, y_train_target)


# ## PREDICTION

# In[ ]:


##### Predict in the training
y_train_pred = rf.predict(X_train_features)

###### Predict in the test
y_test_pred = rf.predict(X_test)

y_test_pred.mean()


# # 6 ------------------SUBMISSION-------------------------
# 

# In[35]:


#!rm submission.csv


# In[ ]:


submission["trip_duration"] = y_test_pred
#Convertir notre fichier en csv
submission.to_csv('submission.csv', index=False)


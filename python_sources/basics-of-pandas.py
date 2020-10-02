#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost
import numpy as np
import pandas as pd
from math import sqrt

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


# In[2]:


train = pd.read_csv("../input/train_4aqQp50.csv")
test = pd.read_csv("../input/test_VJP2kVH.csv")


# In[3]:


train.head()


# In[4]:


train.info()


# In[5]:


train


# In[6]:


train.describe()


# In[7]:


train.columns


# In[ ]:


train.values


# In[8]:


train['INT_SQFT'].plot()


# In[9]:


train['INT_SQFT']


# In[10]:


train.AREA.value_counts()


# In[11]:


#replacing errors to the corresponding values
train.AREA.replace(['Chrompt', 'Chrmpet','Chormpet','TNagar', 'Ana Nagar','Ann Nagar', 'Karapakam', 'Velchery', 'KKNagar', 'Adyr'], ['Chrompet', 'Chrompet', 'Chrompet','T Nagar', 'Anna Nagar', 'Anna Nagar', 'Karapakkam', 'Velachery', 'KK Nagar', 'Adyar'], inplace=True)


# In[12]:


train.AREA.value_counts()


# In[13]:


test.AREA.replace(['Velchery','Karapakam','Chrmpet','Ann Nagar', 'Chormpet', 'Chrompt'],['Velachery','Karapakkam','Chrompet','Anna Nagar','Chrompet','Chrompet'], inplace=True)

train.SALE_COND.replace(['Ab Normal', 'Adj Land','PartiaLl','Partiall'], ['AbNormal', 'AdjLand', 'Partial','Partial'], inplace=True)

test.SALE_COND.replace(['Adj Land','PartiaLl','Partiall'], ['AdjLand', 'Partial','Partial'], inplace=True)

train.PARK_FACIL.replace(['Noo'], ['No'], inplace=True)

test.PARK_FACIL.replace(['Noo'], ['No'], inplace=True)

train.BUILDTYPE.replace(['Other','Comercial'], ['Others','Commercial'], inplace=True)

test.BUILDTYPE.replace(['Other','Comercial', 'Commercil'], ['Others','Commercial', 'Commercial'], inplace=True)

train.UTILITY_AVAIL.replace(['All Pub'], ['AllPub'], inplace=True)

test.UTILITY_AVAIL.replace(['All Pub'], ['AllPub'], inplace=True)

train.STREET.replace(['Pavd', 'NoAccess'], ['Paved', 'No Access'], inplace=True)

test.STREET.replace(['Pavd', 'NoAccess'], ['Paved', 'No Access'], inplace=True)


# In[14]:


train.AREA.value_counts()


# In[15]:


train.info()


# In[16]:


#Dropping table
train = train.drop(['PRT_ID', 'DATE_SALE', 'DATE_BUILD'], axis=1)
# we assign
#verticle axis = 1
#horizontal axis = 0


# In[17]:


train  = train.fillna(0)
#train = train.dropna(0)


# In[18]:


train.info()


# In[19]:


train.describe()


# In[20]:


scale_list = ['INT_SQFT','DIST_MAINROAD','REG_FEE','COMMIS','QS_ROOMS','QS_BATHROOM','QS_BEDROOM','QS_OVERALL']
sc = train[scale_list]


# In[21]:


sc.head()


# In[22]:


scaler = StandardScaler()
sc = scaler.fit_transform(sc)


# In[23]:


train[scale_list] = sc


# In[24]:


train[scale_list].head()


#  fit matlab direct algorithm per run karna 
#  
#  transform matlab run karke changes lana
#  
#  har ek scaling corresponding to its own column

# In[25]:


sc[0]


# In[26]:


sc.head()


# this error is because the data type is no further a data frame

# 

# In[27]:


encoding_list = ['AREA', 'SALE_COND','PARK_FACIL','BUILDTYPE','UTILITY_AVAIL', 'STREET','MZZONE']
train[encoding_list] = train[encoding_list].apply(LabelEncoder().fit_transform)


# In[28]:


train.head()


# #Linear Regression

# In[29]:


y = train['SALES_PRICE']
x = train.drop('SALES_PRICE', axis=1)


# In[30]:


y.head()


# In[31]:


x.head()


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.3)


# In[33]:



X_train.shape


# In[34]:


X_test.shape


# In[35]:


y_test.shape


# In[36]:


logreg=LinearRegression()


# In[37]:


#training
logreg.fit(X_train,y_train)


# In[38]:


LinearRegression().fit(X_train,y_train)


# In[39]:


y_pred=logreg.predict(X_test)


# In[40]:


y_test


# In[41]:


y_pred[0:6]


# In[42]:


print(metrics.mean_squared_error(y_test, y_pred))


# In[43]:


xgb = xgboost.XGBRegressor(n_estimators=25000, learning_rate=0.06, gamma=0, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=4, max_depth=3)


# In[ ]:


xgb.fit(X_train,y_train)


# In[47]:


predictions = xgb.predict(X_test)


# In[48]:


print(sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:





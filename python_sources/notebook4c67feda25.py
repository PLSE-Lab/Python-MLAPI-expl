#!/usr/bin/env python
# coding: utf-8

# Data Exploration

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

df_train.head()


# In[ ]:


df_train.shape


# In[ ]:


def feature_summary(data):
    n_row=data.shape[0]
    features=pd.DataFrame()
    features_names=[]
    features_type = []
    features_counts=[]
    features_missing=[]
    names=data.columns.values
    for i in names:
        features_names.append(i)
        features_type.append(type(data.ix[1,i]))
        features_counts.append(data[i].value_counts().count())
        features_missing.append(data[data[i].isnull()].shape[0])
    features['name']=features_names
    features['type'] = features_type
    features['value counts']=features_counts
    features['missing']=features_missing
    features['percentage_missing']=features['missing']/n_row
    return (features)


# In[ ]:


describe = feature_summary(df_train)


# In[ ]:


describe


# In[ ]:


# filter columns that have percentage missing <0.40 
describe['type'].value_counts()


# In[ ]:


df_train_quant = df_train.select_dtypes(include=['int64', 'floating', 'datetime64'])


# In[ ]:


df_train_quant.shape


# In[ ]:


#now that we have all the quantitative variables 


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(df_train_quant.corr())


# In[ ]:


import numpy as np
from sklearn.decomposition import PCA


# In[ ]:


#PCA = PCA(n_components = 3)


# In[ ]:


#handling missing values using mean values? we can explore more methods for this at a later
#stage
#df_train_quant = df_train_quant.drop('timestamp', 1)
#for i in range(1,len(df_train_quant.columns)):
#       df_train_quant.iloc[:,i] = df_train_quant.iloc[:,i].fillna(df_train_quant.iloc[:,i].mean())


# In[ ]:


#y = df_train_quant['price_doc']
#df_train = df_train_quant.drop('price_doc', 1)


# In[ ]:


#df_train_quant = df_train_quant.drop('timestamp', 1)
#PCA.fit(df_train_quant)


# In[ ]:


#print(PCA.explained_variance_ratio_)


# In[ ]:


#comp1 = PCA.components[:0]


# In[ ]:


#df_train_quant.shape


# In[ ]:


#df_quant = PCA.fit_transform(df_train_quant)


# In[ ]:


#df_quant.shape


# In[ ]:


#we will use the above training data for predictive modeling.


# In[ ]:


#y.shape


# In[ ]:


# simple linear regression
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm 
#X_train,X_test,y_train,y_test = train_test_split(df_quant,y,test_size=0.2)# simple linear regression
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm 
#X_train,X_test,y_train,y_test = train_test_split(df_quant,y,test_size=0.2)


# In[ ]:


#X_train = pd.DataFrame(X_train)
#X_test = pd.DataFrame(X_test)


# In[ ]:


#model = lm.LinearRegression()
#model.fit(X_train, y_train)


# In[ ]:


#predy = model.predict(X_test)


# In[ ]:


#from sklearn.metrics import r2_score
#r2_score(y_test,predy)


# In[ ]:


#y_test


# In[ ]:


#df_samplesub = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


#df_samplesub.shape


# In[ ]:


# Transforming the test data
#df_test_quant = df_test.select_dtypes(include=['int64', 'floating', 'datetime64'])
#df_test_quant = df_test_quant.drop('timestamp', 1)
#for i in range(1,len(df_test_quant.columns)):
#       df_test_quant.iloc[:,i] = df_test_quant.iloc[:,i].fillna(df_test_quant.iloc[:,i].mean())
#y = df_test_quant['price_doc']
#df_train = df_train_quant.drop('price_doc', 1) 
#test = PCA.fit_transform(df_test_quant)


# In[ ]:


#test.shape


# In[ ]:


#y_values = model.predict(test)


# In[ ]:


#y_values


# In[ ]:


##result = pd.DataFrame(y_values)


# In[ ]:


#result.head()


# In[ ]:


#result['id'] = df_test['id']


# In[ ]:


#result['price_doc'] = result.iloc[:,0]


# In[ ]:


#result.head()


# In[ ]:


#result = result.drop(0,1)


# In[ ]:


#result['price_doc'] = result['price_doc'].round(2)


# In[ ]:


##result.head()


# In[ ]:


#result.to_csv("output_4.csv", index = False)


# In[ ]:


#result.head()


# In[ ]:


# XG Boost because everyone else is doing it
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])


# In[ ]:


def feature_summary(data):
    n_row=data.shape[0]
    features=pd.DataFrame()
    features_names=[]
    features_type = []
    features_counts=[]
    features_missing=[]
    names=data.columns.values
    for i in names:
        features_names.append(i)
        features_type.append(type(data.ix[1,i]))
        features_counts.append(data[i].value_counts().count())
        features_missing.append(data[data[i].isnull()].shape[0])
    features['name']=features_names
    features['type'] = features_type
    features['value counts']=features_counts
    features['missing']=features_missing
    features['percentage_missing']=features['missing']/n_row
    return (features)


# In[ ]:


df_train_quant.shape


# In[ ]:


#describe = feature_summary(df_train)


# In[ ]:


df_train_quant = df_train.select_dtypes(include=['int64', 'floating', 'datetime64'])


# In[ ]:


import numpy as np
from sklearn.decomposition import PCA
PCA = PCA(n_components = 275)
#handling missing values using mean values? we can explore more methods for this at a later
#stage
df_train_quant = df_train_quant.drop('timestamp', 1)
for i in range(1,len(df_train_quant.columns)):
       df_train_quant.iloc[:,i] = df_train_quant.iloc[:,i].fillna(df_train_quant.iloc[:,i].mean())
y = df_train_quant['price_doc']
df_train = df_train_quant.drop('price_doc', 1)
df_quant = PCA.fit_transform(df_train_quant)


# In[ ]:


# splitting into training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_quant,y,test_size=0.2)


# In[ ]:


import xgboost as xgb
xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 0
}

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test)


# In[ ]:


model = xgb.train(xgb_params, dtrain, num_boost_round=300)


# In[ ]:


predy = model.predict(dtest)


# In[ ]:


#finding r2 value
from sklearn.metrics import r2_score
r2_score(y_test,predy)


# In[ ]:


# Transforming the test data
df_test_quant = df_test.select_dtypes(include=['int64', 'floating', 'datetime64'])
df_test_quant = df_test_quant.drop('timestamp', 1)
for i in range(1,len(df_test_quant.columns)):
       df_test_quant.iloc[:,i] = df_test_quant.iloc[:,i].fillna(df_test_quant.iloc[:,i].mean())
#y = df_test_quant['price_doc']
#df_train = df_train_quant.drop('price_doc', 1) 
test = PCA.fit_transform(df_test_quant)


# In[ ]:


dtest = xgb.DMatrix(test)


# In[ ]:


result = model.predict(dtest)


# In[ ]:


result = pd.DataFrame(result)
result['id'] = df_test['id']
result['price_doc'] = result.iloc[:,0]
result = result.drop(0,1)
result['price_doc'] = result['price_doc'].round(2)
#result.to_csv("output_5.csv", index = False)


# In[ ]:


result.to_csv("output_6.csv", index = False)


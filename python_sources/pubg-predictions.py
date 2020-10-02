#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 500)


# In[ ]:


df_train = pd.read_csv('../input/train_V2.csv')


# In[ ]:


df_test = pd.read_csv('../input/test_V2.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.columns[df_train.isna().any()].tolist()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


enc = LabelEncoder()
enc2 = LabelEncoder()


# In[ ]:


matchType = df_train['matchType']
matchTypeTest = df_train['matchType']


# In[ ]:


enc.fit(matchType)
enc2.fit(matchTypeTest)


# In[ ]:


enc.classes_


# In[ ]:


matchTypes  = enc.transform(matchType)
matchTypes2 = enc.transform(matchTypeTest)


# In[ ]:


dataframesMatchType = pd.DataFrame(matchTypes)
dataframesMatchType2 = pd.DataFrame(matchTypes2)


# In[ ]:


df_train.drop('matchType', axis=1, inplace=True)
df_test.drop('matchType', axis=1, inplace=True)
df_train.head()


# In[ ]:


df_train['matchType'] = dataframesMatchType[0]
df_test['matchType'] = dataframesMatchType2[0]


# In[ ]:


df_train.head()


# In[ ]:


df_train.dropna(inplace=True)


# In[ ]:


df_train.columns[df_train.isna().any()].tolist()


# In[ ]:


plt.figure(figsize = (25,25))
sns.heatmap(df_train.corr(), annot=True)


# In[ ]:


df_train.describe()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df_train.drop(['winPlacePerc', 'Id', 'groupId', 'matchId'], axis=1)
y = df_train['winPlacePerc']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[ ]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.30)


# In[ ]:


from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.neural_network import MLPRegressor


# In[ ]:


bayes = MLPRegressor()


# In[ ]:


bayes.fit(X_train2, y_train2)


# In[ ]:


predictions = bayes.predict(X_test2)


# In[ ]:


from sklearn.metrics import mean_absolute_error, classification_report, mean_squared_error


# In[ ]:


predictions1 = bayes.predict(X_test)


# In[ ]:


print(mean_absolute_error(y_test, predictions1))


# In[ ]:


print(mean_absolute_error(y_test2, predictions))


# In[ ]:


myPredictions = df_test.drop(['Id', 'groupId', 'matchId'], axis=1)


# In[ ]:


predictions2 = bayes.predict(myPredictions)


# In[ ]:


predictions2


# In[ ]:


df_test['winPlacePerc'] = predictions2
dfFinal = df_test[['Id', 'winPlacePerc']]
dfFinal.head()


# In[ ]:


dfFinal.to_csv('pubgKaggle.csv', index=False)


# In[ ]:





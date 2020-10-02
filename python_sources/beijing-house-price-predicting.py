#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

df = pd.read_csv('../input/new.csv', encoding ='iso-8859-1')


# In[ ]:


df.info()


# In[ ]:


df = df.drop(['url', 'id', 'price', 'Cid', 'DOM'], axis = 1)


# In[ ]:


df.head(5)


# In[ ]:


def str2int(s):
    try:
        return int(s)
    except:
        return np.nan
df[['livingRoom', 'drawingRoom', 'bathRoom']] = df[['livingRoom', 'drawingRoom', 'bathRoom']].applymap(str2int)


# In[ ]:


df.isnull().sum().sort_values(ascending = False).head(10)


# In[ ]:


df['buildingType'].value_counts()


# In[ ]:


df['buildingType'] = df['buildingType'].map(lambda x: x if x >= 1 else np.nan)


# In[ ]:


df['buildingType'].value_counts()


# In[ ]:


df['floor'].head()


# In[ ]:


def floorType(s):
    return s.split(' ')[0]
def floorHeight(s):
    try:
        return int(s.split(' ')[1])
    except:
        return np.nan
df['floorType'] = df['floor'].map(floorType)
df['floorHeight'] = df['floor'].map(floorHeight)


# In[ ]:


df.isnull().sum().sort_values(ascending = False).head(10)


# In[ ]:


df['floorType'].value_counts()


# In[ ]:


df['constructionTime'].value_counts()


# In[ ]:


def changeconstructionTime(s):
    if len(s) < 4:
        return np.nan
    try:
        return int(s)
    except:
        return np.nan
df['constructionTime'] = df['constructionTime'].map(changeconstructionTime)


# In[ ]:


def usedTime(buy, build):
    buy = int(buy.split('-')[0])
    try:
        return buy - build
    except:
        np.nan
df['UsedTime'] = df.apply(lambda x: usedTime(x['tradeTime'], x['constructionTime']), axis = 1)


# In[ ]:


df = df.drop('constructionTime', axis = 1)


# In[ ]:


mean_communityAverage = df['communityAverage'].mean()
df['communityAverage'] = df['communityAverage'].fillna(mean_communityAverage)


# In[ ]:


mode_col = ['buildingType', 'elevator', 'livingRoom', 'drawingRoom', 'floorHeight', 
             'fiveYearsProperty', 'subway','bathRoom', 'UsedTime']
df_mode = df[mode_col].median()
df_mode


# In[ ]:


df[mode_col] = df[mode_col].fillna(df_mode)


# In[ ]:


df.isnull().sum().sum()


# In[ ]:


y_train = df.pop('totalPrice')


# In[ ]:


str_col = ['buildingType','buildingStructure', 'renovationCondition', 'district']
df[str_col] = df[str_col].astype(str)


# In[ ]:


df['tradeTime'] = df['tradeTime'].map(lambda x: x.split('-')[0])


# In[ ]:


df = df.drop('floor',axis = 1)
df.head()


# In[ ]:


df_dummy = pd.get_dummies(df)


# In[ ]:


df_dummy.head()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[ ]:


X = df_dummy.values
y = y_train.values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 5)


# In[ ]:


RFR = RandomForestRegressor(n_estimators=200, max_features=0.3)
RFR.fit(X_train, y_train)


# In[ ]:


y_predict = RFR.predict(X_test)


# In[ ]:


np.sqrt(mean_squared_error(y_test, y_predict))


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(y_test[:100], color = 'blue')
plt.plot(y_predict[:100], color = 'red')


# In[ ]:


from sklearn.externals import joblib
joblib.dump(RFR,'BeijingHousingPricePredicter.pkl')


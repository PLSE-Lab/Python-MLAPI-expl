#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


df = pd.read_csv('../input/rossmann-store-sales/train.csv', parse_dates = ['Date'], low_memory = False)
df.head()


# In[ ]:


df['Date']=pd.to_datetime(df['Date'],format='%Y-%m-%d')


# In[ ]:


df['Hour'] = df['Date'].dt.hour
df['Day_of_Month'] = df['Date'].dt.day
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month


# In[ ]:


print(df['Date'].min())
print(df['Date'].max())


# In[ ]:


test = pd.read_csv('../input/rossmann-store-sales/test.csv', parse_dates = True, low_memory = False)
test.head()


# In[ ]:


test['Date']=pd.to_datetime(test['Date'],format='%Y-%m-%d')


# In[ ]:


test['Hour'] = test['Date'].dt.hour
test['Day_of_Month'] = test['Date'].dt.day
test['Day_of_Week'] = test['Date'].dt.dayofweek
test['Month'] = test['Date'].dt.month


# In[ ]:


print(test['Date'].min())
print(test['Date'].max())


# In[ ]:


sns.pointplot(x='Month', y='Sales', data=df)


# In[ ]:


sns.pointplot(x='Day_of_Week', y='Sales', data=df)


# In[ ]:


sns.countplot(x = 'Day_of_Week', hue = 'Open', data = df)
plt.title('Store Daily Open Countplot')


# In[ ]:


sns.pointplot(x='Day_of_Month', y='Sales', data=df)


# In[ ]:


df['SalesPerCustomer'] = df['Sales']/df['Customers']
df['SalesPerCustomer'].describe()


# In[ ]:


df.Open.value_counts()


# In[ ]:


np.sum([df['Sales'] == 0])


# In[ ]:


#drop closed stores and stores with zero sales
df = df[(df["Open"] != 0) & (df['Sales'] != 0)]


# In[ ]:


store = pd.read_csv('../input/rossmann-store-sales/store.csv')
store.head(30)


# In[ ]:


store.isnull().sum()


# In[ ]:


store['CompetitionDistance'] = store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0]) #try 0
store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0]) #try 0
store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0) #try 0
store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0]) #try 0
store['PromoInterval'] = store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0]) #try 0
store.head()


# In[ ]:


df_store = pd.merge(df, store, how = 'left', on = 'Store')
df_store.head()


# In[ ]:


df_store.groupby('StoreType')['Sales'].describe()


# In[ ]:


df_store.groupby('StoreType')['Customers', 'Sales'].sum()


# In[ ]:


#sales trends
sns.catplot(data = df_store, x = 'Month', y = "Sales", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
               color = 'c') 


# In[ ]:


#customer trends
sns.catplot(data = df_store, x = 'Month', y = "Customers", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
               color = 'c')


# In[ ]:


#sales per customer
sns.catplot(data = df_store, x = 'Month', y = "SalesPerCustomer", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
               color = 'c')


# In[ ]:


sns.catplot(data = df_store, x = 'Month', y = "Sales", 
               col = 'DayOfWeek', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'StoreType', # per store type in rows
               color = 'c') 


# In[ ]:


#stores open on sunday
df_store[(df_store.Open == 1) & (df_store.DayOfWeek == 7)]['Store'].unique()


# In[ ]:


sns.catplot(data = df_store, x = 'DayOfWeek', y = "Sales", 
               col = 'Promo', 
               row = 'Promo2',
               hue = 'Promo2',
               palette = 'RdPu') 


# In[ ]:


df_store['StateHoliday'] = df_store['StateHoliday'].map({'0':0 , 0:0 , 'a':1 , 'b':2 , 'c':3})
df_store['StateHoliday'] = df_store['StateHoliday'].astype(int)


# In[ ]:


df_store['StoreType'] = df_store['StoreType'].map({'a':1 , 'b':2 , 'c':3 , 'd':4})
df_store['StoreType'] = df_store['StoreType'].astype(int)


# In[ ]:


df_store.isnull().sum()


# In[ ]:


df_store['Assortment'] = df_store['Assortment'].map({'a':1 , 'b':2 , 'c':3})
df_store['Assortment'] = df_store['Assortment'].astype(int)


# In[ ]:


df_store['PromoInterval'] = df_store['PromoInterval'].map({'Jan,Apr,Jul,Oct':1 , 'Feb,May,Aug,Nov':2 , 'Mar,Jun,Sept,Dec':3})
df_store['PromoInterval'] = df_store['PromoInterval'].astype(int)


# In[ ]:


df_store.to_csv('df_merged.csv', index=False)


# In[ ]:


df_store.isnull().sum()


# In[ ]:


len(df_store)


# In[ ]:


test = pd.merge(test, store, how = 'left', on = 'Store')
test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


test.fillna(method='ffill', inplace=True)


# In[ ]:


test['StateHoliday'] = test['StateHoliday'].map({'0':0 , 0:0 , 'a':1 , 'b':2 , 'c':3})
test['StateHoliday'] = test['StateHoliday'].astype(int)
test['StoreType'] = test['StoreType'].map({'a':1 , 'b':2 , 'c':3 , 'd':4})
test['StoreType'] = test['StoreType'].astype(int)
test['Assortment'] = test['Assortment'].map({'a':1 , 'b':2 , 'c':3})
test['Assortment'] = test['Assortment'].astype(int)
test['PromoInterval'] = test['PromoInterval'].map({'Jan,Apr,Jul,Oct':1 , 'Feb,May,Aug,Nov':2 , 'Mar,Jun,Sept,Dec':3})
test['PromoInterval'] = test['PromoInterval'].astype(int)


# In[ ]:


test.to_csv('test_merged.csv', index=False)


# In[ ]:


test = test.drop(['Id','Date'],axis=1)


# In[ ]:


test.head()


# Machine Learning

# In[ ]:


X = df_store.drop(['Date','Sales','Customers', 'SalesPerCustomer'],1)
#Transform Target Variable
y = np.log1p(df_store['Sales'])

from sklearn.model_selection import train_test_split
X_train , X_val , y_train , y_val = train_test_split(X, y , test_size=0.30 , random_state = 1 )


# In[ ]:


X_train.shape, X_val.shape, y_train.shape, y_val.shape


# Machine Learning

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=10, n_estimators=200, random_state=42)
gbrt.fit(X_train, y_train)
print(gbrt.score(X_train, y_train))


# In[ ]:


y_pred = gbrt.predict(X_val)


# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error
print(r2_score(y_val , y_pred))
print(np.sqrt(mean_squared_error(y_val , y_pred)))


# In[ ]:


df1 = pd.DataFrame({'Actual': y_val, 'Predicted': y_pred})
df1.head(25)


# Make Prediction CSV File

# In[ ]:


test_pred=gbrt.predict(test[X.columns])
test_pred_inv=np.exp(test_pred)-1


# In[ ]:


test_pred_inv


# In[ ]:


#make submission df
prediction = pd.DataFrame(test_pred_inv)
submission = pd.read_csv('../input/rossmann-store-sales/sample_submission.csv')
prediction_df = pd.concat([submission['Id'], prediction], axis=1)
prediction_df.columns=['Id','Sales']
prediction_df.to_csv('Sample_Submission.csv', index=False)


# In[ ]:


prediction_df.head()


# In[ ]:





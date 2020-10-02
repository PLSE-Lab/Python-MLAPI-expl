#!/usr/bin/env python
# coding: utf-8

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


# Load Data in a Pandas Dataframe
import seaborn as sns #just to make our visualization prettier ;-) 
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('../input/NYC-BikeShare-2015-2017-combined.csv')

# checks for first 5 values of dataframe - df
df.drop(['Unnamed: 0'],axis=1, inplace=True)
df = df[df['Gender']!=0].copy()
df.head()


# In[ ]:


# checks for length of dataframe - df
print(len(df))
ndf = df[['Trip_Duration_in_min', 'Start Time', 'Stop Time',
       'Start Station Name', 'End Station Name',
      'Bike ID', 'User Type','Birth Year', 'Gender']]
ndf.loc[:,('Birth Year')] = ndf['Birth Year'].astype(int)
ndf.head()


# ## EDA

# In[ ]:


ndf.describe()


# In[ ]:


ndf.info()


# In[ ]:


fig = plt.figure(figsize=(3,3), dpi=200)
ax = plt.subplot(111)
df['Gender'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=270, fontsize=10)


# ### 1 = Male (71.5%)
# ### 2 = Female (20.5%)
# ### 3 = Unknown (8.1 %)

# In[ ]:


ndf['Start Date'] = pd.to_datetime(ndf['Start Time'])
ndf['Stop Date'] = pd.to_datetime(ndf['Stop Time'])


# In[ ]:


ndf['Start Month'] = ndf['Start Date'].dt.month
ndf['Start Day'] = ndf['Start Date'].dt.day
ndf['Start Minute'] = ndf['Start Date'].dt.minute
ndf['Start Week'] = ndf['Start Date'].dt.minute
ndf['Start Weekday'] = ndf['Start Date'].dt.minute
ndf['Start Week'] = ndf['Start Date'].dt.week
ndf['Start Weekofyear'] = ndf['Start Date'].dt.weekofyear
ndf['Start weekday'] = ndf['Start Date'].dt.weekday
ndf['Start dayofyear'] = ndf['Start Date'].dt.dayofyear
ndf['Start weekday_name'] = ndf['Start Date'].dt.weekday_name
ndf['Start quarter'] = ndf['Start Date'].dt.quarter
ndf['Stop Month'] = ndf['Stop Date'].dt.month
ndf['Stop Day'] = ndf['Start Date'].dt.day
ndf['Stop Minute'] = ndf['Stop Date'].dt.minute
ndf['Stop Week'] = ndf['Stop Date'].dt.minute
ndf['Stop Weekday'] = ndf['Stop Date'].dt.minute
ndf['Stop Week'] = ndf['Stop Date'].dt.week
ndf['Stop Weekofyear'] = ndf['Stop Date'].dt.weekofyear
ndf['Stop weekday'] = ndf['Stop Date'].dt.weekday
ndf['Stop dayofyear'] = ndf['Stop Date'].dt.dayofyear
ndf['Stop weekday_name'] = ndf['Stop Date'].dt.weekday_name
ndf['Stop quarter'] = ndf['Stop Date'].dt.quarter
ndf.head()


# In[ ]:


ndf.info()


# In[ ]:


cols = ndf.columns
data = ndf[['Trip_Duration_in_min', 'Start Station Name',
       'End Station Name', 'User Type', 'Birth Year', 
      'Start Month', 'Start Day', 'Start Minute',
       'Start Week', 'Start Weekofyear', 
       'Start dayofyear', 'Start quarter', 'Stop Month',
       'Stop Day', 'Stop Minute',
       'Stop Weekofyear', 'Stop weekday', 'Stop dayofyear',
        'Stop quarter', 'Gender']]
data.head()


# In[ ]:


data.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data = data.apply(le.fit_transform)
data.head()


# In[ ]:


X = data.iloc[:-10000,:-1]
test_X = data.iloc[-10000:,:-1]
print(len(X), len(test_X))

y = data.iloc[:-10000,-1:]
test_y = data.iloc[-10000:,-1:]
print(len(y),len(test_y))


# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

clf = DecisionTreeClassifier()
kf = KFold(n_splits=3,random_state=2)
model = DecisionTreeClassifier()


# In[ ]:


results = cross_val_score(model, X, y, cv=kf)
print(results.mean()*100)


# In[ ]:


clf.fit(X_train, y_train)
prediction = clf.predict(test_X)
# pred_t = le.transform(prediction)
# acc = (pred_t==test_y['Gender']).value_counts()
# acc


# In[ ]:


print('Accuracy on Unseen data %f' %(acc[1]/10000*100)+' %')


# In[ ]:


y_true = np.ravel(test_y).copy()
y_pred = clf.predict(test_X)


# In[ ]:


from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_true, y_pred)


# In[ ]:


print('AUC Score = %.2f' %(auc*100)+'')


# In[ ]:





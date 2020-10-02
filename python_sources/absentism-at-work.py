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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# This is my first own kernel, I did this a long time ago. So if you have any suggestions and recommedations, leave them in comment section. Thank you!
# Link to dataset, if anyone intersted
# 
# Abstract:
# The database was created with records of absenteeism at work from July 2007 to July 2010 at a courier company in Brazil.
# 
# Citation Request/Acknowledgements:
# Martiniano, A., Ferreira, R. P., Sassi, R. J., & Affonso, C. (2012). Application of a neuro fuzzy network in prediction of absenteeism at work. In Information Systems and Technologies (CISTI), 7th Iberian Conference on (pp. 1-4). IEEE.
# 
# Acknowledgements: Professor Gary Johns for contributing to the selection of relevant research attributes. Professor Emeritus of Management Honorary Concordia University Research Chair in Management John Molson School of Business Concordia University Montreal, Quebec, Canada Adjunct Professor, OB/HR Division Sauder School of Business, University of British Columbia Vancouver, British Columbia, Canada
# 
# In this dataset task is to predict hours of absenteeism at work, using features, provided in database.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[ ]:


df = pd.read_csv('../input/Absenteeism_at_work.csv', delimiter=";")


# In[ ]:


df.head()


# In[ ]:


print('Shape of dataset is:{}'.format(df.shape))
print('Type of features is:{}'.format(df.dtypes))


# In[ ]:


df['Absenteeism time in hours'].mean()


# In[ ]:


sns.jointplot(x='Absenteeism time in hours',y='Seasons',data=df)


# In[ ]:


sns.jointplot(x='Age',y='Absenteeism time in hours',data=df)


# In[ ]:


plt.figure(figsize=(12,6))
sns.lmplot(x='Age',y='Absenteeism time in hours',data=df,hue='Day of the week',size=5,aspect=2)


# In[ ]:


df[df['Day of the week']==6]['Absenteeism time in hours'].mean()


# In[ ]:


df['Transportation expense'].mean()


# In[ ]:


sns.jointplot(x='Transportation expense',y='Month of absence',data=df,kind='hex',color='red')


# In[ ]:


df['Son'].value_counts()


# In[ ]:


plt.figure(figsize=(10,5))
df[df['Son']!=0]['Absenteeism time in hours'].plot.hist(bins=30)


# In[ ]:


plt.figure(figsize=(10,6))
df[df['Son']==0]['Absenteeism time in hours'].plot.hist(bins=30)


# In[ ]:


g = sns.FacetGrid(data=df,col='Son')
g.map(plt.hist,'Absenteeism time in hours')


# In[ ]:


plt.figure(figsize=(14,6))
df[df['Son']==0]['Age'].plot.hist(bins=30)


# In[ ]:


plt.figure(figsize=(14,6))
df[df['Son']!=0]['Age'].plot.hist(bins=30)


# In[ ]:


reason_27 = df['Reason for absence']==27


# In[ ]:


reasons = df['Reason for absence']


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['Reason for absence'])


# In[ ]:


df[df['Reason for absence']==27].count()


# In[ ]:


df[df['Absenteeism time in hours']==0].count()


# In[ ]:


df.head(10)


# In[ ]:


roa = df.groupby('Reason for absence')


# In[ ]:


roa['Absenteeism time in hours'].max()


# In[ ]:


X = df.iloc[:, 1:20].values
y = df.iloc[:,20:].values.reshape(-1,1)


# In[ ]:


from sklearn.svm import SVR


# In[ ]:


svr_regressor = SVR(kernel='rbf')


# In[ ]:


svr_regressor.fit(X,y.ravel())


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# In[ ]:


prediction = sc_y.inverse_transform(svr_regressor.predict(sc_X.transform(X)))


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


y_pred = svr_regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)


# In[ ]:


from sklearn.metrics import mean_squared_error, explained_variance_score


# In[ ]:


print('MSE:{}'.format(mean_squared_error(y_test,y_pred)))
print('Explained variance score:{}'.format(explained_variance_score(y_test,y_pred)))


#!/usr/bin/env python
# coding: utf-8

# # Introduction COVID19
# ## COVID-19 Visualizations and ML models With sklearn / scikit-learn
# 
# ### ML models: KNN, Linear Regression, Logistic Regression and RandomForestRegressor 
# 
# The first case may be traced back to 17 November 2019.[12] As of 7 June 2020, more than 6.91 million cases have been reported across 188 countries and territories, resulting in more than 400,000 deaths.
# 
# Source: https://en.wikipedia.org/wiki/Coronavirus_disease_2019
# 
# <font color = 'blue'>
# Contents:
# 1. [Importing data and libraries](#1)
# 1. [Load and Check Data](#2)
#     * [Combining Data](#3)
# 1. [Visualizations](#4)
# 1. [Modelling](#5)
#     * [preparing data for train test split](#6)
#     * [train test split](#7)
#     * [KNN](#8)
#     * [Linear regression](#9)
#     * [LogisticRegression](#10) 
#     * [RandomForestRegressor](#11)
# 
# 

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


# <a id = '1'> </a> </br>
# # Importing data and libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# <a id = '2'> </a> </br>
# # Load and Check Data

# In[ ]:


train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv") 
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv") 
sub=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')


# In[ ]:


train.head()


# In[ ]:


data = train.drop(labels=['Id','Province_State','Country_Region','Date'], axis=1)


# In[ ]:


print(data.head(2))
print(test.head(2))


# In[ ]:


data2 = train.drop(labels=['Province_State','Country_Region','Date'], axis=1)


# In[ ]:


print(data2.head(2))


# <a id = '3'> </a> </br>
# # Combining Data
# * recheck data

# In[ ]:


pd.merge(data2,data)


# In[ ]:


print(data.shape)
print(data2.shape)
print(data2.head(2))


# In[ ]:


df_new = data2.rename(columns={'Id': 'ForecastId'})


# In[ ]:


print(df_new.head())
print(df_new.info())


# In[ ]:


df_new.isna()


# In[ ]:


df_new.shape


# <a id = '4'> </a> </br>
# # Visualizations

# In[ ]:


Group_features = ['ConfirmedCases','Fatalities']
fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(df_new[Group_features].corr(), annot = True, fmt = '.2f')
plt.show()


# In[ ]:


corr = df_new[Group_features].corr()
sns.set(style="white")
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


sns.pairplot(df_new)


# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df_new['Fatalities'],color='red')
sns.kdeplot(df_new['ConfirmedCases'],color='blue')
plt.title(' ConfirmedCases - Fatalities',size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(10,8))
df = df_new
sns.lineplot(x="ConfirmedCases", y="Fatalities",data=df_new,label='ConfirmedCases')

plt.title("ConfirmedCases")


# In[ ]:


# line plot 
# first of all check columns 'data.columns'
# budget vs revenue 
df_new.ConfirmedCases.plot(kind = 'line', color = 'r', label = ' ConfirmedCases', linewidth=1,alpha =0.8  ,grid = True, linestyle = ':')
df_new.Fatalities.plot(color = 'b',label = 'Fatalities',linewidth=1,alpha = 0.8,grid = True, linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel(' ConfirmedCases ')
plt.ylabel(' Fatalities ')
plt.title('ConfirmedCases vs Fatalities')
plt.show()


# In[ ]:


# x ConfirmedCases , y Fatalities
df_new.plot(kind='scatter',x='ConfirmedCases',y='Fatalities',alpha=0.5,color='red')
plt.xlabel('ConfirmedCases')
plt.ylabel('Fatalities')
plt.title('ConfirmedCases Count / Fatalities Scatter plot')
plt.show()


# In[ ]:


sns.set(style="white")
df = df_new.loc[:,['ConfirmedCases','Fatalities']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)


# In[ ]:


# histogram
# values of ConfirmedCases 

plt.plot(df_new.ConfirmedCases,df_new.Fatalities)
plt.xlabel('ConfirmedCases')
plt.ylabel('Fatalities')
plt.show()


# In[ ]:


# histogram
# values of Fatalities 
df_new.Fatalities.plot(kind = 'hist',bins = 10,figsize = (10,10),color='r')
plt.show()


# In[ ]:


fig, ax = plt.subplots()
for a in [df_new.ConfirmedCases, df_new.Fatalities]:
    sns.distplot(a, bins=range(1, 110, 10), ax=ax, kde=False)
ax.set_xlim([0, 100])


# In[ ]:


plt.hist([df_new.Fatalities, df_new.ConfirmedCases], color=['r','b'], alpha=0.5)


# <a id = '5'> </a> </br>
# # Modelling 
# * preparing data for train test split
# * train test split
# * KNN
# * Linear regression
# * LogisticRegression
# * RandomForestRegressor

# <a id = '6'> </a> </br>
# ### Preparing data for train test split

# In[ ]:


df_new = df_new[:-22536]


# In[ ]:


X = df_new['ConfirmedCases'].values
y = df_new['Fatalities'].values


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


X = X.reshape(-1,1)
y = y.reshape(-1,1)


# <a id = '7'> </a> </br>
# ### train_test_split

# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42)


# <a id = '8'> </a> </br>
# ### KNN 

# In[ ]:


# !!! DO NOT FORGET TO LIBRARIES


# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=12)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print('Score', knn.score(X_test, y_test))


# <a id = '9'> </a> </br>
# ### LinearRegression

# In[ ]:


# !!! DO NOT FORGET TO LIBRARIES
reg = LinearRegression()

reg.fit(X_train,y_train)
preds = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,preds))
print(rmse)
print('Score',reg.score(X_test,y_test))


# <a id = '10'> </a> </br>
# ### LogisticRegression

# In[ ]:


# !!! DO NOT FORGET TO LIBRARIES

log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
log_reg.predict(X_test)
print("Score :",log_reg.score(X_test, y_test))


# <a id = '11'> </a> </br>
# ### RandomForestRegressor

# In[ ]:


# !!! DO NOT FORGET TO LIBRARIES


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train);
# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
#Score
print("Score :",rf.score(X_test, y_test))


# If you like it, please vote. (:

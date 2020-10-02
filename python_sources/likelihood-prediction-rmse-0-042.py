#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import seaborn as sns

# Modeling packages as needed

from sklearn.linear_model import Ridge,RidgeCV

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV

import xgboost as xgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Let's take the latest file for now;

data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')


# In[ ]:


data.shape


# In[ ]:


data.head()


# Chance of Admit is our dependent variable. It's a continuous variable

# In[ ]:


# Any missing values?

data.describe()


# 1. No missing values in the data
# 2. The maximum value in the dependent variable doesn't cross 1. No negative values found
# 3. Research field is a binary variable with 56% of the entries being from researchers
# 4. Max CGPA is 9.92 on 10 - Must be an outlier. We should look at the distribution
# 5. GRE and TOEFL scores don't cross their limits. Data seems not to have any data entry issues 

# In[ ]:


# distribution of variables in the data
sns.distplot(data['Chance of Admit '])


# Data looks left skewed a little bit. To fix left skewed-ness, we need to perform power transformation (square or cube etc.)

# In[ ]:


sq = np.power(data['Chance of Admit '],2)

sns.distplot(sq)

# Looks good now; it has 2 modes but that shouldn't be a problem for now


# In[ ]:


# Let's create a grid and look at the scatters and distributions of all other variables in the data;
sns.pairplot(data)

plt.show()


# We can clearly notice that Most of the variables show a high correlation with the dependent variable, and also amongst themselves. This might lead to multi-collinearity in the model

# In[ ]:


# Let's look at a box plot of admit column by research

fig = go.Figure()
# Use x instead of y argument for horizontal plot
fig.add_trace(go.Box(x= data['Chance of Admit '][data['Research'] == 0],name="Research = 0"))
fig.add_trace(go.Box(x=data['Chance of Admit '][data['Research'] == 1],name="Research = 1"))

fig.update_layout(title='Box by Research',template='plotly_white')

fig.show()


# A clear distinction of having research experience vs not having research experience. If a candidate has research experience, the likelihood of getting an admit increases by almost 1.2 times

# In[ ]:


pd.crosstab(data['LOR '],1)


# In[ ]:


data[['LOR ','Chance of Admit ']].groupby('LOR ').mean()

#  Clearly noticeable, let's look at a boxplot


# In[ ]:


fig = go.Figure()

unq = list(data.sort_values('LOR ')['LOR '].drop_duplicates())

for i in unq:
    
    fig.add_trace(go.Box(y = data['Chance of Admit '][data['LOR '] == i],name="LOR = "+str(i)))


fig.update_layout(title='Box by LOR score',template='plotly_white')

fig.show()


# Previously, we have noticed that the averages show a clear distintion, now we also notice that the distributions show clear distinction

# In[ ]:


# University Rating by Chance of Admit 

data[['University Rating','Chance of Admit ']].groupby('University Rating').mean()


# In[ ]:


fig = go.Figure()

unq = list(data.sort_values('University Rating')['University Rating'].drop_duplicates())

for i in unq:
    
    fig.add_trace(go.Box(y = data['Chance of Admit '][data['University Rating'] == i],name="University Rating = "+str(i)))


fig.update_layout(title='Box by University Rating score',template='plotly_white')

fig.show()


# In[ ]:


#Calculate the correlations between the independent and dependent variables

corr = data.corr()
fig, ax = plt.subplots(figsize=(8, 8))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.title('Correlation Plots')
plt.show()


# Almost all the variables show high correlations with dependent variable
# The variables are also correlated amongst themselves, inducing multi-collinearity
# 

# In[ ]:


#  I want to try ridge as it reduces standard errors;

# split the data into train and test

data_for_model = data.drop(['Serial No.'],axis = 1)

data_for_model['dependent'] =  np.power(data_for_model['Chance of Admit '],2)

n = data_for_model.shape[0]

train = data_for_model[0:(n-100)]

test = data_for_model[400:n]

print(train.shape)
print(test.shape)


# In[ ]:


train.columns


# In[ ]:


x = train.drop(['Chance of Admit ','dependent'],axis = 1)
y = train['dependent']

xt = test.drop(['Chance of Admit ','dependent'],axis = 1)
yt = test['dependent']


# In[ ]:


ridgecv = RidgeCV(alphas = 10**np.linspace(10,-2,100)*0.5, scoring = 'neg_mean_squared_error', normalize = True)

ridgecv.fit(x, y)
ridgecv.alpha_


# In[ ]:


ridge = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridge.fit(x, y)


m1 = mean_squared_error(yt, ridge.predict(xt))

print(m1)
print(np.sqrt(m1))

#  This would be WRT the transformed variable


# In[ ]:


plt.plot(np.sqrt(yt), np.sqrt(ridge.predict(xt)), '*')

# The predictions look good


# In[ ]:


# Calculate the RMSE on the actual variable by taking the square root as yt is a squared variable

m2 = mean_squared_error(np.sqrt(yt), np.sqrt(ridge.predict(xt)))

np.sqrt(m2)

print("MSE: {0:.5f}\nRMSE: {1:.5f}".format(m2, np.sqrt(m2)))


# # The final RMSE of the model is 0.042

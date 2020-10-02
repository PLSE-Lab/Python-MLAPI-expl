#!/usr/bin/env python
# coding: utf-8

# ### 1. [Data Cleaning](#Data-Cleaning)
# ### 2. [Data Visualization](#Data-visualization)
# ### 3. [Training and testing](#Training-and-Testing)
# ### 4. [Model performance comparison](#Comparison-of-regression-model-performance)

# In[ ]:


get_ipython().system('pip install --upgrade plotly # piecharts require plotly 4.4.1')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import metrics
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


file='/kaggle/input/top50spotify2019/top50.csv'
data = pd.read_csv(file, encoding = 'ISO-8859-1', index_col=0)


# # Data Cleaning

# In[ ]:


data.info()


# In[ ]:


# Clean columns - remove the dots from column names
data.columns = data.columns.str.replace('.','')


# In[ ]:


data.info()


# In[ ]:


# get first 10 rows
data.head(10)


# In[ ]:


# get dimensions of dataset
data.shape


# In[ ]:


# check for any null values
data.isnull().sum()


# In[ ]:


pd.set_option('precision', 3)
data.describe()


# # Data visualization

# In[ ]:


fig = px.pie(data, names='ArtistName', title='Songs by Artist')
fig.update_traces(textposition='inside', textinfo='label+value', showlegend=False)


# In[ ]:


fig = px.pie(data, names='Genre')
fig.update_traces(textposition='inside', textinfo='label+value', showlegend=False)


# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(y='ArtistName', data=data, order=data.ArtistName.value_counts().index)
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))

# sorting artists by popularity
order=data.groupby('ArtistName')['Popularity'].mean().sort_values().index;
sns.barplot(x='Popularity', y='ArtistName', data=data, order=order)
plt.show()


# In[ ]:


sns.pairplot(data)
plt.show()


# In[ ]:


pd.set_option('precision', 3)
corr = data.corr(method='spearman')
print(corr)


# In[ ]:


plt.figure(figsize=(15,10))
plt.title('Correlation Heatmap')
sns.heatmap(corr, annot=True)
plt.show()


# In[ ]:


# certain features such as BeatsPerMinute and Speechiness have higher correlation than others
# and features like Acousticness have very little correlation
figure = sns.pairplot(data, x_vars=['BeatsPerMinute','Acousticness'], y_vars=['Popularity'], kind='reg')
figure.fig.set_size_inches(20, 10)


# In[ ]:


sns.distplot(data['Popularity'], bins=10, kde=True)
plt.show()

# Popularity (response variable) is skewed


# In[ ]:


# Handle the skewness
# Keep number of bins same in both plots, else this plot will be a pure normal distribution
transformed,_ = stats.boxcox(data.Popularity)
sns.distplot(transformed, bins=10, kde=True)
plt.show()


# # Training and Testing

# In[ ]:


# drop categorical columns
# predict popularity as a function of other predictors
x = data.drop(['TrackName','ArtistName','Genre','Popularity'],axis=1)
y = data['Popularity'].values


# ### Selection and Scaling

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[ ]:


# Perform feature scaling so that each feature has mean 0 and standard deviation 1
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)


# ## Linear Regression

# In[ ]:


lr = LinearRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
lr_err = metrics.mean_squared_error(lr_pred, y_test)
print(lr.coef_)


# ## Support Vector Regression

# In[ ]:


svr=SVR(kernel='rbf', gamma='scale')
svr.fit(x_train,y_train)
svr_pred=svr.predict(x_test)
svr_err = metrics.mean_squared_error(svr_pred, y_test)


# ## Random Forest Regression

# In[ ]:


rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x_train, y_train)
rfr_pred = rfr.predict(x_test)
rfr_err = metrics.mean_squared_error(rfr_pred, y_test)


# ### Comparison of regression model performance

# In[ ]:


sns.barplot(x=['LinearRegression', 'SVR', 'RandomForest'], y=[lr_err, svr_err, rfr_err])
plt.show()


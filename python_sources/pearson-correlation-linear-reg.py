#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
auto_mpg_df=pd.read_csv('../input/autompg-dataset/auto-mpg.csv', na_values='?')
auto_mpg_df.pop('car name')
auto_mpg_df.dropna(inplace=True)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


auto_mpg_df.head()


# In[ ]:


auto_mpg_df.describe()


# auto_mpg_df.corr() finds the correlation each variable with all the other variables in the dataframe. In the following step we are storing the correlation values to the variable corr_matrix. Since in our dataframe mpg is the target variable and we are more interested to know how other variables are correlated to this, we are just displaying the correlation values with respect to mpg using corr_matrix['mpg']

# In[ ]:


# method = pearson is by default and can be avoided to mention
corr_matrix = auto_mpg_df.corr(method='pearson')
corr_matrix


# In[ ]:


corr_matrix['mpg']


# From the above correlation marix it can be observered that weight, acceleration & cylinders have strong negative relation with mpg where as model year, origin has not so strong positive relation with mpg and acceleration has lower positive relation with mpg. This relation can also be visualized using the heat map as shown below stronger the relation darker the color

# In[ ]:


sns.scatterplot(data=auto_mpg_df, x='mpg', y = 'weight');


# In[ ]:


sns.heatmap(auto_mpg_df.corr());


# In[ ]:


sns.scatterplot(data=auto_mpg_df, y='mpg', x= 'weight');


# In[ ]:


sns.pairplot(auto_mpg_df, vars=['mpg','weight','cylinders','acceleration','displacement','origin','model year'])


# In[ ]:


Y = auto_mpg_df.pop('mpg')


# In[ ]:


Y = np.asarray(Y)
X_train,X_test,y_train,y_test = train_test_split(auto_mpg_df,Y,test_size=0.3,random_state=42)


# In[ ]:


linearRegModel = LinearRegression()
linearRegModel.fit(X_train,y_train)
from sklearn.metrics import accuracy_score
from sklearn import metrics
y_pred = linearRegModel.predict(X_test)
linearRegModel.score(X_test,y_test)
metrics.explained_variance_score(y_test,y_pred)


# 

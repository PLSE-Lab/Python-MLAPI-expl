#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis of the Dataset

# <center><img src="https://dwkujuq9vpuly.cloudfront.net/news/wp-content/uploads/2020/03/shutterstock_152463302-1-960x480.jpg"></center>

# ## This notebook covers basic exploration of this dataset.
# ## The motivation behind every plot is given before to have a clarity of the aim.

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


# In[ ]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


df


# In[ ]:


#1460 Rows
#81 Columns


# # Exploring the target variable, SalePrice

# In[ ]:


import plotly.express as px
fig = px.histogram(df, x="SalePrice")
fig.show()


# # We can look at different dimensions of Sales Price, ie how does Price look in each neighborhood

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(20, 6))
sns.boxplot(x='Neighborhood', y='SalePrice', data=df, showfliers=False);


# ### How prices differ by Quality?

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(20, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=df, showfliers=False);


# ### Quality is going to be a super important feature!

# ### How prices differ by Condition?
# 

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(20, 6))
sns.boxplot(x='OverallCond', y='SalePrice', data=df, showfliers=False);


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(20, 6))
sns.boxplot(x='MSZoning', y='SalePrice', data=df, showfliers=False);


# ### Relation b/w Lot Area and Price 

# In[ ]:


import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(y=df['LotArea'], x=df['SalePrice'],
                    mode='markers', name='markers'))


# ### Also yearbuilt will influence the price of the property, lets explore that also!

# ### Only considering years > 1950 for simplicity.

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(40, 6))
temp = df[['YearBuilt','SalePrice']]
temp = temp[temp['YearBuilt']>1950]
sns.boxplot(x='YearBuilt', y='SalePrice', data=temp, showfliers=False);


# ### Notice that more variation in the years post 1985. So year built can be an important feature

# In[ ]:


import plotly.express as px
fig = px.histogram(df, x="LotArea")
fig.show()


# ### Density Plots for numerical variables

# In[ ]:


pd.set_option('display.max_columns', None)

df.head()


# 

# In[ ]:


numerical_list = ['LotArea','MasVnrArea','GrLivArea','BedroomAbvGr','GarageArea','YrSold']
for i in numerical_list:
    sns.distplot(df[i], hist=False, rug=True)
    plt.show();


# ### How much does Garage features influence the price?

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(20, 6))
sns.boxplot(x='GarageCond', y='SalePrice', data=df, showfliers=False);


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(20, 6))
sns.boxplot(x='GarageQual', y='SalePrice', data=df, showfliers=False);


# In[ ]:


import plotly.express as px
fig = px.scatter(df, x="GarageArea", y="SalePrice",color='GarageArea')
fig.show()


# In[ ]:


#GarageType	GarageYrBlt	GarageFinish	GarageCars


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(20, 6))
sns.boxplot(x='GarageType', y='SalePrice', data=df, showfliers=False);


# In[ ]:


import plotly.express as px
fig = px.scatter(df, x="GarageYrBlt", y="SalePrice",color="GarageYrBlt")
fig.show()


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(20, 6))
sns.boxplot(x='GarageFinish', y='SalePrice', data=df, showfliers=False);


# In[ ]:



fig, axes = plt.subplots(1, 1, figsize=(20, 6))
sns.boxplot(x='GarageCars', y='SalePrice', data=df, showfliers=False);


# ### Marks the end of this notebook!

# In[ ]:





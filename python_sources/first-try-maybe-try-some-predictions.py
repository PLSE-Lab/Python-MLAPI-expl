#!/usr/bin/env python
# coding: utf-8

# my first notebook on Kaggle, try some fun here.
# -----------------------------------------------

# In[ ]:


# import pandas
import pandas as pd


# In[ ]:


# create a DataFrame
df = pd.read_csv('../input/2016.csv', header=0, sep=',', index_col=False)


# In[ ]:


# see data shape
df.shape


# In[ ]:


# see data type
df.dtypes


# In[ ]:


# See data head
df.head()


# In[ ]:


df.describe()


# 
# 
# **Visualization**
# -----------------

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


country = df['Country']
Region = df['Region']
Happiness_Rank = df['Happiness Rank']


# In[ ]:


df_copy = df
df.drop(['Country', 'Region','Happiness Rank'], axis = 1, inplace = True)


# In[ ]:


pd.scatter_matrix(df, alpha = 0.8, figsize = (15,15), diagonal = 'kde');


# Firstly, analyze the obvious linear correlations.

# In[ ]:


df.drop(['Freedom', 'Trust (Government Corruption)','Generosity','Dystopia Residual'], axis = 1, inplace = True)


# In[ ]:


pd.scatter_matrix(df, alpha = 0.8, figsize = (12,12), diagonal = 'kde');


# Try some predictions with this data set. This problem looks like a regression problem.
# 
# 
# ----------

# In[ ]:


df.head()


# In[ ]:


# import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical = ['Lower Confidence Interval', 'Upper Confidence Interval', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)']
df[numerical] = scaler.fit_transform(df[numerical])

df.head()


# In[ ]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

plt.figure(figsize=(16, 10))
for i, key in enumerate(['Lower Confidence Interval', 'Upper Confidence Interval', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)']):
    plt.subplot(2, 3, i+1)
    plt.xlabel(key)
    plt.scatter(df[key], df['Happiness Score'], alpha=0.5)


# All data are numerical, that is good. Don't need to do one-hot coding.

# In[ ]:


features = df.drop(['Happiness Score'],axis=1)
label = df['Happiness Score']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, random_state = 0)

print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))


# In[ ]:


import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

exported_pipeline = GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="ls",
                                              max_features=0.9, min_samples_leaf=5,
                                              min_samples_split=6)
exported_pipeline.fit(X_train, y_train)


# In[ ]:


results = exported_pipeline.predict(X_test)
results_ground = np.array(y_test)
error = results_ground - results


# In[ ]:


x_axis = range(0, len(error))
y_axis = error

_ = plt.scatter(x_axis,y_axis)
_ = plt.xlabel('Number of testing samples')
_ = plt.ylabel('Happiness score error')


# Now we see the nice result, also we can calculate the numerical score.

# In[ ]:


score = exported_pipeline.score(X_test, y_test)
print(score)


# Ok, using the linear components, we almost get a perfect result. Next step, what about using the "not-so-linear" features?

# In[ ]:





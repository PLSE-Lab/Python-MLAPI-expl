#!/usr/bin/env python
# coding: utf-8

# # Predicting the effectiveness of a Froth floating process
# 
# 
# **June 2019**
# 
# 
# **Project Description:**
# 
# We will use this dataset to analyse and predict the Froth floating process having the two aims:
# * What is the best predictor for the iron concentration of the product?
# * Can the data set be used to predict the impurity of the product (by silicate concentration)?
# 
# **Data Description:** 
# 
# This notebook deals with the analysis of a reverse cationic flotation process of a real production environment. The data (including its documentation) is accessible through kaggle: https://www.kaggle.com/edumagalhaes/quality-prediction-in-a-mining-process
# 
# ---

# ## The Froth flotation process
# 
# The froth floatation is used to seperate the iron contents in the ore from other contaminations. The whole process usually contains for steps:
# 
# 1. Contioning of the ore feed pulp (mixture of ore and water) and other reagents
# 2. Separation of hydrophobic and hydrophilic materials: binding particles attach to the bubbles
# 3. The bubbles transport the particles upwards until they float on the surface (froth)
# 4. Collection of the froth by mechanical separation (e.g. by an impeller)

# ---
# 
# ## Data Analysis
# 
# We start our analysis by importing required libraries:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn import metrics

# include fasti.ai libraries
from fastai.tabular import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from IPython.display import display
pd.set_option('display.max_columns', None) # display all columns
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/MiningProcess_Flotation_Plant_Database.csv", parse_dates = True, index_col = 'date',decimal=',')


# Check if we have missing (nan) values:

# In[ ]:


shape1 = df.shape
df = df.dropna()
shape2 = df.shape
if shape1 == shape2:
    print('Data contains no nan values.')
else:
    print('Data contains nan values.')


# Good. Let's look at the first couple of rows in our dataframe:

# In[ ]:


df.head()


# The dataframe contains data about:
# * Quality measures of the iron ore pulp before feeding it into the process (inputs)
# * Features that can effect the quality of the product (process parameters)
# * Quality measures of the iron ore pulp as product of the process (outputs)

# OK, this looks good so far. Lets start with visualizing the data to see flaws in the data. We start by plotting our most improtant variables '% Iron Concentrate' and '% Silica Concentrate':

# In[ ]:


plt.figure(figsize=(25,8))
plt.subplot(1, 2, 1)
plt.plot(df['% Iron Concentrate']);
plt.xlabel('Date')
plt.title('Iron Concentrate in %')
plt.subplot(1, 2, 2)
plt.plot(df['% Silica Concentrate']);
plt.xlabel('Date')
plt.title('Silica Concentrate in %')


# We can see that our data misses data packages of a couple of days. Based on the documentation at Kaggle, this was caused by a production shutdown. In order to rule out any influences from potentially corrupted data, we will remove the data earlier of the restart of production ("2017-03-29 12:00:00").
# 
# We can also see that the quality of the products does not seem to follow a clear temporal dependency.

# In[ ]:


sep_date = "2017-03-29 12:00:00"
ind_date = df.index<sep_date #boolean of earlier dates
df.drop(df.index[ind_date],inplace=True)
df.head(1)


# Now, we quickly look at pearson correlations between our features (independent variables) to get a better understanding of our dataset:

# In[ ]:


plt.figure(figsize=(30, 25))
p = sns.heatmap(df.corr(), annot=True)


# Wow, that revealed a high (negative) correlations between the 'Iron Feed' and 'Silica Feed' (both Inputs of the process) as well as 'Iron Concentrate' and 'Silica Concentrate' (both Outputs of the process from the lab measurement). The later basically says, the higher the quality of the Iron, the smaller the less Silica it contains.

# ## Modeling
# 
# Now let's apply a model to check if we can predict the dependent variable '% Concentrate Silica'. First, we split our dataframe into train and validation set (train: first 80% of dataframe, test: last 20% of dataframe). Then we need to remove our '% Silica Concentrate' and '% Iron Concentrate' columns, since the first one is the dependent variable and the later is not available for the online implementation, as these values come from a lab measurement and takes roughly 1h 40 minutes.

# In[ ]:


train, test = train_test_split(df, test_size=0.2)
x = train.drop(['% Silica Concentrate','% Iron Concentrate'], axis=1)
y = train['% Silica Concentrate']


# **Train Random Forest**

# In[ ]:


model = RandomForestRegressor(n_estimators=50, min_samples_leaf=1, max_features=None, n_jobs=-1)
model.fit(x,y)


# These settings can be tuned a little bit to improve performance...

# **Check Train Set**

# In[ ]:


y_hat = model.predict(x)
mse = metrics.mean_squared_error(y,y_hat)
print('Train Set')
print('RMSE:',math.sqrt(mse),'   R2:',model.score(x,y))


# **Check Test Set**

# In[ ]:


x_test = test.drop(['% Silica Concentrate','% Iron Concentrate'], axis=1)
y_test = test['% Silica Concentrate']
y_hat_test = model.predict(x_test)
mse_test = metrics.mean_squared_error(y_test,y_hat_test)
print('TEST Set')
print('RMSE:',math.sqrt(mse_test),'   R2:',model.score(x_test,y_test))


# ## Model interpretation

# **Feature importance**
# 
# Let's look at the importance of each feature and plot the 10 most important features:[](http://)

# In[ ]:


feat_importances = pd.Series(model.feature_importances_, index=df.columns[:-2])
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# We can see the ten most important features, '% Iron Feed' and '% Silica Feed' as well as the pH level of the ore pulp seem to be substantial parameters to control. 

# **Identifiying redundant features**

# In[ ]:


from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)
plt.show()


# The plot above shows how each feature correlates with the others. The features which merge on the right hand side are closer to each other, suggesting that potentially only one of both would be sufficient for training the model. This could be useful to reduce the colinearity of different features.

# ------

# ## Averaging dataset to account for differently sampled features

# Based on the documentation of the dataset, some columns are sampled every 20 seconds, some every hour. For instance, the feature 'Ore Pulp Flow' changes continously during the process while the features '% Iron Feed' and '% Silica Feed' are sample only every hour. Thus, I think it is not really helpful to use every row (sampled every 20s) including the less sampled features (held constant over the duration of one hour), since this assumes that every row is an individual observation - which it isnt. Using all samples to train our model does not really represent the reality. What we can try to do is to mean the rows (observations) for every hour and create a new dataframe which uses the average of the 20s samples. This however, will strongly reduce our data size (by factor 180). What we can do to not lose all information of the 20s sampled features, is to also include their variations during one hour (e.g. by calculating also the standard deviation of the meaned columns).

# In[ ]:


df_mean = df.copy()
mean_grpby = df_mean.groupby(['date']).mean() # calculate mean
std_grpby = df_mean.groupby(['date']).std() # calculate std
std_grpby = std_grpby.loc[:, (std_grpby != 0).any(axis=0)] # delete null columns (columns with zero variance)
std_grpby = std_grpby.add_prefix('STD_') # add prefix to column names
df_merge = pd.merge(mean_grpby, std_grpby, on='date') # merge both dataframes
df_merge.describe()


# Repeat training model with new dataframe 'df_merge':

# In[ ]:


train, test = train_test_split(df_merge, test_size=0.2)
x_aver = train.drop(['% Silica Concentrate','% Iron Concentrate','STD_% Silica Concentrate','STD_% Iron Concentrate'], axis=1)
y_aver = train['% Silica Concentrate']


# In[ ]:


model = RandomForestRegressor(n_estimators=50, min_samples_leaf=1, max_features=None, n_jobs=-1)
model.fit(x_aver,y_aver)


# **Check Averaged Train Set**

# In[ ]:


y_aver_hat = model.predict(x_aver)
mse = metrics.mean_squared_error(y_aver,y_aver_hat)
print('Train Set')
print('RMSE:',math.sqrt(mse),'   R2:',model.score(x_aver,y_aver))


# **Check Averaged Test Set**

# In[ ]:


x_aver_test = test.drop(['% Silica Concentrate','% Iron Concentrate','STD_% Silica Concentrate','STD_% Iron Concentrate'], axis=1)
y_test = test['% Silica Concentrate']
y_hat_test = model.predict(x_aver_test)
mse_test = metrics.mean_squared_error(y_test,y_hat_test)
print('TEST Set')
print('RMSE:',math.sqrt(mse_test),'   R2:',model.score(x_aver_test,y_test))


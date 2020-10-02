#!/usr/bin/env python
# coding: utf-8

# # Scale, Standardize, or Normalize with scikit-learn
# ## When to use MinMaxScaler, RobustScaler, StandardScaler, and Normalizer
# ## By Jeff Hale

# ## Please upvote if you find this Kernel helpful :)

# # Why scale, standardize, or normalize?
# 
# Many machine learning algorithms, such as neural networks, regression-based algorithms, K-nearest neighbors, support vector machines with radial bias kernel functions, principal components analysis, and algorithms using linear discriminant analysis don't perform as well if the features are not on relatively similar scales. 
# 
# Sometimes you'll want a more normally distributed distribution. 
# 
# Some of the methods below dilute the effects of outliers. 

# ## TLDR
# 
# * Use MinMaxScaler as your default
# * Use RobustScaler if you have outliers and can handle a larger range
# * Use StandardScaler if you need normalized features
# * Use Normalizer sparingly - it normalizes rows, not columns

# Here's a [cheat sheet I made in a google sheet](https://docs.google.com/spreadsheets/d/1woVi7wq13628HJ-tN6ApaRGVZ85OdmHsDBKLAf5ylaQ/edit?usp=sharing) to help folks keep the options straight. 
# 
# Let's set things up and start making some distributions!

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn import preprocessing

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')

np.random.seed(34)


# # Original Distributions

# Let's make several types of random distributions.

# In[ ]:


#create columns of various distributions
df = pd.DataFrame({ 
    'beta': np.random.beta(5, 1, 1000) * 60,        # beta
    'exponential': np.random.exponential(10, 1000), # exponential
    'normal_p': np.random.normal(10, 2, 1000),      # normal platykurtic
    'normal_l': np.random.normal(10, 10, 1000),     # normal leptokurtic
})

# make bimodal distribution
first_half = np.random.normal(20, 3, 500) 
second_half = np.random.normal(-20, 3, 500) 
bimodal = np.concatenate([first_half, second_half])

df['bimodal'] = bimodal

# create list of column names to use later
col_names = list(df.columns)


# Let's plot our original distributions.

# In[ ]:


# plot original distribution plot
fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title('Original Distributions')

sns.kdeplot(df['beta'], ax=ax1)
sns.kdeplot(df['exponential'], ax=ax1)
sns.kdeplot(df['normal_p'], ax=ax1)
sns.kdeplot(df['normal_l'], ax=ax1)
sns.kdeplot(df['bimodal'], ax=ax1);


# In[ ]:


df.head()


# Let's see what are the means are.

# In[ ]:


df.mean()


# If you'd like more summary statistics:

# In[ ]:


df.describe()


# In[ ]:


df.plot()


# These values are all in the same ballpark.

# ## Add a feature with much larger values

# This feature could be home prices, for example.

# In[ ]:


normal_big = np.random.normal(1000000, 10000, (1000,1))  # normal distribution of large values
df['normal_big'] = normal_big


# In[ ]:


col_names.append('normal_big')


# In[ ]:


df['normal_big'].plot(kind='kde')


# In[ ]:


df.normal_big.mean()


# We've got a normalish distribution with a mean near 1,000,0000.

# If we put this on the same plot as the original distributions, you can't even see the earlier columns.

# In[ ]:


# plot original distribution plot with larger value feature
fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title('Original Distributions')

sns.kdeplot(df['beta'], ax=ax1)
sns.kdeplot(df['exponential'], ax=ax1)
sns.kdeplot(df['normal_p'], ax=ax1)
sns.kdeplot(df['normal_l'], ax=ax1)
sns.kdeplot(df['bimodal'], ax=ax1);
sns.kdeplot(df['normal_big'], ax=ax1);


# The new, high-value distribution is way to the right. And here's a plot of the values.

# In[ ]:


df.plot()


# In[ ]:


df.describe()


# Now let's see what happens when we do some scaling. Let's apply MinMax Scaler first.

# # MinMaxScaler 

# MinMaxScaler subtracts the column mean from each value and then divides by the range.

# In[ ]:


mm_scaler = preprocessing.MinMaxScaler()
df_mm = mm_scaler.fit_transform(df)

df_mm = pd.DataFrame(df_mm, columns=col_names)

fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title('After MinMaxScaler')

sns.kdeplot(df_mm['beta'], ax=ax1)
sns.kdeplot(df_mm['exponential'], ax=ax1)
sns.kdeplot(df_mm['normal_p'], ax=ax1)
sns.kdeplot(df_mm['normal_l'], ax=ax1)
sns.kdeplot(df_mm['bimodal'], ax=ax1)
sns.kdeplot(df_mm['normal_big'], ax=ax1);


# Notice how the shape of each distribution remains the same, but now the values are between 0 and 1.

# In[ ]:


df_mm['beta'].min()


# In[ ]:


df_mm['beta'].max()


# Let's look at the minimums and maximums for each column prior to scaling.

# In[ ]:


mins = [df[col].min() for col in df.columns]
mins


# In[ ]:


maxs = [df[col].max() for col in df.columns]
maxs


# Let's check the minimums and maximums for each column after MinMaxScaler.

# In[ ]:


mins = [df_mm[col].min() for col in df_mm.columns]
mins


# In[ ]:


maxs = [df_mm[col].max() for col in df_mm.columns]
maxs


# Looks close enough to 0 to 1 intervals to me. Our feature with much larger values was brought into scale with our other features. 
# 
# Now let's look at RobustScaler.

# # RobustScaler

# RobustScaler subtracts the column median and divides by the interquartile range.

# In[ ]:


r_scaler = preprocessing.RobustScaler()
df_r = r_scaler.fit_transform(df)

df_r = pd.DataFrame(df_r, columns=col_names)

fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title('After RobustScaler')

sns.kdeplot(df_r['beta'], ax=ax1)
sns.kdeplot(df_r['exponential'], ax=ax1)
sns.kdeplot(df_r['normal_p'], ax=ax1)
sns.kdeplot(df_r['normal_l'], ax=ax1)
sns.kdeplot(df_r['bimodal'], ax=ax1)
sns.kdeplot(df_r['normal_big'], ax=ax1);


# Let's check the minimums and maximums for each column after RobustScaler.

# In[ ]:


mins = [df_r[col].min() for col in df_r.columns]
mins


# In[ ]:


maxs = [df_r[col].max() for col in df_r.columns]
maxs


# Although the range of values for each feature is much smaller than for the original features, it's larger and varies more than for MinMaxScaler. The bimodal distribution values are now compressed into two small groups.

# Now let's look at StandardScaler.

# # StandardScaler

# StandardScaler is scales each column to have 0 mean and unit variance.

# In[ ]:


s_scaler = preprocessing.StandardScaler()
df_s = s_scaler.fit_transform(df)

df_s = pd.DataFrame(df_s, columns=col_names)

fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title('After StandardScaler')

sns.kdeplot(df_s['beta'], ax=ax1)
sns.kdeplot(df_s['exponential'], ax=ax1)
sns.kdeplot(df_s['normal_p'], ax=ax1)
sns.kdeplot(df_s['normal_l'], ax=ax1)
sns.kdeplot(df_s['bimodal'], ax=ax1)
sns.kdeplot(df_s['normal_big'], ax=ax1);


# Qutie a nice chart, don't you think? You can see that all features now have 0 mean.

# Let's check the minimums and maximums for each column after StandardScaler.

# In[ ]:


mins = [df_s[col].min() for col in df_s.columns]
mins


# In[ ]:


maxs = [df_s[col].max() for col in df_s.columns]
maxs


# The ranges are fairly similar to RobustScaler. 
# 
# Now let's look at Normalizer.

# # Normalizer
# 
# Note that normalizer operates on the rows, not the columns. It applies l2 normalization by default.

# In[ ]:


n_scaler = preprocessing.Normalizer()
df_n = n_scaler.fit_transform(df)

df_n = pd.DataFrame(df_n, columns=col_names)

fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title('After Normalizer')

sns.kdeplot(df_n['beta'], ax=ax1)
sns.kdeplot(df_n['exponential'], ax=ax1)
sns.kdeplot(df_n['normal_p'], ax=ax1)
sns.kdeplot(df_n['normal_l'], ax=ax1)
sns.kdeplot(df_n['bimodal'], ax=ax1)
sns.kdeplot(df_n['normal_big'], ax=ax1);


# Let's check the minimums and maximums for each column after scaling.

# In[ ]:


mins = [df_n[col].min() for col in df_n.columns]
mins


# In[ ]:


maxs = [df_n[col].max() for col in df_n.columns]
maxs


# Normalizer also moved the features to similar scales. Notice that the range for our much larger feature's values is now extremely small and clustered around .9999999999. 

# Let's look at our original and transformed distributions together. We'll exclude Normalizer because you generally want to tranform your features, not your samples.

# # Combined Plot

# In[ ]:


# Combined plot.

fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(20, 8))


ax0.set_title('Original Distributions')

sns.kdeplot(df['beta'], ax=ax0)
sns.kdeplot(df['exponential'], ax=ax0)
sns.kdeplot(df['normal_p'], ax=ax0)
sns.kdeplot(df['normal_l'], ax=ax0)
sns.kdeplot(df['bimodal'], ax=ax0)
sns.kdeplot(df['normal_big'], ax=ax0);


ax1.set_title('After MinMaxScaler')

sns.kdeplot(df_mm['beta'], ax=ax1)
sns.kdeplot(df_mm['exponential'], ax=ax1)
sns.kdeplot(df_mm['normal_p'], ax=ax1)
sns.kdeplot(df_mm['normal_l'], ax=ax1)
sns.kdeplot(df_mm['bimodal'], ax=ax1)
sns.kdeplot(df_mm['normal_big'], ax=ax1);


ax2.set_title('After RobustScaler')

sns.kdeplot(df_r['beta'], ax=ax2)
sns.kdeplot(df_r['exponential'], ax=ax2)
sns.kdeplot(df_r['normal_p'], ax=ax2)
sns.kdeplot(df_r['normal_l'], ax=ax2)
sns.kdeplot(df_r['bimodal'], ax=ax2)
sns.kdeplot(df_r['normal_big'], ax=ax2);


ax3.set_title('After StandardScaler')

sns.kdeplot(df_s['beta'], ax=ax3)
sns.kdeplot(df_s['exponential'], ax=ax3)
sns.kdeplot(df_s['normal_p'], ax=ax3)
sns.kdeplot(df_s['normal_l'], ax=ax3)
sns.kdeplot(df_s['bimodal'], ax=ax3)
sns.kdeplot(df_s['normal_big'], ax=ax3);


# You can see that after any transformation the distributions are on a similar scale. Also notice that MinMaxScaler doesn't distort the distances between the values in each feature.

# I hope you found this Kernel to be a helpful introduction to Scaling, Standardizing, and Normalizing with scikit-learn. Please upvote it if you found it helpful!

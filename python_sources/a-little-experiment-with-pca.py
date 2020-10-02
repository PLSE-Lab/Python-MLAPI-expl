#!/usr/bin/env python
# coding: utf-8

# # A little experiment with PCA

# Principal Component Analysis (PCA) is a well known technique for dimensionality reduction. In simple terms, it consists in the construction of a change of basis transformation such that the vectors in the new basis correspond to directions of maximum variance in the data set, and are in decreasing order in terms of variance.

# #### Goal

# Suppose we are in the setting of a regression problem, i.e., our goal is to predict a continuously ranging variable, such as the canonical housing prices example. Furthermore, suppose we have a certain (large) number of numerical features that are relevant to use in our model (if there are just a few numerical features, maybe PCA is not necessary).
# 
# We want to understand what is the outcome if we fit our PCA instance with the target variable (which is, in this case, numerical) in the data set. Can this introduce data leakage in our model?
# 
# To test this, we'll construct a supervised learning problem and proceed with 2 versions of the model:
# 
# 1. We fit PCA including the target variable in the data set
# 2. We fit PCA without the target variable in the data set
# 
# In the end, we compute some metrics to investigate the difference.

# ## Imports

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ## Generating Data

# In order to test our assumptions, we'll create a data set with ten thousand instance as follows:
# 
# 1. Each column of the data is a normally distributed sample. Specifically, column j follows a normal distribution of mean 0 and standard deviation j.
# 2. The target variable will be the sum of all variables with added noise. This noise is also normally distributed, with mean 0 and standard deviation 0.05.
# 
# Since our data is artificially generated, there is no need to use 'train_test_split', we just create 2 sets (train and test).

# #### Creating Features

# In[ ]:


from scipy.stats import norm


# In[ ]:


#10 columns, 100000 samples
l = [norm.rvs(size = 10**5, loc = 0, scale = i+1) for i in range(10)]
l_test = [norm.rvs(size = 10**5, loc = 0, scale = i+1) for i in range(10)]
df = pd.DataFrame(l).T
df_test = pd.DataFrame(l_test).T


# In[ ]:


c = {}
for i in range(10):
    c[i] = 'Variable ' + str(i+1)

df.rename(columns = c, inplace = True)
df_test.rename(columns = c, inplace = True)


# #### Creating Target Variable

# In[ ]:


noise = norm.rvs(size = 10**5, loc = 0, scale = 0.05)
noise_test = norm.rvs(size = 10**5, loc = 0, scale = 0.05)


# In[ ]:


df['y'] = df['Variable 1']
df_test['y'] = df_test['Variable 1']
for i in range(2,11):
    df['y'] = df['y'] + df['Variable ' + str(i)]
    df_test['y'] = df_test['y'] + df_test['Variable ' + str(i)]
    
df['y'] = df['y'] + noise
df_test['y'] = df_test['y'] + noise_test


# In[ ]:


df.head()


# In[ ]:


df_test.head()


# ## PCA Time

# Lets start by creating two copies of the training set, one to each case.

# In[ ]:


df1 = df.copy()
df2 = df.copy()
y = df.y


# From the second copy, we drop the target variable!

# In[ ]:


df2.drop('y', axis = 1, inplace = True)


# In[ ]:


df1.head()


# In[ ]:


df2.head()


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


# We create two instances of the PCA
pca1 = PCA(n_components = 5)
pca2 = PCA(n_components = 5)


# In[ ]:


# Fit transform of the PCA in each case
principalcomponents1 = pca1.fit_transform(df1)
principalcomponents2 = pca2.fit_transform(df2)


# In[ ]:


# PCA explained variance in each dimension
_, ax = plt.subplots(figsize = (10,5))
ax.plot(pca1.explained_variance_ratio_, color = 'b')
ax.plot(pca2.explained_variance_ratio_, color = 'r')


# In[ ]:


# We take the data frame with the data represented in the principal components
principalDf1 = pd.DataFrame(data = principalcomponents1, columns = ['1 - principal component {0}'.format(i+1) for i in range(5)])
principalDf2 = pd.DataFrame(data = principalcomponents2, columns = ['2 - principal component {0}'.format(i+1) for i in range(5)])


# ## Linear Regression

# We fit a linear regression model for each case.

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


linreg1 = LinearRegression()
linreg2 = LinearRegression()

linreg1.fit(principalDf1, y)
linreg2.fit(principalDf2, y)


# In[ ]:


linreg1.coef_, linreg2.coef_


# The models acually fit our data very differently!

# #### We now make prediction with each model and test the results.

# In[ ]:


df_test_1 = df_test.copy()
df_test_2 = df_test.copy()

y_test = df_test.y

df_test_2.drop('y', axis = 1, inplace = True)


# In[ ]:


principalcomponents_test_1 = pca1.transform(df_test_1)
principalcomponents_test_2 = pca2.transform(df_test_2) # Transforming our data with PCA

# Constructing our data frames.
principalDf1_test = pd.DataFrame(data = principalcomponents_test_1, columns = ['1 - principal component {0}'.format(i+1) for i in range(5)])
principalDf2_test = pd.DataFrame(data = principalcomponents_test_2, columns = ['2 - principal component {0}'.format(i+1) for i in range(5)])

predict_test_1 = linreg1.predict(principalDf1_test)
predict_test_2 = linreg2.predict(principalDf2_test) # Making our predicitions in each case


# #### Final tests and metrics

# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


e1 = mean_squared_error(predict_test_1, y_test)
e2 = mean_squared_error(predict_test_2, y_test)

e1, e2


# Now, note:

# In[ ]:


r = 0
for i in range(5):
    r = r + np.std(df_test_2['Variable {0}'.format(i+1)])**2


# In[ ]:


r


# The mean squared error of the second model approximates the sum of the variances of the features with least variance (which by the structure of our data, correspond to the first features). This means that error for the second model is exactly what we should expect, since we selected just 5 dimensions for the PCA, meaning that the features used to train the second models can't capture the smaller variance in the first features.
# 
# Now, we get a unexpectedly good result for the first model. This most likely means that the first features of the PCA encoded most of the information about the target variable since it has the largest variance. 

# In[ ]:





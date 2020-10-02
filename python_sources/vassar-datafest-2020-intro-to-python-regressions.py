#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries and Read in Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Read in Data
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# In[ ]:


df = df.sample(10000, random_state=42)


# In[ ]:


# Look at first 5 lines of data
df.head()


# ### Simple Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr = LinearRegression()

X = df[['number_of_reviews']]
y = df[['price']]
lr.fit(X,y) # train the model


# In[ ]:


# Coefficient(s)
lr.coef_


# If you add more features to X and then run LinearRegression(), that becomes Mutlilnear Regression (MLR)

# ### OLS Regression

# In[ ]:


import statsmodels.api as sm

X = df[['number_of_reviews']].values
y = df[['price']].values

# fit a OLS model with intercept
X = sm.add_constant(X)
mod = sm.OLS(y, X).fit()

print(mod.summary())


# ### Categorical Variables...? Encoding...?

# In Python (unlike R or STATA), you will get an error if you try to run a regression without encoding categorical variables. "Encoding" means the process of converting categorical variables into some form of vector so that the program / software can understand and be able to process the regression

# In[ ]:


from sklearn.preprocessing import LabelEncoder

lblenc = LabelEncoder()

df['neighbourhood_group_enc'] = lblenc.fit_transform(df['neighbourhood_group'].ravel())


# In[ ]:


# Dummy Variables!

borough_dummy = pd.get_dummies(df[['neighbourhood_group']], drop_first=True)


# In[ ]:


lr2 = LinearRegression()

X2 = pd.concat([ df[['number_of_reviews']], borough_dummy ], axis=1)
y2 = df[['price']]
lr2.fit(X2,y2) # train the model


# In[ ]:


lr2.coef_


# ### How do I evaluate my model?

# In[ ]:


from sklearn.model_selection import train_test_split

X = df[['number_of_reviews']].values
y = df[['price']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[ ]:


lr3 = LinearRegression()

lr3.fit(X_train, y_train)


# residual = real actual value - predicted

# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


# Mean Squared Error (MSE)
mean_squared_error(lr3.predict(X_test), y_test)


# In[ ]:


# RMSE
np.sqrt(mean_squared_error(lr3.predict(X_test), y_test))


# Other more rigorous Methods:
# 
# **Cross Validation**

# In[ ]:


# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import mean_squared_error, median_absolute_error, mean_absolute_error, r2_score
# from sklearn.model_selection import KFold


# ### Next steps...

# There are various kind of regression algorithms already implemented on Python's Sklearn! Look up blog posts, watch online tutorials and read SKlearn's official documentation to master them all!
# 
# https://scikit-learn.org/stable/user_guide.html

# - Regularized Linear Regressions: Ridge, Lasso, Elastic Net
# - Tree/Forest Algorithms: Decision Tree, Random Forest
# - Other: Support Vector Regression (SVR), K-Nearest Neighbors Regressor(KNN)
# 
# AND SO MUCH MORE!

# Overfitting: You are training you model to closely for the training set but for new unseen data, your predictive ability for the model you created is very weak

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Hai kaggler, in this kernel i'm going to perform prediction on california housing Dataset. I'm using Linear Regression, Random Forest Regressor and Decision Tree Regressor. Let's get started.

# ## Import Modules

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import inv_boxcox
from scipy.stats import boxcox


# ## Quick Look
# 
# Take a quick look at the dataset

# In[ ]:


df = pd.read_csv('../input/california-housing-prices/housing.csv')
df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# ## Fill Missing Value
# 
# We got a few missing value on total_rooms columns, let's fill the missing value.

# In[ ]:


df = df.fillna(method='ffill')
df.isnull().sum()


# In[ ]:


df.ocean_proximity.value_counts()


# ## Feature Engineering
# 
# Performing feature engineering to change categorical value to numerical value

# In[ ]:


df = pd.get_dummies(df)
df.head()


# In[ ]:


#Correlation map to see how features are correlated with SalePrice
Cor_heat = df.corr()
plt.figure(figsize=(16,16))
sns.heatmap(Cor_heat, cmap = "RdBu_r", vmax=0.9, square=True)


# The visualization above is correlation between every columns, it's interesting how total_rooms, total_bedrooms, household, and population are strongly correlated. The target column is median house value, let's see what most important feature we have.

# In[ ]:


## Lets see what most important features we have

IF = Cor_heat['median_house_value'].sort_values(ascending=False).head(10).to_frame()
IF.head(5)


# median house value is strongly correlated with median income, let's see what most important feature for median income.

# In[ ]:


IF = Cor_heat['median_income'].sort_values(ascending=False).head(10).to_frame()
IF.head(5)


# ## Transformation
# 
# Before performing prediction, we need to make sure the target dataset is not skewed.

# In[ ]:


# Basic Distribution

print('Skew Value : ' + str(df.median_house_value.skew()))
sns.distplot(df.median_house_value)


# 0.97 is pretty bad, let's see if we can find a better skew value.

# In[ ]:


f = plt.figure(figsize=(16,16))

# log 1 Transform
ax = f.add_subplot(221)
L1p = np.log1p(df.median_house_value)
sns.distplot(L1p,color='b',ax=ax)
ax.set_title('skew value Log 1 transform: ' + str(np.log1p(df.median_house_value).skew()))

# Square Log Transform
ax = f.add_subplot(222)
SRT = np.sqrt(df.median_house_value)
sns.distplot(SRT,color='c',ax=ax)
ax.set_title('Skew Value Square Transform: ' + str(np.sqrt(df.median_house_value).skew()))

# Log Transform
ax = f.add_subplot(223)
LT = np.log(df.median_house_value)
sns.distplot(LT, color='r',ax=ax)
ax.set_title('Skew value Log Transform: ' + str(np.log(df.median_house_value).skew()))

# Box Cox Transform
ax = f.add_subplot(224)
BCT,fitted_lambda = boxcox(df.median_house_value,lmbda=None)
sns.distplot(BCT,color='g',ax=ax)
ax.set_title('Skew Value Box Cox Transform: ' + str(pd.Series(BCT).skew()))


# We have box cox transformation as the best transformation with skew value near to 0, but to make sure let's perform prediction on all of them

# ## Spliting Data

# In[ ]:


Train = df.drop('median_house_value', axis=1)
Test = df.median_house_value


# In[ ]:


# Assign the distribution of Sale Price

feature_SP = {'Log Transform': LT,
              'Square Root Transform': SRT,
              'Box-Cox Transform':BCT,
              'Log 1 Transform': L1p}


# ## Modeling
# 
# I'm using Linear Regression, Decision Tree Regressor, and Random Forest Regressor.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# In[ ]:


## Linear Regression

from sklearn.linear_model import LinearRegression
LR = LinearRegression()


# In[ ]:


## Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=25,max_depth=5)


# In[ ]:


## Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
DTR =DecisionTreeRegressor(random_state=10)


# In[ ]:


alg = [LR, RFR, DTR]
predicted_value = {}

for y in alg:
    print(str(y.__class__.__name__) + ' results')
    for key, value in feature_SP.items():
        Test = value
        X_train, X_test, y_train, y_test = train_test_split(Train, Test, test_size=0.33, random_state=42)
        y.fit(X_train, y_train)
        predicted_value[str(y.__class__.__name__) + ' with ' + str(key)] = y.predict(X_test)
        score = y.score(X_test, y_test)
        print('Accuracy with ' + str(key) + ' transformation : ' + str(np.mean(score)))


# In[ ]:


predicted_value


# ## End
# 
# The best result is when we using Decision Tree Regressor and applying Log Transformation on the target data.
# that is all for this kernel, thank you.

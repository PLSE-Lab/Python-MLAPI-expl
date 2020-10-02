#!/usr/bin/env python
# coding: utf-8

# > *Please provide feedback and suggestions and help me to improve.*
# If any other point could be concluded then share it in comments section.
# Leave an upvote to encourage me.

# 
# **About the dataset:** Columns:
# 
# * age: age of primary beneficiary
# 
# * sex: insurance contractor gender, female, male
# 
# * bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
# objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
# 
# * children: Number of children covered by health insurance / Number of dependents
# 
# * smoker: Smoking
# 
# * region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# 
# * charges: Individual medical costs billed by health insurance

# In[ ]:


#   importing libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/insurance/insurance.csv')


# In[ ]:


# Getting essence of our data!
data.info()


# There is no missing value in this dataset which is really rare in practical world.

# In[ ]:


data.head()


# ## EDA

# In[ ]:


sns.set()


# In[ ]:


sns.countplot(data.region);


# In[ ]:


sns.countplot(data.sex);


# Our data is quite balanced with respect to sex and region features.

# In[ ]:


sns.countplot(data.smoker);


# There are very less smokers as compared to non-smokers.

# In[ ]:


sns.jointplot(data.bmi,data.charges);


# There is not clear linear relationship between target variable 'charges' and feature 'bmi'.
# 
# Let's explore our target variable which is _charges_. Let's create a histogram to see if the target variable is Normally distributed. If we want to create any linear model, it is essential that the features are normally distributed. This is one of the assumptions of multiple linear regression.

# In[ ]:


sns.distplot(data.charges);


# In[ ]:


sns.boxplot(data.charges);


# Target variable is not normally distributed.Let's apply logarithmic transformation to solve this problem.

# In[ ]:


data.charges = np.log(data.charges)


# In[ ]:


plt.figure(figsize=(6,6))
sns.distplot((data.charges));


# In[ ]:


sns.boxplot(data.charges);


# Seems better.

# In[ ]:


sns.boxplot(x='sex',y='charges',data=data)


# In[ ]:


sns.regplot(data.bmi,data.charges)


# Assumption of homoscedasticity seems satisfied.
# 
# ### Converting categorical columns into numerical ones using dummy variables

# In[ ]:


data.children.value_counts()


# In[ ]:


dummies = pd.get_dummies(data[['sex','smoker','region']],drop_first=True)
dummies.head()


# In[ ]:


df_dummies = pd.concat([data,dummies],axis=1)


# In[ ]:


df_dummies.drop(['sex','smoker', 'region','charges'],axis=1,inplace=True)


# In[ ]:


df_dummies.head(3)


# Now, our dataset is ready for regression.
# 
# ### Splitting data into train and test sets

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X=df_dummies
y=data.charges
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# ## Applying regression using sklearn

# In[ ]:


# importing 
from sklearn.linear_model import LinearRegression

lm=LinearRegression()
lm.fit(X_train,y_train)
pred_lm = lm.predict(X_test)

# Our predictions
plt.scatter(y_test,pred_lm)

# Perfect predictions
plt.plot(y_test,y_test,'r');


# ### Evaluating the model

# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score


# In[ ]:


mean_absolute_error(y_test,pred_lm)


# In[ ]:


np.sqrt(mean_squared_error(y_test,pred_lm))


# In[ ]:


explained_variance_score(y_test,pred_lm)


# In[ ]:


r2_score(y_test,pred_lm)


# So, this model explains 79% variance of the target variable.

# ## Applying regression using statsmodel

# In[ ]:


import statsmodels.api as sm


# In[ ]:


results = sm.OLS(y,X).fit()
results.summary()


# **CONCLUSION :**
# 
# * R-squared (0.977) which is close to 1 implies that our regression line explains a good amount of variation of y.
# * A predictor that has a low p-value is likely to be a meaningful addition to our model because changes in the predictor's value are related to changes in the response variable. Seeing p-values in table we can conclude that all our variables are significant except 'region_southeast' which is infact surprising.
# * Durbin-Watson test suggests that there is negligible autocorrelation as it is close to 2. Assumption of autocorrelation is also satisfied.

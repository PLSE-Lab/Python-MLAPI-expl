#!/usr/bin/env python
# coding: utf-8

# # Testing different models based on the preprocessed dataset
# 
# Hi all, I'm continuing my slighly different approach to the housing problem, which I began with the kernel [New approach to EDA and feature creation](https://www.kaggle.com/pretorias/new-approach-to-eda-and-feature-creation). This kernel imports the training dataset obtained after the preprocessing. Note that I have skipped applying the transformations to the test dataset, as the goal of both of my kernels is to try new ideas and learn Python commands. You will have to make some modifications if your ultimate goal is to make a submission.
# 
# In this kernel I will fit different common models (without ensembling) and I will compare their cross-validated performance. After that I will optimize the hyper-parameters of one or two promising models.
# 
# Lets first load the common data analysis libraries, we will load modelling libraries as we go.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[30]:


from sklearn.model_selection import KFold, cross_val_score, train_test_split

#Validation function
def rmsle_cv(X, y, model, n_folds = 10):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# # GLM model
# 
# We will begin with GLMs, that are simple, flexible and powerful and thus a good starting point.
# 
# I have seen that unfortunately the integration of GLMs in scikit is a bit poor, therefore we will calculate the cross-validated RMSE by hand. Luckily it is quite easy!
# 
# We will begin with a GLM with a normal probability distribution and the identity as link function, i.e. an ordinary linear model.

# In[ ]:


import statsmodels.api as sm
train = pd.read_csv('../input/preprocessed_training.csv')
print('Feature types: {}'.format(train.dtypes.unique())) # all features are numerical

price = train['SalePrice']
train.drop('SalePrice', axis=1, inplace=True);


# In[ ]:


kf = KFold(n_splits=10, random_state=42, shuffle=True)
print(kf)  

results = []

for train_index, test_index in kf.split(train):
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    y_train, y_test = price.iloc[train_index], price.iloc[test_index]
    glm_gaussian = sm.GLM(y_train, X_train)
    mod = glm_gaussian.fit()
    pred_test = mod.predict(X_test)
    results.append(np.sqrt(np.mean((pred_test-y_test)**2)))
    
print("GLM score: {:.4f} ({:.4f})\n".format(np.mean(results), np.std(results)))


# The RMSE is quite high! We will get better results using the log(x+1) transformation, as discussed in the EDA kernel.

# In[32]:


log_price = np.log1p(price)

kf = KFold(n_splits=10, random_state=42, shuffle=True)
print(kf)  

results = []

for train_index, test_index in kf.split(train):
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    y_train, y_test = log_price.iloc[train_index], log_price.iloc[test_index]
    glm_gaussian = sm.GLM(y_train, X_train)
    mod = glm_gaussian.fit()
    pred_test = mod.predict(X_test)
    results.append(np.sqrt(np.mean((pred_test-y_test)**2)))
    
print("GLM score: {:.4f} ({:.4f})\n".format(np.mean(results), np.std(results)))


# We obtain a value similar to the RMSE from more complex models in other kernels. It is possible to obtain better results by ensembling, but we will not explore it here.
# 
# But what does a rmse of 0.11 represent? 

# In[ ]:


(np.exp(0.1168)-1)*100


# We are speaking about a 12% error "on average" (larger errors have a bigger weight in the RMSE). Just out of curiosity lets calculate the MAE (mean absolute error), which is easier to interpret.

# In[17]:


from sklearn.metrics import mean_absolute_error

kf = KFold(n_splits=10, random_state=42, shuffle=True)
print(kf)  

results = []

for train_index, test_index in kf.split(train):
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    y_train, y_test = log_price.iloc[train_index], log_price.iloc[test_index]
    glm_gaussian = sm.GLM(y_train, X_train)
    mod = glm_gaussian.fit()
    pred_test = mod.predict(X_test)
    results.append(mean_absolute_error(pred_test, y_test))
    
print("GLM score: {:.4f} ({:.4f})".format(np.mean(results), np.std(results)))
print("Mean percentage error: {:.4f}\n".format((np.exp(np.mean(results))-1)*100))


# To evaluate the goodness of fit, lets fit a model with all the data and analyze the residuals. Keep in mind that a good model fit doesn't automatically translate to good predictions (that's why we crossvalidate).

# In[25]:


glm_gaussian = sm.GLM(log_price, train)
mod = glm_gaussian.fit()
pred_test = mod.predict(train)

fig, ax = plt.subplots()
ax.scatter(x=log_price, y=log_price - pred_test)
plt.ylabel('Residuals', fontsize=13)
plt.xlabel('log(y)', fontsize=13)
ax.axhline(y=0)
plt.title('Residual plot')
plt.show()

sns.distplot(log_price - pred_test)
plt.title('Residual distribution')


# The residuals look quite good, there are some extreme values, especially for the cheaper houses, but the distribution looks normal.

# # Gradient Boosting Regression
# 
# Lets train a more complex model for comparison. I will use the same parameters used in the kernel of Serigne.

# In[33]:


from sklearn.ensemble import GradientBoostingRegressor

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[31]:


score = rmsle_cv(train, log_price, GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# Almost no difference in comparison with the linear model!

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(train)
scaled_df = pd.DataFrame(scaled_df, columns=train.columns)

glm_gaussian = sm.GLM(price, scaled_df)
mod = glm_gaussian.fit()
mod.summary()


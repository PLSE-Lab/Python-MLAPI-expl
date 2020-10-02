#!/usr/bin/env python
# coding: utf-8

# We use a part of Alexandru Papiu's notebook

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


# ## Data preprocessing: ##

# We're not going to do anything fancy here:
# First I'll transform the skewed numeric features by taking log(feature + 1) - this will make the features more normal
# Create Dummy variables for the categorical features
# Replace the numeric missing values (NaN's) with the mean of their respective columns
# In [5]:

# In[ ]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()


# In[ ]:


#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# In[ ]:


all_data = pd.get_dummies(all_data)


# In[ ]:


#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())


# In[ ]:


#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


# ## Models ##

# In[ ]:


from sklearn.linear_model import Ridge, RidgeCV, LassoCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 3))
    return(rmse)


# **Gradient Boosting Regressor**

# In[ ]:


n_estimators = [150, 170 , 200, 400, 500]
cv_rmse_gb = [rmse_cv(GradientBoostingRegressor(n_estimators = n_estimator)).mean() 
            for n_estimator in n_estimators]


# In[ ]:


print (cv_rmse_gb)


# In[ ]:


cv_gb = pd.Series(cv_rmse_gb , index = n_estimators)
cv_gb.plot(title = "Validation Gradient Boosting")
plt.xlabel("n_estimator")
plt.ylabel("rmse")


# In[ ]:


cv_gb.min()


# In[ ]:


model_gb = GradientBoostingRegressor(n_estimators = 400).fit(X_train, y)


# In[ ]:


#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds_gb = pd.DataFrame({"preds Gradient Boost":model_gb.predict(X_train), "true":y})
preds_gb["residuals"] = preds_gb["true"] - preds_gb["preds Gradient Boost"]
preds_gb.plot(x = "preds Gradient Boost", y = "residuals",kind = "scatter")


# **Ridge**

# In[ ]:


alphas_ridge = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_rmse_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas_ridge]


# In[ ]:


print (cv_rmse_ridge)


# In[ ]:


cv_ridge = pd.Series(cv_rmse_ridge, index = alphas_ridge)
cv_ridge.plot(title = "Validation Ridge")
plt.xlabel("alphas")
plt.ylabel("rmse")


# In[ ]:


cv_ridge.min()


# In[ ]:


model_ridge = Ridge(alpha = 5).fit(X_train, y)


# In[ ]:


#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds_ridge = pd.DataFrame({"preds Ridge":model_ridge.predict(X_train), "true":y})
preds_ridge["residuals"] = preds_ridge["true"] - preds_ridge["preds Ridge"]
preds_ridge.plot(x = "preds Ridge", y = "residuals",kind = "scatter")


# **Lasso**

# In[ ]:


model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)


# In[ ]:


cv_rmse_lasso = rmse_cv(model_lasso).mean()


# In[ ]:


print (cv_rmse_lasso)


# In[ ]:


cv_lasso = pd.Series(model_lasso.coef_, index = X_train.columns)
cv_lasso.plot(title = "Validation Lasso")
plt.xlabel("coef")
plt.ylabel("rmse")


# In[ ]:


#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds_lasso= pd.DataFrame({"preds Lasso":model_lasso.predict(X_train), "true":y})
preds_lasso["residuals"] = preds_lasso["true"] - preds_lasso["preds Lasso"]
preds_lasso.plot(x = "preds Lasso", y = "residuals",kind = "scatter")


# **xgboost model**

# In[ ]:


import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":6, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)


# In[ ]:


model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()


# In[ ]:


model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=6, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)


# **Prediction**

# In[ ]:


gb_preds = np.expm1(model_gb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))
ridge_preds = np.expm1(model_ridge.predict(X_test))
xgb_preds = np.expm1(model_xgb.predict(X_test))


# In[ ]:


preds = 0.5*lasso_preds + 0.25*ridge_preds + 0.25*xgb_preds


# In[ ]:


solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("lasso_ridge_xgb_v2.csv", index = False)


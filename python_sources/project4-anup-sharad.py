#!/usr/bin/env python
# coding: utf-8

# In[99]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[100]:


get_ipython().system('pip install regressors')


# In[101]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.formula.api as sm
import statsmodels.sandbox.tools.cross_val as cross_val
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model as lm
from regressors import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, LeaveOneOut

print(os.listdir("../input"))


# In[102]:


df = pd.read_csv('../input/train.csv')
df.columns


# Through our initial data exploration, we came up with following variables that we think are most influential while determining Sale price :
# **Neighborhood, OverallQual, GrLivArea, GarageArea, TotalBsmtSF, FullBath, YearBuilt, HouseStyle, TotRmsAbvGrd**

# In[103]:


cols = ['SalePrice','Neighborhood', 'OverallQual','GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt','HouseStyle','TotRmsAbvGrd']
df_train = df[cols]
df_train.head()


# In[104]:


sns.pairplot(df_train)


# In[105]:


df_train.corr()


# In[106]:


print("Check for NaN/null values:\n",df_train.isnull().values.any())
print("Number of NaN/null values:\n",df_train.isnull().sum())


# **Part A - Interactions between variables**

# In[107]:


main = sm.ols(formula="SalePrice ~ (OverallQual*GrLivArea)+(YearBuilt*OverallQual)+(Neighborhood*HouseStyle*TotalBsmtSF)+(TotalBsmtSF*TotRmsAbvGrd)+(FullBath*TotRmsAbvGrd*GarageArea)",data=df_train).fit()
print(main.summary())


# Findings :
# * Interaction between TotalBsmtSF, TotRmsAbvGrd gives better p-value than individual ones.
# * Some of the interactions between Neighborhood, HouseStyle and TotalBsmtSF have good p-values, so we can keep the Neighborhood, HouseStyle and TotalBsmtSF interaction.
# * Interaction between OverallQual, GrLivArea gives better p-value than individual ones.
# * Interaction between YearBuilt, OverallQual doesn't better the p-value than individual effects, so we can remove this interaction and add only main effects of the involved variables.
# * Interaction between FullBath, TotRmsAbvGrd and GarageArea improves p-value than individual ones.

# **Simplified Model**

# In[108]:


main = sm.ols(formula="SalePrice ~ (OverallQual*GrLivArea)+(Neighborhood*HouseStyle*TotalBsmtSF)+(TotalBsmtSF*TotRmsAbvGrd*GarageArea)",data=df_train).fit()
print(main.summary())


# **Part B - Numerical Transformations**

# In[120]:


main = sm.ols(formula="SalePrice ~ OverallQual+I(OverallQual*OverallQual)+I(OverallQual*OverallQual*OverallQual)",data=df_train).fit()
print(main.summary())


# * Cubic transformation of OverallQual improved the p-value.

# In[119]:


main = sm.ols(formula="SalePrice ~ GrLivArea+I(GrLivArea*GrLivArea)+I(GrLivArea*GrLivArea*GrLivArea)",data=df_train).fit()
print(main.summary())


# * Cubic transformation for GrLivArea improved p-value.

# **Part C - Forward Selection**

# In[113]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs


# In[121]:


df_dum = pd.get_dummies(df_train)
df_dum.head()


# In[122]:


df_dum.columns


# In[146]:


inputDF = df_dum[['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF',
       'FullBath', 'YearBuilt', 'TotRmsAbvGrd', 'Neighborhood_Blmngtn',
       'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide',
       'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor',
       'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR',
       'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes',
       'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge',
       'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU',
       'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst',
       'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker',
       'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_1Story',
       'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story',
       'HouseStyle_SFoyer', 'HouseStyle_SLvl']]
outputDF = df_dum[["SalePrice"]]

model_fwd = sfs(LinearRegression(),k_features=8,forward=True,verbose=2,cv=8,n_jobs=-1,scoring='r2')
model_fwd.fit(inputDF,outputDF)


# In[147]:


model_fwd.k_feature_names_


# **Part D - Backward Model Selection**

# In[144]:


model_bkd = sfs(LinearRegression(),k_features=8,forward=False,verbose=2,cv=8,n_jobs=-1,scoring='r2')
model_bkd.fit(inputDF,outputDF)


# In[145]:


model_bkd.k_feature_names_


# **Part E - Cross validation**

# In[133]:


from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[165]:


inputDF_looc = df_dum[['OverallQual',
 'GrLivArea',
 'GarageArea',
 'YearBuilt',
 'Neighborhood_NoRidge',
 'Neighborhood_NridgHt',
 'Neighborhood_StoneBr',
 'HouseStyle_1Story']]
outputDF_looc = df_dum[["SalePrice"]]
model_fwd_sfs = LinearRegression()
loocv = LeaveOneOut()

rmse = np.sqrt(-cross_val_score(model_fwd_sfs, inputDF_looc, outputDF_looc, scoring="neg_mean_squared_error", cv = loocv))
print(rmse.mean())


# In[168]:


kf = KFold(5, shuffle=True, random_state=42).get_n_splits(inputDF_looc)
rmse = np.sqrt(-cross_val_score(model_fwd_sfs, inputDF_looc, outputDF_looc, scoring="neg_mean_squared_error", cv = kf))
print(rmse.mean())


# In[169]:


kf = KFold(10, shuffle=True, random_state=42).get_n_splits(inputDF_looc)
rmse = np.sqrt(-cross_val_score(model_fwd_sfs, inputDF_looc, outputDF_looc, scoring="neg_mean_squared_error", cv = kf))
print(rmse.mean())


# In[166]:


inputDF_looc_bkd = df_dum[['OverallQual',
 'GrLivArea',
 'GarageArea',
 'Neighborhood_NAmes',
 'Neighborhood_NWAmes',
 'Neighborhood_OldTown',
 'Neighborhood_SWISU',
 'HouseStyle_1Story']]
outputDF_looc_bkd = df_dum[["SalePrice"]]
model_bkd_sfs = LinearRegression()
loocv = LeaveOneOut()

rmse = np.sqrt(-cross_val_score(model_bkd_sfs, inputDF_looc_bkd, outputDF_looc_bkd, scoring="neg_mean_squared_error", cv = loocv))
print(rmse.mean())


# In[170]:


kf = KFold(5, shuffle=True, random_state=42).get_n_splits(inputDF_looc_bkd)
rmse = np.sqrt(-cross_val_score(model_bkd_sfs, inputDF_looc_bkd, outputDF_looc_bkd, scoring="neg_mean_squared_error", cv = kf))
print(rmse.mean())


# In[171]:


kf = KFold(10, shuffle=True, random_state=42).get_n_splits(inputDF_looc_bkd)
rmse = np.sqrt(-cross_val_score(model_bkd_sfs, inputDF_looc_bkd, outputDF_looc_bkd, scoring="neg_mean_squared_error", cv = kf))
print(rmse.mean())


# **Part E - Final Model**

# In[179]:


inputDF_final = df_dum[['OverallQual',
 'GrLivArea',
 'GarageArea',
 'YearBuilt',
 'Neighborhood_NoRidge',
 'Neighborhood_NridgHt',
 'Neighborhood_StoneBr',
 'HouseStyle_1Story']]
outputDF_final = df_dum[["SalePrice"]]
model_final = LinearRegression()
loocv = LeaveOneOut()
rmse = np.sqrt(-cross_val_score(model_final, inputDF_final, outputDF_final, scoring="neg_mean_squared_error", cv = loocv))
print(rmse.mean())


# In[181]:


model_lm = lm.LinearRegression()
results = model_lm.fit(inputDF_final,outputDF_final)

print("P-value:\n",stats.coef_pval(model_lm, inputDF_final, outputDF_final))
print("Adjusted R-Squared:\n",stats.adj_r2_score(model_lm, inputDF_final, outputDF_final))


# Model from the SFS forward selection method gave R-Squared of 0.790

# In[185]:


main = sm.ols(formula="SalePrice ~ I(OverallQual*OverallQual*OverallQual)+I(GrLivArea*GrLivArea)+(OverallQual*GrLivArea)+(Neighborhood*HouseStyle*TotalBsmtSF)+(TotalBsmtSF*TotRmsAbvGrd*GarageArea)",data=df_train).fit()
print(main.summary())


# The statmodel built above provides better R-Squared than SFS forward selection model. The final model that we want to propose is Statmodel stated above.

# * Was the model what you would have expected based on your original understanding of data? Were there any surprise findings?
# - It is inline with expectations that we had come up after the initial data exploration. Using interactions, we think, made it more accurate than using only main effects only model.
# 
# * Based on this new model, would you make any changes to your Validation strategy proposed in your previous project deliverable?
# - The validation strategy was to verify predicted values of Sale price with actual sale values happened in the area. We believe there won't be any change in that strategy.

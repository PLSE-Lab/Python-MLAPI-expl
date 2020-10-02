#!/usr/bin/env python
# coding: utf-8

# 1. Imports
# ----------
# 

# In[ ]:


import numpy as np 
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor   

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
combined = pd.concat((train,test))


# 2. Data Perusal
# ---------------

# In[ ]:


print(train.shape)
print(test.shape)
print(combined.shape)


# In[ ]:


combined.head()


# In[ ]:


combined.describe()


# In[ ]:


print(combined.isnull().sum())


# In[ ]:


combined.info()


# In[ ]:


#Lists the amount of unique features in each object column
a_temp = []
for col in combined:
	if combined[col].dtype == 'O':
		a_temp.append(len(combined[col].unique()))
print(a_temp)


# 3. Feature Manipulation
# ---------------

# In[ ]:


#Changes all object columns with two unique variables into binary
le = LabelEncoder()
for col in combined:
    if len(combined[col].unique()) == 2 and combined[col].dtype == 'O':
        combined[col] = le.fit_transform(combined[col])
        print(col)
        print(list(le.classes_))
        print([0, 1])


# In[ ]:


#Create dummies for all object columns with >2 unique variables
#combined = pd.get_dummies(combined)
# ^ is easier but I wanted to preserve the column names as a prefix
for col in combined:
    if combined[col].dtype == 'O':
    	pd.get_dummies(combined[col], prefix=col)
    	combined = pd.concat([combined, pd.get_dummies(combined[col], prefix=col)],axis=1)
    	combined = combined.drop([col], 1)


# In[ ]:


print(combined.loc[:,combined.isnull().any()].isnull().sum())
combined = combined.drop(['SalePrice'],1)
nancolumns = combined.loc[:,combined.isnull().any()].columns


# In[ ]:


#Feature Assumption 1
#Seems likely that these people don't have basements so I'll fill with 0
print(combined.loc[(combined['BsmtFullBath'].isnull()),nancolumns])
combined.loc[(combined['BsmtFullBath'].isnull()),:] = combined.loc[(combined['BsmtFullBath'].isnull()),:].fillna(0)


# In[ ]:


#Feature Assumption 2
#Seems likely that these people don't have a garage so I'll fill with 0
print(combined.loc[(combined['GarageArea'].isnull()),nancolumns])
combined.loc[(combined['GarageArea'].isnull()),:] = combined.loc[(combined['GarageArea'].isnull()),:].fillna(0)

print(combined.loc[:,combined.isnull().any()].isnull().sum())


# In[ ]:


#Check the vif score to see if any of the columns have high multicollinearity
nancolumns = combined.loc[:,combined.isnull().any()].columns
vif = []
for x in nancolumns:
	vif.append(variance_inflation_factor(combined.dropna().values, combined.columns.dropna().get_loc(x)))
print(vif)


# In[ ]:


#Feature Assumption 3
#Since GarageYrBlt has a large vif, I will drop the whole column
combined.drop('GarageYrBlt', inplace=True, axis=1)

print(combined.loc[:,combined.isnull().any()].isnull().sum())


# In[ ]:


#Feature Assumption 4
#Since no one in this group has any data on the MasVnrType I will assume the MasVnrArea to be 0
combined.loc[(combined['MasVnrType_None'] == 0), 'MasVnrArea'].isnull().sum()
vnr_mask = ['MasVnrType_None', 'MasVnrType_Stone', 'MasVnrType_BrkFace', 'MasVnrType_BrkCmn']
print(combined.loc[(combined['MasVnrArea'].isnull()), vnr_mask])
combined.loc[(combined['MasVnrArea'].isnull()),:] = combined.loc[(combined['MasVnrArea'].isnull()),:].fillna(0)


# In[ ]:


#Feature Assumption 5
#Imputed the regression yhat for LotFrontage as it had a decent r2_score
#Can change from the linear regression to a stochastic on future update to factor in unexplained variance
testa = combined.loc[(combined['LotFrontage'].isnull()),:]
traina = combined.loc[~(combined['LotFrontage'].isnull()),:]
X = traina.drop('LotFrontage', axis=1)
y = traina['LotFrontage']
linreg = LinearRegression()
linreg_yhat = linreg.fit(X,y).predict(X)
print(r2_score(y,linreg_yhat))
print(r2_score(y,np.full((y.shape),y.median())))
print(r2_score(y,np.full((y.shape),y.mean())))
X2 = testa.drop('LotFrontage', axis=1)
linreg_yhat = linreg.fit(X,y).predict(X2)
testa['LotFrontage'] = linreg_yhat
combined = pd.concat((testa,traina))


# In[ ]:


combined.set_index('Id',drop=True,inplace=True)
train.set_index('Id',drop=True,inplace=True)
test.set_index('Id',drop=True,inplace=True)

combined['SalePrice'] = train['SalePrice']

train = combined.loc[train.index]
test = combined.loc[test.index]


# In[ ]:


print(train.corr()['SalePrice'].nlargest(10))
print(train.corr()['SalePrice'].nsmallest(10))


# In[ ]:


X = train.drop('SalePrice', axis=1)
y = train.SalePrice


# ##4. Models##

# In[ ]:


#ElasticNet as an intial test. 
#Reveals through the l1_ratio that the Lasso Regression is likely the true model.
elastic = ElasticNetCV(cv=10, normalize=True, n_jobs=-1, l1_ratio=np.linspace(0.1,1,10))
elastic_fit = elastic.fit(X,y)
print(elastic.score(X,y))
elastic_coefficients = pd.DataFrame(list(zip(train.columns, elastic_fit.coef_)), 
	columns=['Feature', 'Coefficient'])
elastic_coefficients.sort_values('Coefficient', inplace=True)
print(elastic_coefficients[elastic_coefficients.Coefficient != 0])
print(elastic_fit.l1_ratio_)
print(elastic_fit.alpha_)


# In[ ]:


lasso = LassoCV(cv=10, normalize=True, n_jobs=-1)
lasso_fit = lasso.fit(X,y)
print(lasso.score(X,y))
lasso_coefficients = pd.DataFrame(list(zip(train.columns, lasso_fit.coef_)), 
	columns=['Feature', 'Coefficient'])
lasso_coefficients.sort_values('Coefficient', inplace=True)
print(lasso_coefficients[lasso_coefficients.Coefficient != 0])
print(lasso_fit.alpha_)


# In[ ]:


ridge = RidgeCV(cv=10, normalize=True, fit_intercept=True,alphas=(0.1, 1.0, 10.0, lasso_fit.alpha_))
ridge_fit = ridge.fit(X,y)
print(ridge.score(X,y))
ridge_coefficients = pd.DataFrame(list(zip(train.columns, ridge_fit.coef_)), 
	columns=['Feature', 'Coefficient'])
ridge_coefficients.sort_values('Coefficient', inplace=True)
print(ridge_coefficients[ridge_coefficients.Coefficient != 0])
print(ridge_fit.alpha_)


# ##5. Submission##

# In[ ]:


Xtest = test.drop('SalePrice', 1)
test_yhat = lasso_fit.predict(Xtest)
submit = pd.DataFrame(list(zip(test.index, test_yhat)), columns = ['Id', 'SalePrice'])
submit.to_csv("../working/submit.csv", index=False)
print(submit.head())


# ## Future Updates ##

# In[ ]:


######################################################################################################

# Plot some relevant graphs
# fancyimpute for the stochasitic regression imputation
# Normalize/Standardize data in preprocessing
# Extract RMSE score
# Linear Regression
# Gradient Boost
# Format the prints
# Copy vs Slice repair
# Tidy Up


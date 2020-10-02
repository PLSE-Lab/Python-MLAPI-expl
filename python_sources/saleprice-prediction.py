#!/usr/bin/env python
# coding: utf-8

# My name is Martin C. I am submitting this solution as part of my application at turingly.
# 
# This notebooks is structured as follows:
# - Introduction
# - Data cleaning/preprocessing
# - Modeling/prediction

# ## Introduction
# In this notebook we look at the kaggle problem titled "House Prices: Advanced Regression Techniques". We are given data about houses and are asked to predict the saleprices of the houses. There are 1460 training examples and 1459 test examples. The data has 80 columns (i.e. 80 features), including 
# -    LotArea: Lot size in square feet
# -    Street: Type of road access
# -    Alley: Type of alley access
# -    LotShape: General shape of property
# -    LandContour: Flatness of the property
# -    BldgType: Type of dwelling  
# -    OverallQual: Overall material and finish quality
# -    OverallCond: Overall condition rating
# -    YearBuilt: Original construction date
#   

# ## Data cleaning/preprocessing

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


def tocsv(df, filename='tocsv.csv', index = False):
    df.to_csv(filename, index=index)


# In[ ]:


pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# ### Import data

# In[ ]:


df_train = pd.read_csv('../input/train.csv') 
df_test = pd.read_csv('../input/test.csv')

id_col = df_test['Id']


# In[ ]:


print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
p = df_train.SalePrice.hist()
t = p.set_title("SalePrice distribution")


# In[ ]:


y_train = np.log10(df_train.SalePrice)
X_train = df_train.drop('SalePrice', axis=1)
X_test = df_test
X = pd.concat([X_train, X_test])


# In[ ]:





# ### Dealing with missing data 

# In[ ]:


categoricals = X_train.select_dtypes(include='object').columns
numericals = X_train.select_dtypes(exclude='object').columns
print(f'{len(categoricals)} categorical features')
print(f'{len(numericals)} numerical features')


# Categorical features that have missing data:

# In[ ]:


X[categoricals].isna().sum().sort_values(ascending=False)


# ### Imputation 

# For the categorical features, most of the "missing" values are not actually missing. E.g. PoolQC is NA when there is no pool. That does not mean that data is missing. We replace the "missing" values with the string "absent":

# In[ ]:


X[categoricals] = X[categoricals].fillna("absent")


# In[ ]:


X[numericals].isna().sum().sort_values(ascending=False)


# Now use one-hot encoding for the categorical variables:

# In[ ]:


X = pd.get_dummies(X)


# Now for the rest of the features (numerical) we simply use the mean value. This is far from optimal obviously, but it is simple. 

# In[ ]:


from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy='mean')
imp_mean.fit(X)
X_imp = imp_mean.transform(X)

X_imp = pd.DataFrame(X_imp)


# In[ ]:


X_train_model = X_imp[0:1460]
X_test_model = X_imp[1460:]


# In[ ]:


from sklearn import linear_model
#regr = linear_model.Lasso(alpha=0.1)
regr = linear_model.LinearRegression()
regr = regr.fit(X_train_model, y_train)
out = regr.predict(X_test_model)

out = 10**out

out = pd.DataFrame(out,columns=['SalePrice'])

out.insert(0,"Id", id_col) 
out
tocsv(out)


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
print('Modules loaded')


# In[ ]:


X = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# # Analysing the dependent variable: 'SalePrice'

# In[ ]:


X.SalePrice.describe()


# In[ ]:


sns.distplot(X.SalePrice)


# In[ ]:


print("Skewness: %f" % X.SalePrice.skew())
print("Kurtosis: %f" % X.SalePrice.kurt())


# # Analysing the different independent variables
# 

# ### Dependent variables overview

# In[ ]:


X.columns


# ### Data Leakage: Variables acquired after the sale

# In[ ]:


X = X.drop(['SaleType','SaleCondition','Id','MoSold', 'YrSold'], axis = 1)


# ### Missing Values

# In[ ]:


total = X.isnull().sum().sort_values(ascending = False)
percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis =1, keys=['Total', 'Percent'])
missing_data[(missing_data.Percent > 0)]


# In[ ]:


X = X.drop(missing_data[missing_data.Percent > 0.1].index, axis = 1)


# ### Categorical versus Numerical data

# In[ ]:


numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and 
                    X[cname].dtype == "object"]
my_cols = categorical_cols + numerical_cols
print(categorical_cols,numerical_cols)


# #### Changing the categorical data to numerical data and imputing the left over missing values

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')
# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[ ]:


X_prepro = preprocessor.fit_transform(X)


# In[ ]:


prepro_cols = numerical_cols + preprocessor.named_transformers_.cat.named_steps.onehot.get_feature_names(X[categorical_cols].columns).tolist()


# In[ ]:


X_prepro = pd.DataFrame(data = X_prepro, columns = prepro_cols, index = X.index)


# ### Transformed into numpy array, needs to go back to pd dataframe

# # Analysing the relationship between the variables

# In[ ]:


corrmat = X_prepro.corr()
f, ax = plt.subplots(figsize=(18, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


# ## Variables with highest correlation regarding 'SalePrice'

# In[ ]:


k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(X_prepro[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(18, 9))
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


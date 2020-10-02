#!/usr/bin/env python
# coding: utf-8

# Hi everyone. Below is my first submission to the ever famous housing regression. The apt dataset for beginners into kaggle, I realized a lot of kernels to be full of long codes that intimidated me as a beginner.
# 
# Here I've tried to optimize my code and still providing the best results which I believe will not be intimidating for beginners and learners who are trying to get their first submission. 
# 
# To anyone who views the code - an upvote will be appreciated. You know how motivating it is to keep you going. Feel free to comment out any errors or suggestions without any hesitation.

# # Importing libraries and reading the data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import skew
import math
from sklearn.metrics import mean_squared_error
import sklearn.metrics as sklm
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# Getting the basic information about the data
# 

# In[ ]:


df_train.shape


# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# In[ ]:


df_train.columns


# In[ ]:


#merging the test and train data
frames = [df_train,df_test]
df = pd.concat(frames, keys=['x', 'y'])

df


# # Missing values

# Checking for the number and percentage of missing values

# In[ ]:


total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)


# Filling certain categorical columns about which we have an intuition using the forward fill method 

# In[ ]:


for column in ['Electrical', 'SaleType', 'KitchenQual', 'Exterior1st','Exterior2nd','Functional','Utilities','MSZoning']:
    df[column].fillna(method='ffill',inplace=True)


# Extracting the column names of the numerical and categorical features separately and filling the rest of the missing values.

# In[ ]:


num_features = df.select_dtypes(include=np.number).columns.tolist()
cat_features = df.select_dtypes(exclude=np.number).columns.tolist()
num_features.remove('SalePrice')


# In[ ]:


df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace = True)
df[cat_features] = df[cat_features].fillna("none")
df[num_features] = df[num_features].fillna(0)


# Making sure the no other values are missed. 

# In[ ]:


total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)


# Creating new features as per our intuition and dropping the other columns.

# In[ ]:


df['TotalArea'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GrLivArea'] +df['GarageArea']

df['Bathrooms'] = df['FullBath'] + df['HalfBath']*0.5 

df['Year average']= (df['YearRemodAdd'] + df['YearBuilt'])/2


# # Data Visualization

# Checking the correlation of the numerical features

# In[ ]:


corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,cmap="RdYlGn");


# Zooming up the map to list the top correlations with SalePrice

# In[ ]:


corrmat = df_train.corr()
top_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(df_train[top_features].corr(),annot=True,cmap="RdYlGn")


# Getting the scatterplot for the top correlated features

# In[ ]:


#scatterplot
sns.set()
columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF']
sns.pairplot(df_train[columns], size = 3)
plt.show();


# checking for the outliers and dropping them

# In[ ]:


plt.scatter(df_train.GrLivArea, df_train.SalePrice, c = 'b')
plt.scatter(df_train.TotalBsmtSF, df_train.SalePrice, c = 'g')


# In[ ]:


df.shape


# In[ ]:


#df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df.drop(df[df['Id'] == 1299].index, inplace = True)
df.drop(df[df['Id'] == 524].index, inplace = True)


# # Transforming the data

# Taking a look at the target variable. It looks skewed so we transform it to fit normal distribution

# In[ ]:


sns.distplot(df['SalePrice']);
fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)


# In[ ]:


df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
sns.distplot(df_train['SalePrice']);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# In[ ]:


df['MSSubClass'] = df['MSSubClass'].apply(str)
df['YrSold'] = df['YrSold'].astype(str)


# Checking the skewness of other variables and treating it.

# In[ ]:


df.skew(axis=0).sort_values(ascending= False).head(10)


# In[ ]:


num_features = df.select_dtypes(include=np.number).columns.tolist()
df[num_features] = np.log1p(df[num_features])


# Converting the categorical columns into numerical

# In[ ]:


df1 = pd.get_dummies(df.drop('SalePrice', axis=1))
X_train = df1.xs('x')
X_test = df1.xs('y')


# In[ ]:


df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
y_train = df_train.iloc[:,80]


# In[ ]:


y_train.shape


# # Fitting the Model

# In[ ]:


from sklearn.linear_model import Lasso
import sklearn.model_selection as ms
parameters= {'alpha':[0.0001,0.0009,0.001,0.01,0.1,1,10],
            'max_iter':[100,500,1000]}


lasso = Lasso()
lasso_model = ms.GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=10)
lasso_model.fit(X_train,y_train)

print('The best value of Alpha is: ',lasso_model.best_params_)


# In[ ]:


lasso_mod=Lasso(alpha=0.0009,max_iter = 500)
lasso_mod.fit(X_train,y_train)
y_lasso_train=lasso_mod.predict(X_train)
y_lasso_test=lasso_mod.predict(X_test)
math.sqrt(sklm.mean_squared_error(y_train, y_lasso_train))


# In[ ]:





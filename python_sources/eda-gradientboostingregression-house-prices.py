#!/usr/bin/env python
# coding: utf-8

# ******House prices: Advanced Regression Techniques** 
# 
# Author Notebook: PG
# 
# Description: 
# 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, in this competition challenges we need to predict the final price of each home.
# This is an example of supervised dataset with numerical target.
# The target variable in the testset is the variable named "SalePrice"

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#libraries for visualizations
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999



sns.set_style("white")
import scipy.stats as stats

import numpy as np
from sklearn.impute import SimpleImputer


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor


from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **EXPLORATORY DATA ANALYSIS:**
# 
# 1) Defining Training set and Test set and exploring shape and initial data. Descriptive Statistics on all the variables. 
# 
# 2) Creating hystograms, other graphs and correlation matrix to understand how the variables are correlated. 
# 
# 3) Analysing missing data. 
# 
# 4) Applying machine learning techniques.
# 

# In[ ]:


trainset=pd.read_csv('../input/train.csv')
trainset.shape


# In[ ]:


trainset.info()


# In[ ]:


trainset.head()


# In[ ]:


testset=pd.read_csv('../input/test.csv')
testset.shape


# In[ ]:


testset.head()


# In[ ]:


trainset.describe()


# In[ ]:


testset.describe()


# The datasets have some categorical variables, we are transorming them in numerical in order to use ML on them. Ideally, if we use a non based tree model, we should transform them using one-hot encoding in order to have a binary output for each category. To use one-hot encoding we should first merge trainset and testset, apply one-hot encoding and then split them again. We are leaving this method to future updates of this kernel. 

# In[ ]:


trainset_cat = trainset.select_dtypes(include=['object']).copy()

trainset_cat.head()

lista=list(trainset_cat)
print(lista)


testset_cat = testset.select_dtypes(include=['object']).copy()

testset_cat.head()

lista_test=list(testset_cat)
print(lista_test)

#trainset_onehot =trainset.copy()
#trainset_onehot_f = pd.get_dummies(trainset_onehot, columns=['Neighborhood'], prefix = ['Neighborhood'])

#print(trainset_onehot_f.head())


# Transformation of all the categorical variables in numerical variables with one easy loop.

# In[ ]:


for name in lista:
 trainset[name]=trainset_cat[name].astype('category').cat.codes


for nametest in lista_test:
 testset[nametest]=testset_cat[nametest].astype('category').cat.codes


# In[ ]:


trainset.head()


# Histogram on the target variable "SalePrice"

# In[ ]:


print(trainset['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(trainset['SalePrice'], color='c', bins=100, hist_kws={'alpha': 0.4});


# Target is  right skewed and some outliers are above 500000. We delete these outliers to get a normal distribution for my target variable. 

# In[ ]:


trainset_c=trainset[trainset.SalePrice<500000]


# Histogram of the cleaned trainset (trainset_c), we can see that the records are deleted

# In[ ]:


print(trainset_c['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(trainset_c['SalePrice'], color='c', bins=100, hist_kws={'alpha': 0.4});


# Creating graph showing the highly correlated variable with the target
# 

# In[ ]:


labels = []
values = []
for col in trainset_c.columns:
    if col not in ["Id", "SalePrice"] and trainset_c[col].dtype!='object':
        labels.append(col)
        values.append(np.corrcoef(trainset_c[col].values, trainset_c["SalePrice"].values)[0,1])
corr_df = pd.DataFrame({'columns_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
 
corr_df = corr_df[(corr_df['corr_values']>0.25) | (corr_df['corr_values']<-0.25)]
ind = np.arange(corr_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,6))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='gold')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.columns_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()


# **Missing Values Analysis**

# In[ ]:


missingdata = trainset_c.isnull().sum(axis=0).reset_index()
missingdata.columns = ['column_name', 'missing_count']
missingdata = missingdata.ix[missingdata['missing_count']>0]
missingdata = missingdata.sort_values(by='missing_count')

ind = np.arange(missingdata.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missingdata.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missingdata.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# We can see that the variables with the highest number of missing values don't have an high correlation with the target. We can avoid to consider them. 

# In[ ]:


temp_df = trainset_c[corr_df.columns_labels.tolist()]
corrmat = temp_df.corr(method='pearson')
f, ax = plt.subplots(figsize=(12, 12))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True, cmap="YlOrRd")
plt.title("Correlation Matrix", fontsize=15)
plt.show()


# Some variables are high correlated among themselves. GarageCars and GarageArea, 1stFlrSF and TotalBsmfSF, Fireplaces and Fireplacequ have high correlation. We should think about removing one of each of these groups before applying Machine Learning Techniques. 

# Let's analyse the relationship between some variables having high correlation with our target variable. 

# **Overall quality vs Target variable**

# In[ ]:


trainset_c['OverallQual'].loc[trainset_c['OverallQual']>7] = 7
plt.figure(figsize=(12,8))
sns.violinplot(x='OverallQual', y='SalePrice', data=trainset_c)
plt.xlabel('Overall Quality', fontsize=12)
plt.ylabel('SalePrice', fontsize=12)
plt.show()


# In[ ]:


col = "GrLivArea"
ulimit = np.percentile(trainset_c[col].values, 99.5)
llimit = np.percentile(trainset_c[col].values, 0.5)
trainset_c[col].loc[trainset_c[col]>ulimit] = ulimit
trainset_c[col].loc[trainset_c[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=trainset_c[col].values, y=trainset_c.SalePrice.values, height=10, color=color[4])
plt.ylabel('SalePrice', fontsize=12)
plt.xlabel('GrLivArea', fontsize=12)
plt.title("GrLivArea Vs SalePrice", fontsize=15)
plt.show()


# A clear linear correlation is visible. 

# **Machine Learning Tecniques**
# 
#  Gradient Boosting Regressor using as predictors the variables with the highest correlations with the Target.  We don't consider, for example, GarageArea because (as previously demonstrated) the correlation with GarageCars is suggesting that they contain the same information. 
# Gradient Boosting Regressor is a tree-based model and it doesn't benefit from scaling each feature or from using the one-hot encoding. These techniques should be used in non tree based models. 
#         
# 

# Replacing missing value using Method Parameter in order to apply Machine Learning Techniques

# In[ ]:


trainset_c.fillna( method ='ffill', inplace = True)


# In[ ]:



testset.fillna( method ='ffill', inplace = True)


# Train the model 

# In[ ]:


trainset_y = trainset_c.SalePrice
x_col = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt','YearRemodAdd','TotRmsAbvGrd','Fireplaces','Foundation','BsmtFinSF1','OpenPorchSF','WoodDeckSF','GarageCond','2ndFlrSF','HalfBath','LotArea','LotShape','GarageFinish','HeatingQC','BsmtQual','KitchenQual','ExterQual']

trainset_x = trainset_c[x_col]



my_model = GradientBoostingRegressor()
my_model.fit(trainset_x, trainset_y)


# Test the model and create prediction, a good ideas is to use crossvalidation to improve our prediction. We will leave this method to future updates of this kernel.

# In[ ]:


testset_X = testset[x_col]
testset_X.head()
# Use the model to make predictions
predicted_prices = my_model.predict(testset_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)


# In[ ]:


my_submission = pd.DataFrame({'Id': testset.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


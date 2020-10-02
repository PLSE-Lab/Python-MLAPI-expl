#!/usr/bin/env python
# coding: utf-8

# # **Please Consider Upvote if you like this Kernel.**

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sub=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


train.columns


# # Analysing SalePrice

# In[ ]:


train['SalePrice'].describe()


# In[ ]:


plt.figure(figsize=(14,8))
px.histogram(train,x='SalePrice',nbins=80,title='Selling Price Distribution')


# > *Above visual shows that*
# > * There is positive skewness
# > * The plot is not normally distributed

# # Exploring SalePrice with other Variables

# > *Relationship with 'GrLivArea'*

# In[ ]:


plt.figure(figsize=(14,8))
px.scatter(train,x='GrLivArea',y='SalePrice',title='SalePrice vs GrLivArea',render_mode='auto',)


# > Relationship with 'TotalBsmtSF'

# In[ ]:


plt.figure(figsize=(14,8))
px.scatter(train,x='TotalBsmtSF',y='SalePrice',title='SalePrice vs TotalBsmtSF',render_mode='auto')


# > Relationship with 'OverallQual'

# In[ ]:


plt.figure(figsize=(14,8))
px.box(train,x='OverallQual',y='SalePrice',title='SalePrice vs OverallQual')


# > Relationship with 'YearBuilt'

# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=train,x='YearBuilt',y='SalePrice')
plt.title('SalePrice vs YearBuilt')
plt.xlabel('Year')
plt.ylabel('Price')
plt.xticks(rotation=90)
plt.tight_layout()


# # Finding the correlation between the different features.

# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(train.corr(),square=True)


# > SalePrice Correlation

# In[ ]:


#saleprice correlation matrix
plt.figure(figsize=(10,10))
k = 10 #number of variables for heatmap
cols = train.corr().nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# > Here there are many identical features, which means that they convey same information, so it is worthy to consider them only 1 time.
# * TotalBsmtSF and 1stFlrSF are identical
# * GarageCars and GarageArea are identical
# * TotRmsAbvGrd and GrLivArea are identical
# 
#  

# In[ ]:


cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();


# # Dealing with Missing Data

# In[ ]:


total = train.isnull().sum().sort_values(ascending=False)
percent = (total*100) / train.shape[0]
missingData = pd.concat([total,percent],keys=['Total','Percentage'],axis=1)
missingData.head(20)


# > We will remove the columns which will have more than 15% of missing data.
# 
# > As far as GarageX and BsmtX are concerned GarageX has the same number of missing values and they all are very identical to GarageCars. As we will consider GarageCars we can drop GarageX.
# 
# > For BsmtX, all this columns are identical to TotalBsmtSF and we can also remove all the BsmtX columns.
# 
# > MasVnrType and MasVnrArea are very closely associated with YearBuilt thus we dont need this columns as well.
# 
# > As far as one entry of Electrical is concerned we will just delete this observation and keep the variable.

# In[ ]:


train.drop((missingData[missingData['Total'] > 1]).index,1,inplace=True)


# In[ ]:


train=train.dropna()


# # Removing Outliers

# In[ ]:


plt.figure(figsize=(14,8))
px.scatter(train,x='GrLivArea',y='SalePrice',title='SalePrice vs GrLivArea',render_mode='auto',)


# > In the above scatter plot the two points where GrLivArea is very high and SalePrice is less, this two points are outliers and needed to be removed

# In[ ]:


train.loc[train['GrLivArea']==4676]


# In[ ]:


train.loc[train['GrLivArea']==5642]


# In[ ]:


train = train.drop([1298,523],axis=0)


# # Creating Dummy Variables.

# In[ ]:


test = test[train.columns[0:62]]


# In[ ]:


idTest = pd.DataFrame(test['Id'])


# In[ ]:


data = pd.concat([train, test], sort=False)
data = data.reset_index(drop=True)


# In[ ]:


data=pd.get_dummies(data)


# In[ ]:


train, test = data[:len(train)], data[len(train):]

X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']

test = test.drop(columns=['SalePrice', 'Id'])


# > Spilling the training data

# # Training by XGBoost algorithm with default Parameters

# In[ ]:


model = XGBRegressor()
model.fit(X,y)
print(r2_score(model.predict(X),y))


# > This is initial phase of the Kernel, I'll update the kernel. I'll be looking forward to following tasks:
# 1. Fine tuning the parameters to improve the performance.
# 2. Applying the model to test data and check performance.
# 3. Will explore the data more and see if more Data cleaning is possible or not.

# # Your Upvote will be appreciated if you liked this Kernel.
# ## Thanks

# In[ ]:





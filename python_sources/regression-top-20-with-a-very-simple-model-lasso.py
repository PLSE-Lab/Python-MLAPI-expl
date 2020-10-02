#!/usr/bin/env python
# coding: utf-8

# # Regression:top 20% with a very simple model-lasso
# 
# Please, upvote if you find useful.
# 
# ### Steps:
# * 1- Preprocessing and exploring
#     * 1.1 - Imports
#     * 1.2 - Checking Types
#     * 1.3 - Missing Values
#     * 1.4 - Remove some features high correlated and outliers
#     * 1.5 - Transformations
#     * 1.6 - Prepare for model
# * 2- Model
# * 3- Submission
# 

# ### 1.1- Imports

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, LassoCV


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
test2=pd.read_csv("../input/test.csv")
len_train=train.shape[0]
houses=pd.concat([train,test], sort=False)
print(train.shape)
print(test.shape)


# ### 1.2- Checking Types
# 

# In[ ]:


houses.select_dtypes(include='object').head()


# In[ ]:


houses.select_dtypes(include=['float','int']).head()


# #### When we read the data description file we realize that "MSSubClass", a numerical features (not ordinal), should be transformed into categorical. I'll do this later in this kernel.

# ### 1.3 - Missing Values

# ### Categorical
# 

# In[ ]:


houses.select_dtypes(include='object').isnull().sum()[houses.select_dtypes(include='object').isnull().sum()>0]


# Depending on the categorical variable, missing value can means "None" (which I will fill with "None") or "Not Available" (which I will fill with the mode).

# In[ ]:


for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
           'PoolQC','Fence','MiscFeature'):
    train[col]=train[col].fillna('None')
    test[col]=test[col].fillna('None')


# In[ ]:


#for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):
    #train[col]=train[col].fillna(train[col].mode()[0])
    #test[col]=test[col].fillna(test[col].mode()[0])


# In[ ]:


for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):
    train[col]=train[col].fillna(train[col].mode()[0])
    test[col]=test[col].fillna(train[col].mode()[0])


# ### Numerical

# In[ ]:


houses.select_dtypes(include=['int','float']).isnull().sum()[houses.select_dtypes(include=['int','float']).isnull().sum()>0]


# Some NAs means "None" (which I will fill with 0) or means "Not Available" (which I will fill with mean)

# In[ ]:


for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea'):
    train[col]=train[col].fillna(0)
    test[col]=test[col].fillna(0)


# In[ ]:


#train['LotFrontage']=train.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.mean()))
#test['LotFrontage']=test.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.mean()))


# In[ ]:


train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())
test['LotFrontage']=test['LotFrontage'].fillna(train['LotFrontage'].mean())


# In[ ]:


print(train.isnull().sum().sum())
print(train.isnull().sum().sum())


# ### 1.4 - Remove some features high correlated and outliers

# In[ ]:


plt.figure(figsize=[30,15])
sns.heatmap(train.corr(), annot=True)


# In[ ]:


#from 2 features high correlated, removing the less correlated with SalePrice
train.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)
test.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)


# In[ ]:


#removing outliers recomended by author
train = train[train['GrLivArea']<4000]


# In[ ]:


len_train=train.shape[0]
print(train.shape)


# In[ ]:


houses=pd.concat([train,test], sort=False)


# ### 1.5 - Transformations

# Numerical to categorical

# In[ ]:


houses['MSSubClass']=houses['MSSubClass'].astype(str)


# Skew

# In[ ]:


skew=houses.select_dtypes(include=['int','float']).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skew_df=pd.DataFrame({'Skew':skew})
skewed_df=skew_df[(skew_df['Skew']>0.5)|(skew_df['Skew']<-0.5)]


# In[ ]:


skewed_df.index


# In[ ]:


train=houses[:len_train]
test=houses[len_train:]


# In[ ]:


lam=0.1
for col in ('MiscVal', 'PoolArea', 'LotArea', 'LowQualFinSF', '3SsnPorch',
       'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch',
       'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF',
       'LotFrontage', 'GrLivArea', 'BsmtFinSF1', 'BsmtUnfSF', 'Fireplaces',
       'HalfBath', 'TotalBsmtSF', 'BsmtFullBath', 'OverallCond', 'YearBuilt',
       'GarageYrBlt'):
    train[col]=boxcox1p(train[col],lam)
    test[col]=boxcox1p(test[col],lam)


# In[ ]:


train['SalePrice']=np.log(train['SalePrice'])


# Categorical to one hot encoding

# In[ ]:


houses=pd.concat([train,test], sort=False)
houses=pd.get_dummies(houses)


# ### 1.6 - Prepare for model
# 

# In[ ]:


train=houses[:len_train]
test=houses[len_train:]


# In[ ]:


train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)


# In[ ]:


x=train.drop('SalePrice', axis=1)
y=train['SalePrice']
test=test.drop('SalePrice', axis=1)


# In[ ]:


sc=RobustScaler()
x=sc.fit_transform(x)
test=sc.transform(test)


# # 2 - Model

# In[ ]:


model=Lasso(alpha =0.001, random_state=1)


# In[ ]:


model.fit(x,y)


# # 3- Submission

# In[ ]:


pred=model.predict(test)
preds=np.exp(pred)


# In[ ]:


output=pd.DataFrame({'Id':test2.Id, 'SalePrice':preds})
output.to_csv('submission.csv', index=False)


# In[ ]:


output.head()


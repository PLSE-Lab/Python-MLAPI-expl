#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[ ]:


import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# Data visualization and analyize  sales price
# 

# In[ ]:


sns.distplot(train['SalePrice'])


# it seems to be positive skew

# In[ ]:


plt.figure(figsize=(12,9))
sns.heatmap(train.corr(), cmap='coolwarm')


# there are many items that are correlated to sales price ('OverallQual','YearBuilt','FullBath','TotRmsAbvGrd',
#              'GarageCars','GarageArea')
# 

# In[ ]:



plt.scatter(train['GrLivArea'], train['SalePrice'])


# In[ ]:


sns.scatterplot(x="SalePrice", y="OverallQual" , data=train)


# In[ ]:


total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# the features that have null values are not affected in SALES PRICE , so we can drop it

# In[ ]:


#dealing with missing data
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max() #just checking that there's no missing data missing


# In[ ]:


train.head()


# 

# In[ ]:


saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);


# In[ ]:


data = train[['OverallQual','YearBuilt','FullBath','TotRmsAbvGrd',
             'GarageCars','GarageArea','PoolArea']]


# In[ ]:


y=train["SalePrice"]
X=data
X.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)


# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)
print(lm)


# In[ ]:


print(lm.intercept_)


# In[ ]:


print(lm.coef_)


# In[ ]:


predictions=lm.predict(X_test)


# In[ ]:





# In[ ]:


data_test = test[['OverallQual','YearBuilt','FullBath','TotRmsAbvGrd',
             'GarageCars','GarageArea','PoolArea']]


# In[ ]:


data_test.head()


# In[ ]:


data_test.isnull().sum()


# In[ ]:


data_test.info()


# In[ ]:



data_test.fillna(0, inplace=True)


# In[ ]:


data_test.isnull().sum()


# In[ ]:


scaler=StandardScaler()
X=scaler.fit_transform(data_test)


# In[ ]:


predictions1 = lm.predict(X)


# In[ ]:


output = pd.DataFrame({'Id':test['Id'],'SalePrice':predictions1})


# In[ ]:


output.to_csv('submission.csv', index=False)


# In[ ]:





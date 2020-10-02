
# coding: utf-8

# In[52]:

import pandas as pd
import numpy as np
from sklearn import preprocessing


# In[65]:

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.set_index('Id',inplace=True)
train_x = train.drop('SalePrice',axis=1)
train_y = train.SalePrice
result = pd.DataFrame()
result['Id'] = test.Id
test.set_index('Id',inplace=True)


# In[66]:

train_x.fillna('0',inplace=True)
test.fillna('0',inplace=True)
print(test.head())
print(train_x.head())


# In[67]:

le = preprocessing.LabelEncoder()
for cols in ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
            'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
            'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu',
            'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType',
            'SaleCondition']:
    le.fit(np.array(pd.concat([train_x[cols],test[cols]])))
    train_x[cols] = le.transform(np.array(train_x[cols]))
    test[cols] = le.transform(np.array(test[cols]))


# In[68]:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[69]:

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)


# In[70]:

regr = LinearRegression()


# In[71]:

regr.fit(x_train,y_train)


# In[72]:

print(regr.score(x_test,y_test))


# In[75]:

result['SalePrice'] = pd.DataFrame(regr.predict(np.array(test)))


# In[76]:




# In[77]:

result.set_index('Id',inplace=True)


# In[78]:

print(result.head())


# In[79]:

result.to_csv('Submission.csv')


# In[ ]:




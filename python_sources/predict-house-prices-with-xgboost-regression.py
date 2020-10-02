#!/usr/bin/env python
# coding: utf-8

# ##This is my base code for predicting the house price using XGBoost regression. From this I will iterate to improve my results.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost
import csv as csv
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, mean_squared_error,precision_score


# In[ ]:


train_dataset=pd.read_csv('../input/train.csv', header=0)
test_dataset=pd.read_csv('../input/test.csv', header=0)


# #Study features

# In[ ]:


categorical_features=['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities',
                      'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                      'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond',
                      'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating',
                      'HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu',
                     'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence',
                     'MiscFeature','SaleType','SaleCondition']
train_dataset.describe()


# #Preparing the dataset

# #Cleaning

# In[ ]:


features_with_nan=['Alley','MasVnrType','BsmtQual','BsmtQual','BsmtCond','BsmtCond','BsmtExposure',
                   'BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish']
#function that creates a column for every value it might have
def ConverNaNToNAString(data, columnList):
    for x in columnList:       
        data[x] =str(data[x])              
            

ConverNaNToNAString(train_dataset, features_with_nan)
ConverNaNToNAString(test_dataset, features_with_nan)


# ##Creating columns from each categorical feature value

# In[ ]:


#function that creates a column for every value it might have
def CreateColumnPerValue(data, columnList):
    for x in columnList:

        values=pd.unique(data[x])
        
        for v in values:
            column_name=x+"_"+str(v)   
            data[column_name]=(data[x]==v).astype(float)
    
        data.drop(x, axis=1, inplace=True)
        


# In[ ]:


CreateColumnPerValue(train_dataset,categorical_features)
CreateColumnPerValue(test_dataset,categorical_features)


# In[ ]:


y = train_dataset["SalePrice"]
train_dataset.drop("SalePrice", axis=1, inplace=True)


# #Looking for most relevant features

# In[ ]:


model = xgboost.XGBRegressor()


# In[ ]:


xgb_model=model.fit(train_dataset,y)
model.booster().get_fscore()


# In[ ]:


#Let's remove the less important ones
most_relevant_features= list( dict((k, v) for k, v in model.booster().get_fscore().items() if v >= 10).keys())
print(most_relevant_features)
xgb_model=model.fit(train_dataset[most_relevant_features],y)


# #Let's predict for test data

# In[ ]:


test_dataset['Prediction'] = xgb_model.predict(test_dataset[most_relevant_features])


# #Generate the submission file

# In[ ]:


filename = 'submission2.csv'
pd.DataFrame({'Id': test_dataset.Id, 'SalePrice': test_dataset.Prediction}).to_csv(filename, index=False)


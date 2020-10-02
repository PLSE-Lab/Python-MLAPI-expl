#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore  warning (from sklearn)

#print(os.listdir("../input"))


# Let's Load the Data

# In[ ]:


house_data = pd.read_csv('../input/train.csv')
house_data_test = pd.read_csv('../input/test.csv')


# **Data Cleaning**

# Will store the Id column(information) from test dataframe ,in test_parent dataframe.
# 
# Then let's drop the Id column from both test and train data. 

# In[ ]:


train_parent=house_data
test_parent=house_data_test 
house_data = house_data.drop('Id', axis=1)
house_data_test = house_data_test.drop('Id', axis=1)


# Now we will analyze the NaN values present in the dataset and deal with them .

# In[ ]:


#We will find all the columns which have more than 40 % NaN data and drop then
threshold=0.4 * len(house_data)
df=pd.DataFrame(len(house_data) - house_data.count(),columns=['count'])
df.index[df['count'] > threshold]


# In[ ]:


house_data = house_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
house_data_test = house_data_test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)


# Will verify that the Label is a numeric data

# In[ ]:


house_data['SalePrice'].describe()


# Find all the numeric columns and replace the NaN values with 0 ,
# and for categorical columns ,replace NaN values with 'None'.

# In[ ]:


house_data.select_dtypes(include=np.number).columns #will give all numeric columns ,we will remove the SalePrice column 
for col in ('MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold'):
    
    house_data[col] = house_data[col].fillna(0)
    house_data_test[col] = house_data_test[col].fillna('0')


# In[ ]:


house_data.select_dtypes(exclude=np.number).columns
for col in ('MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition'):
    
    house_data[col] = house_data[col].fillna('None')
    house_data_test[col] = house_data_test[col].fillna('None')


# Verify that there are no null values in the data set

# In[ ]:


house_data[house_data.isnull().any(axis=1)]


# In[ ]:


house_data_test[house_data_test.isnull().any(axis=1)]


#  **Combining the two datasets and then doing One Hot Encoding on the combined dataset.**

# In[ ]:


train=house_data
test=house_data_test

#Assigning a flag to training and testing dataset for segregation after OHE .
train['train']=1 
test['train']=0

#Combining training and testing dataset

combined=pd.concat([train,test])


# In[ ]:


#Applying One Hot Encoding to categorical data
ohe_data_frame=pd.get_dummies(combined, 
                           columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition'],
      )


# In[ ]:


#Splitting the combined dataset after doing OHE .
train_df=ohe_data_frame[ohe_data_frame['train']==1]
test_df=ohe_data_frame[ohe_data_frame['train']==0]
train_df.drop(['train'],axis=1,inplace=True)             #Drop the Flag(train) coloumn from training dataset
test_df.drop(['train','SalePrice'],axis=1,inplace=True)     #Drop the Flag(train),Label(SalePrice) coloumn from test dataset


# In[ ]:


house_data=train_df
house_data_test=test_df


# **Data Cleaning is now complete We can now use our data to build our models**

# In[ ]:


X_train = house_data.drop('SalePrice', axis=1)
# Taking the labels (price)
Y_train = house_data['SalePrice']
X_test = house_data_test


# Let's apply Gradient Boosting for regression and find the best parameter for GBR using GridSearchCV    

# In[ ]:


"""
from sklearn.model_selection import GridSearchCV

num_estimators = [500,1000,3000]
learn_rates = [0.01, 0.02, 0.05, 0.1]
max_depths = [1, 2, 3, 4]
min_samples_leaf = [5,10,15]
min_samples_split = [2,5,10]

param_grid = {'n_estimators': num_estimators,
              'learning_rate': learn_rates,
              'max_depth': max_depths,
              'min_samples_leaf': min_samples_leaf,
              'min_samples_split': min_samples_split}

grid_search = GridSearchCV(GradientBoostingRegressor(loss='huber'),
                           param_grid, cv=3, return_train_score=True)
grid_search.fit(X_train, Y_train)

grid_search.best_params_  
"""


# In[ ]:


#GardientBoosting
params = {'n_estimators': 3000, 'max_depth': 1, 'min_samples_leaf':15, 'min_samples_split':10, 
          'learning_rate': 0.05, 'loss': 'huber','max_features':'sqrt'}
gbr_model = GradientBoostingRegressor(**params)
gbr_model.fit(X_train, Y_train)


# In[ ]:


gbr_model.score(X_train, Y_train)


# In[ ]:


#Predicting the SalePrice for the test data
y_grad_predict = gbr_model.predict(X_test)
print(y_grad_predict)


# In[ ]:


#Submission 
my_submission = pd.DataFrame({'Id': test_parent.Id, 'SalePrice': y_grad_predict})
print(my_submission)

my_submission.to_csv('submission.csv', encoding='utf-8', index=False)


# If you found this notebook helpful or you just liked it , some upvotes would be very much appreciated - That will keep me motivated .
# 
# Please drop down suggestions and comments if any, so that i can learn to build better solutions.
# 
# **Thank You** :-)

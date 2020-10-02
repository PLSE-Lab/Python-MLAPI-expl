#!/usr/bin/env python
# coding: utf-8

# # Hello Community!
# This is my attempt at predicting the house prices using Random Forest regression and some basic cleaning techniques.
# My very frist notbook so it would be really helpful to know if there is any way to improve the model and please do correct me if I've gone wrong somewhere. I would like to learn:)

# First Things First ..... importing the libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Read the Data

# In[ ]:


trainData = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")


# In[ ]:


testData = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


sampleData = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# # Data Exploration
# Now that the data has been loaded, let's explore the data

# In[ ]:


trainData.head()


# In[ ]:


trainData.info()


# As we can see there are a lot of null values.
# We need to take care of that either by deleting completely or by replacing the null values with the mean or mode of that column.
# Before we do that let's explore our Dependent Variable - SalePrice

# In[ ]:


sns.distplot(trainData.SalePrice)
plt.show()


# Since the distribution is not normal we'll take the log.

# In[ ]:


SalePrice = trainData.SalePrice


# In[ ]:


SalePrice_log = np.log(SalePrice)


# In[ ]:


sns.distplot(SalePrice_log)
plt.show()


# Okay it looks much better

# In[ ]:


numeric_var = trainData.select_dtypes(exclude='object')


# In[ ]:


trainData.drop(columns = ['SalePrice']).corrwith(SalePrice,axis=0).plot.bar(figsize = (12,12))


# We can see that variables OverallCond,GrlivArea,GarageCars(for some reason) have higher correlation with the price as compare to others.
# Its always good to know.
# We can consider dropping these variables.

# Also we can also check which variables have high correlation amongst each other so that we can drop that variable.
# 

# In[ ]:


plt.figure(figsize=(12,12))
fig = sns.heatmap(trainData.corr(), linewidth = 0.3)


# # Data Cleaning
# Let's check the  null values

# In[ ]:


train_id = trainData.Id
test_id = testData.Id


# In[ ]:


trainData.set_index('Id')
testData.set_index('Id')


# In[ ]:


data = pd.concat([trainData, testData])


# In[ ]:


with pd.option_context('display.max_rows',None,'display.max_columns',None):
    display(data.isnull().sum())


# In[ ]:


data.drop(columns=['Alley','PoolQC','Fence','MiscFeature'], inplace = True)


# In[ ]:


num_col = data.select_dtypes(exclude='object').columns
obj_cat = data.select_dtypes(include='object').columns


# In[ ]:


for i in num_col:
    data[i] = data[i].fillna(data[i].mean())


# In[ ]:


for i in obj_cat:
    data[i] = data[i].fillna(data[i].mode()[0])


# In[ ]:


with pd.option_context('display.max_rows',None,'display.max_columns',None):
    display(data.isnull().sum())


# In[ ]:


data.drop(columns = ['SalePrice'], inplace = True)


# Also lets drop the columns with high correlation

# In[ ]:


data.drop(['GrLivArea', '1stFlrSF', 'OverallQual', 'GarageCars'], axis=1, inplace=True)


# In[ ]:


data = data.set_index('Id')


# In[ ]:


data


# Let's get the dummies

# In[ ]:


data2 = data.copy(deep=True)


# In[ ]:


obj_col = data2.select_dtypes(include='object').columns


# In[ ]:


data_dummy = pd.get_dummies(data2, columns = obj_col, drop_first=True)


# Okay let's split the data back into train and test

# In[ ]:


train = data_dummy.iloc[:train_id.shape[0],:]
test = data_dummy.iloc[train_id.shape[0]:,:]


# In[ ]:


train


# In[ ]:


test


# Looks good to me

# # Feature scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc = StandardScaler()


# In[ ]:


x = pd.DataFrame(sc.fit_transform(train), columns = train.columns.values)
x_test = pd.DataFrame(sc.transform(test), columns = test.columns.values)


# In[ ]:


x


# In[ ]:


from sklearn.model_selection import train_test_split

x_train,x_test_1, y_train, y_test_1 = train_test_split(x,SalePrice_log,test_size = 0.2, random_state = 0)


# # Training the model
# Here I'm going to use Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor()


# In[ ]:


regressor.fit(x_train,y_train)


# In[ ]:


y_pred= regressor.predict(x_test_1)


# In[ ]:


from sklearn.metrics import r2_score,mean_squared_error


# In[ ]:


r2_score(y_test_1,y_pred)


# In[ ]:


mean_squared_error(y_test_1,y_pred)


# Seems pretty good

# Let's see if we can improve our model by changing the parameters.

# In[ ]:


param = {'max_depth': [3,5,8],
        'n_estimators': [100,300,500],
        'criterion': ['mse', 'mae'],
        'max_features': ['sqrt','log2','auto']}


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


regressor = RandomForestRegressor()
random = RandomizedSearchCV(estimator = regressor, param_distributions=param, n_iter = 5,scoring='neg_mean_squared_error',n_jobs=-1,cv=5)


# In[ ]:


random.fit(x_train,y_train)


# In[ ]:


random.best_params_


# In[ ]:


regressor2 = RandomForestRegressor(n_estimators=100,
 max_features='auto',
 max_depth=8,
 criterion='mse')


# In[ ]:


regressor2.fit(x_train,y_train)


# In[ ]:


y_pred = regressor2.predict(x_test_1)


# In[ ]:


r2_score(y_test_1,y_pred)


# The previous model was better:p
# So we'll use that one

# In[ ]:


prediction = regressor.predict(x_test)


# In[ ]:


pred = np.exp(prediction)


# In[ ]:


Id = pd.DataFrame(test_id, columns = ['Id'])

predi = pd.DataFrame(pred, columns = ['SalePrice'])


# In[ ]:


result = pd.concat([Id,predi],axis = 1)


# In[ ]:





# # So this is it .
# As I said it is a simple model with basic cleaning techniques.
# # Looking forward to learn through your comments.
# 

# In[ ]:





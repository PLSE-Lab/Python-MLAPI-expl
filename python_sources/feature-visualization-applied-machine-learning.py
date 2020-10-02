#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Work Flow of This Notebook

# 1. Data Understanding 
# 
# 2. Future Selection
# 
# 3. Missing Values
# 
# 4. Data Visualization 
# 
#    *Histogram   
#    
#    *Box Plot
#    
#    *Violin
#    
#    *ScatterPlot
#    
#    
#    
# 5. Test-Train-Split
# 
# 6. Machine Learning Model
# 
#    *SVR
#    
#    *Random Forest
#    
#    *XGBoost
#    
# 7. Model Validations   
# 
#  

# In[ ]:


# Load the Data

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


# About the Data

train.head()


# In[ ]:


train.dtypes.sample(60)


# In[ ]:


train.describe().T


# In[ ]:


train.info()


# In[ ]:


train.shape


# ### Correlation Matrix

# In[ ]:


corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)


# ## Feature Selection

# In[ ]:


features = ["OverallQual" , "GrLivArea","GarageCars" ,"GarageArea","TotalBsmtSF" ,"1stFlrSF" ,"FullBath"]

X = train[features]
y = train["SalePrice"]


# In[ ]:


X.head()


# ## Missing Values

# In[ ]:


X.isnull().sum()


# In[ ]:


y.isnull().sum()


# # Data Visualization

# In[ ]:


import seaborn as sns


# In[ ]:


sns.barplot(x="GrLivArea" , y="SalePrice", data=train)


# In[ ]:


sns.barplot(x="OverallQual", y="SalePrice", data=train)


# In[ ]:


sns.distplot(train.SalePrice,kde=False)


# In[ ]:


sns.boxplot(x="OverallQual",y="SalePrice",data=train)
              
             


# In[ ]:


sns.catplot(x="OverallQual", y = "SalePrice" , kind= "violin" ,data=train)


# In[ ]:


sns.catplot(x="GrLivArea", y = "SalePrice" , kind= "violin" ,data=train)


# In[ ]:


sns.scatterplot( x = "OverallQual" , y = "SalePrice" , data = train)


# In[ ]:


sns.scatterplot( x = "GrLivArea" , y = "SalePrice" , data = train)


# In[ ]:


sns.lmplot(x = "SalePrice" , y = "GrLivArea" , hue = "OverallQual" , data = train)


# In[ ]:


# test_train_split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)


# # Machine Learning Models

# In[ ]:


# SVR

from sklearn.svm import SVR
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_train,y_train)


# In[ ]:


# svr prediction
predicted_svr = svr_reg.predict(x_test)


# In[ ]:


# MAE
from sklearn.metrics import mean_absolute_error


mean_absolute_error(y_test , predicted_svr)


# In[ ]:


# Root mean Square Error (R2)

from sklearn.metrics import r2_score

r2_score(y_test, svr_reg.predict(x_test)) 


# In[ ]:


# Random Forest

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)

rf_reg.fit(x_train,y_train)


# In[ ]:


# RF prediction

predicted_rf = rf_reg.predict(x_test)


# In[ ]:


#MAE
mean_absolute_error(y_test,predicted_rf)


# In[ ]:


#R2
r2_score(y_test, rf_reg.predict(x_test)) 


# In[ ]:


#XGBOOST

from xgboost import XGBRegressor

xgb_regressor = XGBRegressor()
xgb_regressor.fit(x_train, y_train)


# In[ ]:


#prediction
predicted_xgb = xgb_regressor.predict(x_test)


# In[ ]:


#mae
mean_absolute_error(y_test,predicted_xgb)


# In[ ]:


#r2
r2_score(y_test,predicted_xgb)


# In[ ]:


#SUBMISSION

test_X = test[features]
predicted_xgb_test = xgb_regressor.predict(test_X)


# In[ ]:


output = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_xgb_test})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:





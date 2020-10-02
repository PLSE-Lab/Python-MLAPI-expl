#!/usr/bin/env python
# coding: utf-8

# * My first attempt to work on Regression model 
# * My Data Analytics skills are still in beginner level. Hence little Data Analytics
# * Most of the Notebooks are overwhelming with lots of concepts, but this one will be simple
# Happy Hunting !
# ![all-my-lifei-have-lived-by-a-code-and-the-46408606.png](attachment:all-my-lifei-have-lived-by-a-code-and-the-46408606.png)
# 
# Please upvote if you like this Notebook 

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


# In[ ]:


df_train  = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')   


# In[ ]:


df_train[df_train.columns[1:]].corr()['SalePrice'][:]        


# The corelation data shown above doesn't provide the accurate information as it contains lots of missing values 

# In[ ]:


df_train.describe()


# **Dealing with missing Values**
# * I made it simple by the filling the float and int type data with Mean of respective column
# * For Data type with Object fill with Mode
# 
# 
# **First check the Train Dataset and Test Dataset for Missing values**

# In[ ]:



df_train.isnull().sum().sort_values(ascending = False)[0:20]


# In[ ]:



df_test.isnull().sum().sort_values(ascending = False)[0:40]


# In[ ]:


for col in df_train:
    if df_train[col].dtype == 'object':
      df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
    else:
       df_train[col].fillna(round(df_train[col].mean()),inplace = True)


for col in df_test:
      if df_test[col].dtype == 'object':
          df_test[col] = df_test[col].fillna(df_test[col].mode()[0])
      else:
        df_test[col].fillna(round(df_test[col].mean()),inplace = True)


# * Check the Train and Test data now for any missing values

# In[ ]:


df_train.isnull().any().any()


# In[ ]:


df_test.isnull().any().any()


# In[ ]:


df_train.drop(columns = ['Id', ],axis = 1, inplace =True)
df_test.drop(columns = ['Id'],axis = 1, inplace =True)


# In[ ]:


import seaborn as sns
sns.pairplot(df_train, x_vars=["SalePrice"],
                 y_vars=[ "TotalBsmtSF", 'YearBuilt','1stFlrSF','GrLivArea','2ndFlrSF','GarageArea'], 
             corner = True)


# **Now we have to deal with Categorical Data types**
# * One hot Encoded is applied to all the columns with Categorical Data
# 
# * Below function is used for one hot encoding

# In[ ]:



def cat_onehotencoder(df_concat):
    df_temp = df_concat
    for col in df_temp:
        if df_temp[col].dtype =='object':
            df1 = pd.get_dummies(df_concat[col], drop_first = True)
            df_concat.drop([col], axis = 1, inplace = True)
            
            df_concat = pd.concat([df_concat,df1], axis = 1)
        
    
        
    
    return df_concat


# In[ ]:


y = df_train.iloc[:,-1].values
df_t = df_train
y


# In[ ]:


df_train.drop(columns = ['SalePrice'], axis = 0, inplace = True)


# Concatenate Test and Train data to develop the categorical data

# In[ ]:


df_concat = pd.concat([df_train,df_test], axis = 0)
df_final =  cat_onehotencoder(df_concat)


# In[ ]:


df_final =df_final.loc[:,~df_final.columns.duplicated()]
df_final.shape


# In[ ]:


df_final


# * Now categorical columns has been taken care 

# In[ ]:


import seaborn as sns
correlations = df_train[df_train.columns].corr(method='pearson')
sns.heatmap(correlations, cmap="YlGnBu", annot = True)

import heapq

print('Absolute overall correlations')
print('-' * 30)
correlations_abs_sum = correlations[correlations.columns].abs().sum()
print(correlations_abs_sum, '\n')

print('Weakest correlations')
print('-' * 30)
print(correlations_abs_sum.nsmallest(5))


# **Split the Train and Test set from df_final**

# In[ ]:


train = df_final.iloc[:1460,:]
test = df_final.iloc[1460:,:]


# In[ ]:


X= train.iloc[:,:].values


# In[ ]:



from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2)



# In[ ]:


#from sklearn.preprocessing import StandardScaler#
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_val = sc.transform(X_val)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

gbreg=GradientBoostingRegressor()
parameters= {'n_estimators':[100,200,300, 600],
             'max_depth':[3,4,6,7]
    }

gbreg=GridSearchCV(gbreg, param_grid=parameters)
gbreg.fit(X_train,y_train)
print("The best value of leanring rate is: ",gbreg.best_params_, )


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_log_error
GB_reg = GradientBoostingRegressor(max_depth = 4, n_estimators = 300)
GB_reg.fit(X_train, y_train)

y_predgb = GB_reg.predict(X_val)
score_gb = r2_score(y_predgb, y_val)
MSL_gb = mean_squared_log_error(y_predgb,y_val)
print(score_gb, MSL_gb)


# In[ ]:


from xgboost import XGBRegressor
reg_xgb = XGBRegressor()
reg_xgb.fit(X_train,y_train)
ypred_xgb = reg_xgb.predict(X_val)

score_xgb = r2_score(ypred_xgb, y_val)
MSL_xgb = mean_squared_log_error(ypred_xgb,y_val)
print(score_xgb, MSL_xgb)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators =75, random_state = 42)
regressor_rf.fit(X_train,y_train)
ypred_rf = regressor_rf.predict(X_val)
score_rf = r2_score(ypred_rf, y_val)
MSL_rf = mean_squared_log_error(ypred_rf,y_val)
print(score_rf, MSL_rf)


# In[ ]:


y_pred_final = GB_reg.predict(test)
y_pred_final


# In[ ]:


sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


SalePrice = pd.DataFrame(y_pred_final, columns = ['SalePrice'])
len(SalePrice)
SalePrice.insert(0, 'Id', sub['Id'], True)
SalePrice.to_csv('sample_submissions1.csv', index = False)


# I have a better score but not the best 
# Hyper parameters tuning is to be done on this to improve the results 
# Also i haven't dealt with data skewedness 
# 
# **If you find this useful Please UPVOTE this notebook, Thanks**

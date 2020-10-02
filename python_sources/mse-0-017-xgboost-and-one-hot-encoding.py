#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pandas_profiling
from sklearn import preprocessing
import xgboost


# **Read the data and perform some descriptive statistics**

# In[ ]:


train_data = pd.read_csv("../input/train.csv")


# In[ ]:


train_data.head()


# In[ ]:


profile = pandas_profiling.ProfileReport(train_data)
profile


# **Data transfomation & one hot encoding**

# In[ ]:


#Identify columns type and extract type to array
col = [c for c in train_data.columns]
numclasses=[]
for c in col:
    numclasses.append(train_data[c].dtypes)
numclasses


# In[ ]:


# from previous extraction find dtype = object and keep column name
categorical_variables = list(np.array(col)[np.array(numclasses)==np.object])
categorical_variables


# In[ ]:


# from extracted names of categorical variables, dummify them and drop the categorical variable from main dataframe
collectdf2=[]
for name2 in categorical_variables:
    df2 = pd.get_dummies(train_data[name2],prefix=name2,dummy_na=True)
    train_data.drop(name2,axis=1,inplace=True)
    collectdf2=pd.concat([pd.DataFrame(collectdf2),df2],axis=1)
collectdf2


# In[ ]:


# we realise that centralAir is already dummy but with object dtype (Yes or No). Drop one of two created variables
collectdf2.drop("CentralAir_N",axis=1,inplace=True)


# In[ ]:


# Derive two new variables from YearBuilt and YearRemodAdd
train_data["YrAdd"]= train_data["YrSold"]-train_data["YearRemodAdd"]
train_data["Yrbuilt"]= train_data["YrSold"]-train_data["YearBuilt"]
train_data.drop("YearRemodAdd",axis=1,inplace=True)
train_data.drop("YearBuilt",axis=1,inplace=True)


# In[ ]:


# After fetching data find out 5 addtional potential categorical variables to dummify
collectdf1=[]
categorical_variables=['MSSubClass','OverallQual','OverallCond','YrSold','MoSold']
for name1 in categorical_variables:
    df1 = pd.get_dummies(train_data[name1],prefix=name1,dummy_na=True)
    train_data.drop(name1,axis=1,inplace=True)
    collectdf1=pd.concat([pd.DataFrame(collectdf1),df1],axis=1)
collectdf1


# **Dealing with missing values**

# In[ ]:


# lets replace missing values with zero
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(0)
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(0)
train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(0)


# In[ ]:


# drop id (useless)
train_data.drop("Id",axis=1,inplace=True)


# **Correlation analysis for continuous variables**

# In[ ]:


train_data.corr(method='pearson')


# **Let's scale the data ! (continuous variables)**

# In[ ]:


X_scaled = preprocessing.scale(train_data.drop("SalePrice",axis=1))


# In[ ]:


data_ready = pd.concat([pd.DataFrame(X_scaled),collectdf1,collectdf2],axis=1)


# **Too many variables, let's use feature extraction based on feature importance using Trees**

# In[ ]:


X = data_ready
Y = train_data["SalePrice"]

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)


# In[ ]:


#let's concatenate feature importance and columns names
Import_feat = pd.DataFrame([X.columns,model.feature_importances_]).T
Import_feat.sort_values(by=1, inplace=True, ascending=False)
Import_feat


# In[ ]:


#drop data that has zero in feature importance
dropcolmn = Import_feat[0][Import_feat[1]==0].values

for d in dropcolmn:
    data_ready.drop(d,axis=1,inplace=True)


# **Data is ready !! let's convert the target to logarithm**

# In[ ]:


X = data_ready

Y = np.log(train_data["SalePrice"])


# In[ ]:


from sklearn.model_selection import train_test_split

test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# **Now we can use xgboost and fit**

# In[ ]:


model = xgboost.XGBRegressor(learning_rate =0.01,
 n_estimators=5000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 reg_alpha=0,
 subsample=0.8,
 colsample_bytree=0.8,
 nthread=4,
 objective='reg:linear',                           
 seed=27)
results = model.fit(X_train, Y_train,eval_metric="rmse")


# **Result performance**

# In[ ]:


Y_HAT = model.predict(X_test)

from sklearn.metrics import mean_squared_error

mean_squared_error(Y_test, Y_HAT)


#!/usr/bin/env python
# coding: utf-8

# **Hackathon 5:E-commerce Price Prediction**
# 
# **Platform**: Analytics India Magazine
# 
# **Link**:[E-commerce Price Prediction](https://www.machinehack.com/course/e-commerce-price-prediction-weekend-hackathon-8/)
# 
# **Pulic Leaderboard Rank**:7
# 
# **Private Leaderboard Rank**:9

# ## Description : 
# 
# E-commerce platforms have been in existence for more than 2 decades now. The popularity and its preference as a common choice for buying and selling essential products have grown rapidly and exponentially over the past few years. E-commerce has impacted the lifestyle of common people to a huge extent. Many such platforms are competing over each other for dominance by providing consumer goods at a competitive price. In this hackathon, we challenge data science enthusiasts to predict the price of commodities on an e-commerce platform.
# 
# ## AIM:
# 
# **Given are 7 distinguishing factors that can influence the price of a product on an e-commerce platform. Your objective as a data scientist is to build a machine learning model that can accurately predict the price of a product based on the given factors.**

# # Data Analysis

# In[ ]:


#import necessary libraries

import pandas as pd
pd.set_option('display.max_rows', 500)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:


pip install eli5


# In[ ]:


#ELI5 is a Python package which helps to debug machine learning classifiers and explain their predictions.
from eli5.sklearn import PermutationImportance


# In[ ]:


#import datasets

train = pd.read_csv('../input/Train.csv')
test = pd.read_csv('../input/Test.csv')


# In[ ]:


#Lets look at the train and test datasets
print(train.head(),test.head())


# In[ ]:


print(train.info(),test.info())


# **Some Observations**:
# 
# 
# *   We have 6 features(1 float nd 5 object type)
# *   There are no null values in the dataset
# *   Date feature maybe used to create some additional features of interest
# 
# 
# 
# 
# 

# **Target Variable Distribution**

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of Selling_Price")
ax = sns.distplot(train["Selling_Price"])


# 
# 
# > The Distribution of our target variable is highly left skewed, hence we will be performing log-transformation to make it more normal.
# 
# 
# 

# In[ ]:


train['Selling_Price'] = np.log(train['Selling_Price'])


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of Selling_Price after transformation")
ax = sns.distplot(train["Selling_Price"])


# **Item Rating vs Selling Price**

# In[ ]:


sns.scatterplot('Item_Rating','Selling_Price',data=train)


# 
# 
# > No linear relationship can be seen whatsoever bw the two variables,hence 'Iten Rating' is not much of a use for us.
# 
# 

# *Let us now combine our dataset and perform some analysis/feature engineering*

# In[ ]:


combine = train.append(test)


# In[ ]:


combine.head()


# In[ ]:


print(combine.info())


# **General Observations**
# 
# -The features 'Product' and 'Product_Brand' though object type , can easily be converted into numeric by removing the char and separator to avoid encoding the data
# 
# -Item Category and Sucategory 1 & 2 features seem to have a lot of unique categorical values.

# 
# 
# > Converting 'Product' and 'Product_Brand' to numeric type.
# 
# 

# In[ ]:


combine['Product']=combine['Product'].str.split('-', n=1, expand=True)[1]
combine['Product'] = combine['Product'].astype(int)
combine['Product'] = np.log(combine['Product'])


# In[ ]:


combine['Product_Brand']=combine['Product_Brand'].str.split('-', n=1, expand=True)[1]
combine['Product_Brand'] = combine['Product_Brand'].astype(int)
combine['Product_Brand'] = np.log(combine['Product_Brand'])


# In[ ]:


combine.dtypes


# **Now let us look at probably the 3 most important and interesting features (Item_category,subcategory1 and subcategory2) in detail**

# In[ ]:


cols = ['Item_Category','Subcategory_1','Subcategory_2']
for i in cols:
  print("============================")
  print("Categories in",i,": ")
  print(combine[i].value_counts())


# In[ ]:


for i in cols:
  print("Total distinct categories in",i, ":",len(combine[i].unique()))


# -As observed, there are a lot of distinct categories in all the three features above hence OHE will not make sense here.
# 
# -We will be using frequency encoding for converting these features into numeric type
# 
# **One major issue to Adress**
# 
# 
# 
# > How to handle 'Unknown' category in subcategory1 and subcategory2?
# 
# **SOL**: Here is a simple and elegant solution provided by a friend...
# 
# And I quote......
# 
# *'Unkown is subcategory1/subcategory2 means that there is no hierarchial relationship bw Item_Category-Subcategory1-Subcategory2 which affects the selling price and hence if we find an unknown we can simply replace it by the category one level above'*
# 
# 
# 

# **Performing the above operations**

# In[ ]:


combine['Subcategory_1'] = np.where((combine['Subcategory_1']=='unknown'),combine['Item_Category'],combine['Subcategory_1'])
combine['Subcategory_2'] = np.where((combine['Subcategory_2']=='unknown'),combine['Subcategory_1'],combine['Subcategory_2'])


# In[ ]:


enc_nom = (combine.groupby('Item_Category').size()) / len(combine)
enc_nom
combine['Item_Category_encode'] = combine['Item_Category'].apply(lambda x : enc_nom[x])


# In[ ]:


enc_nom_1 = (combine.groupby('Subcategory_1').size()) / len(combine)
enc_nom_1
combine['Subcategory_1_encode'] = combine['Subcategory_1'].apply(lambda x : enc_nom_1[x])


# In[ ]:


enc_nom_2 = (combine.groupby('Subcategory_2').size()) / len(combine)
enc_nom_2
combine['Subcategory_2_encode'] = combine['Subcategory_2'].apply(lambda x : enc_nom_2[x])


# **Extracting the additional Month feature from Date as it may be useful**

# In[ ]:


from datetime import datetime
combine['Date'] = pd.to_datetime(combine['Date'])
combine['Month'] = [date.month for date in combine.Date]


# **Let us have a look at the final dataset after dropping unnecessary columns**

# In[ ]:


combine.drop(['Item_Category','Subcategory_1','Subcategory_2','Date','Item_Rating'],axis=1,inplace = True)


# In[ ]:


combine.head()


# # Model Building

# In[ ]:


#Separating train and test datasets
train=combine[combine['Selling_Price'].isnull()!=True]
test=combine[combine['Selling_Price'].isnull()==True]

test=test.drop(['Selling_Price'], axis=1)


# In[ ]:


print(train.shape,test.shape)


# In[ ]:


#train-test split
Y = train['Selling_Price']
X = train.drop('Selling_Price',axis=1)


X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.1,random_state=0)


# In[ ]:


print(X_train.shape,Y_train.shape)


# **I used 4 of the following tree based/boosting algorithms for training purpose:**
# 
# 
# 
# 1.   CatBoost
# 2.   RandomForestRegressor
# 3.   XGBoost
# 4.   LightGBM
# 
# 
# 
# > Random Forest performed the best as a standalone model based on 10 fold CV score and public dataset results and hence hyper-parameter tuning was done only for it.
# 
# Score summary is provided at the end for each model
# 
# 
# 
# 
# 
# 

# ## CatBoost

# In[ ]:


pip install catboost


# In[ ]:


from catboost import CatBoostRegressor
cb = CatBoostRegressor(
    n_estimators = 500,
    learning_rate = 0.1,
    loss_function = 'MAE',
    eval_metric = 'RMSE')

cb.fit(X_train,Y_train)


# In[ ]:


import eli5
perm = PermutationImportance(cb,random_state=100).fit(X_val, Y_val)
eli5.show_weights(perm,feature_names=X_val.columns.tolist())


# In[ ]:


from sklearn.metrics import mean_squared_error
kf=KFold(n_splits=10, random_state=100, shuffle=True)

y_test_predict=0
mse = 0
j=1
result={}

for i, (train_index, test_index) in enumerate(kf.split(train)):
    
   Y_train, Y_valid = Y.iloc[train_index], Y.iloc[test_index]
   X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]
   
   print( "\nFold ", j)
   cb = CatBoostRegressor(
    n_estimators = 1000,
    learning_rate = 0.05,
    max_depth = 6,
    boosting_type = 'Ordered',
    loss_function = 'RMSE',
    eval_metric = 'RMSE',verbose = 0)
   
  #  xg=XGBRegressor(booster='gbtree', max_depth=5, learning_rate=0.05, reg_alpha=0,
  #                 reg_lambda=1, n_jobs=-1, random_state=100, n_estimators=5000)
    
   model=cb.fit(X_train,Y_train)
   pred = model.predict(X_valid)
   
   print(np.sqrt(mean_squared_error(Y_valid, np.abs(pred))))
   mse+=np.sqrt(mean_squared_error(Y_valid,np.abs(pred)))
    
   #y_test_predict+=model.predict(test)  
   #result[j]=model.predict(X_main_test)
   j+=1

results=y_test_predict/10

print(mse/10)


# In[ ]:


# model3=cb.fit(x,y)
# pred_cb = model3.predict(test)
# pred_cb = np.abs(pred_cb)
# pred_cb


# ## RandomForestRegressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,criterion='mse',
                           
                           min_samples_leaf=1, 
                           min_samples_split = 5, 
                           random_state=100)
rf.fit(X_train,Y_train)


# In[ ]:


import eli5
perm = PermutationImportance(rf,random_state=100).fit(X_val, Y_val)
eli5.show_weights(perm,feature_names=X_val.columns.tolist())


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 300, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['auto','sqrt']
#max_features = ['sqrt']
# Maximum number of levels in tree
#max_depth = [int(x) for x in np.linspace(6, 30, num = 5)]
#max_depth.append(None)
# Minimum number of samples required to split a node
#min_samples_split = [2, 4,6]
min_samples_split = [4,5,6,7]
# Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2, 4]
min_samples_leaf = [1]
# Method of selecting samples for training each tree
#bootstrap = [True, False]
bootstrap = [True]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               #'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[ ]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, Y_train)


# In[ ]:


rf_random.best_params_


# In[ ]:


# from sklearn.model_selection import GridSearchCV
# # Create the parameter grid based on the results of random search 
# param_grid = {
#     'bootstrap': [True],
#     'max_features': [1,2,3],
#     'min_samples_leaf': [1],
#     'min_samples_split': [4,5],
#     'n_estimators': [300,350,400]
# }
# # Create a based model
# rf = RandomForestRegressor()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)


# In[ ]:


# grid_search.fit(X_train, Y_train)


# In[ ]:


# grid_search.best_params_


# In[ ]:


#Calsulating cv score using the best parameters
kf=KFold(n_splits=10, random_state=100, shuffle=True)

y_test_predict=0
mse = 0
j=1
result={}

for i, (train_index, test_index) in enumerate(kf.split(train)):
    
   Y_train, Y_valid = Y.iloc[train_index], Y.iloc[test_index]
   X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]
   
   print( "\nFold ", j)
   
   rf = RandomForestRegressor(n_estimators=455,
                           max_features='sqrt',
                           bootstrap='True',
                           min_samples_leaf=1,
                           min_samples_split=4,
                           random_state=100)
  #  xg=XGBRegressor(booster='gbtree', max_depth=5, learning_rate=0.05, reg_alpha=0,
  #                 reg_lambda=1, n_jobs=-1, random_state=100, n_estimators=5000)
    
   model=rf.fit(X_train,Y_train)
   pred = model.predict(X_valid)
   
   print(np.sqrt(mean_squared_error(Y_valid, np.abs(pred))))
   mse+=np.sqrt(mean_squared_error(Y_valid,np.abs(pred)))
    
   #y_test_predict+=model.predict(test)  
   #result[j]=model.predict(X_main_test)
   j+=1

results=y_test_predict/10

print(mse/10)


# In[ ]:


x = X_train.append(X_valid)
y = Y_train.append(Y_valid)
x.head()


# In[ ]:


#Training the model on whole dataset
model=rf.fit(x,y)
pred = model.predict(test)
pred = np.abs(pred)
pred


# ## XGBOOST

# In[ ]:


from xgboost import XGBRegressor
#xgb = xgb.XGBRegressor(objective ='reg:squarederror',  learning_rate = 0.1,
#                max_depth = 8, alpha = 10, n_estimators = 400)

xgb=XGBRegressor(booster='gbtree', max_depth=6, learning_rate=0.1, reg_alpha=0,
                  reg_lambda=1, n_jobs=-1, random_state=100, n_estimators=500)
xgb.fit(X_train,Y_train)


# In[ ]:


import eli5
perm = PermutationImportance(xgb,random_state=100).fit(X_val, Y_val)
eli5.show_weights(perm,feature_names=X_val.columns.tolist())


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
kf=KFold(n_splits=10, random_state=100, shuffle=True)

y_test_predict=0
mse = 0
j=1
result={}

for i, (train_index, test_index) in enumerate(kf.split(train)):
    
   Y_train, Y_valid = Y.iloc[train_index], Y.iloc[test_index]
   X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]
   
   print( "\nFold ", j)
   
   xg=XGBRegressor(booster='gbtree', max_depth=6, learning_rate=0.06,
                  n_jobs=-1, random_state=100, n_estimators=800)
    
   model=xg.fit(X_train,Y_train)
   pred = model.predict(X_valid)
   
   print(np.sqrt(mean_squared_error(Y_valid, np.abs(pred))))
   mse+=np.sqrt(mean_squared_error(Y_valid,np.abs(pred)))
    
   #y_test_predict+=model.predict(test)  
   #result[j]=model.predict(X_main_test)
   j+=1

results=y_test_predict/10

print(mse/10)


# In[ ]:


model2=xg.fit(x,y)
pred_xg = model2.predict(test)
pred_xg = np.abs(pred_xg)
pred_xg


# ## LightGBM

# In[ ]:


from lightgbm import LGBMRegressor
lgb = LGBMRegressor(boosting_type='gbdt', objective='regression',metric = 'rmsle',
                      max_depth=6, learning_rate=0.1, 
                      n_estimators=500, nthread=-1, silent=True)
lgb.fit(X_train,Y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
predictions = lgb.predict(X_val)
np.sqrt(mean_squared_error(Y_val, predictions))


# In[ ]:


import eli5
perm = PermutationImportance(lgb,random_state=100).fit(X_val, Y_val)
eli5.show_weights(perm,feature_names=X_val.columns.tolist())


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
kf=KFold(n_splits=10, random_state=100, shuffle=True)

y_test_predict=0
mse = 0
j=1
result={}

for i, (train_index, test_index) in enumerate(kf.split(train)):
    
   Y_train, Y_valid = Y.iloc[train_index], Y.iloc[test_index]
   X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]
   
   print( "\nFold ", j)
   
   lg = LGBMRegressor(boosting_type='gbdt', objective='regression',metric = 'rmsle',
                      max_depth=8, learning_rate=0.025, 
                      n_estimators=750, nthread=-1, silent=True)
    
   model=lg.fit(X_train,Y_train)
   pred = model.predict(X_valid)
   
   print(np.sqrt(mean_squared_error(Y_valid, np.abs(pred))))
   mse+=np.sqrt(mean_squared_error(Y_valid,np.abs(pred)))
    
   #y_test_predict+=model.predict(test)  
   #result[j]=model.predict(X_main_test)
   j+=1

results=y_test_predict/10

print(mse/10)


# In[ ]:


model4=lg.fit(x,y)
pred_lg = model4.predict(test)
pred_lg = np.abs(pred_lg)
pred_lg


# # Summary

# 
# **Public Score(30% dataset)**
# 
# **Metric-rmsle**
# *   Catboost : 0.698
# *   Random Forest: 0.646
# *   Light GBM : 0.687
# *   XGBoost: 0.66
# 
# 
# 
# 

# **Private Score(100% dataset): 0.63579**
# 
# **Model- RandomForestRegressor**
# 

# # Submission File

# In[ ]:


Dataset_Submission=pd.read_excel('/content/Sample_Submission.xlsx')
Dataset_Submission['Selling_Price']=np.exp(pred)
Dataset_Submission.head(10)

Dataset_Submission.to_excel('submission15.xlsx', index=False)


# # Remarks
# 
# ## <a>Do Upvote if you liked the approach and comment if you have any suggestions</a>
# 
# THANKS!!!

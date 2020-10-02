#!/usr/bin/env python
# coding: utf-8

# Author: Gaurav Mishra
# Date: 01/22/2018
# 
# This is my first competition and tried to implement basic regression model to get indepth of the dataset.
# The purpose of this  Multi Regression Techniques is to find insights of Data cleaning/preparation/transformation which will ultimately be used into a regression model.
# 
# This work was influenced by some kernels of the same competition.
# 
# **Regression techniques**
# 
# LASSO Regression
# 
# Elastic Net Regression
# 
# Gradient Boosting Regression
# 
# XGBoost
# 
# LightGBM
# 
# **Score :**
# Root Mean Squared Error
# 
# 
# **Data cleaning features:**
# 
# Imputing missing values by proceeding sequentially through the data.
# 
# Transforming some numerical variables that seem really categorical
# 
# Label Encoding some categorical variables that may contain information in their ordering set
# 
# Getting dummy variables for categorical features.
# 

# Import requred libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


df=pd.read_csv("../input/train.csv")


# **Data Visualization**:
# Let see the basic relations between data before applying Regression Model

# **Sales Description with Overall Quality**

# In[ ]:


sns.regplot(x = 'OverallQual', y = 'SalePrice', data = df, color = 'Blue')


# **Distribution plot of Sale price**

# In[ ]:


sns.distplot(df['SalePrice'], kde = False, color = 'b', hist_kws={'alpha': 0.9})


# **Relation between Neighborhood and SalePrice**

# In[ ]:


plt.figure(figsize = (12, 6))
sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = df)
xt = plt.xticks(rotation=45)


# **Regression Models**

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import time
import matplotlib.pyplot as plt
from sklearn import metrics


# In[ ]:


#Now let's import and put the train and test datasets in  pandas dataframe
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
allData  = [train , test]
train.head(5)


# In[ ]:


train.shape


# **Define data cleaning functions**

# In[ ]:


def numericalCol(x):            
     return x.select_dtypes(include=[np.number]).columns.values

# Delete the columns
def colDelete(x,dropColumn):
    x.drop(dropColumn, axis=1, inplace = True)

def colWithNAs(x):            
    z = x.isnull()
    df = np.sum(z, axis = 0)       # Sum vertically, across rows
    col = df[df > 0].index.values 
    return (col)


# In[ ]:


y = train['SalePrice']  
testIds = test['Id']
train.drop( ['SalePrice', 'Id'], inplace = True, axis = 'columns')
test.drop( ['Id'], inplace = True, axis = 'columns')  
print(" Sale Price and Id has been dropped from the dataset")


# In[ ]:


ntrain = train.shape[0]
ntest= test.shape[0]
ntrain,ntest


# **Data Cleaning**

# In[ ]:


dropColumn = ['MiscFeature','PoolQC', 'Alley','Fence','FireplaceQu']

for dataset in allData:   
    colDelete(dataset,dropColumn)
    columnsReplaceToNumeric = numericalCol(dataset)
    dataset[columnsReplaceToNumeric]=dataset[columnsReplaceToNumeric].fillna(dataset[columnsReplaceToNumeric].mean(), inplace = True)
    columnWithNAs = colWithNAs(dataset)
    dataset[columnWithNAs] = dataset[columnWithNAs].fillna(value = "other")
  


# Setting categorical dataset to numeric

# In[ ]:


allDataDF = pd.concat(allData, axis = 'index')
allDataDF.shape 
allDataAllNumeric = pd.get_dummies(allDataDF)
print(allDataAllNumeric.shape)


# In[ ]:


train = allDataAllNumeric[:ntrain]
test = allDataAllNumeric[ntrain:]
train.shape
test.shape
y_train = y
X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(
                                     train, y_train,
                                     test_size=0.25,
                                     random_state=42
                                     )


# Defining Regression function to be applied for different models

# In[ ]:


def regression(regr,X_test_sparse,y_test_sparse):
    start = time.time()
    regr.fit(X_train_sparse,y_train_sparse)
    end = time.time()
    rf_model_time=(end-start)/60.0
    print("Time taken to model: ", rf_model_time , " minutes" ) 
    
def regressionPlot(regr,X_test_sparse,y_test_sparse,title):
    predictions=regr.predict(X_test_sparse)
    plt.figure(figsize=(10,6))
    plt.scatter(predictions,y_test_sparse,cmap='plasma')
    plt.title(title)
    plt.show()
    
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(np.log1p(y_test_sparse), np.log1p(predictions))))


# **Base models**
# Below basic models may be very sensitive to outliers. So we need to made it more robust on them. For that we use the sklearn's Robustscaler() method on pipeline

# **LASSO Regression** :least absolute shrinkage and selection operator

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# In[ ]:


regression(lasso,X_test_sparse,y_test_sparse)
regressionPlot(lasso,X_test_sparse,y_test_sparse,"Lasso Model")


# **Elastic Net Regression :**The elastic net is a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods.

# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# In[ ]:


regression(ENet,X_test_sparse,y_test_sparse)
regressionPlot(ENet,X_test_sparse,y_test_sparse,"Elastic Net Regression")


# **Gradient Boosting Regression** :It produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.
# 
# With huber loss that makes it robust to outliers

# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[ ]:


regression(GBoost,X_test_sparse,y_test_sparse)
regressionPlot(GBoost,X_test_sparse,y_test_sparse,"Gradient Boosting Regression")


# **XGBoost :**xgboost is short for eXtreme Gradient Boosting package. It is an efficient and scalable
# implementation of gradient boosting framework by (Friedman, 2001) (Friedman et al., 2000).

# In[ ]:


modelXgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# In[ ]:


regression(modelXgb,X_test_sparse,y_test_sparse)
regressionPlot(modelXgb,X_test_sparse,y_test_sparse,"XGBoost")


# **LightGBM: ** A fast, distributed, high performance gradient boosting (GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, used for ranking, classification and many other machine learning tasks.

# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[ ]:


regression(model_lgb,X_test_sparse,y_test_sparse)
regressionPlot(model_lgb,X_test_sparse,y_test_sparse,"LightGBM")


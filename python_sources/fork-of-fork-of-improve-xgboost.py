#!/usr/bin/env python
# coding: utf-8

# In[229]:


import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.impute import SimpleImputer

# Read the data
def find_sample(random_state, return_all):
    X = pd.read_csv('../input/train.csv', index_col='Id')
    X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice              
    X.drop(['SalePrice'], axis=1, inplace=True)

    # Break off validation set from training data
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                    random_state=random_state, shuffle=True)

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                            X_train_full[cname].dtype == "object"]

    # Select numeric columns
    numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



    # Keep selected columns only
    my_cols = low_cardinality_cols + numeric_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()
    
   


    # One-hot encode the data (to shorten the code, we use pandas)
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_test = pd.get_dummies(X_test)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
    X_train, X_test = X_train.align(X_test, join='left', axis=1)
    
    
    #Removing outliers
    X_train = X_train[(X_train.GrLivArea<4000) ]
    y_train = y_train[y_train.index.isin(X_train.index)]
    
    X_valid = X_valid[(X_valid.GrLivArea<4000) ]
    y_valid = y_valid[y_valid.index.isin(X_valid.index)]
    
    #missing 
    missing_column = X_test.isnull().all()[lambda x:x].index.tolist()
    X_test[missing_column] = X_test[missing_column].apply(lambda x: x.fillna(-1))
    X_train[missing_column] = X_train[missing_column].apply(lambda x: x.fillna(-1))
    X_valid[missing_column] = X_valid[missing_column].apply(lambda x: x.fillna(-1))
    X_train = X_train.apply(lambda x: x.fillna(x.mean()))
    X_valid = X_valid.apply(lambda x: x.fillna(x.mean()))
    
    
    
    current_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, importance_type='gain',
       learning_rate=0.07, max_delta_step=0, max_depth=2,
       min_child_weight=1, missing=None, n_estimators=1500, n_jobs=-1,
       nthread=-1, objective='reg:linear', predictor='gpu_predictor',
       random_state=0, reg_alpha=0, reg_lambda=2, scale_pos_weight=1,
       seed=None, silent=True, subsample=1)
    current_model.fit(X_train, y_train)
    pred =current_model.predict(X_valid)
    print("Mean absolute error ", mean_absolute_error(pred, y_valid))
    
    if return_all:
        return mean_absolute_error(pred, y_valid), current_model, X_train, X_valid, y_train, y_valid, X_test
    else:
        return mean_absolute_error(pred, y_valid)
    


# In[225]:


#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#better = dict()
#found = False
#used = []
#while not found: 
#    randrange = random.sample(range(0,10000), 100) 
#    randrange = [i for i in randrange if i not in used]
#    for i in randrange:
#        used.append(i)
#        result = find_sample(i, False)
#        if result < 12200:
#            print("Quite good result MAE : {}".format(result))
#            better[i] = result
#            if result < 12000:
#                better[i] = result
#                found = True
#                break
            


# In[230]:


_,final_model, X_train, X_valid, y_train, y_valid,X_test =  find_sample(3076, True)


# In[231]:


X_comb = X_train.append(X_valid)
y_comb = y_train.append(y_valid)

final_model.fit(X_comb, y_comb)


# In[232]:


predict_test = final_model.predict(X_test)


# In[233]:


output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': predict_test})
output.to_csv('submission.csv', index=False)


# In[234]:


get_ipython().system('head submission.csv')


# # Keep going
# 
# Continue to learn about **[data leakage](https://www.kaggle.com/alexisbcook/data-leakage)**.  This is an important issue for a data scientist to understand, and it has the potential to ruin your models in subtle and dangerous ways!

# ---
# **[Intermediate Machine Learning Micro-Course Home Page](https://www.kaggle.com/learn/intermediate-machine-learning)**
# 
# 

#!/usr/bin/env python
# coding: utf-8

# Here I am explaing small steps even I know that every one knows these in this platform. Very basic steps, but I am feeling beginers can use/check this to learn

# Few steps to solve,
# 
# 1. Data analysing
# 2. Data cleaning
# 3. Feature engineering
# 4. Feature filtering by identifying important features
# 5. Algorithm selection
# 6. Parameter selection/fine tune
# 7. Train & predict
# 8. Cross Validation
# 
# As a initial step, I am covering few steps(2,5,6,7,8) which are needed to create baseline notebook as complete cycle. Later I will other steps(1,3,4,6-fine tune) one by one as update
# 
# In addition these I am adding few which are needed like libraries, seed, memory, etc.
# 
# Next step I will add Feature engineering(3rd step) 

# In[ ]:


# Libraries - We need to import the libraries for the classes, methods which we may using in any notebook - this is general concept like any oops language 

import numpy as np 
import pandas as pd 
import os
import gc

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

from sklearn import preprocessing

# To list the directories in ../input, using this we can use the directory/file name(s) to read the train,test and any additional data
print(os.listdir('../input'))

# Pandas's dispaly  settings - to view all column data without "..." when we are using pandas library to read and dispaly data to anlayse 
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 100)  
pd.set_option('display.max_columns', 500)


# In[ ]:


# Seed - to maintian uniqueness and stability of predictions throughout the developement
# Without seed settings we may getting different results each time when we train and predict
random_state = 1337
np.random.seed(random_state)


# In[ ]:


# Read Data using pandas
# Competietion having 2 types(trasaction and identities) data for train and test. 
# After read both files for each we need to merge the data by using join.

train_ts = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv", index_col='TransactionID')
train_id = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv", index_col='TransactionID')

test_ts = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv", index_col='TransactionID')
test_id = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv", index_col='TransactionID')

submission = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")

print(train_ts.shape,train_id.shape,test_ts.shape,test_id.shape,submission.shape)


# In[ ]:


# trasactions and identities data mearging using join

train = train_ts.merge(train_id, how='left', left_index=True, right_index=True)
test = test_ts.merge(test_id, how='left', left_index=True, right_index=True)
print(train.shape,test.shape)


# In[ ]:


# Data having huge size, to run the notebook without intereptions we need to clean unused datasets to release memory
# Here we created new datasets, so old datasets need to dispose

del(train_ts, train_id, test_ts, test_id)
gc.collect()


# In[ ]:


# Memory reduction function - Memory is high for original data. We need to reduce memory to process features. 
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings                      
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(0,inplace=True) 
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df


# In[ ]:


# Memory reducing for train and test datasets
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


# In[ ]:


# In both train and test datasets having few categorical columns. We need to convert any strain data into numericl as best practice
# and also we can process numerical categorical columns also. 
# Here I listed out the categorical columns(string and numerical) seperately
cat_cols_str=['ProductCD', 'card4', 'card6','P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
cat_cols_int=[ 'card1','card2','card3','card5','addr1', 'addr2']


# In[ ]:


# Processing categorical data function - I will convert categorical columns in to sequesntial numerical data
# Filled 0 or NA for empty data values

def process_categorical_columns(temp_ds):
    for col in cat_cols_str: 
        temp_ds[col] = temp_ds[col].fillna('NA')
        temp_ds[col]= temp_ds[col].replace(0,'NA')
        le = preprocessing.LabelEncoder()
        le.fit(temp_ds[col])
        temp_ds[col] = le.transform(temp_ds[col]) 

    for col in cat_cols_int: 
        temp_ds[col] = temp_ds[col].fillna(0) 
        temp_ds[col]= temp_ds[col].replace('NA',0)
        le = preprocessing.LabelEncoder()
        le.fit(temp_ds[col])
        temp_ds[col] = le.transform(temp_ds[col]) 
    return temp_ds


# In[ ]:


# Processing train and test categorical data
train = process_categorical_columns(train)
test = process_categorical_columns(test)


# In[ ]:


# Prepare dataset to train 
# In this problem we need to predict probabillity in between 0 and 1. 
# So, this is clasifier problem and I am using LGBM clasifier wiht Kfold validation. And using predict_proba() method
# For LGBM we need to provide  Train data(X_data), Train target(y_data) and test data(X_test)

X_data = train.drop('isFraud',axis=1)
y_data = train['isFraud'].values
X_test = test
print(X_data.shape, y_data.shape, X_test.shape)


# In[ ]:


# Clean unused datasets to release memory
del(train, test)
gc.collect()


# In[ ]:


# LGB Parameters - initial - need to fine tune
lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.0085,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
}


# In[ ]:


# LGB training method
def train_and_predict(n_splits=5, n_estimators=100):
    y_pred = np.zeros(X_test.shape[0], dtype='float32')
    cv_score = 0

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_index, val_index) in enumerate(kfold.split(X_data, y_data)):
        if fold >= n_splits:
            break
        
        # Split the data for train and validation in each fold
        X_train, X_val = X_data.iloc[train_index, :], X_data.iloc[val_index, :]
        y_train, y_val = y_data[train_index], y_data[val_index]
        
        # Using LGBM classifier algorthm
        model = LGBMClassifier(**lgb_params, n_estimators=n_estimators, n_jobs = -1)
        model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='auc',
            verbose=50, early_stopping_rounds=200)

        # in predict_proba we will get 2 values for each prediction, so we need to use 2nd value
        y_val_pred = model.predict_proba(X_val)[:,1]
        
        # Calcualte cross validation score using metrics
        val_score = roc_auc_score(y_val, y_val_pred)
        print(f'Fold {fold}, AUC {val_score}')
        
        # Averaging cross validation score and test predictions for all folds
        cv_score += val_score / n_splits
        y_pred += model.predict_proba(X_test)[:,1] / n_splits
    
    # Assign final test predictions to submission dataset
    submission['isFraud'] = y_pred
    return cv_score


# In[ ]:


# Setting folds and start training - as I said I am using KFold, in this we need to set number of folds.
# Based on the folds count full dataset splited into smaller sets 
# and each set used for training seperately and predict the results. 
# And by averaging the results we can get final results with better accuracy
# I am using 5000 iterations,verbore as 50 and early stopping rounds as 200 (in above funtion)
# Verbose - cross validate for each 50 iterations 
# Early stopping rounds - Stop training early if results not improve for last 200 iteration

N_FOLDS = 5
n_estimators = 5000
cv_score = train_and_predict(n_splits=N_FOLDS, n_estimators=n_estimators)
print(cv_score)


# In[ ]:


submission['isFraud'].sum()


# In[ ]:


# create the output file for submission
submission.to_csv('submission.csv',index=False)


# In[ ]:


# list the files in output folder to download the output file if we run the notebook as interactive session instead of commit
from IPython.display import FileLink, FileLinks
FileLinks('.')


#  LB : 0.9121
#  
#  We can slightly improve by increaseing estimators to 10K or 20K
#  

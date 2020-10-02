#!/usr/bin/env python
# coding: utf-8

# **Importing packages**

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
train_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
test_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')


# **Merging the two datasets**

# In[ ]:


train = pd.merge(train_trans, train_id, on='TransactionID', how='left')
test = pd.merge(test_trans, test_id, on='TransactionID', how='left')


# **Reducing Memory usage**

# In[ ]:


def memory_usage_mb(train, *args, **kwargs):
    """Dataframe memory usage in MB. """
    return train.memory_usage(*args, **kwargs).sum() / 1024**2

def reduce_memory_usage(train, deep=True, verbose=True):
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(train, deep=deep)

    for col, col_type in train.dtypes.iteritems():
        best_type = None
        if col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            train[col] = pd.to_numeric(train[col], downcast=downcast)
            best_type = train[col].dtype.name
        # Log the conversion performed.
        if verbose and best_type is not None and best_type != str(col_type):
            print(f"Column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(train, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")
        
    return train


# In[ ]:


def memory_usage_mb(test, *args, **kwargs):
    """Dataframe memory usage in MB. """
    return train.memory_usage(*args, **kwargs).sum() / 1024**2

def reduce_memory_usage(test, deep=True, verbose=True):
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(test, deep=deep)

    for col, col_type in test.dtypes.iteritems():
        best_type = None
        if col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            test[col] = pd.to_numeric(test[col], downcast=downcast)
            best_type = test[col].dtype.name
        # Log the conversion performed.
        if verbose and best_type is not None and best_type != str(col_type):
            print(f"Column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(test, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")
        
    return test


# In[ ]:


del train_id, train_trans, test_id, test_trans


# In[ ]:


train.columns


# **Treating Missing Values**

# In[ ]:


train.dropna(thresh = 0.5*len(train),how ='all', axis=1, inplace = True)


# In[ ]:


train.isnull().sum()


# In[ ]:


list(train.dtypes)


# **Splitting data into numeric and categorical columns**

# In[ ]:


train_cat =train.select_dtypes(include=['object', 'O']).copy()
train_num =train.select_dtypes(exclude=['object', 'O']).copy()


# In[ ]:


mm_scaler = MinMaxScaler()
mm_scaler.fit_transform(train_num)


# **Replacing Null Values**

# In[ ]:


train_cat.isnull().sum()


# In[ ]:


for col in train_cat:
   train_cat[col] = train_cat[col].replace(np.nan, train_cat[col].mode()[0])


# In[ ]:


for col in train_num:
   train_num[col] = train_num[col].replace(np.nan, train_num[col].mean())


# In[ ]:


train_num = reduce_memory_usage(train_num, deep=True, verbose=True)
print(train_num.head(10))


# **Creating dummy values for the categorical column values**

# In[ ]:


dummy = LabelEncoder()
train_catg = train_cat.apply(dummy.fit_transform)
train_catg.head()


# Select upper triangle of correlation matrix

# **Dropping columns with correlation greater than 0.95**

# In[ ]:


corr_matrix = train_num.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


# In[ ]:


plt.matshow(train_num.corr())


# In[ ]:


to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]


# In[ ]:


train_num.drop(train_num[to_drop], axis=1, inplace=True)


# In[ ]:


train_num.drop(['isFraud'], axis=1, inplace=True)


# **Defining the target and the features**

# In[ ]:


frames = [train_catg,train_num]
features = pd.concat(frames, axis=1)


# In[ ]:


features


# In[ ]:


target = train['isFraud']


# **Test Data**

# In[ ]:


test = pd.get_dummies(test)


# **Data Modelling**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)


# In[ ]:


xgboost_classifier = XGBClassifier()


# In[ ]:


xgboost_classifier.fit(X_train, y_train)


# In[ ]:


predictions = xgboost_classifier.predict(X_test)


# In[ ]:


test["isFraud"] = xgboost_classifier.predict_proba(test)[:,1]


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(predictions, y_test))


# In[ ]:


print(classification_report(predictions, y_test))


# In[ ]:


submission = test[["TransactionID", "isFraud"]]
submission.head()
submission.to_csv("submission.csv", index = False)


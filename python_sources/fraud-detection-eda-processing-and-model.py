#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# several prints in one cell.
import warnings
warnings.filterwarnings(action="ignore")

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

for c, o,files in os.walk("/kaggle/input"):
    print(files)


# In[ ]:


path = "/kaggle/input/ieee-fraud-detection/"
train_df_id = pd.read_csv(path + 'train_identity.csv')
train_df_tran = pd.read_csv(path + "train_transaction.csv")


# In[ ]:


test_df_id = pd.read_csv(path + 'test_identity.csv')
test_df_tran = pd.read_csv(path + "test_transaction.csv")


# # Merge the train DF

# In[ ]:


train_df = pd.merge(train_df_tran, train_df_id, on = "TransactionID", how = "left")
test_df = pd.merge(test_df_tran, test_df_id, on = "TransactionID", how = "left")
test_df["isFraud"] = 0


# In[ ]:


train_df.head()


# In[ ]:


short_df = train_df[["TransactionID", "isFraud"]]
short_df.sort_values("TransactionID", ascending = True)
short_df.set_index("TransactionID", inplace = True)


# In[ ]:


# code for @niefangchao

# converting to nan so that we can see where the nans
short_df["isFraud"] = short_df["isFraud"].apply(lambda x: np.nan if x == 1 else 0)
plt.figure(figsize = (20, 10))
sns.heatmap(short_df.isnull(), cbar=False)
plt.title("Check the line 3352092 and 3405782");


# **EDA functions that we will use in our descriptive analysis**

# In[ ]:


def df_grouper(df, columns_to_group, level = 1):
    '''
    A functions that groups the df by the columns you pass in.
    '''
    
    for column in columns_to_group:
        assert column in list(df.columns), "Column {} not contained in DF".format(column)
    
    df_to_group = df[columns_to_group]
    if level == 1:
        grouped_df = pd.DataFrame(df_to_group.groupby(columns_to_group).size()).unstack(level = 1)[0]
    elif level == 0:
        grouped_df = pd.DataFrame(df_to_group.groupby(columns_to_group).size()).unstack(level = 0)[0]
    else:
        try:
            grouped_df = pd.DataFrame(df_to_group.groupby(columns_to_group).size()).unstack(level = level)[0]
        except:
            pass
            
    return grouped_df


# In[ ]:


# https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt#Mapping-emails
# thanks for the emails mappings

emails = {
'gmail': 'google', 
'att.net': 'att', 
'twc.com': 'spectrum', 
'scranton.edu': 'other', 
'optonline.net': 'other', 
'hotmail.co.uk': 'microsoft',
'comcast.net': 'other', 
'yahoo.com.mx': 'yahoo', 
'yahoo.fr': 'yahoo',
'yahoo.es': 'yahoo', 
'charter.net': 'spectrum', 
'live.com': 'microsoft', 
'aim.com': 'aol', 
'hotmail.de': 'microsoft', 
'centurylink.net': 'centurylink',
'gmail.com': 'google', 
'me.com': 'apple', 
'earthlink.net': 'other', 
'gmx.de': 'other',
'web.de': 'other', 
'cfl.rr.com': 'other', 
'hotmail.com': 'microsoft', 
'protonmail.com': 'other', 
'hotmail.fr': 'microsoft', 
'windstream.net': 'other', 
'outlook.es': 'microsoft', 
'yahoo.co.jp': 'yahoo', 
'yahoo.de': 'yahoo',
'servicios-ta.com': 'other', 
'netzero.net': 'other', 
'suddenlink.net': 'other',
'roadrunner.com': 'other', 
'sc.rr.com': 'other', 
'live.fr': 'microsoft',
'verizon.net': 'yahoo', 
'msn.com': 'microsoft', 
'q.com': 'centurylink', 
'prodigy.net.mx': 'att', 
'frontier.com': 'yahoo', 
'anonymous.com': 'other', 
'rocketmail.com': 'yahoo',
'sbcglobal.net': 'att',
'frontiernet.net': 'yahoo', 
'ymail.com': 'yahoo',
'outlook.com': 'microsoft',
'mail.com': 'other', 
'bellsouth.net': 'other',
'embarqmail.com': 'centurylink',
'cableone.net': 'other', 
'hotmail.es': 'microsoft', 
'mac.com': 'apple',
'yahoo.co.uk': 'yahoo',
'netzero.com': 'other', 
'yahoo.com': 'yahoo', 
'live.com.mx': 'microsoft',
'ptd.net': 'other',
'cox.net': 'other',
'aol.com': 'aol',
'juno.com': 'other',
'icloud.com': 'apple'
}

# number types for filtering the columns
int_types = ["int8", "int16", "int32", "int64", "float"]


# **Functions that we will use in our pipeline/preprocessing part**

# In[ ]:


# Let's check how many missing values has each column.

def check_nan(df, limit):
    '''
    Check how many values are missing in each column.
    If the number of missing values are higher than limit, we drop the column.

    NOTE: that it modifies the df inplace.
    '''
    
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    
    total_dropped = 0
    col_to_drop = []
    
    for col in df.columns:

        null_sum = df[col].isnull().sum()
        perc_over_total = round((null_sum/total_rows), 2)
        
        if perc_over_total > limit:
            
            col_to_drop.append(col)
            total_dropped += 1            
    
    df.drop(col_to_drop, axis = 1, inplace = True)
    
    print("We have dropped a total of {} columns.\nIt's {} of the total"          .format(total_dropped, round((total_dropped/total_cols), 2)))
    
    return df


# In[ ]:


def binarizer(df_train, df_test):
    '''
    Work with cat features and binarize the values.
    Works with 2 dataframes at a time and returns a tupple of both.
    '''
    cat_cols = df_train.select_dtypes(exclude=int_types).columns

    for col in cat_cols:
        
        # creating a list of unique features to binarize so we dont get and value error
        unique_train = list(df_train[col].unique())
        unique_test = list(df_test[col].unique())
        unique_values = list(set(unique_train + unique_test))
        
        enc = LabelEncoder()
        enc.fit(unique_values)
        
        df_train[col] = enc.transform((df_train[col].values).reshape(-1 ,1))
        df_test[col] = enc.transform((df_test[col].values).reshape(-1 ,1))
    
    return (df_train, df_test)


# In[ ]:


def cathegorical_imputer(df_train, df_test, strategy, fill_value):
    '''
    Replace all cathegorical features with a constant or the most frequent strategy.
    '''

    cat_cols = df_train.select_dtypes(exclude=int_types).columns
    
    for col in cat_cols:
        
        # select the correct inputer
        if strategy == "constant":
            # input a fill_value of -999 to all nulls
            inputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        elif strategy == "most_frequent":
            inputer = SimpleImputer(strategy=strategy)
        
        # replace the nulls in train and test
        df_train[col] = inputer.fit_transform(X = (df_train[col].values).reshape(-1, 1))
        df_test[col] = inputer.transform(X = (df_test[col].values).reshape(-1, 1))
        
    return (df_train, df_test)


# In[ ]:


def numerical_inputer(df_train, df_test, strategy, fill_value):
    '''
    Replace NaN in the numerical features.
    Works with 2 dataframes at a time (train & test).
    Return a tupple of both.
    '''
    
    # assert valid strategy
    message = "Please select a valid strategy (mean, median, constant (and give a fill_value) or most_frequent)"
    assert strategy in ["constant", "most_frequent", "mean", "median"], message
    
    # int_types defined earlier in the kernel
    num_cols = df_train.select_dtypes(include = int_types).columns
    
    for col in num_cols:

        # select the correct inputer
        if strategy == "constant":
            inputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        elif strategy == "most_frequent":
            inputer = SimpleImputer(strategy=strategy)
        elif strategy == "mean":
            inputer = SimpleImputer(strategy=strategy)
        elif strategy == "median":
            inputer = SimpleImputer(strategy=strategy)

        # replace the nulls in train and test
        try:
            df_train[col] = inputer.fit_transform(X = (df_train[col].values).reshape(-1, 1))
            df_test[col] = inputer.transform(X = (df_test[col].values).reshape(-1, 1))
        except:
            pass
            
    return (df_train, df_test)


# # PROCESSING PART OF THE KERNEL

# In[ ]:


def pipeline(df_train, df_test):
    '''
    We define a personal pipeline to process the data and fill with processing functions.
    NOTE: modifies the df in place.
    '''
    # We have set the limit of 30%. If a column contains more that 30% of it's values as NaN/Missing values we will drop the column
    # Since it's very unlikely that it will help our future model.
    df_train = check_nan(df_train, limit=0.3)
    
    # Select the columns from df_train with less nulls and asign to test.
    df_test = df_test[list(df_train.columns)]
          
    # mapping emails
    df_train["EMAILP"] = df_train["P_emaildomain"].map(emails)
    df_test["EMAILP"] = df_test["P_emaildomain"].map(emails)
          
    # replace nulls from the train and test df with a value of "Other"
    df_train, df_test = cathegorical_imputer(df_train, df_test, strategy = "constant", fill_value = "Other")
          
    # now we will make a one hot encoder of these colums
    df_train, df_test = binarizer(df_train, df_test)
          
    # working with null values in numeric columns
    df_train, df_test = numerical_inputer(df_train, df_test, strategy = "constant", fill_value=-999)
          
    return (df_train, df_test)


# In[ ]:


train_df, test_df = pipeline(train_df, test_df)


# In[ ]:


# check for null values
columns = train_df.columns
for col in  columns:
    total_nulls = train_df[col].isnull().sum()
    if total_nulls > 0:
        pass
        
columns = test_df.select_dtypes(exclude=int_types).columns
columns = test_df.select_dtypes(include=int_types).columns


# > # Modelling part: to be continued

# In[ ]:


#list(train_df.columns)
cols_to_drop = ["TransactionID", "isFraud", "TransactionDT"]
useful_cols = list(train_df.columns)

for col in cols_to_drop:
    useful_cols.remove(col)

Y = train_df["isFraud"]
X = train_df[useful_cols]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)


# In[ ]:


xgboost_classifier = XGBClassifier()

xgboost_classifier.fit(X_train, y_train)


# In[ ]:


predictions = xgboost_classifier.predict(X_test)


# In[ ]:


print(confusion_matrix(predictions, y_test))


# In[ ]:


print(classification_report(predictions, y_test))


# In[ ]:


test_df["isFraud"] = 0


# In[ ]:


proba = xgboost_classifier.predict_proba(test_df[useful_cols])
proba[:,1]

test_df["isFraud"] = proba[:,1]

submission = test_df[["TransactionID", "isFraud"]]
submission.to_csv("submission.csv", index = False)


# In[ ]:


sub_check = pd.read_csv("submission.csv")
sub_check.head()


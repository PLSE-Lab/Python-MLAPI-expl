#!/usr/bin/env python
# coding: utf-8

# # Fraud Detection
# ### August 2019

# This is the second kernel created within the competition and was started on 28 August 2019. My aims are:
# 1. Cut out some of the initial data checks, as these were completed in a previous kernel
# 2. Analyse each feature to understand the type of data, the one-way relationship to target and key correlations with other features
# 3. Propose how to bring each feature through to the model (or whether to remove)
# 4. Consider possible engineered features and profile as in Step 2
# 
# I hope to reach this stage by 8 September 2019. This kernel will then inform a kernel where a more complex model is fitted.

# In[ ]:


# Import packages
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import os
import time
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import ipywidgets as widgets
from IPython.display import display


# In[ ]:


t_Start = time.time()


# ## 1) Import, check and prepare data

# In[ ]:


t1_Start = time.time()


# In[ ]:


# Created dataframes from input files
train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
train_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
sample_sub = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

print('Data imported')


# The training data contains 590,540 rows, with features split across two dataframes:
# - train_trans contains 392 features (plus an identifier and a fraud flag) which appear to relate to the transaction details (e.g. amount, card type)
# - train_id contains a further 40 features (plus an identifier) for around 25% of transactions. The features appear to relate to the online route of the transaction (e.g. browser, device)

# The test data contain 506,691 rows, with around 28% also having ID features. All columns are consistent between the training and the test data (other than the target variable).

# In[ ]:


t1_End = time.time()
if (t1_End - t1_Start) < 60:
    print('Step 1 completed - Time taken %3.1f seconds' % (t1_End - t1_Start))
else:
    print('Step 1 completed - Time taken %i minutes and %i seconds' % ((t1_End - t1_Start)//60,(t1_End - t1_Start)%60))


# ## 2) Analysis of the Data

# In[ ]:


t2_Start = time.time()


# In the training dataset, 3.5% of transactions are fraudulent

# In[ ]:


avg_fraud_rate = train_trans.isFraud.sum()/train_trans.shape[0]


# In[ ]:


# Merge the ID data and the transaction data

train_merged = pd.merge(train_trans, train_id, how='outer', on='TransactionID')


# Use the dropdown box to look at a basic dashboard for each feature

# In[ ]:


w = widgets.Dropdown(options=train_merged.columns.drop(['TransactionID','isFraud']), value='card6', description='column:', disabled=False)


# In[ ]:


# Function to create variables based on selected column

def Display_Features(column):
    ''' (str) --> None
    
    For the selected column, displays various features
    '''
    # Data type
    data_type = train_merged[column].dtype

    # Proportion of missing values
    missing_proportion = train_merged[column].isna().sum()/train_merged[column].count()
    
    # Dataframe of most common values and total other values 
    top_9 = train_merged.groupby(column).count()['TransactionID'].sort_values(ascending=False).head(9)
    top_10 = top_9.copy()
    top_10.loc['other'] = train_merged.groupby(column).count()['TransactionID'].sum()-top_9.sum()
    
    print('The data type is %s' % str(data_type))
    print('%3.1f%% of the data is missing' % (missing_proportion*100))
    display(top_10.to_frame())
# Bar chart of frequencies

def Display_Chart(column):
    ''' (str) --> None
    
    For the selected column, displays a plot of frequencies and fraud rate
    '''
    fig, ax = plt.subplots(1, 1, figsize=(8, 5));

    # bar chart of frequencies
    train_merged.groupby(column).count()['TransactionID'].plot.bar(color='blue',alpha=0.5,ax=ax);
    bx = ax.twinx()
    ax.set_title('Frequency of values (fraud % overlaid)');
    ax.set_xlabel('Value');
    ax.set_ylabel('Frequency');
    ax.set_xticks(range(len(train_merged.groupby(column).count())))
    ax.set_xticklabels(train_merged.groupby(column).count()['TransactionID'].index)
    # line chart of fraud rate
    (1-train_merged[train_merged['isFraud']==0].groupby(column).count()/train_merged.groupby(column).count())['TransactionID'].plot.line(ax=bx,color='red')
    bx.set_ylabel('Fraud %');
    
    plt.show()
    


# In[ ]:


selection = widgets.Output()

def Widger_Event_Handler(change):
    ''' None --> None
    Function to automatically display dashboard when widget changed
    '''
    selection.clear_output()
    with selection:
        Display_Features(change.new)
        Display_Chart(change.new)
    
w.observe(Widger_Event_Handler, names='value')

display(w)


# In[ ]:


display(selection)


# Most of the columns are numeric (although not clear if the numbers may represent categories). There are a lot of missing values with some columns very sparsely populated.

# In[ ]:


t2_End = time.time()
if (t2_End - t2_Start) < 60:
    print('Step 2 completed - Time taken %3.1f seconds' % (t2_End - t2_Start))
else:
    print('Step 2 completed - Time taken %i minutes and %i seconds' % ((t2_End - t2_Start)//60,(t2_End - t2_Start)%60))


# ## 3) Fit ML Model

# In[ ]:


t3_Start = time.time()


# In[ ]:


# Set up target variable

train_trans = train_trans.set_index('TransactionID')
y_train = train_trans.isFraud


# In[ ]:


# Define a function to manipulate the data for input into the model
# At this stage, just cut the data down to fully populated, numeric columns

def DataForInput(trans_data,ID_data):
    ''' (df, df) --> df
    
    Takes the transaction and ID dataframes and combines them to create a single dataframe suitable for input into the model.
    
    Set up as a function so can be repeated for training and test dataset
    '''
    
    return trans_data['TransactionAmt'] 


# In[ ]:


# Set up the training data ready for model training

X_train = DataForInput(train_trans,train_id)


# In[ ]:


# Fit a model

#xgboost = xgb.XGBClassifier().fit(X_train,y_train)


# In[ ]:


# Work out some performance statistics using cross-validation
'''
xgb_accuracy = cross_val_score(xgboost, X_train, y_train, cv=5).mean()
xgb_auc = cross_val_score(xgboost, X_train, y_train, scoring='roc_auc', cv=5).mean()

print('The CV model accuracy score is %3.1f%%' % (xgb_accuracy*100))
print('The CV model AUC score is %3.5f' % (xgb_auc))
'''


# In[ ]:


t3_End = time.time()
if (t3_End - t3_Start) < 60:
    print('Step 3 completed - Time taken %3.1f seconds' % (t3_End - t3_Start))
else:
    print('Step 3 completed - Time taken %i minutes and %i seconds' % ((t3_End - t3_Start)//60,(t3_End - t3_Start)%60))


# ## 4) Create Submission File

# In[ ]:


t4_Start = time.time()


# In[ ]:


# Create predictions

test_trans = test_trans.set_index('TransactionID')
X_test = DataForInput(test_trans,test_id)
#X_test.preds = xgboost.predict_proba(X_test)
X_test.preds = 0.5
sample_sub.isFraud = 1 - X_test.preds
sample_sub.to_csv('submission.csv',index=False)


# The predictions are distributed at the lower end of the distribution. The predicted fraud rate of 4.24% is possibly a little high but appears reasonable.

# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(sample_sub.isFraud,bins=100);
plt.title('Distribution of prediction probabilities in submission');
plt.xlabel('Probability');
plt.ylabel('Count');


# In[ ]:


mean = sample_sub.isFraud.mean()

print('The predicted fraud rate is %3.2f%%' % (mean*100))


# In[ ]:


t4_End = time.time()
if (t4_End - t4_Start) < 60:
    print('Step 4 completed - Time taken %3.1f seconds' % (t4_End - t4_Start))
else:
    print('Step 4 completed - Time taken %i minutes and %i seconds' % ((t4_End - t4_Start)//60,(t4_End - t4_Start)%60))


# In[ ]:


t_End = time.time()
print('Notebook finished - Total run time = %i minutes and %i seconds' % ((t_End - t_Start)//60,(t_End - t_Start)%60))


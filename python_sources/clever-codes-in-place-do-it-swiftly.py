#!/usr/bin/env python
# coding: utf-8

# ## What are "Clever Codes In Place" All About? 
# 
# Hi Kaggle Community, 
#          I am working on various coding scripts for a while and a thought occured that I could write better scripts with fewer lines and improve my personal codebank with short as well as clever codes. The main ideology is to improve the readability and the speed of the code. This notebook should iteratively add all the CCIP (Clever Code In Place) for this competition. Please do feel free share your clever code and approaches in the comments for the data wrangling, loading, quick EDA and ASAP stuffs for ML. 

# In[8]:


import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))


# ## Data Wrangling 
# When there are many CSV files around. I tend to use python dictionaries format. Its simple to grab specific dataframes later as well as saves variable name spacespaces. 

# In[9]:


data = {}
data['application_train'] = pd.read_csv('../input/application_train.csv')
data['POS_CASH_balance'] = pd.read_csv('../input/POS_CASH_balance.csv')
data['bureau_balance'] = pd.read_csv('../input/bureau_balance.csv')
data['previous_application'] = pd.read_csv('../input/previous_application.csv')
data['installments_payments'] = pd.read_csv('../input/installments_payments.csv')
data['credit_card_balance'] = pd.read_csv('../input/credit_card_balance.csv')
data['sample_submission'] = pd.read_csv('../input/sample_submission.csv')
data['application_test'] = pd.read_csv('../input/application_test.csv')
data['bureau'] = pd.read_csv('../input/bureau.csv')


# In[11]:


data['application_train'].head()


# Yeah! Now we have succesfully, wrangled data into a python dict object. There are other ways to simply this better using for loop and *glob* library. But, for now I will stick with this in the favour of understanding and readability. 

# ### Mr. Inspector Frame

# One of my favourite yet simple dataframe making tool to understand and have better glimpse of the frame. I call it Mr. Inspector as the function implies the same. Let's say hi.. to Mr. Inspector Frame

# In[12]:


def mr_inspect(df):
    """Returns a inspection dataframe"""
    print ("Length of dataframe:", len(df))
    inspect_dataframe = pd.DataFrame({'dtype': df.dtypes, 'Unique values': df.nunique() ,
                 'Number of missing values': df.isnull().sum() ,
                  'Percentage missing': (df.isnull().sum() / len(df)) * 100
                 }).sort_values(by='Number of missing values', ascending = False)
    return inspect_dataframe


# In[15]:


mr_inspect(data['credit_card_balance'])


# That's our inspection. You see this tiny dataframe saves lot of single lined commands to dtypes, values counts and what not.. Although this is not a magic wand. There are dataframes with high dimensionality that Mr. Inspector looses his mind inspecting and such a dataframe is so long that we cant conclude on the summary.

# ## Gotta get them all !
# Often we require features of specific types. Here's a short code function snippet. Most of the experienced user might not need this functions but a beginner will cheer the use. BTW, select_dtype functionality of pandas is fewer known technique.

# In[27]:


def get_num_cols(df):
    """Returns list of columns that are numeric"""
    return list(df.select_dtypes(include=['int']).columns)

def get_cat_cols(df):
    """Returns list of columns that are non-numeric"""
    return list(df.select_dtypes(exclude=['int']).columns)


# ## Getting 15 numerical columns for the simplicity

# In[28]:


get_num_cols(data['application_train'])[:15]


# ## Getting 15 categorical columns 

# In[30]:


get_cat_cols(data['application_train'])[:15]


# ### That's all for Day 1. The notebook shall be updated with feature enginnering and easy EDA steps soon. Mean while feel free to share your "Clever Codes In Place"

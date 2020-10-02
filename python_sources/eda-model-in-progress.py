#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


app_train = pd.read_csv('../input/application_train.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
installments_payments = pd.read_csv('../input/installments_payments.csv')
POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
previous_application = pd.read_csv('../input/previous_application.csv')


# In[ ]:


#Lets check the application_train file
app_train.head()


# In[ ]:


#Lets check how many loans are cash loans and how many of them are Revolving loans
app_train['NAME_CONTRACT_TYPE'].value_counts()


# In[ ]:


#Percentage of cash loan vs revolving loans
app_train['NAME_CONTRACT_TYPE'].value_counts(True)


# Majority of the loans are cash loans (almost 90%)

# In[ ]:


#Now lets observe how many are cash loans and revolving loan out of all the lons where customer with payment difficulties
# for that first we need to convert the values in NAME_CONTRACT_TYPE columns with numerical values, and lets store the values in a new column 
app_train['NUM_NAME_CONTRACT_TYPE'] = app_train['NAME_CONTRACT_TYPE'].map({'Cash loans':1,'Revolving loans':0 })


# In[ ]:


app_train[app_train['TARGET'] == 1]['NUM_NAME_CONTRACT_TYPE'].value_counts(True)


# 93.5% of total number of loans where customer with payment difficulties are cash loans

# In[ ]:


#lets see how many of the loans are given to customers with payment diificulty and other cases
app_train['TARGET'].value_counts()


# In[ ]:


app_train['TARGET'].value_counts(True)


# Only 8% of the loans are given to customer having payment difficulty, it also seems we are dealing with imbalanced classification so we may need to use oversampling/undersampling/SMOTE 

# In[ ]:


#lets see how many customers having payment difficulties are male
app_train['NUM_CODE_GENDER'] = app_train['CODE_GENDER'].map({'M':1,'F':0 })
app_train[app_train['TARGET'] == 1]['NUM_CODE_GENDER'].value_counts(True)


# Wow, females are 57% , i thought the higher perecentage would be male

# In[ ]:


#FLAG_OWN_CAR	FLAG_OWN_REALTY	CNT_CHILDREN
#Similarly lets observe how many customers having payment difficulties are having there own car
app_train['NUM_FLAG_OWN_CAR'] = app_train['FLAG_OWN_CAR'].map({'Y':1,'N':0 })
app_train[app_train['TARGET'] == 1]['NUM_FLAG_OWN_CAR'].value_counts(True)


# So 69% have no car

# In[ ]:


#Similarly lets observe how many customers having payment difficulties are having there own house or flat
app_train['NUM_FLAG_OWN_REALTY'] = app_train['FLAG_OWN_REALTY'].map({'Y':1,'N':0 })
app_train[app_train['TARGET'] == 1]['NUM_FLAG_OWN_REALTY'].value_counts(True)


# 68% have there own house or flats

# In[ ]:


#Similarly, lets check for the number of children of customers having payment difficulty
app_train[app_train['TARGET'] == 1]['CNT_CHILDREN'].value_counts(True)


# So, most of the customers having payment difficulty are having no children

# In[ ]:


#Now lets just filter the cases where Target = 1
app_train_t1 =  app_train[app_train['TARGET'] == 1]


# In[ ]:


#18 July


# In[ ]:





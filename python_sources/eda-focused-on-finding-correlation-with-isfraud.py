#!/usr/bin/env python
# coding: utf-8

# Hello world!
# In this notebook you will find a short EDA focused on correlation between variables and the `isFraud` feature.
# 
# First of all, let's load packages and train data.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

# merge the two dataframe in a single set
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)


# Numerical data are find in all `C`, `D`, `V` columns, some `id_` and `card` columns, `TransactionAmt`, `addr1-2` and `dist1-2`.
# 
# Let's select them and check the correlation among all these variables.

# In[ ]:


# compile list of C columns
c_list = []
for i in range(1,15):    
    c_list.append('C' + str(i))

# compile list of D columns
d_list = []
for i in range(1,16):    
    d_list.append('D' + str(i))

# compile list of V columns
v_list = []
for i in range(1,340):    
    v_list.append('V' + str(i))

# compile list of id columns
id_list = []
for i in range(1,12):
    ii = str(i).zfill(2)    
    id_list.append('id_' + str(ii))
id_list.extend(['id_13','id_14','id_17','id_18','id_19','id_20','id_21',                'id_22','id_24','id_25','id_26','id_32'])

# create list of columns label for correlation check
colums_for_corr = ['isFraud','TransactionAmt','card1','card2','card3','card5','addr1','addr2','dist1','dist2']
colums_for_corr.extend(c_list)
colums_for_corr.extend(d_list)
colums_for_corr.extend(v_list)
colums_for_corr.extend(id_list)

corr_matrix = train[colums_for_corr].corr()


# Let's plot a heatmap to have a sight about correlation. 
# Generally speaking, `C` and `D` show a high internal correlation. The same for `V`, but with some exceptions.

# In[ ]:


sns.set(rc={'figure.figsize':(20,20)})
sns.heatmap(corr_matrix, cmap="RdBu")


# From the global correlation matrix, let's extract just the *isFraud* column and see labels with correlation higher than 0.20. (also correlation < - 0.2 would be interesting, but the minimum value encountered was about -0.14.

# In[ ]:


isFraud_corr = corr_matrix['isFraud']

columns_with_correlation = isFraud_corr.loc[isFraud_corr >= 0.2]

print(columns_with_correlation.sort_values(ascending=False))


# For categorical variables, let's explore via graph the correlation between labels and `isFraud`.
# 
# First of all, let's create a function that counts the 1 and 0 values in `isFraud` for each label. Please note that this function neglect all labels that have at least one value with `isFraud=0` but no one with `isFraud=1`. At this stage, this is not a high limitation as we are interested to check correlation with `isFraud=1`.

# In[ ]:


def check_corr_fraud(df, col):
    
    fr1 = (df.loc[df['isFraud'] == 1][col].value_counts())
    fr0 = (df.loc[df['isFraud'] == 0][col].value_counts())
    
    fr = pd.merge(fr1, fr0, how = 'left',
                  left_on  = fr1.index,
                  right_on = fr0.index)
    
    fr = fr.rename(columns={'key_0': col,
                            f'{col}_x': 'Fraud',
                            f'{col}_y': 'Not Fraud'})
    
    fr['tot count'] = fr['Fraud'] + fr['Not Fraud']
    
    fr['% Fraud'] = 100 * np.divide(fr['Fraud'], fr['tot count'])
    
    fr['% Not Fraud'] = 100 * np.divide(fr['Not Fraud'], fr['tot count'])
    
    return fr


# Let's prepare also a function ready to make plots for categorical variables.
# Note that the `ylim` property was set to `[0,100]`, in order to avoid misleading charts.
# The *large* function is to make a larger graph (for many categorical variables).

# In[ ]:


def plot_corr_fraud(df,col):
    
    a = check_corr_fraud(train,col)

    sns.set(rc={'figure.figsize':(11.7,8.27)})
    plt.bar(a[col], a['% Fraud'])
    plt.ylim([0,100])
    plt.ylabel('% Fraud',fontsize=12)
    plt.xlabel(col,fontsize=12)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.4)
    
    return

def plot_corr_fraud_large(df,col):
    
    a = check_corr_fraud(train,col)

    sns.set(rc={'figure.figsize':(101.7,8.27)})
    plt.bar(a[col], a['% Fraud'])
    plt.ylim([0,100])
    plt.ylabel('% Fraud',fontsize=12)
    plt.xlabel(col,fontsize=12)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.4)
    
    return


# In the following, I will select just the graphs with relevant correlation between labels and `isFraud=1`.

# In[ ]:


plot_corr_fraud(train,'P_emaildomain')


# In[ ]:


plot_corr_fraud(train,'R_emaildomain')


# Some email domains look to be more exposed to fraudulent transactions.

# In[ ]:


plot_corr_fraud_large(train,'id_31')


# *(Right-click and open in a new window to see the full size image)*
# 
# Some browser look to be more exposed to fraudulent transactions.

# In[ ]:


plot_corr_fraud_large(train,'DeviceInfo')


# *(Right-click and open in a new window to see the full size image)*
# 
# Some devices look to be more exposed to fraudulent transactions.

# # Conclusion
# 
# From numerical variables, some from the `V` group showed an appreciable correlation with the `isFraud` label.
# 
# From categorical variables, `emaildomain` and `DeviceInfo` showed an appreciable correlation with the `isFraud` label.
# 
# 

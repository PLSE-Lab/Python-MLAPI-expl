#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')


# In[ ]:


len(train_transaction.isFraud[train_transaction.isFraud==1])/len(train_transaction)


# In[ ]:


# Helper functions
# 1. For calculating % na values in  columns
def percent_na(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_groups': percent_missing.index,
                                 'percent_missing': percent_missing.values})
    return missing_value_df
# 2. For plotting grouped histograms 
def sephist(col):
    yes = train_transaction[train_transaction['isFraud'] == 1][col]
    no = train_transaction[train_transaction['isFraud'] == 0][col]
    return yes, no


# In[ ]:


# Helper function for column value details
def column_value_freq(sel_col,cum_per):
    dfpercount = pd.DataFrame(columns=['col_name','num_values_'+str(round(cum_per,2))])
    for col in sel_col:
        col_value = train_transaction[col].value_counts(normalize=True)
        colpercount = pd.DataFrame({'value' : col_value.index,'per_count' : col_value.values})
        colpercount['cum_per_count'] = colpercount['per_count'].cumsum()
        if len(colpercount.loc[colpercount['cum_per_count'] < cum_per,] ) < 2:
            num_col_99 = len(colpercount.loc[colpercount['per_count'] > (1- cum_per),])
        else:
            num_col_99 = len(colpercount.loc[colpercount['cum_per_count']< cum_per,] )
        dfpercount=dfpercount.append({'col_name': col,'num_values_'+str(round(cum_per,2)): num_col_99},ignore_index = True)
    dfpercount['unique_values'] = train_transaction[sel_col].nunique().values
    dfpercount['unique_value_to_num_values'+str(round(cum_per,2))+'_ratio'] = 100 * (dfpercount['num_values_'+str(round(cum_per,2))]/dfpercount.unique_values)
    dfpercount['percent_missing'] = percent_na(train_transaction[sel_col])['percent_missing'].round(3).values
    return dfpercount

def column_value_details(sel_col,cum_per):
    dfpercount = pd.DataFrame(columns=['col_name','values_'+str(round(cum_per,2)),'values_'+str(round(1-cum_per,2))])
    for col in sel_col:
        col_value = train_transaction[col].value_counts(normalize=True)
        colpercount = pd.DataFrame({'value' : col_value.index,'per_count' : col_value.values})
        colpercount['cum_per_count'] = colpercount['per_count'].cumsum()
        if len(colpercount.loc[colpercount['cum_per_count'] < cum_per,] ) < 2:
            values_freq = colpercount.loc[colpercount['per_count'] > (1- cum_per),'value'].tolist()
        else:
            values_freq = colpercount.loc[colpercount['cum_per_count']< cum_per,'value'].tolist() 
        values_less_freq =  [item for item in colpercount['value'] if item not in values_freq]
        dfpercount=dfpercount.append({'col_name': col,'values_'+str(round(cum_per,2)) : values_freq ,'values_'+str(round(1-cum_per,2)): values_less_freq},ignore_index = True)
    num_values_per =[]
    for i in range(len(dfpercount)):
        num_values_per.append(len(dfpercount['values_'+str(round(cum_per,2))][i]))
    dfpercount['num_values_per'] = num_values_per
    return dfpercount


# Of the 394 columns in train_transaction file 339 columns start with V . In most of the models developed  Vcolumns individually have very low importance and gets removed during feature selection process.The kernel attempts to understand the similiarilties between the various V columns and the distribution of data in these columns inorder to understand various groupings that can be used to create useful features.
# 
# The first grouping is based on the percentage of missing values in the columns . The columns can be divided into 15 groups as below.

# In[ ]:


pd.options.display.max_colwidth =300
Vcols=train_transaction.columns[train_transaction.columns.str.startswith('V')]
train_transaction_vcol_na = percent_na(train_transaction[Vcols])
train_transaction_vcol_na_group= train_transaction_vcol_na.groupby('percent_missing')['column_groups'].unique().reset_index()
num_values_per =[]
for i in range(len(train_transaction_vcol_na_group)):
    num_values_per.append(len(train_transaction_vcol_na_group['column_groups'][i]))
train_transaction_vcol_na_group['num_columns_group'] = num_values_per
train_transaction_vcol_na_group


# Let's check whether similiar pattern exists in the test_transaction data.
# The same groups exist in test data also !!!.

# In[ ]:


pd.options.display.max_colwidth =300
Vcols=test_transaction.columns[test_transaction.columns.str.startswith('V')]
test_transaction_vcol_na = percent_na(test_transaction[Vcols])
test_transaction_vcol_na_group= test_transaction_vcol_na.groupby('percent_missing')['column_groups'].unique().reset_index()
num_values_per =[]
for i in range(len(test_transaction_vcol_na_group)):
    num_values_per.append(len(test_transaction_vcol_na_group['column_groups'][i]))
test_transaction_vcol_na_group['num_columns_group'] = num_values_per
test_transaction_vcol_na_group


# Let's explore the 15 groups of columns to understand how many unique values are there in each of the columns and how many values make 96.5% of the data in each of the columns. (96.5% is chosen as this is the percentage of transactions that is not Fraud.)

# In[ ]:


def vcol_multiplot(col,cum_per,ax1):
    col_freq = column_value_freq(col,cum_per)      
    plot1=col_freq.plot(x='col_name',y=['unique_values','num_values_'+str(round(cum_per,2))],kind='bar',rot=90,ax = ax1)
    for p in plot1.patches[1:]:
        h = p.get_height()
        x = p.get_x()+p.get_width()/2.
        if h != 0:
            plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 
                   textcoords="offset points", ha="center", va="bottom")
    plot1.set(ylabel='Count')
    plot1= plot1.set(title='Data Details  in each V columns with ' + str(round(col_freq.percent_missing.mean(),4)) +'% missing values')
    
def vcol_plot(col,cum_per):
    col_freq = column_value_freq(col,cum_per)      
    plot1=col_freq.plot(x='col_name',y=['unique_values','num_values_'+str(round(cum_per,2))],kind='bar',rot=90)
    for p in plot1.patches[1:]:
        h = p.get_height()
        x = p.get_x()+p.get_width()/2.
        if h != 0:
            plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 
                   textcoords="offset points", ha="center", va="bottom")
    plot1.set(ylabel='Count')
    plot1= plot1.set(title='Data Details  in each V columns with ' + str(round(col_freq.percent_missing.mean(),4)) +'% missing values')


# In[ ]:


cum_per = 0.965
fig, axs = plt.subplots(2,1, figsize=(15, 16), facecolor='w', edgecolor='k',squeeze=False)
axs=axs.ravel()
vcol_multiplot(train_transaction_vcol_na_group.column_groups[0],cum_per,axs[0])
vcol_multiplot(train_transaction_vcol_na_group.column_groups[1],cum_per,axs[1])


# In[ ]:


fig, axs = plt.subplots(4,2, figsize=(15,16), facecolor='w', edgecolor='k',squeeze=False)
#fig.subplots_adjust(hspace = 0.75, wspace=.001)
axs = axs.ravel()
for i in range(2,10):
    vcol_multiplot(train_transaction_vcol_na_group.column_groups[i],cum_per,axs[i-2])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# In[ ]:


fig, axs = plt.subplots(5,1, figsize=(15,16), facecolor='w', edgecolor='k',squeeze=False)
axs=axs.ravel()
vcol_multiplot(train_transaction_vcol_na_group.column_groups[10],cum_per,axs[0])
vcol_multiplot(train_transaction_vcol_na_group.column_groups[11],cum_per,axs[1])
vcol_multiplot(train_transaction_vcol_na_group.column_groups[12],cum_per,axs[2])
vcol_multiplot(train_transaction_vcol_na_group.column_groups[13],cum_per,axs[3])
vcol_multiplot(train_transaction_vcol_na_group.column_groups[14],cum_per,axs[4])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# Based on the  data distribution columns can be divided into 5 types.
# 
# 1. **Boolean** - columns  with only two unique values
# 
# 2. **Pseudo- Boolean**  - columns with  96.5% data covered by  maximum two unique values. Within this there are two types.
#         
#         Pseudo-Boolean-categorical - Columns with 15 or less unique values but 96.5% data covered by  maximum two unique values
#         Pseudo-Boolean-numerical - Columns with more than 15 unique values but 96.5% data covered by  maximum two unique values
# 
# 4. **Pseudo-Categorical**  - Columns with  96.5% data covered by  15 or less unique values
# 
# 5. **Numerical** - All Other columns
# 
# 

# **Boolean Columns**
# 
# 

# In[ ]:


colfreq=column_value_freq(Vcols,cum_per)
colfreqbool = colfreq[colfreq.unique_values==2]
if len(colfreqbool)%3 == 0:
    nrow = len(colfreqbool)/3
else:
    nrow = len(colfreqbool) // 3 + 1 
sns.set(rc={'figure.figsize':(14,16)})
for num, alpha in enumerate(colfreqbool.col_name):
    plt.subplot(nrow, 3, num+1)
    plot1= sns.countplot(data=train_transaction,x=alpha,hue='isFraud')
    for p in plot1.patches[1:]:
        h = p.get_height()
        x = p.get_x()+p.get_width()/2.
        if h != 0:
            plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 
                   textcoords="offset points", ha="center", va="bottom")
    plt.legend(title='isFraud',loc='upper right')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# In all these columns almost all values is 1. However except for V305 all values even though minimal which are not 1 are from fraud transactions.

# **Pseudo Booleans**

# 221 Columns starting with V have 96.5% of their values covered by one or two values
# 

# In[ ]:


def cum_value_count(col):
    col_value = train_transaction[col].value_counts(normalize=True)
    colpercount = pd.DataFrame({'value' : col_value.index,'per_count' : col_value.values})
    colpercount['cum_per_count'] = colpercount['per_count'].cumsum()
    return colpercount


# In[ ]:


def V_doublecat_plot(cols,cum_per,limit):
    Vcol_details=column_value_details(cols,cum_per)
    V_cat = Vcol_details[Vcol_details['num_values_per'] <= limit].reset_index()
    sns.set(rc={'figure.figsize':(14,len(V_cat)*2)})
    x=1
    for num, alpha in enumerate(V_cat.col_name):
        plt.subplot(len(V_cat),2,x)
        sns.countplot(data=train_transaction[train_transaction[alpha].isin (V_cat['values_'+str(round(cum_per,2))][num])],y=alpha,hue='isFraud')
        plt.legend(loc='lower right')
        plt.title('Count of unique values which make '+str(round(cum_per*100,3))+'% of data in column ' + str(alpha) )
        plt.subplot(len(V_cat),2,x+1)
        sns.countplot(data=train_transaction[train_transaction[alpha].isin (V_cat['values_'+str(round(1-cum_per,2))][num])],y=alpha,hue='isFraud')
        plt.legend(loc='lower right')
        plt.title('Count of unique values which make only '+str(round((1-cum_per)*100,3))+'% of data in column ' + str(alpha) )
        x= x+2
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# In[ ]:


def V_cat_plot(cols,cum_per,limit):
    Vcol_details=column_value_details(cols,cum_per)
    V_cat = Vcol_details[Vcol_details['num_values_per'] <= limit].reset_index()
    sns.set(rc={'figure.figsize':(14,len(V_cat)*2)})
    x=1
    for num, alpha in enumerate(V_cat.col_name):
        plt.subplot(len(V_cat),2,x)
        sns.countplot(data=train_transaction[train_transaction[alpha].isin (V_cat['values_'+str(round(cum_per,2))][num])],y=alpha,hue='isFraud')
        plt.legend(loc='lower right')
        plt.title('Count of unique values which make '+str(round(cum_per*100,3))+'% of data in column ' + str(alpha) )
        plt.subplot(len(V_cat),2,x+1)
        yes = train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction[alpha].isin (V_cat['values_'+str(round(1-cum_per,2))][num]))][alpha]
        no = train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction[alpha].isin (V_cat['values_'+str(round(1-cum_per,2))][num]))][alpha]
        plt.hist(yes, alpha=0.75, label='Fraud', color='r')
        plt.hist(no, alpha=0.25, label='Not Fraud', color='g')
        plt.legend(loc='upper right')
        plt.title('Histogram of values which make '+str(round((1-cum_per)*100,3))+'% of data in column ' + str(alpha) )
        x= x+2
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# In[ ]:


def V_num_plot(cols,cum_per,limit):
    Vcol_details=column_value_details(cols,cum_per)
    V_num = Vcol_details[Vcol_details['num_values_per'] > limit].reset_index()
    sns.set(rc={'figure.figsize':(14,len(V_num)*2)})
    x=1
    for num, alpha in enumerate(V_num.col_name):
        plt.subplot(len(V_num),2,x)
        yes = train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction[alpha].isin (V_num['values_'+str(round(cum_per,2))][num]))][alpha]
        no = train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction[alpha].isin (V_num['values_'+str(round(cum_per,2))][num]))][alpha]
        plt.hist(yes, alpha=0.75, label='Fraud', color='r')
        plt.hist(no, alpha=0.25, label='Not Fraud', color='g')
        plt.legend(loc='upper right')
        plt.title('Histogram of  values which make '+str(round(cum_per*100,3))+'% of data in column ' + str(alpha) )
        plt.subplot(len(V_num),2,x+1)
        yes = train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction[alpha].isin (V_num['values_'+str(round(1-cum_per,2))][num]))][alpha]
        no = train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction[alpha].isin (V_num['values_'+str(round(1-cum_per,2))][num]))][alpha]
        plt.hist(yes, alpha=0.75, label='Fraud', color='r')
        plt.hist(no, alpha=0.25, label='Not Fraud', color='g')
        plt.legend(loc='upper right')
        plt.title('Histogram of values which make '+str(round((1-cum_per)*100,3))+'% of data in column ' + str(alpha) )
        x= x+2
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# In[ ]:


colfreqpseudobool = colfreq[(colfreq.unique_values !=2) & (colfreq['num_values_'+str(round(cum_per,2))] <= 2)]


# Of these 108 columns have only less than 15 uniques values . Let's look at data distribution in these columns.These are Pseudo Boolean categorical type columns

# In[ ]:


pseudoboolcat = colfreqpseudobool[colfreqpseudobool.unique_values <=15]['col_name'].values
V_doublecat_plot(pseudoboolcat,cum_per,15)


# It's interesting to note that in many of the columns some of the unique values which fall in the 3.5% of column data the proportion of fraudulent transactions is in the range of 25% -50%

# The remaining 113 have more than 15 unique values. Let's look at data distribution in these columns.These are Pseudo Boolean numerical type columns

# In[ ]:


pseudoboolnum = colfreqpseudobool[colfreqpseudobool.unique_values >15]['col_name'].values


# In[ ]:


V_cat_plot(pseudoboolnum,cum_per,15)


# The histograms of values less than 3.5% of the column data shows a higher proportion of fraud transactions.

# In[ ]:


colfreqcat = colfreq[(colfreq.unique_values <=15) & (colfreq['num_values_'+str(round(cum_per,2))] > 2)]
colfreqcat 


# **Pseudo - Categorical**

# There are 44 Columns in this category

# In[ ]:


colfreqpseudocat = colfreq[(colfreq.unique_values >15) & (colfreq['num_values_'+str(round(cum_per,2))] <= 15) & (colfreq['num_values_'+str(round(cum_per,2))]> 2)]


# Let's look at Data distribution in these columns

# In[ ]:


V_cat_plot(colfreqpseudocat.col_name,cum_per,15)


# In some of these columns  a higher proportion of fraud cases are seen  for values which form less than 3.5% of the column data

# **Numerical**

# There are 67 columns in this category. Let's look at how data is distributed in these columns

# In[ ]:


colfreqnum = colfreq[colfreq['num_values_'+str(round(cum_per,2))]>15]


# In[ ]:


V_num_plot(colfreqnum.col_name,cum_per,15)


# In all these columns the more frequent values are in the lower range in both cases.

# **Conclusion**

# It looks like the Pseudo Boolean and Pseudo Categorical columns are important as in both tpes there is a higher proportion of fraud cases when the values fall with less than 3.5% of column data unique values

# Kernel on Unique identifer based on C & D columns  https://www.kaggle.com/rajeshcv/curious-case-of-c-columns
# 
# Kernel on Null Values  https://www.kaggle.com/rajeshcv/tale-of-nulls

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


def plot_distribution_comp(var,nrow=2):
    
    i = 0
    t1 = Train_Data.loc[Train_Data['TARGET'] != 0]
    t0 = Train_Data.loc[Train_Data['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(nrow,2,figsize=(12,6*nrow))

    for feature in var:
        i += 1
        plt.subplot(nrow,2,i)
        sns.kdeplot(t1[feature], bw=0.5,label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5,label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show();
    
def plot_distribution(feature,title):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % title)
    sns.barplot(Train_Data[feature].dropna(),color=color, kde=True,bins=100)
    plt.show()  
    
def count_plot(feature,title):
    sns.set(style="whitegrid")
    #fig.set_size_inches(30, fig.get_figheight(), forward=True)
    #plt.figure(figsize=(20,3))
    count_plot=Train_Data.filter(items=[feature])
    sns.set(style="darkgrid")
    plt.title("Distribution of %s" % title)
    plt.figure(figsize=(16,6))
    #sns.barplot(x='NAME_CONTRACT_TYPE', y='TARGET', data=Name_contact_counts.reset_index())
    sns.countplot(x=feature, data=Train_Data)
    plt.show()
def count_plot(DF,feature,title):
    sns.set(style="whitegrid")
    count_plot=DF.filter(items=[feature])
    plt.figure(figsize=(16,6))
    sns.set(style="darkgrid")
    plt.title("Distribution of %s" % title)
    #sns.barplot(x='NAME_CONTRACT_TYPE', y='TARGET', data=Name_contact_counts.reset_index())
    sns.countplot(x=feature, data=DF)
    plt.show()
    
def bar_plot(df,x_value,y_value,title):
    sns.set(style="whitegrid")
    plt.figure(figsize=(16,6))
    sns.set(style="darkgrid")
    plt.title("Distribution of %s" % title)
    sns.barplot(x=x_value, y=y_value,  data=df)
    plt.show()
print ("All function loaded")


# CREDIT_ACTIVE STATUS

# In[ ]:


bureau=pd.read_csv("/kaggle/input/home-credit-default-risk/bureau.csv")
mapping = {'Active': 1, 'Closed': 0}
creadit_activestatus=bureau.replace({"CREDIT_ACTIVE":mapping})
bar_plot(bureau,"CREDIT_ACTIVE","AMT_CREDIT_SUM_LIMIT","ATM_CREDIT_LIMIT")


# MOST ACtive Credit type reported by Credit BUREAU

# In[ ]:


popular_Credit_Type=bureau.sort_values('AMT_CREDIT_SUM_LIMIT',ascending = False).groupby('CREDIT_TYPE',as_index=False).mean().head(5)
bar_plot(popular_Credit_Type,"CREDIT_TYPE","AMT_CREDIT_SUM_LIMIT","ATM_CREDIT_LIMIT")


# credit_card_balance.csv

# In[ ]:


credit_card_balance=pd.read_csv("/kaggle/input/home-credit-default-risk/credit_card_balance.csv")
credit_card_balance_Contract=credit_card_balance.sort_values('CNT_DRAWINGS_ATM_CURRENT',ascending = False).groupby('NAME_CONTRACT_STATUS',as_index=False).sum().head(3)


# CREDIT_CARD_ATM_WITHDRAW
# ===========================
# 
# Customers who are using credit card for ATM Withdrawl

# In[ ]:


bar_plot(credit_card_balance_Contract,"NAME_CONTRACT_STATUS","CNT_DRAWINGS_ATM_CURRENT","CREDIT_CARD_ATM_WITHDRAW")


# In[ ]:


count_plot(credit_card_balance,'NAME_CONTRACT_STATUS','Installments left to pay with Home credit card')


# previous_application 
# ========================

#  what types goods did the client apply for loans application
# =========================================================

# In[ ]:


previous_application=pd.read_csv("/kaggle/input/home-credit-default-risk/previous_application.csv")
goods_catagory=previous_application.sort_values('AMT_APPLICATION',ascending = False).groupby('NAME_GOODS_CATEGORY',as_index=False).count().head(5)
print(goods_catagory)


# > Train_Data/test_data.csv 

# In[ ]:


Train_Data=pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
count_plot(Train_Data,'NAME_CONTRACT_TYPE','Distribution of Loan type')


# Gender with payment difficulties

# In[ ]:



count_plot(Train_Data,'CODE_GENDER','client with payment difficulties')


# Check the over all population income type

# In[ ]:


count_plot(Train_Data,'NAME_INCOME_TYPE','Clients income type')


# In[ ]:


count_plot(Train_Data,'NAME_FAMILY_STATUS','Relationship status of client ')


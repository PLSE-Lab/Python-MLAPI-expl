#!/usr/bin/env python
# coding: utf-8

# # Introduction
# We compare and analyze statistics of relationships and data between USD/TRY and GAU/TRY. In general, I will test my own knowledge, make the necessary code explanations and make relevant examples.
# 
# <font color='red'>
# Content
# 1. [Load and Check Data](#1)
# 1. [Variable Descriptions](#2)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-whitegrid")

from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Load and Check Data
#     * Import the data and check

# In[ ]:


#dataframe import pandas.read_csv

df_usd = pd.read_csv("/kaggle/input/USD_TRY0.csv")
df_gau = pd.read_csv("/kaggle/input/GAU_TRY1.csv")


# In[ ]:


df_usd.head()


# In[ ]:


df_gau.head()


# In[ ]:


df_usd.describe()


# In[ ]:


df_usd.info()


# In[ ]:


df_gau.info()


# ## Variable Descriptions
# 
#     * Date: Date Time
#     * Price: Present Price
#     * Open: Open Price
#     * High: High Price
#     * Low: Low Price
#     * Change%: Daily Change

# In[ ]:


df_usd.columns


# ## Numerical Variable

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(df_usd[variable],bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Fre")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


numericVar = ["Price","Open","High","Low"]

for i in numericVar:
    plot_hist(i)


# ## Basic Data Analyis

# In[ ]:


df_usd.tail()


# In[ ]:


df_usd_r = df_usd[::-1]


# In[ ]:


df_gau_r = df_gau[::-1]
df_gau_r


# In[ ]:


plt.figure(figsize = (20,10))
sns.pointplot(x=df_usd_r['Date'][500:546],y='Price',data=df_usd_r,color='r',alpha=0.1)
plt.title('USD/TRY Price Analysis',color='red',fontsize=15,style='italic')
plt.xticks(rotation=90)
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize = (20,10))
sns.pointplot(x=df_gau_r['Date'][590:650],y='Price',data=df_gau_r,color='blue',alpha=0.1)
plt.title('GAU/TRY Price Analysis',color='blue',fontsize=15,style='italic')
plt.xticks(rotation=90)
plt.grid()
plt.show()


# In[ ]:


df_usd_x = df_usd_r['HL_PCT'] = (df_usd_r['High'] - df_usd_r['Low']) / df_usd_r['Open'] * 100.0
df_usd_x = df_usd_r['PCT_CHG'] = (df_usd_r['Price'] - df_usd_r['Low']) / df_usd_r['Price'] * 100.0
df_usd_x = df_usd_r[['Date','Price','HL_PCT','PCT_CHG']]


# In[ ]:


df_usd_x.head()


# In[ ]:


plt.figure(figsize = (20,10))
sns.pointplot(x=df_usd_x['Date'][494:543],y=df_usd_x['PCT_CHG'],data=df_usd_x,color='turquoise',alpha=0.1)
plt.title('USD/TRY PCT_CHG Analysis',color='turquoise',fontsize=15,style='italic')
plt.xticks(rotation=90)
plt.grid()
plt.show()


# In[ ]:


df_gau_x = df_gau_r['HL_PCT'] = (df_gau_r['High'] - df_gau_r['Low']) / df_gau_r['Open'] * 100.0
df_gau_x = df_gau_r['PCT_CHG'] = (df_gau_r['Price'] - df_gau_r['Low']) / df_gau_r['Price'] * 100.0
df_gau_x = df_gau_r[['Date','Price','HL_PCT','PCT_CHG']]


# In[ ]:


df_gau_x.head()


# In[ ]:


plt.figure(figsize = (20,10))
sns.pointplot(x=df_gau_x['Date'][590:649],y='PCT_CHG',data=df_gau_x,color='purple',alpha=0.1)
plt.title('GAU/TRY PCT_CHG Analysis',color='purple',fontsize=15,style='italic')
plt.xticks(rotation=90)
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize = (20,10))
sns.pointplot(x=df_gau_x['Date'][590:649],y='PCT_CHG',data=df_gau_x,color='purple',alpha=0.1)
sns.pointplot(x=df_usd_x['Date'][494:543],y=df_usd_x['PCT_CHG'],data=df_usd_x,color='turquoise',alpha=0.1)
plt.title('GAU/TRY vs USD/TRY PCT_CHG Analysis',color='black',fontsize=15,style='italic')
plt.text(10,4,'USD/TRY PCT_CHG',color='turquoise',fontsize = 17,style = 'italic')
plt.text(10,4.2,'GAU/TRY PCT_CHG',color='purple',fontsize = 17,style = 'italic')
plt.xticks(rotation=90)
plt.grid(2)
plt.show()
plt.savefig("graph.png")


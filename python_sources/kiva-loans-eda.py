#!/usr/bin/env python
# coding: utf-8

# Over here, I have done some exploratory data analysis on the Loans data from Kiva. The idea here is to discover the distribution of the loans based on various factors. This may help in making decisions on where the efforts need to be focussed for further improvement based on Kiva's metrics. 

# First, we load the data and the usual packages

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


kiva_loans = pd.read_csv("../input/kiva_loans.csv")
kiva_loans.head()


# In[3]:


kiva_loans.info()


# What are the sectors under which all of Kiva's loans are allotted?

# In[4]:


kiva_loans['sector'].unique()


# Let us now look at the % distribution in the number of loans made to each sector. Keep in mind this is the number of loans made in each sector, not the loan value.

# In[5]:


plt.subplots(figsize=(16,12))
plt.pie(kiva_loans['sector'].value_counts(),labels=kiva_loans['sector'].unique(),autopct='%1.1f%%')
plt.axis('equal')


# We can see that Transportation, Arts, and Food sectors account for more than two-thirds of the numbmer of loans issued. How does this compare with the total value of the loans issued for each sector? We take a look at that now.

# In[23]:


kiva_loans_by_activity = kiva_loans.groupby(['sector'])['loan_amount'].sum()
kiva_loans_by_activity.reset_index()
kiva_loans_by_activity = pd.DataFrame({'sector':kiva_loans_by_activity.index,'loan_amount':kiva_loans_by_activity.values})
kiva_loans_by_activity = kiva_loans_by_activity.sort_values("loan_amount",ascending=False)
kiva_loans_by_activity.head()


# In[24]:


plt.subplots(figsize=(16,12))
sns.barplot(x="sector",y="loan_amount",data=kiva_loans_by_activity)
plt.xlabel('Sector')
plt.ylabel('Loan amount (USD)')
plt.xticks(rotation="vertical")


# From the above barplot, we can see that in terms of the dollar value of the loans, Agriculture takes the top spot, while Arts, and Transportation are way back in the distance. This implies that there are a large number of small value loans being issued to people in the Arts and Transportation sectors, while loans in the Agriculture sector are distributed among fewer people.
# 
# Let use now look at how much % does the loan value of each sector represent the total pie? 

# In[25]:


plt.subplots(figsize=(16,12))
plt.pie(kiva_loans_by_activity['loan_amount'],labels=kiva_loans_by_activity['sector'],autopct='%1.1f%%')
plt.axis('equal')
plt.tight_layout


# From the pie-chart, we can see that just a little more than two-thirds of the dollars go to loans in the Agriculture, Food, and Retail sector. This is a modified version of the classic 80-20 rule (80 % of your business comes from 20% of your customers). In this case, 37% of the customers account for approximately 64% of the dollars loaned out.

# Let's now look at how are the loans geographically distributed. One way to look at it is to use the currency in which the loan was disbursed. This may be a bit misleading because there may be countries that accept multiple currencies, particularly the USD, GBP, EURO and JPY. Nevertheless, it is worth looking at how many loans Kiva disburses in what currency. May be they can use this to negotiate a better exchange rate.

# In[6]:


plt.subplots(figsize=(16,12))
plt.pie(kiva_loans['currency'].value_counts(),labels=kiva_loans['currency'].unique(),autopct='%1.1f%%')
plt.axis('equal')
plt.tight_layout


# The largest number of loans (not value) are being disbursed in INR, PKR, and KES currencies.

# In[7]:


kiva_loans['borrower_genders'][0:5]


# In[8]:


kiva_loans['num_borrowers'] = kiva_loans['borrower_genders'].str.split().str.len()
kiva_loans['num_borrowers'][0:5]


# In[9]:


plt.subplots(figsize=(16,12))
kiva_loans['num_borrowers'].value_counts().plot.bar()
plt.tight_layout()
plt.xlabel('No. of borrowers')
plt.ylabel('Total no. of loans')


# In[ ]:





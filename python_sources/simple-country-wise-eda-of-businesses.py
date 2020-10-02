#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import missingno as msno
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df_loans = pd.read_csv("../input/kiva_loans.csv")
df_region_locations = pd.read_csv("../input/kiva_mpi_region_locations.csv")
df_theme_ids = pd.read_csv("../input/loan_theme_ids.csv")
df_themes_by_region = pd.read_csv("../input/loan_themes_by_region.csv")
# Any results you write to the current directory are saved as output.


# In[2]:


msno.matrix(df_loans)


# In[3]:


df_loans['posted_time'] = pd.to_datetime(df_loans['posted_time'])
df_loans['disbursed_time'] = pd.to_datetime(df_loans['disbursed_time'])
df_loans['funded_time'] = pd.to_datetime(df_loans['funded_time'])
df_loans['date'] = pd.to_datetime(df_loans['date'])


# In[4]:


df_loans.head(2)


# In[5]:


df_loans['loan_amount'].describe()


# *As we can see in the 3rd quartile (75%) , 75% of the loaned amounts are equal to or below 1000 USDs . Lets validate this by visializing the distribution of the loan amounts *

# In[6]:


plt.figure(figsize=(12,10))
sns.distplot(df_loans['loan_amount'],bins=1000)
plt.title("Loan Amount Distribtion")
plt.xlabel("Loan Amount(Dollars)")
plt.xlim(0,10000)


# The distribution is left skewed , which means the outliers are to the right . Majority of the amounts lie between 100-800$ 

# In[7]:


print("2 Standard deviations from the mean :%s" %(round(df_loans['loan_amount'].std())*2))


# In[8]:


plt.figure(figsize=(12,8))
sns.distplot(df_loans.query("loan_amount<2398")['loan_amount'])


# Things are clearer now . Looking at the above graph we can conclude : 
# 
# - Majority of the loan amounts lie between $25 USD - $800 USD
# - There are sudden peaks around 1000USD,1500USD and relatively smaller peak close to 2000USD which implies the need to explore the fundings at or close to these amounts 
# 
# Now lets quickly explore the sectors in terms of the funded amounts 
# 

# In[9]:


sector_wise_amounts = df_loans.groupby("sector").mean()[["funded_amount"]].sort_values(ascending=False,by="funded_amount")
sector_wise_amounts.reset_index()
sector_wise_amounts = sector_wise_amounts.T.squeeze()

plt.figure(figsize=(12,8))
sns.barplot(y=sector_wise_amounts.index, x=sector_wise_amounts.values, alpha=0.6)
plt.ylabel("Sectors")
plt.xlabel("Funded Amounts")
plt.title("Distribution of Funded Amounts by Sectors")


# Notice only the bottom 5 sectors fall below the 800 USD category .Also keep in mind that the bottom 5 sectors make up for most of the data in the dataset. We can validate this as below 

# In[10]:


print("Percent of the dataset below 800$ Funded amount :{:0.2f} % " .format(100*len(df_loans.query("funded_amount<=800"))/len(df_loans)))


# For ease of analysis i would prefer to split the dataset now into 2 sections ,  as below : 
# 
# -*class A* - Amount Funded is 850+ USD
# 
# -*class B* - Amount Funded is less than 850 USD (842 USDs being the mean )
# 

# In[11]:


df_business_classA = df_loans.query("funded_amount>850")
df_business_classB = df_loans.query("funded_amount<=850")


# In[12]:


sector_wise_amounts = df_business_classA.groupby("sector").mean()[["funded_amount"]].sort_values(ascending=False,by="funded_amount")
sector_wise_amounts.reset_index()
sector_wise_amounts = sector_wise_amounts.T.squeeze()

plt.figure(figsize=(12,8))
sns.barplot(y=sector_wise_amounts.index, x=sector_wise_amounts.values, alpha=0.6)
plt.ylabel("Sectors")
plt.xlabel("Funded Amounts")
plt.title("Distribution of Funded Amounts by Sectors in Business Type A")


# In[13]:


sector_wise_amounts = df_business_classB.groupby("sector").mean()[["funded_amount"]].sort_values(ascending=False,by="funded_amount")
sector_wise_amounts.reset_index()
sector_wise_amounts = sector_wise_amounts.T.squeeze()

plt.figure(figsize=(12,8))
sns.barplot(y=sector_wise_amounts.index, x=sector_wise_amounts.values, alpha=0.6)
plt.ylabel("Sectors")
plt.xlabel("Funded Amounts")
plt.title("Distribution of Funded Amounts by Sectors in Business Type B")


# In[14]:


df_business_classA[['funded_amount','loan_amount']].describe().T


# In[15]:


df_business_classB[['funded_amount','loan_amount']].describe().T


# In[16]:


classA_countries = list(df_business_classA['country'].unique()) 
classB_countries = list(df_business_classB['country'].unique())

temp = [i for i in classA_countries if i not in classB_countries] 
print ("Countries that are in classA type sectors(850+ USD funded loans) and are not in class B type sectors(funded loans<850 USDs) \n")
for i in temp : 
    print(i)


# Goes to show the above 9 countries are only involved in sectors that need high amounts of funds or maybe there is more reliability in terms of repayment which maybe the reason behind these countries falling in the above average funded amount category only . 

# In[17]:


plt.figure(figsize=(12,10))
plt.scatter(df_business_classA['lender_count'],df_business_classA['funded_amount'])
plt.xlabel("Lender Count")
plt.ylabel("Funded Amount")
plt.title("Sector Class A - Scatter plot of Lender Count vs Funded Amount")


# In[18]:


plt.figure(figsize=(12,10))
plt.scatter(df_business_classB['lender_count'],df_business_classB['funded_amount'])
plt.xlabel("Lender Count")
plt.ylabel("Funded Amount")
plt.title("Sector Class B - Scatter plot of Lender Count vs Funded Amount")


# Observations : 
# - Quite strange as one can see the difference between the two plots above . One for Sector class A (Funded amount>850 USDs) and the other for class B (Funded amount < 850 USDs) 
# - We can notice for class A , since the amounts are pretty high , the number of lenders grows with the the amount funded which is so not the case for class B type sectors where the amounts are relatively smaller . 

# **Lets analyse the borrower genders in both classes (A & B)**

# In[19]:


df_business_classA['borrower_genders'] = df_business_classA['borrower_genders'].str.split(",")

df_business_classB['borrower_genders'] = df_business_classB['borrower_genders'].str.split(",")


# In[20]:


df_business_classA['borrower_count'] = df_business_classA['borrower_genders'].str.len()
df_business_classB['borrower_count'] = df_business_classB['borrower_genders'].str.len()


# In[21]:


df_business_classA.head(2)


# More to come very soon ..!!

# In[ ]:





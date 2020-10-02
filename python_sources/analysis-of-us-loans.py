#!/usr/bin/env python
# coding: utf-8

# # Analysis of United States loan Data

# This jupiter notebook contains code that attempts to Analyse UnitedSates Loans data from a subset of data provided Kiva_loans

# ### Import relevant Libraries 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Import the data and Creat a Data frame - Kiva

# In[ ]:


#kiva_df=pd.read_csv('/kaggle/input/kiva_loans.csv')
kiva_df= pd.read_csv("/kaggle/input/kiva-loans-data/kiva_loans.csv")


# ### Creation of a subset of Kiva with only US data

# In[ ]:


us_df=kiva_df[kiva_df["country_code"]=="US"]


# ### Loan Application per Sector Bar Chart Creation using Seaborn 

# 1.From the statistics below we see majority of loans borrowed were agricultural loans which could be an indication we have people investing more in Agricutlure than other businesses. <br>
# 2.The second and third major borrowing is used to finance personal use and food respectively. <br>
# 3.The least investment is in entertainment.

# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x='sector',y='loan_amount',data=us_df,ci=None)
plt.xticks(rotation=90)
plt.title("Loan Application per Sector", weight='bold')
plt.xlabel("Sector",weight="bold",fontsize=14)
plt.ylabel("Loan Amount", weight="bold",fontsize=14)
plt.xticks(rotation=90)
plt.grid(True)


# ### Loan Distribution between Gender

# 1. From the chart below we see men are borrowing more than women albiet the margins are small. 

# In[ ]:


#Defination of the Colors using a dictionary
hue_colors={"male":"blue","female":"magenta"}

#Plot the bar chart
plt.figure(figsize=(4,4))
sns.catplot(x="borrower_genders",y="loan_amount",kind='bar',data=us_df,ci=None,palette=hue_colors)
plt.title("Loan Borrowed by Gender",weight="bold")
plt.xlabel("Gender",weight="bold")
plt.ylabel("Loan Amount ( USD)")
plt.show()


# ### Bar Chart depicting the Payment Interval

# 1.From the statistics below it appears majority of the customers pay regually on a monthly basis.<br>
# 2.The second highest category are paying a one off.<br>
# 3.The third group which seems to be the least are paying irregulary.<br>
# 4.It is worth noting the women are majority of the irregular payers.

# In[ ]:


plt.figure(figsize=(10,5))

sns.catplot(x="repayment_interval",y="loan_amount", data=us_df,kind="bar",ci=None,hue="borrower_genders",palette=hue_colors)
plt.title("Payment Interval Comparision",weight="bold")
plt.xlabel("Payment Schedule",weight="bold")
plt.ylabel("Loan Amount",weight="bold")


# ### Correlation between amount borrowed and frequency of borrowing

# 1. From the correlation chart below there seems to a linear relationship between the frequency of borrowing with the amount borrowed. <br>
# 2. Those who borrowed frequently tend to have larger amounts borrowed.

# In[ ]:


plt.figure(figsize=(10,5))

sns.scatterplot(x="loan_amount",y="lender_count",data=us_df,hue="borrower_genders", sizes=(20, 200),palette=hue_colors)
plt.xlabel("Loan Amount in (dollars)",weight="bold")
plt.ylabel("Lender Count",weight="bold")
plt.title("Lender Counts against Loan Value", weight='bold')
plt.show()


# ### Correlation of amount borrowed in(USD) against repayment period

# 1. From the obervations below there seems to be no correlation between the amount borrowed and payment scheudle.<br>
# 2.However it is worth noting women were the majority amongest those with the longest payment schedule of 60 months.

# In[ ]:


plt.figure(figsize=(10,5))

sns.scatterplot(x="term_in_months",y="loan_amount",data=us_df,hue="borrower_genders",palette=hue_colors)
plt.xlabel("term_in_months",weight="bold")
plt.ylabel("Loan Amount",weight="bold")
plt.title("Loan Term against Loan Amount", weight='bold')
plt.show()


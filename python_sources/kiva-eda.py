#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid');

# Importing data
data = pd.read_csv('../input/kiva_loans.csv')


# In[ ]:


data.head()


# # Frequency of loans in each sector
# 

# In[ ]:


data['sector'].value_counts().plot(kind="area",figsize=(12,12))
plt.xticks(np.arange(15), tuple(data['sector'].unique()), rotation=60)
plt.show()


# # Pie chart of repayment intervals
# (I have not had time to search what kind of repayment interval is called a bullet. Let me know in the comments)

# In[ ]:


data['repayment_interval'].value_counts().plot(kind="pie",figsize=(12,12))


# # Factorplot of different sectors with repayment intervals as categories(hue is responsible for this)

# In[ ]:


f = sns.factorplot(x="sector",data=data,kind="count",hue="repayment_interval",size=12,palette="BuGn_r")
f.set_xticklabels(rotation=90)


# # Now, some statistics...

# In[ ]:


print("Mean funded amount: ",data['funded_amount'].mean())
print("Mean lender count: ",data['lender_count'].mean())
print("Mean of term in months: ",data['term_in_months'].mean())


# # Now, we will plot some jointplots to inspect correlation
# A nice correlation coefficient between two fields gives awesome results when regression is done

# In[ ]:


print("Correlation coefficent of funded amount and lender count: ",data['funded_amount'].corr(data['lender_count']))
sns.jointplot(x="funded_amount", y="lender_count", data=data, kind='reg')


# So, this one has a decent correlation coefficient. We can surely consider this result for regression analysis some other time.

# In[ ]:


print("Correlation coefficent of funded amount and term in months: ",data['funded_amount'].corr(data['term_in_months']))
sns.jointplot(x="funded_amount", y="term_in_months", data=data, kind='reg')


# This one has a more scattered plot so these fields are not really dependent on each other.

# In[ ]:


print("Correlation coefficent of term in months and lender count: ",data['term_in_months'].corr(data['lender_count']))
sns.jointplot(x="term_in_months", y="lender_count", data=data, kind='reg')


# This may be better than the last one but these fields too are not really correlated enough to be dependent on each other.

# In[ ]:


print(data['funded_amount'].corr(data['loan_amount']))
sns.jointplot(x="funded_amount", y="loan_amount", data=data, kind='reg')


# This awesome correlation plot is possible as many values in funded amount and loan amount are exactly same.

# # Feature engineering
# Creating a new column month from date containing only month and year for each loan
# 

# In[ ]:


data['month'] =  data['date'].astype(str).str[0:7]


# In[ ]:


data['month'].head()


# # Creating a pivot_table for more strucured data

# In[ ]:


pivot_data = pd.pivot_table(data,values='funded_amount',index='month',columns='sector')
pivot_data.head()


# # Plotting a heatmap out of the pivot data

# In[ ]:


a4_dims = (15, 15)
fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(pivot_data,ax=ax)


# In[ ]:





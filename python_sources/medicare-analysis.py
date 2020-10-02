#!/usr/bin/env python
# coding: utf-8

# # Analyzing The Distribution of Medicare Payments

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") #who needs those

df = pd.read_csv('../input/Medicare_Provider_Charge_Inpatient_DRGALL_FY2016.csv', thousands=',')


# In[19]:


df.head()


# In[20]:


df.tail()


# In[21]:


print(df.dtypes)
df.columns = df.columns.str.replace(' ','_')

df.Average_Total_Payments = df.Average_Total_Payments.apply(lambda x: x.strip('$'))
df.Average_Total_Payments = df.Average_Total_Payments.apply(lambda x: x.replace(',', ""))
df.Average_Total_Payments = df.Average_Total_Payments.apply(pd.to_numeric, errors='coerce')

df.Average_Covered_Charges = df.Average_Covered_Charges.apply(lambda x: x.strip('$'))
df.Average_Covered_Charges = df.Average_Covered_Charges.apply(lambda x: x.replace(',', ""))
df.Average_Covered_Charges = df.Average_Covered_Charges.apply(pd.to_numeric, errors='coerce')

df.Average_Medicare_Payments = df.Average_Medicare_Payments.apply(lambda x: x.strip('$'))
df.Average_Medicare_Payments = df.Average_Medicare_Payments.apply(lambda x: x.replace(',', ""))
df.Average_Medicare_Payments = df.Average_Medicare_Payments.apply(pd.to_numeric, errors='coerce')

print("\n\n AFTER CONVERTING STRING TO NUMERIC \n\n")
print(df.dtypes)


# In[22]:


des = df.describe()
#delete irrelevant variables
del des['Provider_Id']
del des['Provider_Zip_Code']
des


# # Quantitave Variables

# In[23]:


plt.rcParams['figure.figsize'] = [15, 15]

fig  = plt.figure()

sns.distplot(df['Average_Medicare_Payments'], kde=False, ax = fig.add_subplot(221))
sns.distplot(df['Average_Covered_Charges'], kde=False, ax = fig.add_subplot(222))
sns.distplot(df['Average_Total_Payments'], kde=False, ax = fig.add_subplot(223))
sns.distplot(df['Total_Discharges'], kde=False, ax = fig.add_subplot(224))
fig.suptitle("Quantitative Variable Distributions", fontsize=36)


# We notice that the data is highly skewed. After all, not everyone has 4000 discharges and over $500000 in medicare payments! One thing that we can do is to check if it is distributed exponentially by taking the log of each graph.

# In[24]:


fig2  = plt.figure()

sns.distplot(np.log(df['Average_Medicare_Payments']), kde=True, ax = fig2.add_subplot(221))
sns.distplot(np.log(df['Average_Covered_Charges']), kde=True, ax = fig2.add_subplot(222))
sns.distplot(np.log(df['Average_Total_Payments']), kde=True, ax = fig2.add_subplot(223))
sns.distplot(np.log(df['Total_Discharges']), kde=True, ax = fig2.add_subplot(224))
fig2.suptitle("Quantitative Variable Distributions after log transform", fontsize=36)


# We can notice that there is some of the data is normally distributed while other is still skewed.

# # Categorical Variables

# In[54]:


plt.rcParams['figure.figsize'] = [15, 15]
sns.countplot(df['Provider_State'], order = df['Provider_State'].value_counts().index, palette=sns.color_palette("GnBu_d"))


# We notice here that Florida, California, Texas, and New York have some of the highest counts (because they have a large population), while the opposite is true of Vermont, Arkansas, and Wyoming unsurprisingly.

# In[26]:


def sortByState(column, dataframe):
    groupby = dataframe.groupby('Provider_State', as_index=False).mean()
    sortedStates = groupby.sort_values(column)
    return sortedStates['Provider_State']


# In[56]:


sns.barplot(x = 'Provider_State', y = 'Average_Total_Payments', data = df, order = sortByState('Average_Total_Payments', df), palette=sns.color_palette("Blues"))


# In[58]:


sns.barplot(x = 'Provider_State', y = 'Average_Medicare_Payments', data = df, order = sortByState('Average_Medicare_Payments', df), palette=sns.color_palette("Greens"))


# In[29]:


sns.barplot(x = 'Provider_State', y = 'Total_Discharges', data = df, order = sortByState('Total_Discharges', df))


# Now let us try some distribution plots so we can see the distributions by state to see if there are any abnormalities between states

# In[60]:


gb = df.groupby('Provider_State')
j = 0
i = 0

plt.rcParams['figure.figsize'] = [25, 25]

for index, group in gb:
    
    if i % 8 == 0:
        ax = plt.subplot(2, 3, j % 6 + 1)
        j = j + 1
    
    sns.distplot(group['Average_Medicare_Payments'], hist=False, label = index, ax=ax)
    i = i + 1
plt.suptitle('Distribution of Average Medicare Payments', fontsize = 36) 
    
    


# In[61]:


gb = df.groupby('Provider_State')
j = 0
i = 0

plt.rcParams['figure.figsize'] = [25, 25]

for index, group in gb:
    
    if i % 8 == 0:
        ax = plt.subplot(2, 3, j % 6 + 1)
        j = j + 1
    
    sns.distplot(np.log10(group['Average_Medicare_Payments']), hist=False, label = index, ax=ax)
    i = i + 1

plt.suptitle('Distribution of Average Medicare Payments', fontsize = 36) 


# When we take the log of each distribution we notice that all the distributions have patterns with dips in the middle of the distributions - interesting

# ### Conclusion ###
# 
# It was nice to do this kernel, which is my first. It was an interesting experience and I definitely learned a lot about Pandas and Seaborn (both of which I am new to). Feel free to give me suggestions on how I can improve the kernel, it would be greatly appreciated. I hope to do time series data in my next kernel!

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np 
import os

print(os.listdir("../input"))


# **Load in the instruments pickle, see how big the dataframe is, and what columns it has.**

# In[46]:


import pickle

def unpickle_me(filename):
    with open(filename, 'rb') as fh:
        return pickle.load(fh)
    
instruments = unpickle_me('../input/instruments.p')
print(type(instruments))
print(instruments.shape)
print(instruments.columns.values)

    


# In[47]:


instruments.head(100)


# Scatterplot a few high level metrics against each other for a general sense of the data and reality check.  How about market cap and num employees?  Generally speaking, these should be correlated.

# In[62]:


import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
sns.regplot(x='num_employees', y='market_cap', data=instruments, dropna=True)
plt.show()


# Looks like the trend we were expecting, but the data is skewed into a small area of the plot.  Let's log transform the axes to get a better sense of the relationship.

# In[63]:


fig, ax = plt.subplots()
ax.set(xscale="log", yscale="log")
sns.regplot(x='num_employees', y='market_cap', data=instruments, dropna=True)
plt.show()


# Let's plot market cap vs. shares outstanding.  The relationship should be even closer than with employee numbers.

# In[64]:


fig, ax = plt.subplots()
ax.set(xscale="log", yscale="log")
sns.regplot(x='shares_outstanding', y='market_cap', data=instruments, dropna=True)
plt.show()


# Let's look at when the instruments in our data set were founded, to get an idea of how much historical data we might be able to expect to draw on, and how that will be distributed across the observations.

# In[79]:


fig, ax = plt.subplots()
ax.set(xscale="linear", yscale="linear")
year_no_na = instruments['year_founded'].dropna()
bins = int(max(year_no_na) - min(year_no_na))
sns.distplot(a=year_no_na, bins=bins, kde=False)


# From the above, we can see, as we would expect, more instruments created over time.  There are several pronounced peaks from the late 20th century to the present when many new instruments were created.  Possibly these correspond to periods of economic growth, or possibly also times when particular brokerage firms first bundled together packages of ETFs and other derived securities that greatly increased the number of new instruments created during these periods.  Let's take a look at the exact years when the most securities were created.
# 

# In[83]:


from collections import Counter

year_counts = Counter(year_no_na)
for k,v in sorted(year_counts.items(), key=lambda x: x[1], reverse=True):
    print("{}: {}".format(int(k), v))


# Now let's see some of the instruments created in those years.

# In[87]:


instruments.loc[instruments['year_founded']==2013].head(53)


# In[88]:


instruments.loc[instruments['year_founded']==1997].head(50)


# 
# Let's look at current market cap plotted against the company's date of founding.

# In[67]:


fig, ax = plt.subplots()
ax.set(xscale="linear", yscale="log")
sns.regplot(x='year_founded', y='market_cap', data=instruments, dropna=True)
plt.show()


# As we would expect from survivorship bias, companies/securities established over a century ago that have survived tend toward the higher end of the market cap scale.  Securities at the lowest end of the market cap range have more recent founding years.  Smaller market cap instruments that were foudned earlier have had to either grow or fold over time.
# 
# 

# Let's take a look at the `payout_history` field.  First, how many rows even have that information available?

# In[89]:


with_history = instruments.loc[~instruments.payout_history.isnull()]
print(with_history.shape)
with_history.head()


# Let's see what that payout history object is like.

# In[38]:


ex_history = with_history.iloc[40].loc['payout_history']
print(type(ex_history))
print(ex_history.shape)
ex_history.head()


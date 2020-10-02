#!/usr/bin/env python
# coding: utf-8

# **I try to figure out the relation between tier structure and amount of pledged.** 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# live = pd.read_csv('input/live.csv',index_col=0)
back = pd.read_csv('../input/most_backed.csv',index_col=0)


# In[ ]:


back.info()


# ## Claen data ##

# In[ ]:


# Now I want to turn str to num format, but some data in "pledge.tier" contains "...", so I drop that.

t = []
for i,row in enumerate(back['pledge.tier']):
    if '...' in row:
        t.append(i)
print(t)


# In[ ]:


back['pledge.tier'].iloc[981]


# In[ ]:


back = back.drop(back.index[t])


# In[ ]:


# Convert str to float.
def num_tier(string):    
    return len(string[1:-1].split(', '))

def tier_num(string):
    l = []
    for s in string[1:-1].split(', '):
        l.append(float(s))
    return l

back['num of tier'] = back['pledge.tier'].apply(num_tier)
back['pledge.tier_num'] = back['pledge.tier'].apply(tier_num)
back['num.backers.tier_num'] = back['num.backers.tier'].apply(tier_num)


# In[ ]:


back = back[back['currency']=='usd']
back.reset_index(drop=True,inplace=True)


# In[ ]:


back_clean = back[['title','amt.pledged','category','goal','num.backers','num of tier',
                   'pledge.tier_num', 'num.backers.tier_num']]


# In[ ]:


back_clean.head()


# In[ ]:


back_clean.info()


# ## EDA ##

# In[ ]:


# I build the contribution in each tier in the every case.

tier_contrib = pd.Series()
for i in range(len(back_clean.index)):
    a = np.array(back_clean['pledge.tier_num'].iloc[i])
    b = np.array(back_clean['num.backers.tier_num'].iloc[i])
    contrib = pd.Series(list([a*b*100/np.sum(a*b)]))
    tier_contrib = tier_contrib.append(contrib)


# In[ ]:


tier_contrib.reset_index(drop=True,inplace=True)
tier_contrib.head()


# In[ ]:


# "tier_contrib%" means the contribution in each tier.

back_clean['tier_contrib%'] = tier_contrib
back_clean.head()


# In[ ]:


back_clean['num of tier'].plot.hist(bins=50)


# In[ ]:


# I build a new column to show which tier is the most contribution.
# (pledge.tier_num is in order.)

def argmax(tier):
    return tier.argmax()+1

back_clean['argmax_tier_contrib%'] = back_clean['tier_contrib%'].apply(argmax)
back_clean.head()


# In[ ]:


fig = plt.figure(figsize=(12,6))
sns.barplot(x=back_clean.groupby('argmax_tier_contrib%').count()['amt.pledged'].index,
            y=back_clean.groupby('argmax_tier_contrib%').count()['amt.pledged'])
#plt.xlim([0,10])


# **I found that second to seventh price in "pledge.tier_num" is really important, they make a huge contribution to the funding cases.**

# In[ ]:


# max contribution by position in the whole pricing strategy

sns.distplot(back_clean['argmax_tier_contrib%']/back_clean['num of tier'],kde=False,bins=50)


# **Apparently, the "median price" is very important. Many consumers or supporters maybe don't realize the value of that product, so they give the median price for that.**
# 
# **This figure shows that "1.0" is also important, because there's only one tier in some cases.**

# In[ ]:


# Now I use number scale to locate the position of the most contribution price. 

contrib_in_tier = pd.Series()
for i in range(len(back_clean.index)):
    diff = back_clean['pledge.tier_num'][i][-1]-back_clean['pledge.tier_num'][i][0]
    if diff != 0:
        temp = pd.Series((back_clean['pledge.tier_num'][i][back_clean['argmax_tier_contrib%'][i]-1]-back_clean['pledge.tier_num'][i][0])/diff)
    else:
        temp = pd.Series(1)
        
    contrib_in_tier = contrib_in_tier.append(temp*100)


# In[ ]:


sns.distplot(contrib_in_tier,kde=False,bins=50)


# ## Conclusion
# * These projects on Kickstarter have about 10 tiers is very popular.
# * We found that second to seventh price in "pledge.tier_num" is really important, they make a huge contribution to the funding cases.
# * The "median price" in the project is very important.
# * The contribution of crowd maybe meets the "long tail theory", so the big part of contribution is at the lower price.

# In[ ]:





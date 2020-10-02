#!/usr/bin/env python
# coding: utf-8

# # 2016 US Presidential - Social Media
# (work in progress)
# 
# Kaggle Kernel is good stuff!
# 
# - Learn from original author
# 
# - Raise new questions

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# Any results you write to the current directory are saved as output

import numpy
import pandas
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def get_data(first, last):
    """ get data for person named first, last"""
    df1 = pandas.read_csv('../input/{}.csv'.format(first), low_memory=False,
                 header=0, sep=',', encoding = "ISO-8859-1", parse_dates = ['timestamp'],
                     usecols=['timestamp','network','author','likes'])
    df2 = pandas.read_csv('../input/{}.csv'.format(last), low_memory=False,
                 header=0, sep=',', encoding = "ISO-8859-1", parse_dates = ['timestamp'],
                     usecols=['timestamp','network','author','likes'])
    return pandas.concat([df1, df2]).drop_duplicates()


# In[ ]:


donald = get_data("Donald", "Trump")
hillary = get_data("Hillary", "Clinton")
donald.info()
print("-"*80)
hillary.info()


# In[ ]:


donald.head()


# ## Filter out negative likes, which only happened with Imgur

# In[ ]:


donald = donald[donald['likes'] > 0]
hillary = hillary[hillary['likes'] > 0]
donald.groupby('network')['likes'].describe()
hillary.groupby('network')['likes'].describe()


# ## Visualize Trump and Hillary likes change over time

# In[ ]:


hillary['year'] = hillary['timestamp'].apply(lambda t:t.year)
hillary['month'] = hillary['timestamp'].apply(lambda t:t.month)
hillary_agg = hillary.groupby([hillary['year'], hillary['month']])['likes'].sum()


# In[ ]:


donald['year'] = donald['timestamp'].apply(lambda t:t.year)
donald['month'] = donald['timestamp'].apply(lambda t:t.month)
donald_agg = donald.groupby([donald['year'], donald['month']])['likes'].sum()


# In[ ]:


frames = [hillary_agg, donald_agg]
data = pandas.concat(frames, axis = 1)


# In[ ]:


data.columns = ['Hillary', 'Trump']
data.plot(figsize = [9,6])


# ## Compare Trump and Hillary likes over different network

# In[ ]:


donald_network=donald.groupby([donald['network']])['likes'].sum()
hillary_network=hillary.groupby([hillary['network']])['likes'].sum()


# In[ ]:


network_data = pandas.concat([donald_network, hillary_network], axis=1)
network_data.columns=['Trump','Hillary']


# In[ ]:


network_data.head()


# In[ ]:


network_data.plot(kind='bar')


# ## Examine the Top 10 contributors of each candidates' likes

# In[ ]:


donald_by_author = donald.groupby(['author']).likes.sum().sort_values(ascending=False)


# In[ ]:


donald_by_author.head(10)


# In[ ]:


hillary_by_author = hillary.groupby('author').likes.sum().sort_values(ascending=False)


# In[ ]:


hillary_by_author.head(10)


# **From the above, top contributors such as 3.46937E+14 have posts for both candidates and got 3M and 1M likes. It is hard to tell that if people like the candidates, or like the posts, or like the bloggers** ;-)

# In[ ]:





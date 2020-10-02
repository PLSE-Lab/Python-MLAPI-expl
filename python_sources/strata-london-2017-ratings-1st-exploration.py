#!/usr/bin/env python
# coding: utf-8

# Inspired by Kiyoto Tamura 2015's "[Nobody Likes Sponsored Talks at Strata][1]"
# 
# 
#   [1]: https://blog.treasuredata.com/blog/2015/03/10/sponsored-talks-at-strata/

# In[ ]:


import numpy as np
import pandas as pd
import re
import seaborn as snsx
import matplotlib.pyplot as plt


# In[ ]:


# Read in CSV to dataframe
df = pd.read_csv('../input/Strata London 2017 ratings - results-20170630-170827.csv.csv')
df['responses'] = df['responses'].fillna(0).astype(int)
df['starthour'] = df['starthour'].fillna(np.nan).astype(int)


# In[ ]:


# Examine the fields
df.head()


# In[ ]:


#  best talks
df.loc[
    (df.responses>10) & (df.rating>4),
    ["title", "rating", "responses", "company1"]
].sort_values(["rating"], ascending=False)


# In[ ]:


#  worst talks
df.loc[
    (df.responses>5) & (df.rating<3),
    ["title", "rating", "responses", "company1"]
].sort_values(["rating"])


# In[ ]:


ax = df.plot.scatter(x='rating',y='responses')


# In[ ]:


# do later talks get more ratings?
ax=df[df.starthour>0].plot.scatter(x='starthour',y='responses')


# In[ ]:



# do later talks get better ratings?
ax=df[df.starthour>0].plot.scatter(x='starthour',y='rating')


# In[ ]:


# split multiple topics into multiple rows

df2 = df.copy()
s = df2.topic.str.split(', \t').apply(pd.Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'topic'
del df2['topic']
df2.join(s)


# In[ ]:


# so how do I get the average score and count of talks for each topic?

dfmc = df2.join(s).groupby(['topic']).agg(['mean', 'count'])
dfmc


# In[ ]:


# what topics get the best and worst reviews?

ax = dfmc[(dfmc[('rating', 'count')]>3)]        .sort_values([('rating', 'mean')])        .plot.barh(y=('rating', 'mean'),
                  title='Strata London 2017 - avg rating per topic',
                  legend=False,
                  )


# As Kiyoto Tamura said in 2015 "Nobody Likes Sponsored Talks at Strata" - they still get the worst scores.
# 
# But you can watch the most rated session (35 ratings) with one of the best scores (4.54) now:
# 
# - https://conferences.oreilly.com/strata/strata-eu/public/schedule/detail/58064
# 

#!/usr/bin/env python
# coding: utf-8

# # Similarity of Lithuanian parliamentary groups by votes
# 
# This is an attempt to visualize how similar are Lithuanian parliamentary groups by comparing how they vote in the Lithuanian Parliament.
# 
# To do this comparison I simply used PCA to reduce all votings in a time period to a single dimension. My chosen period of time is one day. But the result was quite noisy and hard to interpret, so on top of that I added smoothing using Hamming window of 14 days. This gave quite readable results.
# 
# PCA was fed with list of parliamentary groups, where each group has average vote from each voting. Votings are the features. In one day there can be many votings on different questions.

# In[ ]:


import pandas as pd
import matplotlib as mpl
import sklearn.decomposition


# In[ ]:


mpl.style.use('seaborn-darkgrid')
mpl.rc('font', size=16)


# In[ ]:


votes = pd.read_csv('../input/votes.csv', parse_dates=['time'], dtype={'question': str})


# In[ ]:


pca = sklearn.decomposition.PCA(n_components=1, random_state=0)

def reduce_manifold(g):
    f = g.groupby(['group', 'voting_id'])['vote'].mean().unstack().fillna(0)
    return pd.DataFrame(pca.fit_transform(f), index=f.index)


# Lithuanian parliament is reelected every 4 years somewher in October.
terms = votes.set_index('time').resample('4AS-OCT')

for term in terms.groups.keys():
    # Get all the votes from one term.
    frame = terms.get_group(term)
    title = '%s - %s Parliament' % (frame.index.min().year, frame.index.max().year)
    
    # In order to make graphs readable, we reduce number of parliamentary groups to 5 major ones.
    frame = frame[frame['group'].isin(frame['group'].value_counts().index[:5])]
    
    # Group all votes by day, unstack all votings and reduce dimensionality to one using PCA. 
    agg = frame.groupby(frame.index.to_period('D')).apply(reduce_manifold).unstack()
    agg.columns = agg.columns.droplevel(0)
    
    # Reduce noise by smoothing data using 14 days window.
    agg = agg.rolling(14, win_type='hamming', center=True).mean()
    ax = agg.plot(figsize=(16, 6))
    ax.set_title(title)
    ax.set_ylabel('vote similarity')


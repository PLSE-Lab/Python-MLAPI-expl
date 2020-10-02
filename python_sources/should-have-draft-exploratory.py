#!/usr/bin/env python
# coding: utf-8

# # The Should Have Draft

# This will attempt to quantify the value of a draft pick (position plus top players available, etc), then retroactively determine what players should have been drafted when. The hope is to be able to better identify those traits that might be common to players undervalued by the NFL draft relative to their production.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

dataframe = pd.read_csv('../input/nfl_draft.csv')
dataframe.head()


# First4AV is going to be a key value for us. This metric is meant to be a way to understand a player's impact with one number. You can read about it here: http://www.sports-reference.com/blog/approximate-value/. Since we want players with 4 years of AV, we need to filter out players drafted after 2011.

# In[ ]:


av_df = dataframe[dataframe.Year <= 2011]
av_df.head()


# In[ ]:


print("total rows: " + str(len(av_df)))
print("Average First4AV: " + str(np.mean(av_df.First4AV)))
print("Median First4AV: " + str(np.median(av_df.First4AV)))


# We see here that after 4 years in the league, 1/2 the players (the median) have a First4AV (henceforth abbreviated F4AV) of less than 4. I'd like to understand the relationship between round drafted, mean F4AV, and median F4AV. It's unfortunate I can't tie contracts to players in this notebook as well; I'd like to try to find a price for a point of F4AV.

# In[ ]:


ts = av_df[["Year", "Rnd", "Pick", "First4AV"]]
ts1 = ts.groupby(['Pick']).agg({'First4AV' : [np.mean, np.median]})
ts1


# In[ ]:


rolledDF = pd.DataFrame(ts1.to_records()) #flatten our multi-index table
rolledDF.columns = ['Pick', 'F4AV_mean', 'F4AV_median']
rolledDF['F4AV_Relative_to_AV'] = rolledDF['F4AV_mean'] - rolledDF['F4AV_median']
rolledDF


# I'd like to better understand the relationship between draft pick and F4AV. I think for now, I'd like to have a df that contains 1st rounders, one for 2nd rounders, one for 3rd rounders, and one for everyone. This should give me some ability to understand the high value day 1 picks (1 - 3) as well as the field at large.

# In[ ]:


first_round_df = av_df[av_df['Rnd'] == 1.0]
second_round_df = av_df[av_df['Rnd'] == 2.0]
third_round_df = av_df[av_df['Rnd'] == 3.0]
the_field_df = av_df

first_round_df.head()


# Let's start trying to visualize First 4 AV, starting with the 1st round picks; we can scale out later

# In[ ]:


# df[['Year', 'YearsPlayed']].boxplot(by='Year')
fg = first_round_df[['Pick', 'First4AV']].boxplot(by = 'Pick')
fg.set_xlabel('Pick')
fg.set_ylabel('F4AV')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
fg


# In[ ]:


fg = first_round_df[['Pick','First4AV']].boxplot(by = 'Pick')


# In[ ]:


fg = second_round_df[['Pick', 'First4AV']].boxplot(by = 'Pick')


# In[ ]:


fg = third_round_df[['Pick', 'First4AV']].boxplot(by = 'Pick')


# Interesting...ish, I guess. I wonder what this looks like when grouped by position instead of round?

# In[ ]:


# create a new temp set pivoting on position this time
ts = av_df[["Year", "Rnd", "Pick", "Position Standard", "First4AV"]]
ts2 = ts.groupby(['Position Standard']).agg({'First4AV' : [np.mean, np.median, np.max]})

#flatten the multi-index table and sort on mean descending
rolled_ts2 = pd.DataFrame(ts2.to_records()) #flatten our multi-index table
rolled_ts2.columns = ['Position', 'F4AV_mean', 'F4AV_median', 'F4AV_max']
rolled_ts2['F4AV_Relative_to_AV'] = rolled_ts2['F4AV_mean'] - rolled_ts2['F4AV_median']
# rolled_ts2.sort(['F4AV_mean'], ascending = [0])
rolled_ts2.sort_values(by='F4AV_mean', ascending = False)


# ## Interesting! ##
# 
# Three of the top four producing positions on average are on the offensive line (Tackle, Guard, and Center), and the fourth is also a linesman (defensive line, specifically Defensive End). What this tells me is that an average offensive lineman is much, much, much more valuable than the average QB. 
# 
# Unfortunately, this dataset doesn't have much more I can go with this. I'll look for better NFL datasets in the future.

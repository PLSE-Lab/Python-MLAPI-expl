#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from time import time
import matplotlib.pyplot as plt
import numpy as np


# # Purpose of this notebook
# 
# In order to develop a predictive model, we need data on the pitchers, and the pitches they throw. The simplest way would be to just group them by year and pitcher, then calculate stats. But this kind of aggregated data shouldn't be used for prediction: it involves data that hasn't been collected at the time of the pitch we're predicting things about. So instead, we'll do rolling means and standard deviations. This has the additional advantage of tracking changes in pitchers' habits over the season. 
# 
# ## Who's the starter?
# 
# The first thing we'll calculate is whether the current pitcher is the starting pitcher. We'll do this by assuming whoever finished the first at-bat in top of the first and bottom of the first are the starters. It's possible that the starter got removed before this, but that would be extremely rare, so let's not worry about it.

# In[ ]:


t = time()
pitches = pd.read_csv('../input/pitches.csv')
atbats = pd.read_csv('../input/atbats.csv', index_col=0)
print('Loaded data in {:4.2f} seconds'.format(time()-t))


# In[ ]:


pitches.loc[pitches['pitch_type'].isna(), 'pitch_type'] = 'UN'
pitches.loc[pitches['pitch_type'].isin(['SC', 'AB', 'EP', 'FA']), 'pitch_type'] = 'UN'
pitches.loc[pitches['pitch_type'] == 'FO', 'pitch_type'] = 'PO'

atbats['year'] = (atbats.index//1e6).astype(int)


# In[ ]:


home_sp = (atbats[(atbats['inning']==1) & 
                  (atbats['top'])].groupby('g_id')
                                  .head(n=1)
                                  .set_index('g_id')['pitcher_id']
                                  .rename('home_sp'))
away_sp = (atbats[(atbats['inning']==1) & 
                  ~(atbats['top'])].groupby('g_id')
                                   .head(n=1)
                                   .set_index('g_id')['pitcher_id']
                                   .rename('away_sp'))

starting_pitchers = home_sp.to_frame().join(away_sp, how='left', on='g_id')
atbats = atbats.join(starting_pitchers, on='g_id', how='left')


atbats['is_starter'] = ((atbats['pitcher_id'] == atbats['home_sp']) | 
                                (atbats['pitcher_id'] == atbats['away_sp']))
atbats.drop(columns=['home_sp', 'away_sp'], inplace=True)


# ## Outing statistics
# 
# Next, it would be useful to know how lmany pitches this pitcher has thrown today, and compare that to how many pitches he typically throws. As the pitcher throws more an more pitches, we might expect their behavior to change: starters might rely on fewer pitches at the start, then reach back for something different later in the game. Velocity might go down.
# 
# Since we're calculating based on a rolling average, these stats will be nan for the first outing. We fix that by filling with 90 for starters and 17 for relievers, typical numbers for the last few seasons.
# 
# Note that only the 'pitch_of_outing' column is attached to the pitches dataframe. Since the outing stats won't change for a pitcher over the course of an at-bat, we can attach those to the at-bats frame to conserve memory. If we're really concerned about memory, we could make a new frame with one row for each pitcher per game, but so far this has not been an issue.

# In[ ]:


outing_stats = pitches[['ab_id']].copy()
outing_stats = outing_stats.join(atbats[['pitcher_id', 'batter_id', 
                                         'g_id', 'inning', 'top']], on='ab_id')
pitches['prev_pitches'] = outing_stats.groupby(['pitcher_id','g_id']).cumcount()
prev_pitch_ab = (pitches.groupby('ab_id')[['ab_id', 'prev_pitches']]
                        .head(n=1).set_index('ab_id'))
pitches.drop('prev_pitches', axis=1)
avg_start_len = 90
avg_relief_len = 17

avg_outing_len = (outing_stats.groupby(['pitcher_id', 'g_id']).size()
                              .rolling(window=10, min_periods=1).mean()
                              .groupby(['pitcher_id'])
                              .transform(lambda x: x.shift(1))) 

atbats = atbats.join(avg_outing_len.rename('avg_outing_len'), 
                     on=['pitcher_id', 'g_id'], how='left')
atbats.loc[atbats['avg_outing_len'].isna() & 
            atbats['is_starter'], 'avg_outing_len'] = avg_start_len
atbats.loc[atbats['avg_outing_len'].isna() & 
           ~atbats['is_starter'], 'avg_outing_len'] = avg_relief_len


# ## Ironing out a wrinkle with spin_direction
# 
# We're going to want to calculate some stats about the distributions of some trajectory variables. Now for something like start_speed, this works just as you'd expect. But for directions, we run into a problem: the variable's range is [0, 360). If a distribution crosses the cut at 0, the mean and standard deviation will be completely wrong.
# 
# Here is an efficient way to solve this problem: first, generate a new column, spin_dir_shifted, where the domain is -180 to 180. Next, compare the standard deviation of the distribution of both the variables spin_dir_shifted and spin_dir. Then use the column whose standard deviation is lower.

# In[ ]:


p = pitches.join(atbats[['pitcher_id']], on='ab_id')[['pitcher_id', 'pitch_type', 
                                                      'spin_dir', 'ab_id', 'pitch_num']]
p['spin_dir_shifted'] = (pitches['spin_dir'] + 180) % 360 - 180
p.loc[p['pitch_type'].isna(), 'pitch_type'] = 'UN'
p_grouped = p.groupby(['pitcher_id', 'pitch_type'])
should_shift = (p_grouped['spin_dir'].std() 
                - p_grouped['spin_dir_shifted'].std()) > 0.01 
                # making this 0.01 instead of 0 b/c floating point errors

p = p.join(should_shift.rename('should_shift'), on=['pitcher_id', 'pitch_type'], how='left')
pitches['spin_dir'] = p['spin_dir'].where(p['should_shift'], p['spin_dir_shifted'])
del p, should_shift, p_grouped


# # Preparation for pitch-level statistics
# 
# First, we need some data from the at-bats dataframe: the batter and pitcher IDs, and the year. Next, we'll one-hot the pitch types using the pd.get_dummies function. Because this is a large dataset (3.6 million pitches), it's pretty easy to run out of memory on the Kaggle cloud. To help prevent this, we'll work with only the columns we need. We're also going to ignore pitchouts and uncategorized pitches, and do as much calculation as possible on the at-bat level.

# In[ ]:


traj_cols = ['start_speed', 'spin_rate', 'spin_dir']
pt = pitches[['ab_id','pitch_type'] + traj_cols]
pt = pt.join(atbats[['pitcher_id', 'batter_id', 'year']], how='left', on='ab_id')
old_cols = pt.columns
pt = pd.get_dummies(pt, columns=['pitch_type'], prefix='')
pt.drop(['_UN', '_PO'], axis=1, inplace=True)
pt_dummy_cols = pt.columns.drop(old_cols.drop('pitch_type'))


# ## Pitch Selection and at-bat lengths
# 
# We need two versions of this: the actual length (columns 'tot') and the number of categorized pitches ('tot_cat'), which will be used for pitch usage stats.
# 
# The calculation of a moving average of at-bat length comes in three parts:
# 
# - group by pitcher ID and year, apply rolling function to get last 50 at-bats, sum the number of pitches of each type
# 
# - shift down by 1, so that we get the previous 50 at-bats (not including this one)
# 
# - divide each by the total number of pitches
# 
# We then do the same for to see what pitches a batter tends to face

# In[ ]:


pitches_grouped = pt.groupby('ab_id')
pitches_during_ab = pitches_grouped[pt_dummy_cols].sum()
pitches_during_ab['tot_cat'] = pitches_during_ab.sum(axis=1)
pitches_during_ab['tot'] = pitches_grouped.size()
pitches_during_ab = pitches_during_ab.merge(atbats[['pitcher_id', 'batter_id', 'year']], 
                                            left_index=True, right_index=True)

pfreq_pitcher = (pitches_during_ab.groupby(['pitcher_id', 'year'])
                                   [list(pt_dummy_cols) + ['tot', 'tot_cat']]
                                  .rolling(window=50, min_periods=1).sum())
pfreq_pitcher = pfreq_pitcher.groupby(['pitcher_id', 'year']).shift(1)
pfreq_pitcher = (pfreq_pitcher[pt_dummy_cols].div(pfreq_pitcher['tot_cat'],axis=0)
                                             .fillna(0).reset_index()
                                             .set_index('ab_id'))

pfreq_batter = (pitches_during_ab.groupby(['batter_id', 'year'])
                                 [list(pt_dummy_cols) + ['tot', 'tot_cat']]
                                 .rolling(window=50, min_periods=1).sum())
pfreq_batter = pfreq_batter.groupby(['batter_id', 'year']).shift(1)
pfreq_batter = (pfreq_batter[pt_dummy_cols].div(pfreq_batter['tot_cat'],axis=0)
                                           .fillna(0).reset_index()
                                           .set_index('ab_id'))


# # Pitch trajectory statistics
# 
# Here we will use the groupby function to group on pitch types, so we don't need the one-hot version. Again, the rolling function allows us to easily grab the last 20 pitches of each pitch type from each pitcher. This operation follows similar steps as before:
# 
# 1. Group by pitcher, year, and pitch type, apply rolling function
# 
# - calculate mean, filling with 0 when nan
# 
# - unstack the groups (now each pitch type, trajectory stat has its own column)
# 
# - use set_axis to make it a single index instead of multi-index columns
# 
# - repeat for stdev
# 
# - merge together
# 
# - fill n/a values with previous observation from this pitcher (see note below)
# 
# - grab the first row from each at-bat
# 
# - Shift the mean spin rate back to range [0, 360)
# 
# note: n/a values come about here because each row will only have trajectory information from one pitch type--the type just thrown

# In[ ]:


#1
pt = pitches[['pitch_type', 'ab_id'] + traj_cols]
pt = pt.join(atbats[['pitcher_id', 'batter_id', 'year']], on='ab_id')
pt.index.name = 'pID'
grouped_rolling = (pt.groupby(['pitcher_id', 'year', 'pitch_type'])[traj_cols]
                     .rolling(window=20, min_periods=1))

#2
traj_means = grouped_rolling.mean().fillna(0)

#3
traj_means = traj_means.unstack(level=2)

#4
traj_means = (traj_means.set_axis([f"{y}_{x}_m" for x, y in traj_means.columns], 
                                  axis=1, inplace=False)
                        .reset_index()
                        .set_index('pID'))

#5
traj_stds = grouped_rolling.std().fillna(0)
traj_stds = traj_stds.unstack(level=2)
traj_stds = (traj_stds.set_axis([f"{y}_{x}_std" for x, y in traj_stds.columns], 
                                axis=1, inplace=False)
                       .reset_index()
                       .set_index('pID'))
#6
traj_stats = (traj_means.merge(traj_stds, left_index=True, 
                               right_index=True, on=['pitcher_id', 'year'])
                         .sort_values(by='pID'))

#7
traj_stats = (traj_stats.groupby(['pitcher_id', 'year'])
                       .fillna(method='ffill')
                       .shift(1).fillna(0))

#8
traj_stats = (traj_stats.merge(pt[['ab_id']], left_index=True, right_index=True)
                        .groupby('ab_id').head(n=1).set_index('ab_id'))

#9
spindir_mean_cols = [ele for ele in atbats.columns if 'spin_dir_m' in ele]
traj_stats[spindir_mean_cols] = atbats[spindir_mean_cols] % 360


# # Merge
# 
# All that's left is to merge all the data into the atbats frame

# In[ ]:


atbats = atbats.merge(pfreq_pitcher, left_index=True, 
                      right_index=True, on=['pitcher_id', 'year'])
atbats = atbats.merge(pfreq_batter, left_index=True, 
                      right_index=True, on=['batter_id', 'year'])
atbats = atbats.merge(traj_stats, left_index=True, right_index=True)
atbats = atbats.merge(prev_pitch_ab, left_index=True, right_index=True)


# In[ ]:


atbats.tail()


# In[ ]:





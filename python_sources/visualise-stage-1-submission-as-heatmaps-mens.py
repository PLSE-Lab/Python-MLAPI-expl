#!/usr/bin/env python
# coding: utf-8

# This is a demo kernel to show a way to visualise the predictions for a single years tournament.
# 
# Evaluation can be tricky, given that only 63 ground truths for 2278 predictions are actually scored.
# 
# By plotting all the predictions it is easy to see model bias over teams, and compare the confidence of different submissions.
# 
# The plot format is introduced [here](http://www.kaggle.com/c/march-machine-learning-mania-2017/forums/t/30333/strategy-heatmaps-for-all-submissions), along with generated heatmaps for all the 2017 entries.
# 
# To recap: it is easiest to read the row for each team, where white means 50:50, red indicates probably winning, blue means probably losing, the deeper the color, the higher the probability.

# In[ ]:


import pandas as pd
import numpy as np
import os, re, sys
from scipy.special import logit, expit
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

teams = pd.read_csv('../input/mens-machine-learning-competition-2018/Teams.csv')
id2team = dict(teams[['TeamID','TeamName']].values)

seeds = pd.read_csv('../input/mens-machine-learning-competition-2018/NCAATourneySeeds.csv')

def do_plot_heatmap(probs, team_labels, filename):
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)
    # try different colormaps: http://matplotlib.org/users/colormaps.html
    heatmap = ax.pcolormesh(probs, vmin=0, vmax=1, cmap=plt.cm.seismic)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.invert_yaxis()
    ax.tick_params(direction='out')
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    plt.xticks(rotation=90)
    
    nteams = len(team_labels)
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(nteams)+0.5, minor=False)
    ax.set_yticks(np.arange(nteams)+0.5, minor=False)
    ax.set_xticklabels(team_labels, fontsize=8)
    ax.set_yticklabels(team_labels, fontsize=8)
    plt.savefig(filename, bbox_inches='tight')
    
def get_sub_year(df):
    return int(df.id.min()[:4])

def filter_sub_year(df, year):
    return df.loc[df.id.str.find(str(year))==0]

def show_heatmap(df):
    year = get_sub_year(df)
    # team ids conveniently ordered by region, seed
    team_ids = seeds.loc[seeds.Season==year].TeamID.values
    # maps global team ids to 0..67 based on target years tournament region & seed
    team2ind = {t:i for i,t in enumerate(team_ids)}
    # labels ordered by seed
    team_labels = list(map(id2team.get, team_ids))
    nteams = len(team_labels)

    parts = df.id.str.split('_')   # year, t1, t2
    t1 = parts.str[1].astype(int).map(team2ind)
    t2 = parts.str[2].astype(int).map(team2ind)
    # diagonal is notionally when a team plays itself - naturally 50/50?
    m = np.ones((nteams, nteams)) * 0.5
    # t1 is always the lower team id
    m[t1, t2] = df.pred
    m[t2, t1] = 1 - df.pred
    
    do_plot_heatmap(m, team_labels, f'plot_{year}_{df.pred.mean():.5f}.png')

def load_sub(name):
    df = pd.read_csv(name)
    df.columns = df.columns.str.lower()
    return df


# Some code to render the seed benchmark for a year:
# 

# In[ ]:


from itertools import combinations

# https://www.kaggle.com/c/march-machine-learning-mania-2014/discussion/6776
# Win % = 0.50 + 0.03 * (weak seed minus strong seed) 
def seed_benchmark_prediction(s1, s2):
    return 0.5 + (s2 - s1) * 0.03

# team -> seed; seed x seed => prob
def seed_benchmark_prediction_df(year):
    yseeds = seeds.loc[seeds.Season==year].set_index('TeamID').sort_index()
    t2s = yseeds.Seed.str[1:3].astype(int)
    def gen():
        for t1, t2 in combinations(yseeds.index, 2):
            yield f'{year}_{t1}_{t2}', seed_benchmark_prediction(t2s[t1], t2s[t2])
    return pd.DataFrame.from_records(gen(), columns=['id','pred'])


# First, the seed benchmark for 2017, notice the play-ins distort the grid a bit, some regions have over 16 teams. This is a standard, default submission, most normal submissions that model team strength will look at least vaguely similar to this...

# In[ ]:


show_heatmap(seed_benchmark_prediction_df(2017))


# In[ ]:


# quick copy/paste to show tournament round structure instead
def team_seed_lookup(year):
    seeds = pd.read_csv('../input/mens-machine-learning-competition-2018/NCAATourneySeeds.csv')
    seeds = seeds.loc[seeds.Season==year]
    return seeds.set_index('TeamID').Seed

def show_tournament_structure(year):
    team2seed = team_seed_lookup(year)
    sub = pd.read_csv('../input/mens-machine-learning-competition-2018/SampleSubmissionStage1.csv')
    sub = sub.loc[sub.ID.str.startswith(str(year))]

    # team ids conveniently ordered by region, seed
    team_ids = seeds.loc[seeds.Season==year].TeamID.values
    # maps global team ids to 0..67 based on target years tournament region & seed
    team2ind = {t:i for i,t in enumerate(team_ids)}
    # labels ordered by seed
    team_labels = list(map(id2team.get, team_ids))
    nteams = len(team_labels)
    
    parts = sub.ID.str.split('_')   # year, t1, t2
    sub['t1'] = parts.str[1].astype(int)
    sub['t2'] = parts.str[2].astype(int)
    sub['s1'] = sub.t1.map(team2seed)
    sub['s2'] = sub.t2.map(team2seed)
    sub['i1'] = sub.t1.map(team2ind)
    sub['i2'] = sub.t2.map(team2ind)
    
    slots = pd.read_csv('../input/mens-machine-learning-competition-2018/NCAATourneySeedRoundSlots.csv')
    d = slots.groupby('Seed').GameSlot.apply(set).to_dict()
    # this is round 1..6
    mc = sub.apply(lambda row: len(d[row.s1] & d[row.s2]), axis=1)
    mc = (mc-1) / 5. # map to 0..1

    m = np.zeros((nteams,nteams))
    m[sub.i1, sub.i2] = mc.values
    m[sub.i2, sub.i1] = mc.values
    do_plot_heatmap(m, team_labels, f'tournament_map_{year}.png')


# To help show which round each matchup between teams would be in, here is a plot, early rounds are red, later rounds darker blue. The later a round is, the more predictions it requires, so the larger its area. It is easiest to look at the color for the top seed in each region and follow that, e.g. seed 1 plays seed 16 in the first round, then the winner of the #7 vs #8 match, etc...
# 
# (The play-ins are shown the same color as round 1, but are not scored.)

# In[ ]:


show_tournament_structure(2017)


# To illustrate, I will use the attached collaborative filtering demo, which comes from [last years kernels](http://www.kaggle.com/aikinogard/cf-starter-with-keras-0-560136), which itself comes from the [collaborative filtering lesson](http://course.fast.ai/lessons/lesson5.html) of the excellent fast.ai course.

# In[ ]:


sub = load_sub('../input/collaborative-filtering/CF.csv')
sub.head()


# Here are each of the years, in reverse.

# In[ ]:


show_heatmap(filter_sub_year(sub, 2017))


# In[ ]:


show_heatmap(filter_sub_year(sub, 2016))


# In[ ]:


show_heatmap(filter_sub_year(sub, 2015))


# In[ ]:


show_heatmap(filter_sub_year(sub, 2014))


# In[ ]:


def subtract_seed_benchmark(sub):
    bm = seed_benchmark_prediction_df(get_sub_year(sub)).set_index('id')
    sub = sub.set_index('id')
    sub.pred -= bm.pred
    sub.pred = (sub.pred + 1) / 2  # remap [-1..1] to [0..1]
    return sub.reset_index()

def cmp_seed_benchmark(sub):
    year = get_sub_year(sub)
    sub = sub.set_index('id')
    sub['bm'] = seed_benchmark_prediction_df(year).set_index('id').pred
    sub['diff'] = sub.pred - sub.bm
    return sub.sort_values('diff')


# New for 2018, here is a function to subtract the seed benchmark for the given year. Now (again describing rows) white means the prediction is the same as the seed benchmark, red means the model favours the team more, blue means unfavoured compared to their seeding.
# 
# For 2016 the CF model clearly prefers Syracuse and Gonzaga, rating their chances higher than their seeding.

# In[ ]:


show_heatmap(subtract_seed_benchmark(filter_sub_year(sub, 2016)))


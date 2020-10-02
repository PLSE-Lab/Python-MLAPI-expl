#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_df = pd.read_csv('../input/Speed Dating Data.csv', encoding="ISO-8859-1")
fields = data_df.columns
# Num of fields and some of their names
# print('Number of fields: {0}\n1-10: {1}\n11-20: {2}\n21-30: {3}'.format(len(fields), 
#       fields[0:11], fields[11:21], fields[21:31]))
# Some samples
# print('Example row: {}'.format(data_df.head(1)))
print('Total number of people that participated, assuming person does not appear in more than one wave: {}'.format(len(data_df['iid'].unique())))
print('Total number of dates occurred: {}'.format(len(data_df.index)))


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(17,5))

# The number of dates per person
num_dates_per_male = data_df[data_df.gender == 1].groupby('iid').apply(len)
num_dates_per_female = data_df[data_df.gender == 0].groupby('iid').apply(len)
axes[0].hist(num_dates_per_male, bins=22, alpha=0.5, label='# dates per male')
axes[0].hist(num_dates_per_female, bins=22, alpha=0.5, label='# dates per female')
# axes[0].suptitle('Number of dates per male/female')
axes[0].legend(loc='upper right')

# The number of matches per person
matches = data_df[data_df.match == 1]
matches_male = matches[matches.gender == 1].groupby('iid').apply(len)
matches_female = matches[matches.gender == 0].groupby('iid').apply(len)
axes[1].hist((matches_male / num_dates_per_male).dropna(), alpha=0.5, label='male match percentage')
axes[1].hist((matches_female / num_dates_per_female).dropna(), alpha=0.5, label='female match percentage')
axes[1].legend(loc='upper right')
# axes[1].suptitle('Matches per person by gender')

print('Avg. dates per male: {0:.1f}\t\tAvg. dates per female: {1:.1f}\nAvg. male match percentage: {2:.2f}\tAvg. female match percentage: {3:.2f}'.format(
        num_dates_per_male.mean(), 
        num_dates_per_female.mean(),
        (matches_male / num_dates_per_male).mean() * 100.0,
        (matches_female / num_dates_per_female).mean() * 100.0))


# The histograms above indicate that most people have had at least 10 dates, suggesting that most
# of them experienced a good diversity of people. Also, there is not much of a difference in the number
# dates that males and females receive, and also both genders have similar match rates. This suggests
# that participants can expect roughly similar amount of success, irrespective of gender.

# # Made for each other - Predicting matches #
# What are the most critical factors for a match to occur ? How do these factors vary across age,
# race, income, primary goal in participation ? Do people in similar professions like STEM or 
# entertainment have a higher/lower preference for each other or no such relation exists ? We will
# call these *objective factors*.
# 
# For the first part, we will exclude the attributes that a person values most, thinks that others
# value most, how well a person thinks that they measure up etc. We will call them *subjective factors*.

# ## Objective factors ##
# 
# The outcome of a date often depends on mutual factors. Let us identify some that are readily available:
# 
# + int\_corr (correlation between interests), 
# + same race, 
# + age difference (derived from age and age\_o)
# 
# Personal traits also determine whether a date will be successful. These are factors that
# are specific to the individual and may significantly influence an individual's decision irrespective
# of shared factors. Similarity of traits are also expected to lead to higher probability of match.
# Some example traits:
# 
# + intelligence (from mn\_sat, are intelligent people more picky ?),
# + goal (how serious or cavalier is the individual about this whole thing),
# + date (how experienced is the person in dating),
# + go\_out (how socially active is this person),
# + income + tuition (financial background),
# + profession (derived from field\_cd)
# + imprace + impreligion (a lower value indicates more open-mindedness towards other cultures)
# 
# Since the trait info for both people are not available through a single record, we will need to
# create a datastore that contains person profiles indexed by iid, allowing us to lookup partner
# traits for similarity computation.

# In[ ]:


# Create a dataframe containing information for each person that needs to be looked up
profiles = data_df[['iid', 'mn_sat', 'goal', 'go_out', 'date', 'tuition', 'income',                     'field_cd', 'imprace', 'imprelig']].set_index(keys='iid').drop_duplicates()
for trait in ['mn_sat', 'tuition', 'income']:
    profiles[trait] = profiles[trait].apply(lambda x: str(x).replace(",", "")).astype('float64')
profiles = profiles.fillna(profiles.mean())  # Fill NaN values with mean


# In[ ]:


# Computing columns for objective factors that have to be derived
data_df['age_diff'] = data_df['age'].sub(data_df['age_o']).abs()

def is_similar_profession(x, profiles):
    if np.isnan(x['field_cd']) or np.isnan(x['pid']) or int(x['pid']) not in profiles.index:
        return False
    else:
        return x['field_cd'] == profiles.loc[int(x['pid'])]['field_cd']
data_df['sim_profession'] = data_df.apply(lambda x: is_similar_profession(x, profiles), axis=1)
data_df['financial_bg'] = data_df['income'].apply(zscore)    # financial background


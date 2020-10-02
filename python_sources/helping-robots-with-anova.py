#!/usr/bin/env python
# coding: utf-8

# Helping Robots with ANOVA
# ==========================
# 
# I had a first look at the data of this competition today and did some basic exploration. Especially the group_id feature made me curious. I was wondering, if all the samples were recorded under similar conditions and decided to carry out an analysis of variants to compare samples with the same targets taken in different recording sessions. This kernel basically gives you the results, leaving room for some interpretation.
# 
# ## Import libraries and data

# In[1]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway

X_train = pd.read_csv('../input/X_train.csv')
X_test = pd.read_csv('../input/X_test.csv')
y_train = pd.read_csv('../input/y_train.csv')


# ## Some data exploration and ANOVA
# Let's have a look at our features and columns. And then carry out the ANOVA across measurement groups (i.e. recording sessions).

# In[2]:


X_train.head()


# In[3]:


# any null values?
X_train.isnull().sum()


# In[4]:


y_train.head()


# In[5]:


# what are our surface materials i.e. targets?
np.unique(y_train['surface'])


# Our dataset consists of series with a length of 128 measurements and the values from the 10 sensor channels. Every series of measurements has one target, which is a description of the surface material. Luckily, we don't have to deal with any missing values. 
# 
# In y_train we're furthermore given a group_id which indicates the recording session this particular series was taken. Across each session, the robot was only driving on one surface. We will make use of that information and see, if there is any variation in means across different recording sessions within the features (because of varying sensor calibrations, environment etc.).

# In[39]:


# encode surface targets
encoder = LabelEncoder()
surfaces = np.unique(y_train['surface'])
y_train['surface'] = encoder.fit_transform(y_train['surface'])

# do we have a strong variation in means across groups?
# let's find out with pairwise t-Tests for groups with same surface
joined = X_train.set_index('series_id').join(
    y_train.set_index('series_id'))

def anova_across_surface(surface, X):
    # helper function to calculate anovas for group samples of surface levels
    records = X[X.loc[:, 'surface']==surface]
    group_nos = np.unique(records.loc[:, 'group_id'])
    anovas = []
    for col in records.columns:
        samples = [list(X[X.loc[:, 'group_id'] == i][col]) for i in group_nos]
        aov = f_oneway(*samples)
        anovas.append(aov[0])
        anovas.append(aov[1])
    return anovas

# calculate all the anovas first
anovas = dict()
for i in range(0, 9):
    # for each surface level
    anovas[i] = anova_across_surface(i, joined)
    
# make nice tables
no_of_columns = 3
new_cols = ['row_id\t', 'measr_no', 'orien_X', 'orient_Y',
       'orient_Z', 'orient_W', 'velocity_X',
       'velocity_Y', 'velocity_Z', 'accel_X',
       'accel_Y', 'accel_Z', 'group_id',
       'surface']
joined.columns = new_cols
line_1 = '\n' + '\t\t|%s' * no_of_columns
line_2 = 'surface\t\t' + '|F\t      p-value\t' * no_of_columns
line_3 = '%12.12s' + '\t|%9.3e   %8.3f' * no_of_columns

for i in range(no_of_columns, len(joined.columns), no_of_columns):
    print(line_1 % tuple(joined.columns[i-no_of_columns:i]))
    print(line_2)
    print('=' * 22 * (no_of_columns + 1))
    for j, surface in zip(range(0,9), encoder.inverse_transform(list(range(0, 9)))):
        row = anovas[j][2*(i-no_of_columns):2*i]
        print(line_3 % tuple([surface] + anovas[j][2*(i-no_of_columns):2*i]))


# Let's see what we can get out of the above table! For each combination of surface material and feature, we have the F-statistic and p-value for a one-way ANOVA that was calculated across samples taken during different recording sessions. In other words, for each surface material we asked, if the means of the features are the same across different recording sessions. A low p-value close to zero corresponds to the answer 'no' and we can assume, that the means are not very similar. A high p-value gives us a hint at similar means.
# 
# At first we notice, that for the hard_tiles material, ANOVA didn't provide us with results. This is due to the fact, we have only one sample for hard_tiles in our dataset. ANOVA is therefore obsolete in this case.
# 
# Looking at the second row in table 1, we can sanity-check our calculations. The p-value for equal means of the measurement_id's is 1. This makes sense, because for each sample the measurement_id's are just integers from 0 to 127 and the means thus equal across all samples.
# 
# We can now have a look at the more interesting parts of the above tables. For many entries, we have very extreme values for F and thus p-values very close to 0 (so close, that for many entries we actually cut off the non-zero decimals while formatting). For these values, we have to reject the hypothesis of equal means.
# 
# But there's also a few tests, that indicate similar means:
# 
# 1. Samples of velocity_X for carpet, concrete, hard_tiles_large and tiled materials
#    (with p-values of 0.725, 0.953, 0.960 and 0.97 respectively).
# 2. Samples of acceleration_Z for carpet, concrete, fine_concrete, hard_tiles_large, tiled and wood 
#    (with p-values of 0.519, 1, 0.951, 0.989, 0.999, 0.997 respectively). 
# 
# Interesting to note are also some of the tests, where our hypothesis of similar means has to be rejected even a very low significance levels. I leave it up to the reader to interpret the above results.

# In[ ]:





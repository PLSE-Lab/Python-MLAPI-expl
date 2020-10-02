#!/usr/bin/env python
# coding: utf-8

# I have been struggling with this competition to find an angle to use traditional machine learning techniques rather than using kalman filters and hough transforms
# 
# The following shows that one *could***** do this challenge by grouping predictions by  the angular momentum along the z axis
# 
# I noticed the track ml stuff is broken - nice one Kaggle ;(
# 
# Thanks to CPMP for providing alternative functions

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[ ]:


def score_event_fast(truth, submission):
    truth = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    df = truth.groupby(['track_id', 'particle_id']).hit_id.count().to_frame('count_both').reset_index()
    truth = truth.merge(df, how='left', on=['track_id', 'particle_id'])
    
    df1 = df.groupby(['particle_id']).count_both.sum().to_frame('count_particle').reset_index()
    truth = truth.merge(df1, how='left', on='particle_id')
    df1 = df.groupby(['track_id']).count_both.sum().to_frame('count_track').reset_index()
    truth = truth.merge(df1, how='left', on='track_id')
    truth.count_both *= 2
    score = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].weight.sum()
    return score

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission


# In[ ]:


path_to_train = "../input/train_1"
event_prefix = "event000001000"

event_id = 0
hits = pd.read_csv('../input/train_1/event00000100%d-hits.csv' % event_id)
particles = pd.read_csv('../input/train_1/event00000100%d-particles.csv' % event_id)
truth = pd.read_csv('../input/train_1/event00000100%d-truth.csv' % event_id)
cell = pd.read_csv('../input/train_1/event00000100%d-cells.csv' % event_id)


# In[ ]:


hits = hits.merge(truth,on='hit_id')
print(hits.shape)
hits = hits.merge(particles,on='particle_id',how='left')
print(hits.shape)
hits['target'] = np.sqrt(hits.px**2+hits.py**2)
hits = hits.fillna(0)


# In[ ]:


le = LabelEncoder()
submission = create_one_event_submission(0, hits, le.fit_transform(hits.target))
score = score_event_fast(truth, submission)
score


#!/usr/bin/env python
# coding: utf-8

# *Thanks Luis Andre Dutra e Silva*
# 
# **## Based on the baseline
# The main difference is the use of HDBSCAN as clustering algorithm

# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

import operator


# Add a custom package:
# 1. Click the [<] button at top-right of Kernel screen
# 2. Click Settings
# 3. Enter "LAL/trackml-library", e.g., into "GitHub user/repo" space at the bottom
# 4. Click the (->) button to the left of that
# 5. Restart Kernel by clicking the circular refresh/recycle-y button at the bottom-right of the screen, in the Console
# 6. Custom libraries will now import when imported

# In[44]:


# Change this according to your directory preferred setting
path_to_train = "../input/train_1"


# #  Working on one event

# In[45]:


# This event is in Train_1
event_prefix = "event000001000"
RZ_SCALE = [0.65, 0.965, 1.418] #1.41
LEAF_SIZE = 50


# ## Read and look

# In[46]:


hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))


# ## Identify tracks 
# 
# In this example the track pattern recognition is solved as clustering problem. Each of the clusters corresponds to one track. 
# Firstly we preprocess hit coordinates in order to highlight the fact that a track is (approximatly) an arc of helix. 
# 
# 
# $$ 
# r_{1} = \sqrt{x^{2}+y^{2}+z^{2}}
# $$
# 
# $$
# x_{2} = x / r_{1}
# $$
# $$
# y_{2} = y / r_{1}
# $$
# 
# $$
# r_{2} = \sqrt{x^{2}+y^{2}}
# $$
# 
# $$
# z_{2} = z / r_{2}
# $$
# 
# 
# 
# ![dbscan_pic.png](attachment:dbscan_pic.png)
# 
# Then, DBSCAN is used to recognize hit clusters. 

# In[47]:


from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.neighbors import NearestNeighbors
from scipy import stats
"""
updated - added self.rz_scale
"""
class Clusterer(object):
    
    def _preprocess(self, hits, rz_scales):
        
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values

        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x/r
        hits['y2'] = y/r

        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z/r

        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
        for i, rz_scale in enumerate(rz_scales):
            X[:,i] = X[:,i] * rz_scale
        
        return X
    
    
    def predict(self, hits, rz_scales):
        X = self._preprocess(hits, rz_scales)    
        cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,cluster_selection_method='leaf',metric='braycurtis', leaf_size=LEAF_SIZE)
        clusters = cl.fit_predict(X)+1
        return clusters


# ## Score
# 
# Compute the score for this event. The dummy submission output of create_one_event_submission  is created only to be the second parameter of the score_event function. It should not be confused with a well-behaved submission for the test set. 

# In[48]:


def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission


# # Recognize tracks in all events of a dataset
# In this example, the dataset is the whole training set.   
# This is a simple loop over the one-event actions: because of the use of DBScan, there is no actual training.
# 
# This may take a very long time. To run on only a subset, use
# 
#      load_dataset(path_to_train, skip=100, nevents=15)
# 
# It will skip the first 100 events, and select the next 15 ones.

# 1. Best first two coefficients for scale is 0.65, 0.97

# In[49]:


dataset_submissions = []
dataset_scores = []
rz_scales = RZ_SCALE
ch_scales = np.linspace(1.418, 1.428, num=10)
ordinate = 2
ch_scores = {}
for ch_scale in ch_scales:
    rz_scales[ordinate] = ch_scale
    print("Test for", rz_scales)
    
    for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=0, nevents=15):
        # Track pattern recognition
        model = Clusterer()
        labels = model.predict(hits, rz_scales)

        # Prepare submission for an event
        one_submission = create_one_event_submission(event_id, hits, labels)
        dataset_submissions.append(one_submission)

        # Score for the event
        score = score_event(truth, one_submission)
        dataset_scores.append(score)

        print("Score for event %d: %.9f" % (event_id, score))
    print('Mean score: %.9f' % (np.mean(dataset_scores)))
    ch_scores[ch_scale] = np.mean(dataset_scores)
    print(ch_scores)
    
sorted_ch_scales = sorted(ch_scores.items(), key=operator.itemgetter(1))
print("Best x ", sorted_ch_scales)
RZ_SCALE[ordinate] = sorted_ch_scales[-1][0]
print("new RZ_SCALE", RZ_SCALE)


# # Create a submission
# 
# Create a submission file. 

# In[50]:


path_to_test = "../input/test"
test_dataset_submissions = []

create_submission = True # True for submission 

if create_submission:
    for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):

        # Track pattern recognition
        model = Clusterer()
        labels = model.predict(hits, RZ_SCALE)

        # Prepare submission for an event
        one_submission = create_one_event_submission(event_id, hits, labels)
        test_dataset_submissions.append(one_submission)
        
        print('Event ID: ', event_id)

    # Create submission file
    submission = pd.concat(test_dataset_submissions, axis=0)
    submission.to_csv('submission.csv', index=False)


# In[ ]:





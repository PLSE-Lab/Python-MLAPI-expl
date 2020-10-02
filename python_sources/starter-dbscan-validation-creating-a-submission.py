#!/usr/bin/env python
# coding: utf-8

# # Starter DBSCAN, Validation, and Creating a Submission
# 
# In this kernel, I used a starter DBSCAN (density not tuned and basic preprocessing) to cluster a single event from the train set and to cluster all events from the test set. I also show how to validate predictions using the trackml library and how to create a valid submission.
# 
# Update: Sorted the test events when creating a submission (required for a valid submission)

# ## Table of Contents
# - Import libraries
# - Starter DBSCAN clustering and validation
# - Clustering the test set
# - Creating a submission

# ## Import libraries
# Learned how to import trackml from: https://www.kaggle.com/wesamelshamy/trackml-problem-explanation-and-data-exploration

# In[ ]:


import os

import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MaxAbsScaler


# ## Load a single event from train set

# In[ ]:


hits, cells, particles, truth = load_event('../input/train_1/event000001000')
hits.head()


# ## Starter DBSCAN clustering and validation
# ### Scale the data

# In[ ]:


X = hits[['x', 'y', 'z']]
scaler = MaxAbsScaler().fit(X)
X = scaler.transform(X)


# ### Cluster the hits

# In[ ]:


eps = 0.002
min_samp = 2
db = DBSCAN(eps=eps, min_samples=min_samp, metric='euclidean').fit(X)
labels = db.labels_


# ### Validate the clustering
# My **LB score of 0.1535** is very consistent with my current best **validation score of ~0.15** using a few events.

# In[ ]:


clustering = pd.DataFrame()
clustering['hit_id'] = truth['hit_id']
clustering['track_id'] = labels

score = score_event(truth, clustering)
print(score)


# ### Other clustering metrics
# Homogeneity and Completeness are actually high when ignoring rejected samples.

# In[ ]:


labels_true = truth['particle_id']
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print('CLUSTERING RESULTS:')
print('Estimated number of clusters: %d' % n_clusters)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
rej_perc = list(labels).count(-1) / float(hits.shape[0]) * 100
rej_perc = round(rej_perc, 2)
print ("Rejected samples %:", str(rej_perc) + '%')
rejected_count = list(labels).count(-1)
print ("Rejected samples:", rejected_count)
print ("Total samples:", hits.shape[0])
print ("Clustered samples:", hits.shape[0] - list(labels).count(-1), '\n')

print ('WITHOUT REJECTED SAMPLES:')
labels_true_wr = labels_true[labels != -1]
labels_wr = labels[labels != -1]
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true_wr, labels_wr))
print(("Completeness: %0.3f" % metrics.completeness_score(labels_true_wr, labels_wr)), '\n')


# ## Clustering the test set

# In[ ]:


def cluster_event(event_filename):
    # function for clustering hits in a given event file
    
    print('clustering {}...'.format(event_filename))
    
    # load event
    event_name = event_filename.split('-')[0]
    test_hits, test_cells = load_event('../input/test/{}'.format(event_name), parts=['hits', 'cells'])
    
    # scale hits
    X_test = test_hits[['x', 'y', 'z']]
    scaler = MaxAbsScaler().fit(X_test)
    X_test = scaler.transform(X_test)
    
    # cluster hits
    eps = 0.002
    min_samp = 2
    db = DBSCAN(eps=eps, min_samples=min_samp, metric='euclidean').fit(X_test)
    labels = db.labels_
    
    # format output
    event = pd.DataFrame()
    event_num = int(event_name.split('event')[-1])
    event['event_id'] = [event_num] * len(test_hits['hit_id'])
    event['hit_id'] = test_hits['hit_id']
    # apparently, track_id values can't be negative when i tried submitting
    # add 1 to labels values because rejected samples are labeled -1
    event['track_id'] = labels + 1
    
    return event


# ## Creating a submission
# ### Cluster all events and append results to DataFrame
# The events must be sorted.

# In[ ]:


submission = pd.DataFrame()
for event_filename in sorted(os.listdir("../input/test")):
    if '-hits.csv' in event_filename:
        clustered_event = cluster_event(event_filename)
        submission = pd.concat([submission, clustered_event])


# ### Write output to csv
# Note that I **removed negative values from the track_id column** in the above function. Negative values in the track_id column are not accepted.

# In[ ]:


submission.to_csv('dbcan_init.csv', index=False)


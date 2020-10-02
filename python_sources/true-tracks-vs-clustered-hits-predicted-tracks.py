#!/usr/bin/env python
# coding: utf-8

# # Visualization: True Tracks VS Clustered Hits (Predicted Tracks)
# I created this kernel because I wanted to know what the tracks look like, what the clusters (made w/ DBSCAN) look like, and if the clustering is working correctly (or working at all) aside from knowing the LB score.
# 
# I will show plots of tracks and plots of clustered hits (or predicted tracks) on the same points. The tracks are plotted using the raw hits coordinates and also using the transformed hits coordinates (based on the DBSCAN benchmark kernel). I will also relate the plots to how the clustering is expected to work. Clustered hits are visualized for 4 different settings using DBSCAN.
# 
# The code for the track visualizations are based on Joshua Bonatt's kernel: https://www.kaggle.com/jbonatt/trackml-eda-etc
# 
# The code for the coordinate transformation are based on Mikhail Hushchyn's DBSCAN benchmark kernel: https://www.kaggle.com/mikhailhushchyn/dbscan-benchmark
# 
# Learned how to import trackml from Wesam Elshamy's kernel: https://www.kaggle.com/wesamelshamy/trackml-problem-explanation-and-data-exploration

# # Table of Contents
# - Import Libraries
# - Loading a Single Event
# - Track Visualization
#     - Raw Hits Coordinates
#     - Preprocessesd Hits Data (w/ Coordinate Transformation)
# - Clustered Hits Visualization

# # Import Libraries

# In[1]:


import os

import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.score import score_event

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, Normalizer

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy import spatial


# ## function to be used later
# a function for arranging hits in a track

# In[2]:


# function for arranging hits in a track (closest to origin, closest track to previous track, and so on...)
def arrange_track(track_points):
    arranged_track = pd.DataFrame()

    pt = [0, 0, 0]
    kdtree = spatial.KDTree(track_points)
    distance, index = kdtree.query(pt)

    arranged_track = arranged_track.append(track_points.iloc[index])
    track_points = track_points.drop(track_points.index[index]).reset_index(drop=True)

    while not track_points.empty:
        pt = arranged_track.iloc[-1]
        kdtree = spatial.KDTree(track_points)
        distance, index = kdtree.query(pt)

        arranged_track = arranged_track.append(track_points.iloc[index])
        track_points = track_points.drop(track_points.index[index]).reset_index(drop=True)
        
    return arranged_track

test_points = pd.DataFrame([[0, 0, 5], [0, 0, 1], [0, 0, 3], [0, 0, 2]])
arrange_track(test_points)


# # Load a single event from train set

# In[3]:


hits, cells, particles, truth = load_event('../input/train_1/event000001000')
hits.head()


# In[4]:


truth.head()


# # Track Visualization
# visualization code is based on the kernel: https://www.kaggle.com/jbonatt/trackml-eda-etc
# ## Raw Hits Data
# The following plots show the tracks as they are. The tracks are identified by color using the **truth** data but the coordinates used are from the **hits** data. I use the term "True Tracks" to distinguish these plots from the predicted tracks later on.
# ### every 500th track

# In[8]:


tracks = truth.particle_id.unique()[1::500]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    ax.plot3D(t.z, t.x, t.y, '.', ms=10)
    
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.set_title("True Tracks (Scatter Plot)", y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    t = arrange_track(t)
    ax2.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('x (mm)')
ax2.set_zlabel('y (mm)')
ax2.set_title("True Tracks (Line Plot)", y=-.15, size=20)

plt.show()


# ### every 100th track

# In[9]:


tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    ax.plot3D(t.z, t.x, t.y, '.', ms=10)
    
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.set_title("True Tracks (Scatter Plot)", y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    t = arrange_track(t)
    ax2.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('x (mm)')
ax2.set_zlabel('y (mm)')
ax2.set_title("True Tracks (Line Plot)", y=-.15, size=20)

plt.show()


# Looking at the plots above, we can see a lot of points near the origin that are very close to each other but belong to different tracks. Because of this, we can expect that using DBSCAN (or other clustering algorithms like K-means) will not be effective because these points will probably be clustered together.

# ## Preprocessed Hits Data
# The following plots show the tracks after going through the coordinate transformation performed in the DBSCAN benchmark kernel: https://www.kaggle.com/mikhailhushchyn/dbscan-benchmark
# ### Preprocessing / Coordinate Transformation

# In[11]:


# DBSCAN benchmark preprocessing / coordinate transformation
x = hits.x.values
y = hits.y.values
z = hits.z.values

r = np.sqrt(x**2 + y**2 + z**2)
hits['x2'] = x/r
hits['y2'] = y/r

r = np.sqrt(x**2 + y**2)
hits['z2'] = z/r


# ### every 500th track

# In[12]:


tracks = truth.particle_id.unique()[1::500]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    ax.plot3D(t.z2, t.x2, t.y2, '.', ms=10)
    
ax.set_xlabel('z2')
ax.set_ylabel('x2')
ax.set_zlabel('y2')
ax.set_title("True Tracks (Scatter Plot)", y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    t = arrange_track(t)
    ax2.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax2.set_xlabel('z2')
ax2.set_ylabel('x2')
ax2.set_zlabel('y2')
ax2.set_title("True Tracks (Line Plot)", y=-.15, size=20)

plt.show()


# Mikhail, the author of the DBSCAN benchmark kernel (https://www.kaggle.com/mikhailhushchyn/dbscan-benchmark) mentioned that "The transformation just make hits from the same track closer to each other." Indeed, that is evident in these plots. Looking at the plots above, DBSCAN or other clustering algorithms might work since the tracks appear separated from each other. However, look at the next plots.

# ### every 100th track

# In[13]:


tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    ax.plot3D(t.z2, t.x2, t.y2, '.', ms=10)
    
ax.set_xlabel('z2)')
ax.set_ylabel('x2')
ax.set_zlabel('y2')
ax.set_title("True Tracks (Scatter Plot)", y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    t = arrange_track(t)
    ax2.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax2.set_xlabel('z2')
ax2.set_ylabel('x2')
ax2.set_zlabel('y2')
ax2.set_title("True Tracks (Line Plot)", y=-.15, size=20)

plt.show()


# As more tracks are included in the plot, we can see that many tracks are again close to each other, which means that clustering is still difficult.

# # Clustered Hits Visualization
# Next, on the same points, let's visualize the resulting clusters (or predicted tracks).

# ## DBSCAN Clustering Setting 1
# ## based on the benchmark: coordinate transformation + dbscan

# In[ ]:


X = hits[['x2', 'y2', 'z2']]
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

eps = 0.008
min_samp = 1
db = DBSCAN(eps=eps, min_samples=min_samp, metric='euclidean').fit(X)
labels = db.labels_

clustering = pd.DataFrame()
clustering['hit_id'] = truth['hit_id']
clustering['track_id'] = labels

score = score_event(truth, clustering)
print('track-ml custom metric score:', round(score, 4))

labels_true = truth['particle_id']
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print('\nOTHER CLUSTERING RESULTS:')
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
print ("Clustered samples:", hits.shape[0] - list(labels).count(-1))


# ## Clustered Hits (Predicted Tracks) vs True Tracks - Transformed Coordinates

# In[ ]:


tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x2', 'y2', 'z2']]
    if cluster == -1:
        ax.plot3D(t.z2, t.x2, t.y2, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax.set_xlabel('z2')
ax.set_ylabel('x2')
ax.set_zlabel('y2')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    t = arrange_track(t)
    ax2.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax2.set_xlabel('z2')
ax2.set_ylabel('x2')
ax2.set_zlabel('y2')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()


# Here we see that the clusters are really small and we can't actually observe any tracks.

# ## Clustered Hits (Predicted Tracks) vs True Tracks - Raw Hits Coordinates

# In[ ]:


tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x', 'y', 'z']]
    if cluster == -1:
        ax.plot3D(t.z, t.x, t.y, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    t = arrange_track(t)
    ax2.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('x (mm)')
ax2.set_zlabel('y (mm)')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()


# Here, we can see a few formed tracks but still many small clusters. The tracks that were formed are straight and do not follow the shape of the true tracks.

# ## DBSCAN Clustering Setting 2
# ## based on the benchmark: coordinate transformation + dbscan
# ## high eps (eps=0.018)
# Increasing the eps means decreasing the density required to form a cluster. eps is the distance that is used to define the neighbors of a sample. Since we increased eps, we expect bigger clusters to be formed.

# In[ ]:


X = hits[['x2', 'y2', 'z2']]
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

eps = 0.018
min_samp = 1
db = DBSCAN(eps=eps, min_samples=min_samp, metric='euclidean').fit(X)
labels = db.labels_

clustering = pd.DataFrame()
clustering['hit_id'] = truth['hit_id']
clustering['track_id'] = labels

score = score_event(truth, clustering)
print('track-ml custom metric score:', round(score, 4))

labels_true = truth['particle_id']
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print('\nOTHER CLUSTERING RESULTS:')
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
print ("Clustered samples:", hits.shape[0] - list(labels).count(-1))


# ## Clustered Hits (Predicted Tracks) vs True Tracks - Transformed Coordinates

# In[ ]:


tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x2', 'y2', 'z2']]
    if cluster == -1:
        ax.plot3D(t.z2, t.x2, t.y2, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax.set_xlabel('z2')
ax.set_ylabel('x2')
ax.set_zlabel('y2')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    t = arrange_track(t)
    ax2.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax2.set_xlabel('z2')
ax2.set_ylabel('x2')
ax2.set_zlabel('y2')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()


# As expected, bigger clusters were formed. We can now observe a few tracks. However, the predicted tracks don't seem to match the true tracks. There is also a strange looking predicted track (orange cluster in the middle).

# ## Clustered Hits (Predicted Tracks) vs True Tracks - Raw Hits Coordinates

# In[ ]:


tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x', 'y', 'z']]
    if cluster == -1:
        ax.plot3D(t.z, t.x, t.y, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    t = arrange_track(t)
    ax2.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('x (mm)')
ax2.set_zlabel('y (mm)')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()


# We also observe more tracks in this coordinate system. The strange orange track is seen more clearly here. These are hits that actually belong to different tracks but were clustered together.

# ## DBSCAN Clustering Setting 3
# ## based on the benchmark: coordinate transformation + dbscan
# ## min_samp=3
# Increasing the minimum samples means decreasing the density required to form a cluster. Let's see the effects of this adjustment to the score and clustering metrics.

# In[ ]:


X = hits[['x2', 'y2', 'z2']]
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

eps = 0.008
min_samp = 3
db = DBSCAN(eps=eps, min_samples=min_samp, metric='euclidean').fit(X)
labels = db.labels_

clustering = pd.DataFrame()
clustering['hit_id'] = truth['hit_id']
clustering['track_id'] = labels

score = score_event(truth, clustering)
print('track-ml custom metric score:', round(score, 4))

labels_true = truth['particle_id']
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print('\nOTHER CLUSTERING RESULTS:')
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


# Oddly enough, the score is exactly the same, while the clustering metrics drastically changed compared to Setting 1. In this setting, we observe a lot of rejected samples (samples that did not join any cluster). This is not suprising since the **min_samp** parameter also determines the minimum number of samples in a cluster. This result means that there are hits that are too far from other hits and were unable to join a cluster.

# ## Clustered Hits (Predicted Tracks) vs True Tracks - Transformed Coordinates

# In[ ]:


tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x2', 'y2', 'z2']]
    if cluster == -1:
        ax.plot3D(t.z2, t.x2, t.y2, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax.set_xlabel('z2')
ax.set_ylabel('x2')
ax.set_zlabel('y2')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    t = arrange_track(t)
    ax2.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax2.set_xlabel('z2')
ax2.set_ylabel('x2')
ax2.set_zlabel('y2')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()


# The rejected hits are plotted as black points.

# ## Clustered Hits (Predicted Tracks) vs True Tracks - Raw Hits Coordinates

# In[ ]:


tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x', 'y', 'z']]
    if cluster == -1:
        ax.plot3D(t.z, t.x, t.y, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    t = arrange_track(t)
    ax2.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('x (mm)')
ax2.set_zlabel('y (mm)')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()


# ## DBSCAN Clustering Setting 4
# ## scaling and normalization + dbscan (no coordinate transformation)

# In[ ]:


X = hits[['x', 'y', 'z']]
scaler = MaxAbsScaler().fit(X)
X = scaler.transform(X)
normalizer = Normalizer(norm='l2').fit(X)
X = normalizer.transform(X)

eps = 0.0022
min_samp = 3
db = DBSCAN(eps=eps, min_samples=min_samp, metric='euclidean').fit(X)
labels = db.labels_

clustering = pd.DataFrame()
clustering['hit_id'] = truth['hit_id']
clustering['track_id'] = labels

score = score_event(truth, clustering)
print('track-ml custom metric score:', round(score, 4))

labels_true = truth['particle_id']
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print('\nOTHER CLUSTERING RESULTS:')
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


# ## Clustered Hits (Predicted Tracks) vs True Tracks

# In[ ]:


tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x', 'y', 'z']]
    if cluster == -1:
        ax.plot3D(t.z, t.x, t.y, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    t = arrange_track(t)
    ax2.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('x (mm)')
ax2.set_zlabel('y (mm)')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()


# In this setting, we observe shorter predicted tracks than when usin a coordinate transformation. Also, the predicted tracks are still straight like in Setting 1.

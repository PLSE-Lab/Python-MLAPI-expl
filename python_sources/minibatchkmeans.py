#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
import time


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


from sklearn.cluster import MiniBatchKMeans, KMeans


# In[ ]:


from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs


# In[ ]:


train1 = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=10000)


# In[ ]:


train = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=1000000)


# In[ ]:


test = pd.read_csv("../input/test.csv", parse_dates=['date_time'], nrows=1000000)


# In[ ]:


train_bookings = train[train['is_booking'] == 1].drop('is_booking', axis=1)
train_clicks = train[train['is_booking'] == 0].drop('is_booking', axis=1)


# In[ ]:


train_bookingsz = train_bookings.drop(['date_time','srch_ci','srch_co'], axis=1)
train_bookingszna = train_bookingsz.fillna(0)


# In[ ]:


c = train_bookingszna.ix[4000:5000]


# In[ ]:


test_bookingsz = test.drop(['date_time','srch_ci','srch_co'], axis=1)
test_bookingszna = test_bookingsz.fillna(0)


# In[ ]:


b1 = train_bookingszna.hotel_cluster.unique()


# In[ ]:


n_clusters = 100
k_means = KMeans(init='k-means++', n_clusters=100, n_init=10)
t0 = time.time()
k_means.fit(train_bookingszna)
t_batch = time.time() - t0
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)


# In[ ]:


k_means.fit(train_bookingszna)


# In[ ]:


test_bookingszna['diff_A_B'] = test_bookingszna['srch_adults_cnt'] - test_bookingszna['srch_children_cnt']


# In[ ]:


a = k_means.predict(test_bookingszna)


# In[ ]:


a[0:80]


# In[ ]:


np.unique(k_means_labels)


# In[ ]:


np.unique(k_means.cluster_centers_)


# In[ ]:


# Plot result

fig = plt.figure(figsize=(50, 20))

#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
# MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
# closest one.


# KMeans
ax = fig.add_subplot(1, 1, 1)
for k in range(n_clusters):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    #ax.plot(train_bookingszna[my_members, 0], train_bookingszna[my_members, 1], 'w', marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
    t_batch, k_means.inertia_))
plt.show()


# In[ ]:





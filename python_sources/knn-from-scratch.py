#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# Let's see what's really inside

# In[ ]:


train_data.head()


# In[ ]:


train_data.describe()


# It appears there are 2 useless columns: Soil_Type7 & Soil_Type15 (std = 0) so we can drop them from both train and test data.

# In[ ]:


train_labels = train_data.Cover_Type.values
test_id = test_data.Id.values

train_data.drop(['Soil_Type7', 'Soil_Type15', 'Id', 'Cover_Type'], axis=1, inplace=True)
test_data.drop(['Soil_Type7', 'Soil_Type15', 'Id'], axis=1, inplace=True)


# In[ ]:


print(train_data.shape, test_data.shape)


# Seems good! Now we need to create a distance matrix but first let's see how much RAM is needed for this given that the array needs to be casted as float64 at the beginning..

# Let's try to find the best K for this case. We will use the leave-one-out approach, so we will try to classify every point in the training data using all other points over a range of K values and determine the best one.
# We need to create a distance matrix for the train_data.

# We will run a quick benchmark to see which function should we use.

# In[ ]:


get_ipython().run_line_magic('timeit', "pairwise_distances(train_data[:150], metric = 'euclidean')")
get_ipython().run_line_magic('timeit', "distance.cdist(train_data[:150], train_data[:150], 'euclidean')")


# pairwise_distances is faster.

# In[ ]:


min_max_scaler = MinMaxScaler() # If you did not use the scaler, you will get higher accuracy
train_data = min_max_scaler.fit_transform(train_data)
test_data = min_max_scaler.fit_transform(test_data)

distance_matrix = pairwise_distances(train_data, metric = 'euclidean')
print(distance_matrix.shape)


# Using the argsort, we can find the indexes of the sorted distance matrix (indexes of nearest neighbours)

# In[ ]:


sorted_distance_index = np.argsort(distance_matrix, axis=1).astype(np.uint16)
print(sorted_distance_index)


# Note that the nearest neighbour (first column) is actually the point itself so we will have to consider filtering the first column from our calculations.
# We have to know the labels of these indexes, luckily, we can use numpy direct indexing for this one.

# In[ ]:


sorted_distance_labels = train_labels[sorted_distance_index].astype(np.uint8)
print(sorted_distance_labels)


# We need to decide the maximum k we would like to try for this case. I think 100 is more than enough.
# We will build an array where the rows are the data points (indexes) and the columns are the Ks, we would then see the classification of every point in the training set over all the Ks using np.bincount and np.argmax to find the most common element in the nearest neighbors.

# In[ ]:


max_k = 100
k_matrix = np.empty((len(sorted_distance_labels), 0), dtype=np.uint8)
for k in range (1, max_k+1):
    k_along_rows = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=sorted_distance_labels[:, 1:k+1]).reshape(len(sorted_distance_labels), -1)
    k_matrix = np.hstack((k_matrix, k_along_rows))
print(k_matrix)


# Perfect! Now we have the prediction for every point over all the Ks from 1 to 100. Let's find the best one. If the prediction was correct we replace the value by 1, otherwise 0.

# In[ ]:


k_truth_table = np.where(k_matrix == train_labels[:, None], 1, 0)
print(k_truth_table)
print(k_truth_table.shape)


# In[ ]:


accuracy_per_k = np.sum(k_truth_table, axis=0)/len(k_truth_table)
best_accuracy = np.amax(accuracy_per_k)
best_k = np.argmax(accuracy_per_k) + 1 # real k = index + 1
print('Best K: {0}, Best Accuracy: {1:4.2f}%'.format(best_k, best_accuracy*100))
plt.plot(range(1, max_k+1), accuracy_per_k)
plt.title('Classification accuracy vs Choice of K')
plt.xlabel('K')
plt.ylabel('Classification Accuracy')
plt.show()


# We will use K=1 which should yield an accuracy around 86.8%, however the size of the test_data is too big we need to claculate the size of the distance matrix first to avoid running out of RAM. Note that the array would be cast as float64 before we can convert it.

# In[ ]:


print("RAM needed for the distance matrix = {:.2f} GB".format(len(train_data)*len(test_data) * 64 / (8 * 1024 * 1024 * 1024)))


# No way we would be able to to store it in 1 numpy array... so we will loop over chunks of the test_data, classify them and reuse the same variable names..

# In[ ]:


# Those variables are no longer needed, Free up some RAM instead
del k_truth_table
del k_matrix
del sorted_distance_labels
del sorted_distance_index
del distance_matrix


# In[ ]:


# ALERT: This code takes some time, it took 8 minutes on a powerful PC but with relatively low RAM usage (around 6.8G)
def classify(unknown, dataset, labels, k):
    classify_distance_matrix = pairwise_distances(unknown, dataset, metric='euclidean')
    nearest_images = np.argsort(classify_distance_matrix)[:, :k]
    nearest_images_labels = labels[nearest_images]
    classification = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=nearest_images_labels[:, :k])
    return classification.astype(np.uint8).reshape(-1, 1)

predict = np.empty((0, 1), dtype=np.uint8)
chunks = 15
last_chunk_index = 0
for i in range(1, chunks+1):
    new_chunk_index = int(i * len(test_data) / chunks)
    predict = np.concatenate((predict, classify(test_data[last_chunk_index : new_chunk_index], train_data, train_labels, best_k)))
    last_chunk_index = new_chunk_index
    print("Progress = {:.2f}%".format(i * 100 / chunks))


# In[ ]:


submission = pd.DataFrame({"Id": test_id, "Cover_Type": predict.ravel()})
submission.to_csv('submission.csv', index=False)


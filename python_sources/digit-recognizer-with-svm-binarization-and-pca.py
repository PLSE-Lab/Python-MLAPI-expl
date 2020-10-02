#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from scipy.sparse import lil_matrix
from sklearn import svm
from sklearn.decomposition import PCA
import sklearn.discriminant_analysis


# # First Attempt

# In[ ]:


data = pd.read_csv("../input/train.csv")
data.shape


# Seeing the images:

# In[ ]:


i = 6
image = np.array(data.iloc[i,1:])
image = image.reshape([28, 28])
plt.imshow(image, cmap='gray')
plt.title(data.iloc[i,0])


# In[ ]:


train_n = 5000
train_labels = np.array(data.iloc[:train_n,0])
train = lil_matrix(np.array(data.iloc[:train_n, 1:]), dtype = 'int32')
train_labels.shape, train.shape


# In[ ]:


test_n = 10000
test_labels = np.array(data.iloc[train_n : train_n + test_n, 0])
test = lil_matrix(np.array(data.iloc[train_n : train_n + test_n, 1:]), dtype = 'int32')
test_labels.shape, test.shape


# In[ ]:


clf = svm.SVC(gamma='scale')
clf.fit(train, train_labels)


# In[ ]:


clf.score(test, test_labels)


# The first score was 15.3%

# # Now let's convert the pixels to binary

# In[ ]:


data_simple = (np.array(data)[:,1:] >= 120).astype(int)
data_simple.shape


# In[ ]:


i = 3
image = data_simple[i,:]
image = image.reshape([28, 28])
plt.imshow(image, cmap='gray')
plt.title(data.iloc[i,0])


# In[ ]:


train_n = 32000
train_labels = np.array(data.iloc[:train_n,0])
train = data_simple[:train_n]
train_labels.shape, train.shape


# In[ ]:


test_n = 10000
test_labels = np.array(data.iloc[train_n : train_n + test_n, 0])
test = data_simple[train_n : train_n + test_n]
test_labels.shape, test.shape


# In[ ]:


clf2 = svm.SVC(gamma='scale')
clf2.fit(train, train_labels)


# In[ ]:


clf2.score(test, test_labels)


# With a pixel treshold of 120 the score jumped to 92.28%!!!
# 
# With threshold in 0 the score is 92.26%
# 
# With threshold in 200 it decreases to 90%

# # Now let's try by applying pca previously

# In[ ]:


pca = PCA(0.65)
pca.fit(train)
pca.n_components_


# In[ ]:


train_pca = pca.transform(train)
test_pca = pca.transform(test)
train_pca.shape, test_pca.shape


# In[ ]:


clf3 = svm.SVC(gamma='scale')
clf3.fit(train_pca, train_labels)


# In[ ]:


clf3.score(test_pca, test_labels)


# With pca at 95% we obtain a score of 93.66%!!!
# 
# At 90% confidence it goes up to 94.09%
# 
# At 80% confidence it goes up to 94.99%!!!
# 
# At 60% confidence we get a score of 95.41%!!!!!
# 
# It starts going down after that...
# 
# With 65% confidence and with the 32000 entries on the training set the accuracy goes up to **97.86%**

# # Time to prepare a submission

# In[ ]:


train = data_simple
train_labels = np.array(data.iloc[:,0])
test = (pd.read_csv('../input/test.csv') >= 120).astype(int)
train.shape, train_labels.shape, test.shape


# In[ ]:


pca = PCA(0.65)
pca.fit(train)
pca.n_components_


# In[ ]:


train_pca = pca.transform(train)
test_pca = pca.transform(test)
train_pca.shape, test_pca.shape


# In[ ]:


clf4 = svm.SVC(gamma='scale')
clf4.fit(train_pca, train_labels)


# In[ ]:


pred = clf4.predict(test_pca)
pred.shape


# In[ ]:


r = np.array([range(1,28001), pred], dtype = int).transpose()
r = pd.DataFrame(r)
r.columns = ["ImageId", "Label"]
r


# In[ ]:


r.to_csv("submit.csv", index = False)


# In[ ]:


help(r.to_csv)


# In[ ]:





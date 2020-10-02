#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../input/Iris.csv')
df.head()


# In[ ]:


types = df.Species.unique()
lookup = dict(zip(types, range(len(types))))
df['SpeciesLabel'] = df['Species'].replace(lookup)
df.head()


# In[ ]:


X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['SpeciesLabel']
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[ ]:


from matplotlib import cm

cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)


# In[ ]:


import matplotlib.pyplot as plt

k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test) * 100)

plt.figure()
plt.xlabel('k')
plt.ylabel('% accuracy')
plt.scatter([k_range], scores)
plt.xticks([0,5,10,15,20]);


# In[ ]:





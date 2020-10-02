#!/usr/bin/env python
# coding: utf-8

# ## Multi-label classification with neural networks
# 
# In some classification tasks we use classes that are not mutually exclusive. In other words, given two possible classes, it is possible for both clases to be assigned, and it is possible for neither class to be assigned. Thus this problem is not strictly a **multi-classification task**. A multi-classification task assumes that exactly one label is assigned out of all possible labels; that is, either the record belongs to the `{0, 1}` class, or it belongs to the `{1, 0}` class. This problem on the other hand also allows for `{1, 1}` and `{0, 0}` outputs (both evidenced in the short snippet above). That makes this problem what is known as a **multi-label task**.
# 
# I dicussed [multi-class and multi-label schemes](https://www.kaggle.com/residentmario/notes-on-multiclass-and-multitask-schemes) in a previous, `sklearn` focused notebook. This notebook is a breif continuation on this subject specific to neural networks. It was written on something that tripped me up while I was working on a bigger model.
# 
# To start with, here's the example data we'll be working with:

# In[ ]:


import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, n_features=5, n_informative=5,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.33, 0.33, 0.33],
                           class_sep=2, random_state=0)

import matplotlib.pyplot as plt
colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y]
kwarg_params = {'linewidth': 1, 'edgecolor': 'black'}
plt.scatter(X[:, 0], X[:, 1], c=colors, **kwarg_params)


# The `X` features consist of a cluster with three easily distinguishable class clusters. The output `y` consists of two columns: one which is true if the record is assigned to class 1, and one which is true if the record is assigned to *either* class 1 or 2. This is a multi-label classification task because it's possible to have any of `{0, 1}, {1, 0}, {0, 0}, {1, 1}` appear in the output.
# 
# We'll start with the following model, which works:

# ### Three non-exclusive output classes, sigmoid activation and binary cross entropy loss

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.metrics import binary_accuracy
from keras.utils import to_categorical

X, y = make_classification(n_samples=10000, n_features=5, n_informative=5,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.33, 0.33, 0.33],
                           class_sep=2, random_state=0)
y = to_categorical(y)
y = np.vstack((y[:, 0], y[:, :2].sum(axis=1))).T

clf = Sequential()
clf.add(Dense(5, activation='relu', input_dim=5))
clf.add(Dense(2, activation='sigmoid'))
clf.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=[binary_accuracy])
clf.fit(X, y, epochs=20, batch_size=100, verbose=0)


# In[ ]:


clf.predict(X)


# In[ ]:


y


# Now here's a model that doesn't work:
# 
# ### Three non-exclusive output classes, softmax activation and categorical cross entropy loss

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.metrics import binary_accuracy
from keras.utils import to_categorical

X, y = make_classification(n_samples=10000, n_features=5, n_informative=5,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.33, 0.33, 0.33],
                           class_sep=2, random_state=0)
y = to_categorical(y)
y = np.vstack((y[:, 0], y[:, :2].sum(axis=1))).T

clf = Sequential()
clf.add(Dense(5, activation='relu', input_dim=5))
clf.add(Dense(2, activation='softmax'))
clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=[binary_accuracy])
clf.fit(X, y, epochs=20, batch_size=100, verbose=0)


# In[ ]:


clf.predict(X)


# In[ ]:


y


# You can see here that this did not work.
# 
# Neural networks intrinsically support multi-label tasks in their design. Each possible class or label is just another node in the output layer of the net. However, not all neural network output layer activation functions work with multi-label tasks.
# 
# An example of an activation function which does not work with multi-label tasks is the `softmax` activation function. A softmax normalizes the value it assigns to a particular class against the value assigned to each *other* class also being predicted. In other words, if we have a three-class problem, the softmax output for class 1 will take into account the outputs for classes 2 and 3. In order to do this effectively, softmax makes the assumption that the set of classes being considered are mutually exclusive.
# 
# By comparing`clf.predict(X)` to `y` in the code cells above, you can see the effect this has on our outputs. Records with mutually exclusive classes are predicted correctly. Records where multiple classes are true (e.g. `{1, 1}`) split the confidence of the prediction between them. And in records where both classes are not true (`{0, 0}`), the classifier will just go ahead and pick a winner anyway!
# 
# With a little bit of post-prediction manipulation we could deal with the `{1, 1}` issue, but the `{0, 0}` issue is unfixable.
# 
# The lesson here is to be very careful about whether your problem is multi-class or multi-label.
# 
# The assumption of mutual exclusivity is an important component of why `softmax` is so effective for classification tasks, as it is essentially an additional piece of information the network can use during the optimization step, one which other popular activation functions do not get.
# 
# The `sigmoid` activation function, which we used in the immediately previous model, doesn't have this problem. That's because the `sigmoid` activation function evaluates each class separately. It has no problem assigning low values to records with double falses, and high values to records with double trues.
# 
# So in summary: do not use `softmax` with multi-label problems!

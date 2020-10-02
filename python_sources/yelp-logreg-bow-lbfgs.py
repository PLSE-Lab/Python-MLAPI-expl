#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time

import numpy as np
import pandas as pd
import scipy
from joblib import dump, load
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


X_train = scipy.sparse.load_npz('../input/yelp-data-preprocessing/X_train.npz')
X_test = scipy.sparse.load_npz('../input/yelp-data-preprocessing/X_test.npz')
y_train = np.load('../input/yelp-data-preprocessing/y_train.npy')
y_test = np.load('../input/yelp-data-preprocessing/y_test.npy')


# ## Training the Model

# ### L-BFGS Solver

# In[ ]:


time_start = time.time()

clf = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    n_jobs=-1
)
clf.fit(X_train, y_train)
dump(clf, 'model.joblib')

print(f'Classifier fit in : {time.time() - time_start:.3f} sec')


# In[ ]:


y_pred = clf.predict(X_test)
score = metrics.f1_score(y_true=y_test, y_pred=y_pred, average='macro')

print("Mean Accuracy:", clf.score(X_test, y_test))
print(f"F1 Score: {score:.4f}")
print(metrics.classification_report(y_true=y_test, y_pred=y_pred))


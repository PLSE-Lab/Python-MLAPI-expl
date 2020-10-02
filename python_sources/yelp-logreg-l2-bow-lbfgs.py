#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time

import numpy as np
import pandas as pd
import scipy
from joblib import dump
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[ ]:


print(os.listdir("../input/"))


# ## Load prepreocessed data

# In[ ]:


X_train = scipy.sparse.load_npz('../input/yelp-data-preprocessing/X_train.npz')
X_test = scipy.sparse.load_npz('../input/yelp-data-preprocessing/X_test.npz')
y_train = np.load('../input/yelp-data-preprocessing/y_train.npy')
y_test = np.load('../input/yelp-data-preprocessing/y_test.npy')


# In[ ]:


time_start = time.time()

clf = LogisticRegressionCV(
    cv=4,
    solver='lbfgs',
    random_state=551,
    penalty='l2',
    multi_class='multinomial',
    verbose=2,
    n_jobs=-1
)
clf.fit(X_train, y_train)

dump(clf, 'logreg-l2-lbfgs.joblib')
print(f'Classifier fit in : {time.time() - time_start:.3f} sec')


# In[ ]:


y_pred = clf.predict(X_test)
score = metrics.f1_score(y_true=y_test, y_pred=y_pred, average='macro')

print("Mean Accuracy:", clf.score(X_test, y_test))
print(f"F1 Score: {score:.4f}")
print(metrics.classification_report(y_true=y_test, y_pred=y_pred))


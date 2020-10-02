#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# other imports
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score


# In[ ]:


# set plot size
pylab.rcParams['figure.figsize'] = 16, 12
plt.style.use('ggplot')


# In[ ]:


# read data
df = pd.read_csv('../input/creditcard.csv')


# In[ ]:


# Plot histogram of classes
class_count = pd.value_counts(df['Class'], sort = True).sort_index()
class_count.plot(kind = 'bar')
plt.title("Histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[ ]:


# Convert data to list and split into test-train
X = df.ix[:, df.columns != 'Class']
y = df.ix[:, df.columns == 'Class']
X=X.values.T.tolist()
X=np.asarray(X).transpose()
y=y.values.T.tolist()
y=np.asarray(y).transpose()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


# Over-sample minority samples using SMOTE
sm = SMOTE(ratio=0.5, random_state=42) 
X_tr_res, y_tr_res = sm.fit_sample(X_train, y_train.ravel())


# In[ ]:


# Plot new distribution
plt.hist(y_tr_res)
plt.ylabel('Probability');
plt.title("Histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


# Train model and test on separate test set
rnd = RandomForestClassifier(n_estimators=512)
rnd.fit(X_tr_res, y_tr_res)
y_test_preds=rnd.predict(X_test)
print(cohen_kappa_score(y_test, y_test_preds))


# In[ ]:





# In[ ]:





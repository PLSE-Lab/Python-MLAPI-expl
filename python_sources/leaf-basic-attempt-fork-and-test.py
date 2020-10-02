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


# ## Get training data into data frame

# In[ ]:


df_train = pd.read_csv('../input/train.csv', index_col = 'id')
df_train.head()


# ## Create X

# In[ ]:


X = df_train.drop('species', axis=1)
y = df_train.species
df_train.shape, X.shape, y.shape


# ## Encode labels into y

# In[ ]:


import sklearn.preprocessing as skpp

# classes used to order the label_binarize call below, 
# and at the bottom to give column names to submission data
classes = df_train.species.unique()
classes.sort()

y = skpp.label_binarize(y, classes = classes)
y[0,]


# ## Pipeline

# In[ ]:


from sklearn.pipeline import Pipeline
import sklearn.preprocessing as skpp
import sklearn.decomposition as skdc
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

norm = skpp.StandardScaler()
pca = skdc.PCA(n_components=25) 
svm = OneVsRestClassifier(SVC(kernel='sigmoid', probability=True))
pipe = Pipeline(steps = [
        ('standardizer', norm), 
        ('decom', pca), 
        ('alg', svm)])


# In[ ]:


import sklearn.model_selection as skms
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

m_comp = [25, 50, 75, 100]
params = [
    {
        'decom__n_components': m_comp,
        'alg__estimator__C': list(np.arange(.4, .6, .1)),
        'alg__estimator__kernel': ['linear']
    },
    {
        'decom__n_components': m_comp,
        'alg': [RandomForestClassifier()],
        'alg__n_estimators': [8,10,12]
    },
    {
        'decom__n_components': m_comp,
        'alg': [OneVsRestClassifier(GaussianNB())]        
    }
]

grid = skms.GridSearchCV(pipe, params)


# ## Cross-validated, stratified, and shuffled

# In[ ]:


import sklearn.model_selection as skms

strat_cv_shuffler = skms.StratifiedShuffleSplit(n_splits = 6, train_size=0.8)


# ## Run on training data

# In[ ]:


#scores = skms.cross_val_score(grid, X, y, cv=strat_cv_shuffler)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ## Train

# In[ ]:


grid.fit(X, y)
grid.best_params_


# ## Create submission file

# In[ ]:


df_test = pd.read_csv('../input/test.csv', index_col = 'id')

pred = grid.predict_proba(df_test)
df_sub = pd.DataFrame(pred, index = df_test.index, columns = classes)
df_sub.to_csv('submission.csv')
df_sub.head()


# In[ ]:


#test
pred


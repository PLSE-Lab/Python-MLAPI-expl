#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from mlxtend.preprocessing import standardize
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from mlxtend.feature_selection import ColumnSelector

import warnings
warnings.filterwarnings('ignore')


# In[47]:


train =  pd.read_csv("../input/train.csv")
train.head()


# In[48]:


y = train.target 
X = train.iloc[:, 2:]


# In[49]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[50]:


my_imputer = Imputer()
X = my_imputer.fit_transform(X)


# In[82]:


# Initializing models
#clf1 = KNeighborsClassifier(n_neighbors=5)
#clf2 = RandomForestClassifier(n_estimators=100, random_state=1)
#clf3 = GaussianNB()
#clf4 = ExtraTreesClassifier(n_estimators=100)
#clf5 = SVC()
#clf6 = RidgeClassifier()
#clf7 = AdaBoostClassifier(n_estimators=1000, learning_rate=0.7)
#clf8 = GradientBoostingClassifier(n_estimators=1000, min_samples_split=5)
#clf9 = MLPClassifier(random_state=1, alpha=1e-1)
#clf10 = LogisticRegression(C=0.1)

#rf = RandomForestClassifier(n_estimators=20, random_state=1)

#sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10],
#                            verbose=1,
#                            meta_classifier=rf)

#print('3-fold cross validation:\n')

#for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10, sclf], 
#                      ['KNN', 
#                       'Random Forest', 
#                       'Naive Bayes',
#                       'Extra Trees',
#                       'Support Vector',
##                       'Ridge',
#                       'AdaBoost',
#                       'GradientBoost',
#                       'Neural Net',
#                       'Logistic',
#                       'Ensemble']):

#    scores = model_selection.cross_val_score(clf, X, y, 
#                                              cv=3, scoring='accuracy')
#    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
#          % (scores.mean(), scores.std(), label))#

#params = {'kneighborsclassifier__n_neighbors': [1, 10],
#          'randomforestclassifier__n_estimators': [10, 20],
#          'extratreesclassifier__n_estimators': [10, 20],
#          'adaboostclassifier__n_estimators':[1000, 2000],
#          'gradientboostingclassifier__n_estimators':[50,100],
#          'logisticregression__C':[0.1, 10.0],
#          'meta-randomforestclassifier__n_estimators':[10, 20]}
#grid = GridSearchCV(estimator=sclf, 
#                    param_grid=params, 
#                    cv=5,
#                    refit=True,
#                    n_jobs=-1)
#grid.fit(X, y)
#print('Best parameters: %s' % grid.best_params_)
#print('Accuracy: %.2f' % grid.best_score_)
#Best parameters: {'adaboostclassifier__n_estimators': 1000, 'extratreesclassifier__n_estimators': 10, 
#'gradientboostingclassifier__n_estimators': 50, 'kneighborsclassifier__n_neighbors': 1, 'logisticregression__C': 0.1, 
#'meta-randomforestclassifier__n_estimators': 20, 'randomforestclassifier__n_estimators': 10}
#Accuracy: 0.73


# In[76]:


# Initializing models
clf2 = RandomForestClassifier(n_estimators=100, random_state=1)
clf3 = GaussianNB()
clf4 = ExtraTreesClassifier(n_estimators=100)
clf7 = AdaBoostClassifier(n_estimators=1000, learning_rate=0.7)
clf8 = GradientBoostingClassifier(n_estimators=1000, min_samples_split=5)
clf9 = GradientBoostingClassifier(n_estimators=2000, max_depth=4)
clf10 = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, subsample=0.5)


rf = RandomForestClassifier(n_estimators=20, random_state=1)

sclf = StackingClassifier(classifiers=[clf2, clf3, clf4, clf7, clf8, clf9, clf10],
                            verbose=1,
                            meta_classifier=rf)
sclf.fit(X, y)


# In[77]:


for clf, label in zip([clf2, clf3, clf4, clf7, clf8, clf9, clf10, sclf], 
                      ['Random Forest', 
                       'Naive Bayes',
                       'Extra Trees',
                       'AdaBoost',
                       'GradientBoost1',
                       'GradientBoost2',
                       'GradientBoost3',
                       'Ensemble']):

    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))


# In[78]:


test = pd.read_csv('../input/test.csv')
test = scaler.transform(test.iloc[:, 1:])
test = my_imputer.transform(test)
preds = sclf.predict_proba(test)


# In[79]:


def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


# In[80]:


preds = softmax(preds, axis=1)[:, 1]


# In[83]:


sub = pd.read_csv("../input/sample_submission.csv")


# In[84]:


sub.target = preds
sub.to_csv('sub.csv', index=False)


# In[85]:


sub.head()


# In[ ]:





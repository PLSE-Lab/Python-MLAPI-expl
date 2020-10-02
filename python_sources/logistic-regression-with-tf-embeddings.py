#!/usr/bin/env python
# coding: utf-8

# One of the most popular and very successful algorithsm and notebooks for Toxic Comments competition was a Logistic Regression based on word and character features. However, these feature speaces are really large and sparse - 50,000+ sparse features! Since the competition ended, a few good sentence embedding schemes have popped up. These allow for an automattic embedding of whole sentences into realtively low-lever vector spaces (a few thousand dimensions at most). I wanted to chekc how well these embeddings could be used for the classification of Toxic comments. It took me a few months of wrestling with Kaggle's onlline environment to get at least one of them working (TF Embeddings), but thus far the results are not too encouraging. The logistic regression, as run below, gives LB score of 0.966x, as opposed to almost 0.98 with TF-IDF. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# In[ ]:


train_features = np.load('../input/tf-embedding-files-joiner/train.npy')
test_features = np.load('../input/tf-embedding-files-joiner/test.npy')
target = np.load('../input/tf-embedding-files-joiner/target.npy')
sample_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')


# In[ ]:


target = pd.DataFrame(columns=class_names, data=target)


# In[ ]:


target.head()


# In[ ]:


scores = []
submission = pd.DataFrame.from_dict({'id': sample_submission['id']})
for class_name in class_names:
    train_target = target[class_name]
    classifier = LogisticRegression(C=5, solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]


# In[ ]:


print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('submission.csv', index=False)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn


# In[ ]:


# from https://www.kaggle.com/c/google-quest-challenge/discussion/126778

from scipy.stats import spearmanr


def mean_spearmanr_correlation_score(y, y_pred):
    spearsum = 0
    cnt = 0 
    for col in range(y_pred.shape[1]):
        v = spearmanr(y_pred[:,col], y[:,col]).correlation
        if np.isnan(v):
            continue
        spearsum += v
        cnt += 1
    res = spearsum / cnt
    return res


# In[ ]:


sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")
train = pd.read_csv("../input/google-quest-challenge/train.csv")


# In[ ]:


train.columns


# In[ ]:


target_columns = [
    'question_asker_intent_understanding',
    'question_body_critical',
    'question_conversational',
    'question_expect_short_answer',
    'question_fact_seeking',
    'question_has_commonly_accepted_answer',
    'question_interestingness_others',
    'question_interestingness_self',
    'question_multi_intent',
    'question_not_really_a_question',
    'question_opinion_seeking',
    'question_type_choice',
    'question_type_compare',
    'question_type_consequence',
    'question_type_definition',
    'question_type_entity',
    'question_type_instructions',
    'question_type_procedure',
    'question_type_reason_explanation',
    'question_type_spelling',
    'question_well_written',
    'answer_helpful',
    'answer_level_of_information',
    'answer_plausible',
    'answer_relevance',
    'answer_satisfaction',
    'answer_type_instructions',
    'answer_type_procedure',
    'answer_type_reason_explanation',
    'answer_well_written'
]


# In[ ]:


len(target_columns)


# In[ ]:


stword=stopwords.words('english')
mycorpus=[]
ps=PorterStemmer()
#len(stword)
for i in range(0,len(train)):
    try:
        q_body= re.sub('[^a-zA-Z]',' ', train['question_body'][i])+re.sub('[^a-zA-Z]',' ', train['answer'][i])
    except KeyError:
        print("KeyError:",i)
    except TypeError:
        print("TypeError",i)
    q_body=q_body.lower()
    q_body=q_body.split()
    q_body=[ps.stem(word) for word in q_body if not word in set(stword)]
    q_body=' '.join(q_body)
    mycorpus.append(q_body)


# In[ ]:


for i in target_columns:
    try:
        train[i]=pd.to_numeric(train[i],errors='coerce')
    except ZeroDivisionError:
        print('Error')


# In[ ]:


#for i in target_columns:
    #print(train[train[i].isnull()])


# In[ ]:


#test
stword=stopwords.words('english')
mycorpus_test=[]
ps=PorterStemmer()
#len(stword)
for i in range(0,len(test)):
    try:
        q_body= re.sub('[^a-zA-Z]',' ', test['question_body'][i])+re.sub('[^a-zA-Z]',' ', test['answer'][i])
    except KeyError:
        print("KeyError:",i)
    except TypeError:
        print("TypeError",i)
    q_body=q_body.lower()
    q_body=q_body.split()
    q_body=[ps.stem(word) for word in q_body if not word in set(stword)]
    q_body=' '.join(q_body)
    mycorpus_test.append(q_body)


# In[ ]:


q_body= re.sub('[^a-zA-Z]',' ', train['question_body'][0])
q_body=q_body.lower()
q_body=q_body.split()
print(q_body)
print(mycorpus[0])


# In[ ]:


cv = CountVectorizer(max_features=14250,binary=True)
X = cv.fit_transform(mycorpus).toarray()
y_train = train[target_columns].copy()
y_train=y_train.values

#scaler=sklearn.preprocessing.StandardScaler()
#scaler.fit(X)
#X=scaler.transform(X)

#print(X[6])
a=cv.get_feature_names()
#a = pd.DataFrame(X)
#a.head()
print(a)


# In[ ]:


len(a)


# In[ ]:


print(X[0])


# In[ ]:


#cv_c = CountVectorizer(max_features =100)
X_test=cv.transform(mycorpus_test).toarray()
scaler=sklearn.preprocessing.StandardScaler()
scaler.fit(X_test)
X_test=scaler.transform(X_test)


# In[ ]:


from sklearn.linear_model import RidgeCV
ridge_grid = RidgeCV(alphas=np.linspace(0.1, 20000.0, num=100)).fit(X, y_train)
best_Alpha = ridge_grid.alpha_
best_Alpha


# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import math

n_splits = 5

scores = []
all_scores=[]

cv_k = KFold(n_splits=n_splits, random_state=42)
trained_estimators = []

for train_idx, valid_idx in cv_k.split(X, y_train):
    
    x_train_train = X[train_idx]
    y_train_train = y_train[train_idx]
    x_train_valid = X[valid_idx]
    y_train_valid = y_train[valid_idx]
    
    estimator = Ridge(alpha=best_Alpha, random_state=42)
    #estimator = LogisticRegression()
    estimator.fit(x_train_train, y_train_train)
    trained_estimators.append(estimator)
    
    oof_part = estimator.predict(x_train_valid)
    score = mean_spearmanr_correlation_score(y_train_valid, oof_part)
    #score=estimator.score(x_train_valid, y_train_valid)
    print('Score:', score)
    scores.append(score)


print('Mean score:', np.mean(scores))
all_scores.extend(scores)


y_pred = []
for estimator in trained_estimators:
    y_pred.append(estimator.predict(X))


# In[ ]:


y_pred = []
for estimator in trained_estimators:
    y_pred.append(estimator.predict(X_test))


# In[ ]:


#print(X_test[1][3000])


# In[ ]:


#len(trained_estimators)
#y_pred[9][475]


# In[ ]:


sum_scores = sum(all_scores)
weights = [x / sum_scores for x in all_scores]


# In[ ]:


from scipy.stats import rankdata


def blend_by_ranking(data, weights):
    out = np.zeros(data.shape[0])
    for idx,column in enumerate(data.columns):
        out += weights[idx] * rankdata(data[column].values)
    out /= np.max(out)
    return out


# In[ ]:


submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv", index_col='qa_id')

out = pd.DataFrame(index=submission.index)
for column_idx,column in enumerate(target_columns):
    
    # collect all predictions for one column
    column_data = pd.DataFrame(index=submission.index)
    for prediction_idx,prediction in enumerate(y_pred):
        column_data[str(prediction_idx)] = prediction[:, column_idx]
    
    out[column] = blend_by_ranking(column_data, weights)


# In[ ]:


out.head()


# In[ ]:


out.to_csv("submission.csv")


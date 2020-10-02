#!/usr/bin/env python
# coding: utf-8

# #### Import Libraries

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
import os


# #### Import Data

# In[ ]:


print(os.listdir("../input"))


# In[ ]:


qn_df = pd.read_csv('../input/Question_Classification_Dataset.csv')
qn_df = qn_df.iloc[:,1:]
qn_df.head()


# ### Category0 Analysis

# In[ ]:


qn_df1 = qn_df[['Questions', 'Category0']]
qn_df1.head()


# #### Vectorization

# In[ ]:


qn_df1['Category Vectors'] = pd.factorize(qn_df1['Category0'])[0]
qn_df1.head()


# In[ ]:


vect = TfidfVectorizer(ngram_range = (1,2)).fit(qn_df1['Questions'])


# #### Train Test Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(qn_df1['Questions'], qn_df1['Category Vectors'], test_size=0.2, random_state=0)


# In[ ]:


train_vector = vect.transform(X_train)


# In[ ]:


test_vector = vect.transform(X_test)


# #### SVM

# In[ ]:


model1 = SVC(kernel='linear', probability = True)


# In[ ]:


model1.fit(train_vector, y_train)


# In[ ]:


pred1 = model1.predict(test_vector)


# In[ ]:


accuracy_score(pred1, y_test)


# #### Apply Threshold

# In[ ]:


max_prob, max_prob_args = [],[]

prob = model1.predict_proba(test_vector)
for i in range(len(prob)):
    max_prob.append(prob[i].max())
    if prob[i].max() > 0.8:
        max_prob_args.append(prob[i].argmax())
    else:
        max_prob_args.append(-1)


# In[ ]:


a = pd.DataFrame(X_test)
a['pred'] = max_prob_args
a['actual'] = y_test
a['max_prob'] = max_prob


# In[ ]:


b = a[a['pred'] != -1]   ### 809 out of 1091 datapoints


# In[ ]:


accuracy_score(b['pred'], b['actual'])


# ### Category2 Analysis

# In[ ]:


qn_df2 = qn_df[['Questions', 'Category2']]
qn_df2.head()


# #### Vectorization

# In[ ]:


qn_df2['Category Vectors'] = pd.factorize(qn_df2['Category2'])[0]
qn_df2.head()


# #### Train Test Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(qn_df2['Questions'], qn_df2['Category Vectors'], test_size=0.2, random_state=0)


# In[ ]:


train_vector = vect.transform(X_train)


# In[ ]:


test_vector = vect.transform(X_test)


# #### SVM

# In[ ]:


model2 = SVC(kernel='linear', probability = True)


# In[ ]:


model2.fit(train_vector, y_train)


# In[ ]:


pred2 = model2.predict(test_vector)


# In[ ]:


accuracy_score(pred2, y_test)


# #### Apply Threshold

# In[ ]:


max_prob, max_prob_args = [],[]

prob = model2.predict_proba(test_vector)
for i in range(len(prob)):
    max_prob.append(prob[i].max())
    if prob[i].max() > 0.8:
        max_prob_args.append(prob[i].argmax())
    else:
        max_prob_args.append(-1)


# In[ ]:


a = pd.DataFrame(X_test)
a['pred'] = max_prob_args
a['actual'] = y_test
a['max_prob'] = max_prob


# In[ ]:


b = a[a['pred'] != -1]   ### 521 out of 1091 datapoints


# In[ ]:


accuracy_score(b['pred'], b['actual'])


# ### Create Reference Dictionary

# In[ ]:


dict_cat0 = {}

for val in qn_df1['Category0'].unique():
    dict_cat0[val] = qn_df1[qn_df1['Category0'] == val]['Category Vectors'].unique()[0]


# In[ ]:


dict_cat0


# In[ ]:


dict_cat1 = {}

for val in qn_df2['Category2'].unique():
    dict_cat1[val] = qn_df2[qn_df2['Category2'] == val]['Category Vectors'].unique()[0]


# In[ ]:


dict_cat1


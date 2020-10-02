#!/usr/bin/env python
# coding: utf-8

# Importing libraries

# In[ ]:


import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier


# Basic information, checking if there is missing data

# In[ ]:


data = pd.read_json('../input/train.json')
data.head()


# In[ ]:


data.info()


# Basic visualizations

# In[ ]:


# Bar Plot of various cusisines appearance frequency 
y = data['cuisine'].value_counts()
x = y/y.sum() * 100
y = y.index
sns.barplot(y, x, data=data, palette="BuGn_r")
plt.xticks(rotation=-60)


# In[ ]:


# The most popular ingredients
n = 6714 # total ingredients
frame= pd.DataFrame(Counter([i for sublist in data.ingredients for i in sublist]).most_common(n))
frame = frame.head(10)
frame


# In[ ]:


sns.barplot(frame[0], frame[1], palette="gist_heat")
plt.xticks(rotation=-60)
plt.show()


# Dataset machine learning preprocessing

# In[ ]:


train = json.load(open('../input/train.json'))
test = json.load(open('../input/test.json'))
train_doc = [" ".join(doc['ingredients']).lower() for doc in train]
test_doc = [" ".join(doc['ingredients']).lower() for doc in test]

# Label Encoding of y - the target kinds of cuisine
y_train = [doc['cuisine'] for doc in train]
lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_train


# In[ ]:


# TFIDF statiscic applying to the data - resulting in sparse matrix
tfidf = TfidfVectorizer(binary=True)
X_train = tfidf.fit_transform(train_doc)
X_test = tfidf.transform(test_doc)
X_train


# Models:

# In[ ]:


#Random Forest
#rf = RandomForestClassifier(n_estimators=100)
#model_rf = OneVsRestClassifier(rf, n_jobs=-1)
#model_rf.fit(X_train, y_train)


# In[ ]:


#Support Vector Machine
svc = SVC(C=100, gamma=0.9, coef0=1, tol=0.001, decision_function_shape=None)
model_svc = OneVsRestClassifier(svc, n_jobs=1)
model_svc.fit(X_train, y_train)


# In[ ]:


#Perceptron
#p= Perceptron()
#model_p = OneVsRestClassifier(p, n_jobs=1)
#model_p.fit(X_train, y_train)


# In[ ]:


#Decision Tree
#df = DecisionTreeClassifier()
#model_df = OneVsRestClassifier(df, n_jobs=-1)
#model_df.fit(X_train, y_train)


# In[ ]:


# Predictions based on supported vector machine
y_test = model_svc.predict(X_test)
print(y_test)
y_pred = lb.inverse_transform(y_test)
print(y_pred)
# Submission
test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('svm_output.csv', index=False)


# In[ ]:


# Saving results to file
test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('svm_output.csv', index=False)


# In[ ]:





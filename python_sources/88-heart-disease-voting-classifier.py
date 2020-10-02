#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[7]:


df = pd.read_csv("../input/heart.csv")
df.head()


# In[8]:


#Check for 0s
df.isna().sum()
print(len(df))


# In[9]:


#train-test split
y = df['target']
x = df.loc[:, :'thal']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[10]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[11]:


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
svm_clf = SVC(probability=True)


# In[12]:


voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting = 'soft')
voting_clf.fit(X_train, y_train)


# In[13]:


from sklearn.metrics import accuracy_score 
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# **Results**
# 
# Voting classifier has the highest accuracy: **89%**

# In[15]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(24,12))
plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False)


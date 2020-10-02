#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np 
import pandas as pd 
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette = 'Set3',style = 'white')


# In[7]:


# read json in pandas
df = pd.read_json('../input/Sarcasm_Headlines_Dataset.json',lines=True)
# add new features to better understanding
df['source'] = df['article_link'].apply(lambda x: re.findall(r'\w+', x)[2])
df['len'] = df['headline'].apply(lambda x: len(x.split(' ')))
df = df.drop('article_link',axis =1)
df.head()


# ### Source distribution and Length distribution

# In[13]:


plt.subplots(1,3,figsize= (18,6))
plt.subplot(131)
sns.countplot('is_sarcastic',data = df)
plt.subplot(132)
sns.boxplot(y = 'len',x = 'is_sarcastic',data = df)
plt.subplot(133)
sns.countplot('source',hue= 'is_sarcastic',data = df)


# ### Preparation

# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ### Modeling

# In[26]:


get_ipython().run_cell_magic('time', '', "y = df['is_sarcastic']\nX = df['headline']\n\nfrom sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state= 101)\n\ntext_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])\ntext_mlp_clf = Pipeline([('tfidf', TfidfVectorizer()), \n                         ('clf', MLPClassifier(hidden_layer_sizes=(300,200,50), \n                                               random_state=101, warm_start=True, solver='lbfgs'))]) \ntext_rf_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier())]) \ntext_clf.fit(X_train, y_train)\ntext_mlp_clf.fit(X_train, y_train)\ntext_rf_clf.fit(X_train, y_train)\n\npred = text_clf.predict(X_test)\npred_mlp = text_mlp_clf.predict(X_test)\npred_rf = text_rf_clf.predict(X_test)")


# In[27]:


print(classification_report(y_test, pred))
print(classification_report(y_test, pred_mlp))
print(classification_report(y_test, pred_rf))


# ### Performance(0:89%,1:85%)is not bad!

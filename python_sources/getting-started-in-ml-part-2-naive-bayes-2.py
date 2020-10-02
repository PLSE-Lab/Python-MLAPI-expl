#!/usr/bin/env python
# coding: utf-8

# # Classification using Naive bayes classifier on sklearn's google fetch_20 news groups

# ## 1. Importing necessary packages

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# ## 2. Importing the dataset

# In[ ]:


from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()


# In[ ]:


data.keys()


# In[ ]:


train = fetch_20newsgroups(subset='train', categories=data.target_names)
test = fetch_20newsgroups(subset='test', categories=data.target_names)


# In[ ]:


print(len(data.data))
print(len(train.data))
print(len(test.data))


# In[ ]:


# Sample entry
print(train.data[5])


# ## 3. Classification algorithm - Naive bayes

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)


# ## 4. Prediction

# In[ ]:


labels = model.predict(test.data)


# ## 5. Accuracy

# In[ ]:


from sklearn.metrics import accuracy_score
score = accuracy_score(test.target, labels)
score


# In[ ]:


pred_label_name=np.empty(0)
for i in labels:
    pred_label_name = np.append(pred_label_name, data.target_names[labels[i]])
print(len(pred_label_name))
true_label_name=np.empty(0)
for i in test.target:
    true_label_name = np.append(true_label_name,data.target_names[test.target[i]])
len(true_label_name)


# ## 5. Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(test.target, labels)

from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=cnf_matrix, colorbar=True, show_absolute=False,
                                show_normed=True,figsize=(30,30),)


plt.xticks(np.arange(0,20,1),data.target_names,rotation=90,size=20)
plt.yticks(np.arange(0,20,1),data.target_names,size=20)
plt.xlabel('Predicted label',color='red',fontsize=30)
plt.ylabel('True label',color='red',fontsize=30)
plt.show()


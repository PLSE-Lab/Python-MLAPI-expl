#!/usr/bin/env python
# coding: utf-8

# # Toxicity Bias Logistic Regression, TfidfVectorizer
# 
# 
# * **1. Introduction**
# * **2. Data preparation**
#     * 2.1 Loading packages
#     * 2.2 Load data
#     * 2.3 Show head of data
#     * 2.4 Drop NAN values
# * **3. Clean data**
#     * 3.1 TfidfVectorizer
# * **4. Machine learning**
#     * 4.1 Split data
#     * 4.2 Define Model
#     * 4.3 Fit Model
# * **4. Evaluate the model**
#     * 4.1 Confusion matrix
#     * 4.2 Classification report
#     * 4.3 Receiver Operating Characteristic
# * **5. Prediction and submition**
#     * 5.1 Prediction validation results
#     * 5.2 Submition
# * **6. References**

# # Introduction
# 
# In this kernel i used simple skleaen model "Logistic Regression" and Clean data with sklean feature extraction text tool called TfidfVectorizer.
# 
# Our goal in this competition we are asked to build a model that recognizes toxicity and minimizes unintended bias

# # 2. Data preparation
# ## 2.1 Loading package

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score


# In[ ]:


import os
print(os.listdir("../input"))


# ## 2.2 Load data

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


df = train_df.copy()


# ## 2.3 Show head of data

# In[ ]:


df.head()


# ## 2.4 Drop NAN values

# In[ ]:


df.dropna(axis=1, inplace=True)


# # 3. Clean data
# ## 3.1 TfidfVectorizer
# 
# Convert a collection of raw documents to a matrix of TF-IDF features.

# In[ ]:


Vectorize = TfidfVectorizer()

X = Vectorize.fit_transform(df["comment_text"])

test_X = Vectorize.transform(test_df["comment_text"])


# In[ ]:


y = np.where(train_df['target'] >= 0.5, 1, 0)


# In[ ]:


X.shape, y.shape, test_X.shape


# # 4. Machine learning
# ## 4.1 Split data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)


# ## 4.2 Define Model

# In[ ]:


lr = LogisticRegression(C=5, random_state=42, solver='sag', max_iter=1000, n_jobs=-1)


# ## 4.3 Fit model

# In[ ]:


lr.fit(X_train, y_train)


# In[ ]:


cv_accuracy = cross_val_score(lr, X, y, cv=5, scoring='roc_auc')
print(cv_accuracy)
print(cv_accuracy.mean())


# # 4. Evaluate the model

# In[ ]:


y_pred = lr.predict(X_test)


# ## 4.1 Confusion matrix

# In[ ]:


plt.figure(figsize=(8, 6))
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False, cmap='Reds')
plt.xlabel('predicted value')
plt.ylabel('true value');


# ## 4.2 Classification report

# In[ ]:


print(classification_report(y_test, y_pred))


# ## 4.3 Receiver Operating Characteristic

# In[ ]:


fpr, tpr, thr = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
#auc = auc(fpr, tpr)
auc = roc_auc_score(y_test, y_pred)
lw = 2
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=lw, label="Curve Area = %0.3f" % auc)
plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Plot')
plt.legend(loc="lower right")
plt.show()


# # 5. Prediction and submition
# ## 5.1 Prediction validation results

# In[ ]:


predictions = lr.predict_proba(test_X)[:,1]


# ## 5.2 Submition

# In[ ]:


sub['prediction'] = predictions
sub.to_csv('submission.csv', index=False)


# In[ ]:


sub.head(15)


# # 6. References
# 
# 1. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

# **Thanks For Being Here, If You Have Any Questions Please Let Me Know.
# **

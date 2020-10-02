#!/usr/bin/env python
# coding: utf-8

# This is a **work in progress**, the model will be tuned using GridSearchCV. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# Dataset from https://www.kaggle.com/mlg-ulb/creditcardfraud
# 
# The goal is to train a model capable of spotting fraud cases. 

# In[ ]:


df = pd.read_csv("../input/creditcard.csv")
df.head()


# ### Variables 
# 
# - TimeNumber of seconds elapsed between this transaction and the first transaction in the dataset
# - V1 to V27 may be result of a PCA Dimensionality reduction to protect user identities and sensitive features
# - AmountTransaction amount
# - Class1 for fraudulent transactions, 0 otherwise

# In[ ]:


df['Class'].value_counts()


# * The dataset is particularly interesting because of the unbalance of variability of the target variable Class. 

# In[ ]:


y = df['Class']
y = np.array(y).astype(np.float)
X = df.drop(['Class'], axis=1)
X = np.array(X).astype(np.float)


# ### Natural accuracy
# The natural accuracy is the baseline, it describes how often one would be correct if it was always assumed that there was no fraud.

# In[ ]:


natural_acc = (1 - y.sum()/len(y)) * 100
print('Anything with an accuracy below %.4f would be useless' % natural_acc)


# In[ ]:


def plot_data(X, y):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    return plt.show()


# In[ ]:


plot_data(X, y)


# The imbalance between the two classes will be a problem when training the model, to adjust the balance it could be possible to oversample the minority class, but then the model would be trained on a lot of duplicates, which is second best or undersample the majority class, which would mean to throw away data. SMOTE (Synthetic Minority Oversampling Technique) uses characteristics of nearest neighbor to create synthetic fraud cases 
# 

# In[ ]:


method = SMOTE(kind='regular')
X_resampled, y_resampled = method.fit_sample(X, y)


# In[ ]:


plot_data(X_resampled, y_resampled)


# In[ ]:


new_ratio = (1 - y_resampled.sum()/len(y_resampled)) * 100
print(new_ratio)


# ## Model 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)

model = RandomForestClassifier(random_state=5, 
                               class_weight='balanced_subsample', criterion= 'entropy') 
# weights are calculated with each iteration of growing a tree in the forest

model.fit(X_train, y_train) # the resampled data are used for training only, not for testing
predicted = model.predict(X) 


# In[ ]:


print(accuracy_score(y, predicted) * 100)
print(confusion_matrix(y, predicted))


# In[ ]:


probabilities = model.predict_proba(X)

print(roc_auc_score(y, probabilities[:,1]) * 100)
print(classification_report(y, predicted))


# Assuming that false alarms (false positives) are more acceptable for a credit card company than unspotted cases of fraud, it is important to maximize recall over precision. 

# In[ ]:





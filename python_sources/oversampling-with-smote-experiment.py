#!/usr/bin/env python
# coding: utf-8

# Okay, here are the libraries we'll use.

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier as RF
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('bmh')


# Get a quick look at the training file:

# In[ ]:


get_ipython().system(" head ../input/exoTrain.csv | cut -d ',' -f 1-5")


# Labels are ones and twos, we'll set to zeros and ones.

# In[ ]:


train_data = np.genfromtxt('../input/exoTrain.csv', delimiter=',', skip_header=1)
X = train_data[:, 1:]
y = train_data[:, 0] - 1


# In[ ]:


sum(y == 1) / len(y)


# The data is wildly unbalanced. We can use the SMOTE algorithm to try to balance the dataset. Info on SMOTE is [here][1].
# 
#   [1]: https://www.jair.org/media/953/live-953-2037-jair.pdf

# In[ ]:


X_sm, y_sm = SMOTE(random_state=0).fit_sample(X, y)
Xtr, Xte, ytr, yte = train_test_split(X_sm, y_sm, test_size=0.2, random_state=0)


# Let's fit a simple model and see how it does.

# In[ ]:


rf = RF(n_estimators=100, n_jobs=-1).fit(Xtr, ytr)
rf_preds = rf.predict(Xte)
rf_probs = rf.predict_proba(Xte)
print(acc(yte, rf_preds))


# The ROC curve is a right angle.

# In[ ]:


def plot_roc(actual, probs):
    tpr = dict()
    fpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(actual, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label='AUC: {:.6f}'.format(roc_auc))
    plt.legend(loc='best')


# In[ ]:


plot_roc(yte, rf_probs)


# This looks good on the validation data, but it's misleading - the oversampling has led the model to memorize the positive observations, which appear in both the training and validation set. To get a more accurate estimate of how the model will do on unseen data, first split the data for training and validation, then oversample both sets.

# In[ ]:


test_data = np.genfromtxt('../input/exoTest.csv', delimiter=',', skip_header=1)
test_labels = test_data[:, 0] - 1
test_features = test_data[:, 1:]

test_preds = rf.predict(test_features)
print(acc(test_labels, test_preds))
print(classification_report(test_labels, test_preds))


# Short answer no, even though it looks good on the validation data, the model still just predicts class 0 on the test set: 565/570 = 0.991228070175. 
# 
# The oversampling above leads to a memorization issue - the small number of positive observations are resampled, and then, after the train-test split, show up in both the training and validation datasets. The model has perfect performance at validation because it's seen all of the observations it's being validated on already. To get a better sense of how the model will perform on new data, the order of operations needs to be reversed: first split the data into train and val, then oversample.
# 
# 
# ----------

# In[ ]:


Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
Xtr_sm , ytr_sm = SMOTE(random_state=0).fit_sample(Xtr, ytr)
Xte_sm, yte_sm = SMOTE(random_state=0).fit_sample(Xte, yte)


# In[ ]:


rf = RF(n_estimators=100, n_jobs=-1).fit(Xtr_sm, ytr_sm)
rf_preds = rf.predict(Xte_sm)
rf_probs = rf.predict_proba(Xte_sm)
print(acc(yte_sm, rf_preds))


# In[ ]:


test_data = np.genfromtxt('../input/exoTest.csv', delimiter=',', skip_header=1)
test_labels = test_data[:, 0] - 1
test_features = test_data[:, 1:]

test_preds = rf.predict(test_features)
print(acc(test_labels, test_preds))
print(classification_report(test_labels, test_preds))


# In[ ]:





# In[ ]:


Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
Xtr_sm, ytr_sm = SMOTE(random_state=0).fit_sample(Xtr, ytr)

rf = RF(n_estimators=100, n_jobs=-1).fit(Xtr_sm, ytr_sm)
rf_preds = rf.predict(Xte)
print(acc(yte, rf_preds))
print(classification_report(yte, rf_preds))


# In[ ]:


test_preds = rf.predict(test_features)
print(acc(test_labels, test_preds))
print(classification_report(test_labels, test_preds))


# The model hasn't learned anything, but at least the switch in order of operations means that validation metrics are good estimators of test metrics. 

# In[ ]:





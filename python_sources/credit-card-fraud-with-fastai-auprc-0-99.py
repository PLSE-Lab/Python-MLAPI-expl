#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud with fastai

# Based on https://docs.fast.ai/tabular.html.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Import the libraries we need.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastai.tabular import *
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score


# Read csv file.

# In[ ]:


orig_df = pd.read_csv('../input/creditcard.csv')
orig_df.head()


# Print number of rows with Class == 0 and Class == 1.

# In[ ]:


orig_df.loc[orig_df['Class'] == 0].shape[0], orig_df.loc[orig_df['Class'] == 1].shape[0]


# Looks like we have an imbalanced dataset.  Use Synthetic Minority Oversampling TEchnique (SMOTE) to oversample the minority class.  See https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets.

# In[ ]:


smote = SMOTE()
X, y = smote.fit_sample(orig_df.drop(columns=['Class']), orig_df['Class'])
y = y[..., np.newaxis]
Xy = np.concatenate((X, y), axis=1)
df = pd.DataFrame.from_records(Xy, columns=list(orig_df))
df.Class = df.Class.astype(int)
df.head()


# Print number of rows with Class == 0 and Class == 1.

# In[ ]:


df.loc[df['Class'] == 0].shape[0], df.loc[df['Class'] == 1].shape[0]


# Define Transforms that will be applied to our variables.  We don't have missing values, so we don't need FillMissing.  We don't have any categorical variable, so we don't need Categorify.

# In[ ]:


#procs = [FillMissing, Categorify, Normalize]
procs = [Normalize]


# Split our data into training and validation sets.

# In[ ]:


idx = np.arange(df.shape[0])
train, valid, train_idx, valid_idx = train_test_split(df, idx, test_size=0.05, random_state=42)
train_idx.shape, valid_idx.shape


# Split our variables into dependent and independent variables.  Then split independent variables into categorical and continuous variables.  "fastai will assume all variables that aren't dependent or categorical are continuous, unless we explicitly pass a list to the cont_names parameter when constructing our DataBunch."  We don't have any categorical variable.

# In[ ]:


dep_var = 'Class'
cat_names = []


# Create DataBunch.

# In[ ]:


data = TabularDataBunch.from_df('.', df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)
print(data.train_ds.cont_names)  # `cont_names` defaults to: set(df)-set(cat_names)-{dep_var}


# Create a Learner.  We don't have any categorical variable, so emb_szs is empty.  fastai doesn't have AUPRC, so metrics is empty.

# In[ ]:


learn = tabular_learner(data, layers=[1000, 500], ps=[0.001, 0.01], emb_szs={}, metrics=[])


# Find learning rate to use.

# In[ ]:


learn.lr_find(stop_div=True, num_it=100)


# In[ ]:


learn.recorder.plot()


# Looks like 1e-2 is a good value to use.  Let's train our model for 1 epoch.

# In[ ]:


learn.fit_one_cycle(1, 1e-2)


# Optional: Plot learning rate schedule.

# In[ ]:


learn.recorder.plot_lr()


# Does the model do well on the training data?  Accuracy is not meaningful for unbalanced classification.  To find how well the model do, plot and compute AUPRC.  See https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/.  Tweak the model until it does well on the training data.

# In[ ]:


[train_preds, train_targets] = learn.get_preds(ds_type=DatasetType.Train)
train_preds = to_np(train_preds)[:, 1]
train_targets = to_np(train_targets)


# In[ ]:


precision, recall, thresholds = precision_recall_curve(train_targets, train_preds)
plt.plot(recall, precision, marker='.')
auprc = auc(recall, precision)
auprc


# Does the model do well on the validation data?

# In[ ]:


[valid_preds, valid_targets] = learn.get_preds(ds_type=DatasetType.Valid)
valid_preds = to_np(valid_preds)[:, 1]
valid_targets = to_np(valid_targets)


# In[ ]:


precision, recall, thresholds = precision_recall_curve(valid_targets, valid_preds)
plt.plot(recall, precision, marker='.')
auprc = auc(recall, precision)
auprc


# Find max F1 score as well as precision, recall, and threshold at max F1.

# In[ ]:


F1 = [2*p*r/(p+r) for p, r in zip(precision, recall)]
idx = np.argmax(F1)
np.max(F1), precision[idx], recall[idx], thresholds[idx]


# As a sanity check, make sure we get the same max F1 score using the threshold we found.

# In[ ]:


preds = valid_preds.copy()
preds[preds >= thresholds[idx]] = 1
preds[preds < thresholds[idx]] = 0
f1_score(valid_targets, preds)


# In[ ]:





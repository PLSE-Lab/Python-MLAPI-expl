#!/usr/bin/env python
# coding: utf-8

# **How much should we trust public LB for DSBowl competition?**
# 
# We have 1000 samples in public test and around 7000 in full test set. So let's simulate the model fit to the complete set with a certain kappa and see, how does the score vary on different folders choosen as public LB.
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.metrics import accuracy_score, cohen_kappa_score
import random
from tqdm.notebook import tqdm

random.seed(42)
np.random.seed(42)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
data_dir = "/kaggle/input/data-science-bowl-2019/"
df_labels = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
# Any results you write to the current directory are saved as output.


# Check the target distribution for train set.

# In[ ]:


probas = df_labels.accuracy_group.value_counts(normalize=True)
probas.head()


# Generate private test set simulation

# In[ ]:


df_gt_simulated = np.random.choice(probas.index.values, 7000, p=probas.values)


# In[ ]:


pd.Series(df_gt_simulated).value_counts(normalize=True)


# In[ ]:


df_pred_simulated = df_gt_simulated.copy()
cohen_kappa_score(df_gt_simulated, df_pred_simulated, weights='quadratic')


# Shuffle part of predictions to simulate a non-overfit model with realistic accuracy

# In[ ]:


inds_to_shuffle = np.random.choice(range(len(df_gt_simulated)), int(len(df_gt_simulated) * 0.6))
df_pred_simulated[inds_to_shuffle] = np.random.permutation(df_pred_simulated[inds_to_shuffle])
cohen_kappa_score(df_gt_simulated, df_pred_simulated, weights='quadratic'), accuracy_score(df_gt_simulated, df_pred_simulated)


# In[ ]:


scores = []
for t, p in zip(np.split(df_gt_simulated, 7), np.split(df_pred_simulated, 7)):
    score = cohen_kappa_score(t, p, weights='quadratic')
    print("fold score: ", score)
    scores.append(score)
print("Mean score: ", np.mean(scores))


# In[ ]:


min(scores), max(scores)


# The model that would score 0.55 on private can give from 0.527 to 0.571 on public

# How much does this trust interval change over the real value for kappa?

# In[ ]:


real_kappas = []
fold_kappas = []
for t in tqdm(range(100, 1, -1)):
    df_pred_simulated = df_gt_simulated.copy()
    inds_to_shuffle = np.random.choice(range(len(df_gt_simulated)), int(len(df_gt_simulated) * t / 100))
    df_pred_simulated[inds_to_shuffle] = np.random.permutation(df_pred_simulated[inds_to_shuffle])
    real_kappa = cohen_kappa_score(df_gt_simulated, df_pred_simulated, weights='quadratic')
    scores = []
    for t, p in zip(np.split(df_gt_simulated, 7), np.split(df_pred_simulated, 7)):
        scores.append(cohen_kappa_score(t, p, weights='quadratic'))
    for s in scores:
        real_kappas.append(real_kappa)
        fold_kappas.append(s)


# In[ ]:


pd.DataFrame({"real_kappas": real_kappas, "fold_kappas": fold_kappas}).plot.scatter(x="real_kappas", y="fold_kappas", grid=True)


# In[ ]:


pd.DataFrame({"real_kappas": real_kappas, "fold_kappas": fold_kappas}).groupby("real_kappas").agg(['min', 'max']).plot()


# Conclusions:
# * In this competition it is very easy to overfit to public LB and that will backfire badly.
# * Hold-out set validation means virtually nothing here.
# * CV can be trusted. In case there is no leaks or train/test domain differences.

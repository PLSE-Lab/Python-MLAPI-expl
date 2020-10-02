#!/usr/bin/env python
# coding: utf-8

# ## Analysis of subgroup AUCs from a BERT Baseline
# 
# The results used for this analysis are based on a straightforward fine-tuning of BERT, using `run_classifier.py`, available [here](https://github.com/google-research/bert). I used a sequence length of 128 tokens and trained for 1 epoch.
# 
# The analysis clearly shows that I need more civil (non-toxic) examples mentioning from the groups `Muslim` and `homosexual_gay_or_lesbian` to improve training.
# 

# In[ ]:


import pandas as pd
from sklearn import metrics


# A random sample of 100k examples was taken from the training set provided for this competition. This was split into a dev set of 50k and `test.csv` in the code below. The `predictions.csv` file here is based predictions from the BERT model described above.

# In[ ]:


pred = pd.read_csv("../input/bert-baseline/predictions.csv")
df = pd.read_csv("../input/bert-baseline/test.csv")
df['prediction'] = pred[' Toxic']
df['target'] = df['target'] >= 0.5
df['bool_pred'] = df['prediction'] >= 0.5


# First, a simple AUC function taking a dataframe as input and using sklearn for the calculation.

# In[ ]:


def auc(df):
    y = df['target']
    pred = df['prediction']
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    return metrics.auc(fpr, tpr)

overall = auc(df)
overall


# Only 9 subgroups have enough examples in the competition test set to be included in the bias calculation. Other subgroups are ignored as part of the background for this competition.

# In[ ]:


groups = ['black', 'white', 'male', 'female',
          'christian', 'jewish', 'muslim',
          'psychiatric_or_mental_illness',
          'homosexual_gay_or_lesbian']

categories = pd.DataFrame(columns = ['SUB', 'BPSN', 'BNSP'], index = groups)


# 
# The Mp function below is based on the [evaluation formula](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation) this competition. This formula is then applied to the 9 subgroups and their BPSN and BNSP datasets.
# 
# * **SUB**
# : Claculates an AUC using only examples from the sub-group.
# * **BPSN**
# : Background Positive Subgroup Negative. Claculates an AUC using a subset of toxic comments outside the sub-group (BP) and non-toxic comments in the sub-group (SN).
# * **BNSP**
# : Background Negative Subgroup Positive. Claculates an AUC using a subset of non-toxic comments outside the sub-group (BN) and toxic comments in the sub-group (SP).

# In[ ]:


import numpy as np
def Mp(data, p=-5.0):
    return np.average(data ** p) ** (1/p)

for group in groups:
    df[group] = df[group] >= 0.5
    categories.loc[group,'SUB'] = auc(df[df[group]])
    bpsn = ((~df[group] & df['target'])    #background positive
            | (df[group] & ~df['target'])) #subgroup negative
    categories.loc[group,'BPSN'] = auc(df[bpsn])
    bnsp = ((~df[group] & ~df['target'])   #background negative
            | (df[group] & df['target']))  #subgrooup positive
    categories.loc[group,'BNSP'] = auc(df[bnsp])

categories.loc['Mp',:] = categories.apply(Mp, axis= 0)
categories


# Interesting results. Clearly I need to modify my input pipeline to include more subgroup examples, especially `homosexual` and `Muslim` examples from the BPSN subset. In other words, I need more civil (non-toxic) examples mentioning Muslims and homosexuals to improve the leader board score.

# Finally, the final leader-board score.

# In[ ]:


leaderboard = (np.sum(categories.loc['Mp',:]) + overall) / 4
leaderboard


#  This same vanilla BERT model gave me an actual score of `0.9223` on the competition leader board. Sy my randomly sampled 50k test set must have easier examples in the sub-groups than the official test set of 97k examples.

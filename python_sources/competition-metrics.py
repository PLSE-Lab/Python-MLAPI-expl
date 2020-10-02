#!/usr/bin/env python
# coding: utf-8

# # Competition Metrics
# 
# 
# Hi everyone!
# 
# I would like to share with you my metrics implementations for this competition. 
# 
# I think correct calculating metrics is very important! 
# 
# ### If you find any imprecision in calculating metrics, please, let me know!

# # CHANGELOG
# 
# - v1, initial
# - v2, fix global average from micro to macro (see [here](https://www.kaggle.com/c/birdsong-recognition/discussion/160320) in discussion), fix explanation `Micro averaged`
# - v3, fix mistake with swapping names FN/FP  (thanks for noticing [@nandhuelan](https://www.kaggle.com/nandhuelan))
# - v4, fix sorting in numpy method
# - v5, added example with usage of ready method from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) (thanks for kindly noticing [@Maxwell](https://www.kaggle.com/maxwell110))
# 
# <img src="https://i.stack.imgur.com/VxiS5.png" width="500" align="left"/>

# # Explanation
# 
# In description:
# > Submissions will be evaluated based on their row-wise micro averaged F1 score.
# 
# - The F1 score is the harmonic mean of the precision and recall (more information [here](https://en.wikipedia.org/wiki/F1_score)). Equation:
# 
# $ F_1 = {2 * precision * recall \over precision + recall} = {2 * TP \over 2*TP + FN + FP} $
# 
# - Row-wise means that TP, FN, FP is calculated using every value (bird) in row (thanks a lot [@dhananjay3](https://www.kaggle.com/dhananjay3) for explanation [here](https://www.kaggle.com/c/birdsong-recognition/discussion/159968#893120) for me)
# 
# - `Micro averaged` means that F1 is caluclated by counting the total TP, FN and FP in one row (!), after F1 for all rows are used as average (thanks a lot [@dhananjay3](https://www.kaggle.com/dhananjay3) and [@carriesmi](https://www.kaggle.com/carriesmi) for explanation and experiment [here](https://www.kaggle.com/c/birdsong-recognition/discussion/160320))

# # Using Birds
# 
# implementation with using string birds

# In[ ]:


import numpy as np

def row_wise_f1_score_micro(y_true, y_pred):
    """ author @shonenkov """
    F1 = []
    for preds, trues in zip(y_pred, y_true):
        TP, FN, FP = 0, 0, 0
        preds = preds.split()
        trues = trues.split()
        for true in trues:
            if true in preds:
                TP += 1
            else:
                FN += 1
        for pred in preds:
            if pred not in trues:
                FP += 1
        F1.append(2*TP / (2*TP + FN + FP))
    return np.mean(F1)


# ### tests:

# In[ ]:


print('[all equal]:', row_wise_f1_score_micro(
    y_true=['nocall', 'ameavo'], 
    y_pred=['nocall', 'ameavo'],
))

print('[nothing]:', row_wise_f1_score_micro(
    y_true=['nocall', 'ameavo'], 
    y_pred=['amebit', 'amebit'],
))

print('[1 correct]:', row_wise_f1_score_micro(
    y_true=['nocall', 'ameavo'], 
    y_pred=['nocall', 'amebit'],
))

print('[double prediction]:', row_wise_f1_score_micro(
    y_true=['nocall', 'ameavo amebit'], 
    y_pred=['nocall', 'ameavo amebit'],
))

print('[double prediction with permutation]:', row_wise_f1_score_micro(
    y_true=['nocall', 'ameavo amebit'], 
    y_pred=['nocall', 'amebit ameavo'],
))


print('[semi prediction]:', row_wise_f1_score_micro(
    y_true=['nocall', 'ameavo amebit'], 
    y_pred=['nocall', 'ameavo'],
))

print('[semi prediction with odd]:', row_wise_f1_score_micro(
    y_true=['nocall', 'ameavo'], 
    y_pred=['nocall', 'ameavo amebit'],
))

print('[semi prediction with double odd]:', row_wise_f1_score_micro(
    y_true=['nocall', 'ameavo'], 
    y_pred=['nocall', 'ameavo amebit amecro'],
))

print('[semi prediction of triple with odd]:', row_wise_f1_score_micro(
    y_true=['nocall', 'ameavo amecro'], 
    y_pred=['nocall', 'ameavo amebit amecro'],
))


# # Using Numpy Vectors
# 
# For example during evaluation of model. 

# In[ ]:


def row_wise_f1_score_micro_numpy(y_true, y_pred, threshold=0.5, count=5):
    """ 
    @author shonenkov 
    
    y_true - 2d npy vector with gt
    y_pred - 2d npy vector with prediction
    threshold - for round labels
    count - number of preds (used sorting by confidence)
    """
    def meth_agn_v2(x, threshold):
        idx, = np.where(x > threshold)
        return idx[np.argsort(x[idx])[::-1]]

    F1 = []
    for preds, trues in zip(y_pred, y_true):
        TP, FN, FP = 0, 0, 0
        preds = meth_agn_v2(preds, threshold)[:count]
        trues = meth_agn_v2(trues, threshold)
        for true in trues:
            if true in preds:
                TP += 1
            else:
                FN += 1
        for pred in preds:
            if pred not in trues:
                FP += 1
        F1.append(2*TP / (2*TP + FN + FP))
    return np.mean(F1)


# ### test

# In[ ]:


y_pred = np.array([
    [0.4,0.6,0.9],
    [0.1,0.9,0.8],
    [0.1,0.4,0.2],
    [0.9,0.9,0.9],
])

y_true = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
])

threshold = 0.5

row_wise_f1_score_micro_numpy(y_true, y_pred, threshold=threshold)


# In[ ]:


y_pred = np.array([
    [0.4,0.6,0.9],
    [0.1,0.9,0.8],
    [0.1,0.4,0.2],
    [0.9,0.9,0.9],
])

y_true = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
])

threshold = 0.6

row_wise_f1_score_micro_numpy(y_true, y_pred, threshold=threshold)


# # Using Sklearn
# 
# Thanks a lot [@Maxwell](https://www.kaggle.com/maxwell110) for noticing simpler implementation in [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) with average "samples".
# 
# 
# P.S.
# This method doesnt have param "count", but it is very good method for usage also! Param "count" provides to restrict prediction count using sorting by confidence (if count = 3, that means "no more than 3") 

# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


y_pred = np.array([
    [0.4,0.6,0.9],
    [0.1,0.9,0.8],
    [0.1,0.4,0.2],
    [0.9,0.9,0.9],
])

y_true = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
])

threshold = 0.5

f1_score(y_true, np.where(y_pred > threshold, 1, 0), average='samples')


# In[ ]:


y_pred = np.array([
    [0.4,0.6,0.9],
    [0.1,0.9,0.8],
    [0.1,0.4,0.2],
    [0.9,0.9,0.9],
])

y_true = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
])

threshold = 0.6

f1_score(y_true, np.where(y_pred > threshold, 1, 0), average='samples')


# # Thank you for attention!
# 
# Don't forget to read my kernel about sample submission using custom check phase (it helps to find and avoid bugs before using button submission): 
# 
# - [[Sample Submission] Using Custom Check](https://www.kaggle.com/shonenkov/sample-submission-using-custom-check)

# ### If you find any imprecision in calculating metrics, please, let me know!

#!/usr/bin/env python
# coding: utf-8

# This notebook will show the usage of the post-processing discussed here: https://www.kaggle.com/c/google-quest-challenge/discussion/130083

# In[ ]:


import pandas as pd 
import numpy as np
from scipy.stats.mstats import hmean
from scipy.stats import spearmanr
from functools import partial
# suppress scientific notation in numpy and pandas
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:.6f}'.format
pd.set_option('display.max_columns', None)


# # Postprocess

# In[ ]:


target_columns=pd.read_csv(f'../input/ensemble-data/fold_0_labels.csv').iloc[:,1:].columns.tolist()

target_columns


# In[ ]:


# labels.npy stores the frequencies of every labels for every column
classes = np.load('../input/labels/labels.npy', allow_pickle=True)

prior_freqs_list = [np.array([classes[i][key] for key in sorted(classes[i])]) for i in range(len(classes))]
prior_probs_list = [freqs / sum(freqs) for freqs in prior_freqs_list]

prior_probs_list


# In[ ]:



def deal_column(s: np.ndarray, freq):
    """
    the idea is illustrated here: https://www.kaggle.com/c/google-quest-challenge/discussion/130083
    s is the original predictions, and freq is the number of every labels from small to large.
    Example: 
    If a column only has 3 lables: 0, 1/3, 2/3 and the distribution is [0.5, 0.2, 0.3]
    assume the original prediction s for this column is [0.01,0.03,0.05,0.02,0.07,0.04,0.09,0.0,0.08,0.06]
    This method will map the lowest 5 predictions to 0 because theoretically this test set has 10*0.5=5 examples that labeled 0.
    The processing for labels 1/3 and 2/3 is similar, and the output will be:
    [0.0,0.0,0.05,0.0,0.07,0.0,0.07,0.0,0.07,0.05]
    """
    res = s.copy()  # use a copy to return
    d = {i: v for i, v in enumerate(s)}  # <index, original_value>
    d = sorted(d.items(), key=lambda item: item[1])
    j = 0
    for i in range(len(freq)):
        if freq[i] > 0 and j < len(d):
            fixed_value = d[j][1]
            while freq[i] > 0:
                res[d[j][0]] = fixed_value
                freq[i] -= 1
                j += 1
    return res


# prob is the distribution of the column in trainning set, n is the number of examples of test set
def estimate_frequency(prob: np.ndarray, n):
    tmp = prob * n
    freq = [int(round(t)) for t in tmp]
    # the prob times #example and and use round operation cannot make sure the sum of freq equals to #example
    # here we consider the error of round operation, e.g. round(1.9)=2 so the error is 0.1, and round(1.5)=2 so error is 0.5
    confidence = {i: np.abs(0.5 - (x - int(x))) for i, x in enumerate(tmp)}  # the smaller the error, the higher the confidence we have in round
    confidence = sorted(confidence.items(), key=lambda item: item[1])
    # fix frequency according to confidence of 'round' operation
    fix_order = [idx for idx, _ in confidence]
    idx = 0
    s = np.sum(freq)
    # fix the frequency of every label, until the sum is #example
    while s != n:
        if s > n:
            freq[fix_order[idx]] -= 1
        else:
            freq[fix_order[idx]] += 1
        s = np.sum(freq)
        # theoretically we can fix the freq in one round, but here we use a loop
        idx = (idx + 1) % len(fix_order)
    # if the resulting freq only has 1 label/class, we change it to 2 labels: one has n-1 examples and the other has 1 example
    if np.sum(np.array(freq) > 0) < 2:  # in case there is only one class
        freq[0], freq[len(freq) - 1] = n - 1, 1
    return freq


def align(predictions: np.ndarray, ban_list=None) -> np.ndarray:
    num_samples = predictions.shape[0]  # number of examples of test set
    predictions_new = predictions.copy()
    for i in range(30):
        # deal with every column but skip the columns that post-processing won't improve the score
        if ban_list is not None and i in ban_list:
            continue
        frequency = estimate_frequency(prior_probs_list[i], num_samples)
        predictions_new[:, i] = deal_column(predictions[:, i], frequency)
    return predictions_new


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(spearmanr(col_trues, col_pred).correlation)
    return np.mean(rhos).item()
    

def cal(arr1, arr2): # calculate column-wise scores
    return np.array([compute_spearmanr(arr1[:, i].reshape(-1, 1), arr2[:, i].reshape(-1, 1)) for i in range(30)])


# In[ ]:



diffs=pd.DataFrame(columns=target_columns)

for FOLD in range(5):
    #Read csv files
    labels=pd.read_csv(f'../input/ensemble-data/fold_{FOLD}_labels.csv').iloc[:,1:]
    base=pd.read_csv(f'../input/ensemble-data/bert_base_uncased_fold_{FOLD}_preds.csv').iloc[:,1:]
    wwm_uncased=pd.read_csv(f'../input/ensemble-data/wwm_uncased_fold_{FOLD}_preds.csv').iloc[:,1:]
    wwm_cased=pd.read_csv(f'../input/ensemble-data/wwm_cased_fold_{FOLD}_preds.csv').iloc[:,1:]
    large_uncased=pd.read_csv(f'../input/ensemble-data/large_uncased_fold_{FOLD}_preds.csv').iloc[:,1:]
    roberta=pd.read_csv(f'../input/ensemble-data/roberta_large_fold_{FOLD}_preds.csv').iloc[:,1:]

    ps=[base.values, wwm_uncased.values, wwm_cased.values, large_uncased.values, roberta.values]

    
    mv=np.average(ps,axis=0)
    original_scores=cal(labels.values,mv)
    
    # post-processing
    mv_1=mv.copy()
    mv_1=align(mv_1)

    relative_scores=cal(labels.values,mv_1)-original_scores
    row = pd.DataFrame(relative_scores).T
    row.columns = target_columns
    diffs=diffs.append(row)
diffs.index=[f'fold-{n}' for n in range(5)]


# In[ ]:


diffs


# In[ ]:


# apply post-processing to the following columns will lower the scores. The numbers are the indices of the column in target_columns
ban_list = [0, 1, 3, 4, 6, 10, 16, 17, 18] + list(range(20, 30))

scores,post_scores,post_ban_scores=[],[],[]
# test the performance of PP
for FOLD in range(5):
    #Read csv files
    labels=pd.read_csv(f'../input/ensemble-data/fold_{FOLD}_labels.csv').iloc[:,1:]
    base=pd.read_csv(f'../input/ensemble-data/bert_base_uncased_fold_{FOLD}_preds.csv').iloc[:,1:]
    wwm_uncased=pd.read_csv(f'../input/ensemble-data/wwm_uncased_fold_{FOLD}_preds.csv').iloc[:,1:]
    wwm_cased=pd.read_csv(f'../input/ensemble-data/wwm_cased_fold_{FOLD}_preds.csv').iloc[:,1:]
    large_uncased=pd.read_csv(f'../input/ensemble-data/large_uncased_fold_{FOLD}_preds.csv').iloc[:,1:]
    roberta=pd.read_csv(f'../input/ensemble-data/roberta_large_fold_{FOLD}_preds.csv').iloc[:,1:]

    ps=[base.values, wwm_uncased.values, wwm_cased.values, large_uncased.values, roberta.values]

    
    mv=np.average(ps,axis=0)
    scores.append(compute_spearmanr(labels.values,mv))
    
    # post-processing
    mv_1=mv.copy()
    mv_2=mv.copy()
    mv_1=align(mv_1)
    mv_2=align(mv_2,ban_list)

    post_scores.append(compute_spearmanr(labels.values,mv_1))
    post_ban_scores.append(compute_spearmanr(labels.values,mv_2))


# In[ ]:


print(f"original score: {np.mean(scores)}\npost without ban: {np.mean(post_scores)}\npost with ban: {np.mean(post_ban_scores)}")


# In[ ]:





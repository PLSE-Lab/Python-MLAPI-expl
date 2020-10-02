#!/usr/bin/env python
# coding: utf-8

# This notebook is an attempt to answer of how gini standard error depends on the percent of the data used for calculation. For example, if one selected 10% of train data for validation, what gini error could be expected? Or what would be gini deviation for public leaderboard comparing to private one?

# It is not possible to calculate gini directly on test data, therefore test data is created from 50% of train data (and increased using  random sampling with replacement to correspond test size in this competition), other 50% is used for training. Note that  goal here is not any specific model or exact gini value, but only behavior of gini standard deviation.

# In[1]:


import pandas as pd
import numpy as np
from numba import jit
from sklearn.model_selection import train_test_split

# obligatory part of any kernel
# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0.0
    delta = 0
    n = len(y_true)
    for i in range(n - 1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (1.0 * ntrue * (n - ntrue))
    return gini


seed = 1685
train_original = pd.read_csv('../input/train.csv', dtype={'target': np.int8, 'id': np.int32})
test_original = pd.read_csv('../input/test.csv', dtype={'id': np.int32})
test_set_size = len(test_original.index)
train_set_size = len(train_original)

train, test_source = train_test_split(train_original, random_state=seed, train_size=0.5, test_size=0.5)
test = test_source.sample(n=test_set_size, replace=True)

del train_original, test_original, test_source


# Fit some model and make prediction on test

# In[2]:


from xgboost import XGBClassifier

model = XGBClassifier(
    n_jobs=4,
    random_state=seed,
    n_estimators=100,
    max_depth=4,
    objective='binary:logistic',
    learning_rate=0.1,
)

fit_model = model.fit(train.drop(['id', 'target'], axis=1), train['target'])

prediction = fit_model.predict_proba(test.drop(['id', 'target'], axis=1))[:, 1]
gini_total = eval_gini(test['target'], prediction)
target_and_prediction = pd.DataFrame({
    'prediction': prediction,
    'target' : test['target']
})


# Check what gini would be for randomly selected 5%, 10%, 15%, etc of test data

# In[ ]:


step = 5
num_rounds = 200
x = []
mean = []
std = []
mins=[]
maxs=[]

test_target = test['target']
percents = range(step, 100 + step, step)
for part_size in percents:
    ginis = []
    for i in range(num_rounds):
        iteration_seed = (seed + num_rounds * part_size + i)
        part = target_and_prediction.sample(frac=part_size / 100, random_state=iteration_seed)
        gini = eval_gini(part['target'], part['prediction'])
        ginis.append(gini)
    x.append(part_size)
    mean.append(np.mean(ginis))
    std.append(np.std(ginis))
    mins.append(np.min(ginis))
    maxs.append(np.max(ginis))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

high = [x + y for x, y in zip(mean, std)]
low = [x - y for x, y in zip(mean, std)]

plt.figure(1, figsize=(12, 12))
plt.axis([0, 100, 0.255, 0.29])

plt.fill_between(x, maxs, mins, interpolate=True, color='#F0F0F0')
plt.fill_between(x, high, low, interpolate=True, color='silver')
plt.plot(x, [gini_total] * len(x), 'r--', x, mean, '-o')

public_index = 30 // step - 1
public_label = mean[public_index] + std[public_index]
plt.annotate('Public leaderboard,\n30%, std = {:8.5f}'.format(std[public_index]),
             xy=(30, public_label + 0.001),
             xytext=(30, public_label + 0.005),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

private_index = 70 // step - 1
private_label = mean[private_index] + std[private_index]
plt.annotate('Private leaderboard,\n70%, std = {:8.5f}'.format(std[private_index]),
             xy=(70, private_label + 0.001),
             xytext=(70, private_label + 0.005),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
plt.title('Gini deviation vs data size')
plt.xlabel('Percent of test data used for gini calculation')
plt.ylabel('Gini mean value and deviation')
plt.show()


# On plot above, gini was calculated 200 times on randomly selected samples of each size.
# Light grey area - gini min/max, grey area - gini within standard deviation, blue - mean value, and red - gini value calculated on full data set.
# 
# Model here has gini about 0.27 on full test set, but public leaderboard may vary within 0.265 - 0.275, and private leaderboard may have range 0.268 - 0.272.

# In[ ]:


results = pd.DataFrame()
results['test, %'] = percents
results['train, %'] = results['test, %'] * test_set_size / train_set_size
results['std'] = std
results['min'] = mins
results['mean'] = mean
results['max'] = maxs

results


# Table above shows that standard error of public leaderboard is about 0.005 and if one use 15% validation set from train data, standard deviation would be about 0.01.
# 
# Note sure I can draw conclusions like this, but If we assume that top of the public leaderboard overfits to 30% of test data, then potential winners have public gini  within 0.281 - 0.291, that is about 2000 teams, and winning solution has gini about 0.286.
# 
# The question is how you deal with such a big standard error? 

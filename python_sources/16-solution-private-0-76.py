#!/usr/bin/env python
# coding: utf-8

# # #16 Solution (Private 0.76)

# First, thank you Kaggle for hosting this awesome competition and congratulations to every one that participated & fought hard in this competition! Especially to [Markus F](https://www.kaggle.com/friedchips), [Thomas Rohwer](https://www.kaggle.com/trohwer64), and [Nanashi](https://www.kaggle.com/jesucristo) for their incredibly helpful contribution in discussions and kernels! I personally would have not made it #16 if it wasnt for their contribution on [The Missing Link](https://www.kaggle.com/friedchips/the-missing-link), ["The Orientation Sensor" or "Science vs. Alchemy" discussion](https://www.kaggle.com/c/career-con-2019/discussion/87239#latest-512162), and [Smart Robots. Complete Notebook](https://www.kaggle.com/jesucristo/1-smart-robots-complete-notebook-0-73).
# 
# 
# 
# 
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import os
from time import time
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from numba import jit
import itertools
from seaborn import countplot,lineplot, barplot
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import random
import math

from numpy.fft import *
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
# import peakutils

le = preprocessing.LabelEncoder()
RAW_DATA_PATH = "../input/career-con-2019"

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# Load data and append test row to train data:

# In[ ]:


data_target = pd.read_csv("{}/y_train.csv".format(RAW_DATA_PATH))
train_X = pd.read_csv('{}/X_train.csv'.format(RAW_DATA_PATH)).iloc[:,3:]
test_X  = pd.read_csv('{}/X_test.csv'.format(RAW_DATA_PATH)).iloc[:,3:]
train_X = train_X.append(test_X)
train_X = train_X.values.reshape(-1,128,10)


# Like what Markus F said in his kernel, we can reconstruct the original data before it's being splitted. Apparently, some of the original data is being splitted between train and test data. So here we go:

# In[ ]:


# Originally from Markus F's kernel: https://www.kaggle.com/friedchips/the-missing-link
def sq_dist(a,b):
    ''' the squared euclidean distance between two samples '''
    
    return np.sum((a-b)**2, axis=1)


def find_run_edges(data, edge):
    ''' examine links between samples. left/right run edges are those samples which do not have a link on that side. '''

    if edge == 'left':
        border1 = 0
        border2 = -1
    elif edge == 'right':
        border1 = -1
        border2 = 0
    else:
        return False
    
    edge_list = []
    linked_list = []
    
    for i in range(len(data)):
        dist_list = sq_dist(data[i, border1, :4], data[:, border2, :4]) # distances to rest of samples
        min_dist = np.min(dist_list)
        closest_i   = np.argmin(dist_list) # this is i's closest neighbor
        if closest_i == i: # this might happen and it's definitely wrong
            print('Sample', i, 'linked with itself. Next closest sample used instead.')
            closest_i = np.argsort(dist_list)[1]
        dist_list = sq_dist(data[closest_i, border2, :4], data[:, border1, :4]) # now find closest_i's closest neighbor
        rev_dist = np.min(dist_list)
        closest_rev = np.argmin(dist_list) # here it is
        if closest_rev == closest_i: # again a check
            print('Sample', i, '(back-)linked with itself. Next closest sample used instead.')
            closest_rev = np.argsort(dist_list)[1]
        if (i != closest_rev): # we found an edge
            edge_list.append(i)
        else:
            linked_list.append([i, closest_i, min_dist])
            
    return edge_list, linked_list


def find_runs(data, left_edges, right_edges):
    ''' go through the list of samples & link the closest neighbors into a single run '''
    
    data_runs = []

    for start_point in left_edges:
        i = start_point
        run_list = [i]
        while i not in right_edges:
            tmp = np.argmin(sq_dist(data[i, -1, :4], data[:, 0, :4]))
            if tmp == i: # self-linked sample
                tmp = np.argsort(sq_dist(data[i, -1, :4], data[:, 0, :4]))[1]
            i = tmp
            run_list.append(i)
        data_runs.append(np.array(run_list))
    
    return data_runs


# Let's find the edges!

# In[ ]:


train_left_edges, train_left_linked  = find_run_edges(train_X, edge='left')
train_right_edges, train_right_linked = find_run_edges(train_X, edge='right')
print('Found', len(train_left_edges), 'left edges and', len(train_right_edges), 'right edges.')


# Let's find the runs!

# In[ ]:


train_runs = find_runs(train_X, train_left_edges, train_right_edges)


# Now find the missing samples from test data

# In[ ]:


lost_samples = np.array([ i for i in range(len(train_X)) if i not in np.concatenate(train_runs) ])
print(lost_samples)
print(len(lost_samples))


# In[ ]:


find_run_edges(train_X[lost_samples], edge='left')[1][0]
lost_run = np.array(lost_samples[find_runs(train_X[lost_samples], [0], [5])[0]])
train_runs.append(lost_run)


# [](http://)As what have been said in Markus F's kernel, there is some train runs (4) having more than 1 surface. We need to separate it manually. In order to do this, we need to track the runs in which the mis-allocated series lies.

# In[ ]:


first_double_surface = 0
for i in range(0, len(train_runs)):
    for x in train_runs[i]:
        if x==821:
            print(i)
            first_double_surface = i


# In[ ]:


train_runs[first_double_surface]


# In[ ]:


new_train_runs = [821,  974,  328, 1548,  172,
        355,  957, 1481, 1046, 1650,  857,  724,  164, 1092, 1017, 1300,
       1212,  536,  531, 1032,  994, 1501,  588,  579, 1177,  812, 1333,
       1253]

train_runs[first_double_surface] = train_runs[first_double_surface][0:-len(new_train_runs)]


# In[ ]:


train_runs.append(np.array(new_train_runs))


# In[ ]:


second_double_surface = 0
for i in range(0, len(train_runs)):
    for x in train_runs[i]:
        if x==3055:
            print(i)
            second_double_surface = i


# In[ ]:


train_runs[second_double_surface]


# In[ ]:


new_train_runs2 = [3055, 3360, 3662,
       3780, 3663, 3091, 3769, 3175, 1957, 2712, 2063, 2708, 3139, 2722]

train_runs[second_double_surface] = train_runs[second_double_surface][0:-len(new_train_runs2)]


# In[ ]:


train_runs.append(np.array(new_train_runs2))


# In[ ]:


third_double_surface = 0
for i in range(0, len(train_runs)):
    for x in train_runs[i]:
        if x==2484:
            print(i)
            third_double_surface = i


# In[ ]:


train_runs[third_double_surface]


# In[ ]:


new_train_runs3 = [2484, 3062, 2290, 3517, 3293, 2651, 3767,
       2029, 2558, 3580, 1874, 3373, 2514, 2308, 3160, 3161, 3613, 2511,
       2469, 2990, 2780, 3756, 2376, 2616, 2540, 2039, 2219, 3743, 3198,
       2584, 2752, 2304, 2887, 2841, 3480, 2517, 3020, 3424, 2027, 2652,
       2648, 3433, 2359, 3392, 3164, 3798, 3642, 2713, 3405, 3673, 2369,
       3411, 3595, 2242, 2307, 1897, 2834, 2350, 3795, 2948, 1856, 3486,
       3353, 1966]

train_runs[third_double_surface] = train_runs[third_double_surface][0:-len(new_train_runs3)]


# In[ ]:


train_runs.append(np.array(new_train_runs3))


# In[ ]:


fourth_double_surface = 0
for i in range(0, len(train_runs)):
    for x in train_runs[i]:
        if x==3501:
            print(i)
            fourth_double_surface = i


# In[ ]:


train_runs[fourth_double_surface]


# In[ ]:


new_train_runs4 = [3501, 2785]
train_runs[fourth_double_surface] = train_runs[fourth_double_surface][0:-len(new_train_runs4)]
train_runs.append(np.array(new_train_runs4))


# After we remove the double surface runs, let's add this knowledge to our train_y. 

# In[ ]:


df_train_y = pd.DataFrame()
df_train_y['run_id'] = 0
df_train_y['run_pos'] = 0

for run_id in range(len(train_runs)):
    for run_pos in range(len(train_runs[run_id])):
        series_id = train_runs[run_id][run_pos]
        df_train_y.at[ series_id, 'run_id'  ] = run_id
        df_train_y.at[ series_id, 'run_pos' ] = run_pos

df_train_y.tail()


# Rename index as series_id.

# In[ ]:


df_train_y['index'] = df_train_y.index
df_train_y = df_train_y.sort_values('index')
df_train_y.rename(columns={'index':'series_id'}, inplace=True)
df_train_y['run_id'] = df_train_y['run_id'].apply(lambda x: int(x))
df_train_y['run_pos'] = df_train_y['run_pos'].apply(lambda x: int(x))
df_train_y.tail()


# Now let's separate train and test.

# In[ ]:


run_id_train = df_train_y['run_id'][0:3810].values
run_id_test = df_train_y['run_id'][3810:7626].values


# Because some runs is linked between train and test data, we can essentially use train's target to map our test's target.

# In[ ]:


data_target['run_id'] = run_id_train
mapping_leak = {}
for i in range(0,3810):
    cur_data = data_target.iloc[i]
    mapping_leak.update({cur_data['run_id']: cur_data['surface']})
    
unknown_run = []
ans_test = []
known_series = []
unknown_series = []

for i in range(0,3816):
    if run_id_test[i] in mapping_leak:
        ans_test.append(mapping_leak[run_id_test[i]])
        known_series.append(i)
    else:
        ans_test.append('unknown')
        unknown_series.append(i)
        unknown_run.append(run_id_test[i])


# As you can see, we automatically get more 69% of our test target classified correctly!

# In[ ]:


print("Number of known series:{}\nNumber of unknown series:{}\n".format(len(known_series),len(unknown_series)))


# In[ ]:


sub = pd.read_csv("{}/sample_submission.csv".format(RAW_DATA_PATH))
sub['surface'] = ans_test
sub.head()


# In[ ]:


sub['surface'].head(10)


# Then, I just combine these data with my best submission (LB 0.73). I keep the known series target values while replacing the unknown one with values from my best submission.

# In[ ]:


best = pd.read_csv('../input/best-73/submission_24.csv')
map_best_ans = {}

for i in range(0, best.shape[0]):
    map_best_ans.update({best.iloc[i]['series_id'] : best.iloc[i]['surface'] })


# In[ ]:


result = []
for i in range(0, sub.shape[0]):
    if (sub.surface[i] == 'unknown'):
        result.append(map_best_ans[i])
    else:
        result.append(sub.surface[i])


# In[ ]:


sub.surface = result
sub.to_csv('final_submission.csv', index=False)


# In[ ]:


sub.surface.value_counts()


# In[ ]:





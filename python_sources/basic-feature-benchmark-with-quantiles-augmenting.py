#!/usr/bin/env python
# coding: utf-8

# The aim of this kernel is to test (and share) the effect of augmenting training set, applying it to the initial kernel "Basic Feature Benchmark with Quantiles" [https://www.kaggle.com/andrekos/basic-feature-benchmark-with-quantiles] and considering first approach of "Basic Feature Benchmark" [https://www.kaggle.com/inversion/basic-feature-benchmark]
# 
# From the initial approach of dividing the training set in (n/150_00) segments, this solution **doubles the number of segments** under the following conditions:
# *  it avoids creating segments very similar to each other (with more than 50% of overlaping regions): each new segment shares 50% of 2 sequential segments (as in Figure 1 below); 
# *  it uses custom split iterator in order to avoid segments in train set which share regions with segments in validation set.
# 
# Figure 1: ![](https://i.imgur.com/rovjuNK_d.jpg?maxwidth=640&fidelity=medium)
# 
# Apparently this augmentation does not change the results significantly (not with the features currently used).
# 
# Not completely tested yet. Open to suggestions and contributions.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold


# In[ ]:


train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})


# In[ ]:


train.head()


# In[ ]:


# pandas doesn't show us all the decimals
pd.options.display.precision = 15


# In[ ]:


# much better!
train.head()


# In[ ]:


# Create a training file with simple derived features
rows = 150_000
# shift_step for augmented training set
shift_step = int(np.floor(rows / 2))
segments = int(np.floor(train.shape[0] / rows))
segments_augmented = 2*segments - 1

X_train = pd.DataFrame(index=range(segments_augmented), dtype=np.float64,
                       columns=['ave', 'std', 'max', 'min','q95','q99', 'q05','q01'])
y_train = pd.DataFrame(index=range(segments_augmented), dtype=np.float64,
                       columns=['time_to_failure'])


for segment in tqdm(range(segments)):
    for do_shift in [False,True]:        
        if(do_shift):
            shift = shift_step
            idx = segments + segment            
            if(segment==segments-1): #last segment would be incomplete for the shifted version
                continue
        else:
            shift = 0
            idx = segment
        
        seg = train.iloc[segment*rows+shift:segment*rows+shift+rows]

        x = seg['acoustic_data'].values
        y = seg['time_to_failure'].values[-1]

        y_train.loc[idx, 'time_to_failure'] = y

        X_train.loc[idx, 'ave'] = x.mean()
        X_train.loc[idx, 'std'] = x.std()
        X_train.loc[idx, 'max'] = x.max()
        X_train.loc[idx, 'min'] = x.min()
        X_train.loc[idx, 'q95'] = np.quantile(x,0.95)
        X_train.loc[idx, 'q99'] = np.quantile(x,0.99)
        X_train.loc[idx, 'q05'] = np.quantile(x,0.05)
        X_train.loc[idx, 'q01'] = np.quantile(x,0.01)

    


# In[ ]:


X_train.shape


# In[ ]:


# check new segments
X_train.loc[[0,1,2,3,4,0+segments,1+segments,2+segments,3+segments]]


# In[ ]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)


# In[ ]:


nFolds = 3
# custom kfold to remove segments in train set which share regions with validation set
customKFoldAvoidLeakToValidation = []
for train_id, valid_id in KFold(nFolds,shuffle=True).split(X_train):
    must_remove = []    
    for v in valid_id:       
        if(v>=segments):                
            must_remove.append(v-segments)
            must_remove.append(v-segments+1)                
        else:                
            must_remove.append(v+segments-1)
            must_remove.append(v+segments)                
    train_id = [t for t in train_id if t not in (must_remove)]
    customKFoldAvoidLeakToValidation.append((train_id, valid_id)) 


# In[ ]:


scorer = make_scorer(mean_absolute_error, greater_is_better=False)


# In[ ]:


parameters = [{ 'gamma': [0.6, 0.7, 0.8],
               'C': [2.35, 2.4, 2.45, 2.5],
              'nu': [0.85, 0.9, 0.95]}]

##best_params_
#parameters = [{'C': [2.35], 'gamma': [0.6], 'nu': [0.9]}]

reg1 = GridSearchCV(NuSVR(kernel='rbf', tol=0.01), parameters, cv = customKFoldAvoidLeakToValidation, scoring=scorer)
reg1.fit(X_train_scaled, y_train.values.flatten())
y_pred1 = reg1.predict(X_train_scaled)

print(reg1.best_params_)
print(reg1.best_score_)


# In[ ]:


parameters = [{'gamma': [0.06, 0.1, 0.08, 0.09], #np.logspace(-2, 2, 5)
               'alpha': [0.005, 0.01, 0.05]}]

#best_params_
#parameters = [{'alpha': [0.05], 'gamma': [0.06]}]

reg2 = GridSearchCV(KernelRidge(kernel='rbf'), parameters, cv = customKFoldAvoidLeakToValidation, scoring=scorer)
reg2.fit(X_train_scaled, y_train.values.flatten())
y_pred2 = reg2.predict(X_train_scaled)

print(reg2.best_params_)
print(reg2.best_score_)


# In[ ]:


plt.tight_layout()
f = plt.figure(figsize=(12, 6))
f.add_subplot(1,2, 1)
plt.scatter(y_train.values.flatten(), y_pred1)
plt.title('reg1', fontsize=20)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
f.add_subplot(1,2, 2)
plt.scatter(y_train.values.flatten(), y_pred2)
plt.title('reg2', fontsize=20)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
plt.show(block=True)


# In[ ]:


score1 = mean_absolute_error(y_train.values.flatten(), y_pred1)
print(f'Score1: {score1:0.3f}')
score2 = mean_absolute_error(y_train.values.flatten(), y_pred2)
print(f'Score2: {score2:0.3f}')
score3 = mean_absolute_error(y_train.values.flatten(), (0.5*y_pred1 + 0.5*y_pred2 ) )
print(f'Score3: {score3:0.3f}')


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')


# In[ ]:


X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)


# In[ ]:


for seg_id in X_test.index:
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    
    X_test.loc[seg_id, 'ave'] = x.mean()
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()
    X_test.loc[seg_id, 'q95'] = np.quantile(x,0.95)
    X_test.loc[seg_id, 'q99'] = np.quantile(x,0.99)
    X_test.loc[seg_id, 'q05'] = np.quantile(x,0.05)
    X_test.loc[seg_id, 'q01'] = np.quantile(x,0.01)


# In[ ]:


X_test_scaled = scaler.transform(X_test)
predictions_submit = 0.5*reg1.predict(X_test_scaled) + 0.5*reg2.predict(X_test_scaled) 
#remove non-positive predictions
predictions_submit = predictions_submit.clip(min=0)
submission['time_to_failure'] = predictions_submit
submission.to_csv('submission.csv')


# In[ ]:





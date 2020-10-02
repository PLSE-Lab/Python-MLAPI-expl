#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The aim of this notebook is to obtain the stack of optimized classifiers which maximise the accuracy in the train set.
# Then use this stack to predict the dependent variable in the test set.
# 
# In the previous notebook [Tactic 03. Hyperparameter optimization](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization),
# there are the predictions (for train and test) of the analised models.
# 
# The first thing to look is the correlations of the predictions.
# 
# ---
# 
# Credits to the Experts (Please like their kernels)  
# Paulo Pinto: [GMEAN of low correlation](https://www.kaggle.com/paulorzp/gmean-of-low-correlation-lb-0-952x)  
# [Nanashi](https://www.kaggle.com/jesucristo)

# In[ ]:


import pip._internal as pip
pip.main(['install', '--upgrade', 'numpy==1.17.2'])
import numpy as np
import pandas as pd

import glob
import re

import warnings
# warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.filterwarnings('ignore')

from itertools import combinations
from scipy.stats import chisquare
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from xgboost import XGBClassifier

from lwoku import get_accuracy

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# # Read predictions

# In[ ]:


tactic_03_results = pd.read_csv('../input/tactic-99-summary/tactic_03_results.csv', index_col='Id', engine='python')

model = []
S_test = []
S_train = []
for index, row in tactic_03_results.iterrows():
    model += [row['Model']]
    S_test += [pd.read_csv('../input/tactic-03-hyperparameter-optimization/submission_' + row['Model'] + '.csv', index_col=0, engine='python')]
    S_train += [pd.read_csv('../input/tactic-03-hyperparameter-optimization/train_' + row['Model'] + '.csv', index_col=0, engine='python')]


# In[ ]:


S_test = pd.concat(S_test, axis=1)
S_test.columns = model
S_test


# In[ ]:


S_train = pd.concat(S_train, axis=1)
S_train.columns = model
S_train


# In[ ]:


tactic_03_results[['Model', 'Score']]


# ## Drop low scored models

# In[ ]:


low_scored_models = tactic_03_results[['Model', 'Score']].query('Score < 0.7')['Model'].tolist()
print('Drop models: {}'.format(low_scored_models))
S_test.drop(low_scored_models, axis='columns', inplace=True)
S_train.drop(low_scored_models, axis='columns', inplace=True)


# # Correlations

# In[ ]:


def acc(p1, p2):
    return np.sum(p1 == p2, axis=0) / float(p1.shape[0])

corr = S_test.corr(method=acc)
f, ax = plt.subplots(figsize=(len(model), len(model)))
sns.heatmap(corr, cmap="Oranges", annot=True, fmt='.3f');


# In[ ]:


mean_corr = corr.mean()
mean_corr


# # 2nd level model

# In[ ]:


X = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')
y = X['Cover_Type'].copy()


# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = {
    'n_estimators': range(5, 15),
    'max_depth': [3, 4, 5],
    'learning_rate': [x/100 for x in range(5, 13)]
}
model = XGBClassifier(n_estimators=12, max_depth=3, learning_rate=0.08, random_state=0, n_jobs=-1)
clf = GridSearchCV(model, parameters, cv=5)
clf.fit(S_train, y)


# In[ ]:


from grid_search_utils import plot_grid_search, table_grid_search
plot_grid_search(clf)


# In[ ]:


clf.best_estimator_


# In[ ]:


results = pd.DataFrame(columns = ['Model combination',
                                  'Accuracy'])

model = XGBClassifier(n_estimators=8, max_depth=4, learning_rate=0.09, random_state=0, n_jobs=-1)

import time
t0 = time.time()

# Loop from single model to n models:
for n in range(1, len(mean_corr) + 1):
    # Get combinations for n models
    n_model_combinations = combinations(mean_corr.index, n)
    # Loop for all the combinations of n models
    for i in list(n_model_combinations):
        n_model_combination = list(i)
        accuracy = get_accuracy(model, S_train[n_model_combination], y, cv=3)
        t1 = time.time()
#         print('Model combination {}: {} in {} seconds'.format(n_model_combination, accuracy, (t1 - t0)))
        t0 = t1
        results = results.append({
            'Model combination': n_model_combination,
            'Accuracy': accuracy
        }, ignore_index = True)

results = results.sort_values('Accuracy', ascending=False).reset_index(drop=True)


# In[ ]:


results = results.sort_values('Accuracy', ascending=False).reset_index(drop=True)
results.to_csv('results.csv', index=True, index_label='Id')
results[0:10]


# # Submit
# For the 10 first model combinations of the results, the model is fitted, predicted and submitted.

# In[ ]:


for index, row in results[0:10].iterrows():
    model.fit(S_train[row['Model combination']], y)
    y_test_pred = pd.Series(model.predict(S_test[row['Model combination']]), index=S_test.index)
    name = '_'.join(row['Model combination'])
    y_test_pred.to_csv('submission_' + name + '.csv', header=['Cover_Type'], index=True, index_label='Id')


#!/usr/bin/env python
# coding: utf-8

# My methodology is as follows:
#  1. Clean the text using my [Tweet Cleaner](https://www.kaggle.com/jdparsons/tweet-cleaner) notebook
#  2. Send the clean text to GPT-2 using my [GPT-2: fake real disasters](https://www.kaggle.com/jdparsons/gpt-2-fake-real-disasters-data-augmentation) notebook. This generates similar tweets with the same label, which doubles the size of my training data.
#  3. In this notebook:
#   * I, um, use USE (Universal Sentence Encoder) to convert the tweets into 512 dimensional vectors
#   * Perform a grid search on many Light GBM parameters to find the best ones
#   * Use K-Fold Cross Validation to train multiple LGB models on different random splits of the data. The prediction function averages the votes into a final answer.
# 
# My best score was 0.82413 with the previously mentioned preprocessing notebooks and the following Light GBM parameters:
#  * K-fold DV=5
#  * 'feature_fraction': 0.9, 'lambda_l1': 1.0, 'lambda_l2': 0.001, 'learning_rate': 0.05, 'max_depth': -1, 'n_estimators': 128, 'num_leaves': 64,
# 
# My next experiment will be to send my GPT-2 augmented data to Google's AutoML and see how it scores compared to this hand-crafted approach. I'll publish that notebook and update this description when I have the results. **If you have any ideas for improving this notebook, please let me know in the comments!**
# 
# Code linted via http://pep8online.com/ and https://yapf.now.sh/ to follow the Google python style guide (mostly).

# In[ ]:


import os
import sys
import torch
import random
import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, KFold
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# set pandas preview to use full width of browser to see more of the column data
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


# In[ ]:


show_files = False

if show_files is True:
    # helper method to quickly see the file paths of your imported data
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))


# # Load data

# In[ ]:


# uncomment these lines to try different types of preprocessed data

#train_file_path = '../input/nlp-getting-started/train.csv'
#train_file_path = '../input/tweet-cleaner/train_df_clean.csv'
train_file_path = '../input/gpt-2-fake-real-disasters-data-augmentation/train_df_combined.csv'

#test_file_path = '../input/nlp-getting-started/test.csv'
test_file_path = '../input/tweet-cleaner/test_df_clean.csv'

train_full = pd.read_csv(train_file_path)
train = train_full[['text']]
target = train_full[['target']]
test = pd.read_csv(test_file_path)
test = test[['id', 'text']]


# # Create embeddings

# In[ ]:


# from https://www.kaggle.com/denychaen/tweets-simple-baseline-tfembed-lgb
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
X_train_embeddings = embed(train['text'].values)
X_test_embeddings = embed(test['text'].values)


# # Grid Search

# In[ ]:


do_grid_search = False

if do_grid_search is True:
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X_train_embeddings['outputs']),
        target,
        test_size=0.2,
        random_state=20)

    estimator = lgb.LGBMClassifier()

    # Try your own ranges here, you may find better ones than me!
    param_grid = {
        'n_estimators': [64, 200],
        'num_leaves': [31, 64],
        'learning_rate': [0.05, 0.1],
        'feature_fraction': [0.8, 1.0],
        'max_depth': [-1],
        'lambda_l1': [0, 1],
        'lambda_l2': [0, 1],
    }

    gridsearch = GridSearchCV(
        estimator, param_grid, refit=True, verbose=0, cv=5)

    gridsearch.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'],
        early_stopping_rounds=10,
        verbose=0)

    print('Best parameters found by grid search are:', gridsearch.best_params_)

    Y_pred = gridsearch.best_estimator_.predict(X_test)

    print(metrics.classification_report(y_test, Y_pred, digits=3),)
    print(metrics.confusion_matrix(y_test, Y_pred))

    # if we just guessed the most common class (0) for every prediction
    print('The null acccuracy is:',
          max(y_test['target'].mean(), 1 - y_test['target'].mean()))


# # Train final model

# In[ ]:


params = {
    'objective': 'binary',
    'feature_fraction': 0.9,
    'lambda_l1': 1.0,
    'lambda_l2': 0.001,
    'learning_rate': 0.05,
    'max_depth': -1,
    'n_estimators': 128,
    'num_leaves': 64,
    'metric': 'binary_logloss',
    'random_seed': 42
}

kf = KFold(n_splits=5, random_state=42)
models = []

X_train = pd.DataFrame(data=np.array(X_train_embeddings['outputs']))
y_train = target

for train_index, test_index in kf.split(X_train):
    train_features = X_train.loc[train_index]
    train_target = y_train.loc[train_index]

    test_features = X_train.loc[test_index]
    test_target = y_train.loc[test_index]

    d_training = lgb.Dataset(
        train_features, label=train_target, free_raw_data=False)
    d_test = lgb.Dataset(test_features, label=test_target, free_raw_data=False)

    evals_result = {}  # to record eval results for plotting

    model = lgb.train(
        params,
        train_set=d_training,
        num_boost_round=200,
        valid_sets=[d_training, d_test],
        verbose_eval=50,
        early_stopping_rounds=5,
        evals_result=evals_result)

    # https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
    print('Plotting metrics recorded during training...')
    ax = lgb.plot_metric(evals_result, metric='binary_logloss')
    plt.show()

    models.append(model)


# # Prediction method

# In[ ]:


# https://www.kaggle.com/rohanrao/ashrae-half-and-half
# averages the votes of each model, then returns the majority vote
def get_predictions(data):

    results = []
    for model in models:
        if results == []:
            results = model.predict(
                data, num_iteration=model.best_iteration) / len(models)
        else:
            results += model.predict(
                data, num_iteration=model.best_iteration) / len(models)

    # if the average value is less than .5, then the majority vote was 0, otherwise it was 1
    results = [int(round(x)) for x in results]

    return results


# # Generate submission

# In[ ]:


preds = get_predictions(X_test_embeddings['outputs'])

ssub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

ssub["target"] = preds
ssub.to_csv("submission.csv", index=False)


# In[ ]:





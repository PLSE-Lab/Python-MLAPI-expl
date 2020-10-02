#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Let's import the necessary libraries

# In[ ]:


# Main Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Utility libs
from tqdm import tqdm
import time
import datetime
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from copy import deepcopy
import pprint
import shap
import os

# You might have to do !pip install catboost
# If you don't have it on your local machine
# nevertheless Kaggle runtimes come preinstalled with CatBoost
import catboost

from pathlib import Path
data_dir = Path('../input/data-science-bowl-2019')
os.listdir(data_dir)


# # Reading the data

# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/data-science-bowl-2019/train.csv')\nlabels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')\ntest = pd.read_csv('../input/data-science-bowl-2019/test.csv')\nspecs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')\nsample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')")


# # Cleaning and extracting features

# How we are going to extract features is through cumulating various features present in the data as grouped by unique sessions. 
# 
# It is being modeled as how many games a user has played, or clips they have watched etc. up until they take an assessment. 
# 
# This truncated history is condensed into one row of the prepared data.
# 
# Let's see how this works out!

# In[ ]:


# thank you for this data preparation function 
# https://www.kaggle.com/mhviraf/a-new-baseline-for-dsb-2019-catboost-model
def get_data(user_sample, test_set=False):
    last_activity = 0
    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy=0
    accumulated_correct_attempts = 0 
    accumulated_uncorrect_attempts = 0 
    accumulated_actions = 0
    counter = 0
    durations = []
    for i, session in user_sample.groupby('game_session', sort=False):
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        if test_set == True:
            second_condition = True
        else:
            if len(session)>1:
                second_condition = True
            else:
                second_condition= False
            
        if (session_type == 'Assessment') & (second_condition):
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            features = user_activities_count.copy()
            features['session_title'] = session['title'].iloc[0] 
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts 
            accumulated_uncorrect_attempts += false_attempts
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1

            features.update(accuracy_groups)
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            features['accumulated_actions'] = accumulated_actions
            accumulated_accuracy_group += features['accuracy_group']
            accuracy_groups[features['accuracy_group']] += 1
            if test_set == True:
                all_assessments.append(features)
            else:
                if true_attempts+false_attempts > 0:
                    all_assessments.append(features)
                
            counter += 1
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type
    if test_set:
        return all_assessments[-1] 
    return all_assessments


# In[ ]:


list_of_user_activities = list(set(train['title'].value_counts().index).union(set(test['title'].value_counts().index)))
activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

train['title'] = train['title'].map(activities_map)
test['title'] = test['title'].map(activities_map)
labels['title'] = labels['title'].map(activities_map)
win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
win_code[activities_map['Bird Measurer (Assessment)']] = 4110

train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp'] = pd.to_datetime(test['timestamp'])
compiled_data = []
for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=17000):
    compiled_data += get_data(user_sample)
new_train = pd.DataFrame(compiled_data)
del compiled_data
print("Train Data:")
new_train.head()


# So as we can see, this is what we have gathered. Let's plot these features and see if there are trends within these features as grouped by their accuracy groups.
# 
# Let's begin.

# In[ ]:


def plot_count(feature, title, df, size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()


# # Now let's plot and visualize each feature

# In[ ]:


plt.figure(figsize=(16,6))
col = 'Clip'
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# We can see here that there is a positively skewed gaussian.
# This is fine. 

# In[ ]:


plt.figure(figsize=(16,6))
col = 'Activity'
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# Another, more prominent positively skewed gaussian.

# In[ ]:


plt.figure(figsize=(16,6))
col = 'Assessment'
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
col = 'Game'
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[ ]:


plot_count('session_title', 'Session Titles', new_train, 3)


# So the count of session_titles is not all that variable afterall.

# In[ ]:


plt.figure(figsize=(16,6))
col = 'accumulated_correct_attempts'
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
col = 'accumulated_uncorrect_attempts'
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
col = 'duration_mean'
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
col = 'accumulated_accuracy'
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# This is an interesting plot, we can see that a lot of users with accuracy group 3 have also a high accumulated accuracy of 0.
# 
# I will leave it up to you guys to find more inferences. 
# Let me know something in the comments if I am missing out on it. 

# In[ ]:


plot_count('accuracy_group', 'Accuracy Groups', new_train, 2)


# In[ ]:


plt.figure(figsize=(16,6))
col = 0
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
col = 1
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
col = 2
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
col = 3
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
col = 'accumulated_accuracy_group'
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
col = 'accumulated_actions'
_accuracy_groups = new_train.accuracy_group.unique()
plt.title("Distribution of %s values (grouped by accuracy group) in the data"%col)
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = new_train.loc[new_train.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df[col], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# # Let's try a Gaussian Mixture Model Classification on these features and find covariances between features

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.externals.six.moves import xrange
import matplotlib as mpl
from itertools import combinations
from sklearn.mixture import GaussianMixture

colors = ['navy', 'turquoise', 'darkorange', 'yellow']
feat1 = 'Game'
feat2 = 'Assessment'

def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold()
skf.get_n_splits(new_train.drop(['accuracy_group'], axis=1), new_train['accuracy_group'])
# Only take the first fold.
for train_index, test_index in skf.split(new_train.drop(['accuracy_group'], axis=1), new_train['accuracy_group']):
    X = new_train.drop(['accuracy_group'], axis=1)
    y = new_train['accuracy_group']
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    n_classes = len(np.unique(y_train))

    # Try GMMs using different types of covariances.
    estimators = {cov_type: GaussianMixture(n_components=n_classes,
              covariance_type=cov_type, max_iter=100, random_state=0)
              for cov_type in ['spherical', 'diag', 'tied', 'full']}

    n_estimators = len(estimators)

    plt.figure(figsize=(3 * n_estimators // 2, 6))
    plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                        left=.01, right=.99)


    for index, (name, estimator) in enumerate(estimators.items()):
        # Since we have class labels for the training data, we can
        # initialize the GMM parameters in a supervised manner.
        estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                        for i in range(n_classes)])

        # Train the other parameters using the EM algorithm.
        estimator.fit(X_train)

        h = plt.subplot(2, n_estimators // 2, index + 1)
        make_ellipses(estimator, h)
        
        for n, color in enumerate(colors):
            data = new_train.loc[new_train['accuracy_group'] == n]
            plt.scatter(data[feat1], data[feat2], s=0.8, color=color,
                        label=n)
            # Plot the test data with crosses
        for n, color in enumerate(colors):
            data = X_test[y_test == n]
            plt.scatter(data[feat1], data[feat2], marker='x', color=color)
    break
plt.xticks(())
plt.yticks(())
plt.title(name)
plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))
plt.show()


# # Thank You
# 
# Next I will be posting a baseline Random Forest and AdaBoost model using these features and perhaps we can extract and engineer new features.
# 
# ## Feel free to ask any questions, I'll be happy to answer.
# 
# ## Please leave an upvote for appreciation if you thought this notebook was helpful so others can see it as well. :) 
# 
# # Happy Kaggling

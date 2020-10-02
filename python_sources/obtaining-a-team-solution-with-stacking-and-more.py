#!/usr/bin/env python
# coding: utf-8

# # Stacking
# 
# A very popular approach for obtaining a joint team solution is to form a linear combination of the individual models (with weights selected via 'common sense') and hope that it will improve the score (it usually does). This notebook shows a more systematic approach to obtaining a joint model with a help of `mlxtend.StackingCVClassifier`. In particular, it allows to automate the process of assessing different model mixtures with cross-validation and helps to select the most promising of them.
# 
# So, let's imagine a situation that two team members (Alice and Bob) came up with different solutions - Alice built a CatBoost model, and Bob crafted a logit one. Now they want to build a merged classifier.
# 

# In[ ]:


import os

import pandas as pd
import numpy as np
import scipy.sparse

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

from mlxtend.feature_selection import ColumnSelector
from mlxtend.classifier import StackingCVClassifier

from catboost import CatBoostClassifier


# In[ ]:


RANDOM_SEED = 42

DATA_DIR = '../input/mlcourse-dota2-win-prediction/'

train_data = pd.read_csv(os.path.join(DATA_DIR, 'train_features.csv'), index_col='match_id_hash')
test_data = pd.read_csv(os.path.join(DATA_DIR, 'test_features.csv'), index_col='match_id_hash')
y_train = pd.read_csv(os.path.join(DATA_DIR, 'train_targets.csv'), index_col='match_id_hash')['radiant_win'].map({True: 1, False:0})


# # Feature Engineering
# 
# Obtaining a competitive submission is not a goal of this kernel, therefore, we use only the most basic features from csv, without even touching jsonl. There are many quality kernels that contain precious ideas about useful features. Some of them are:
# 
#  - https://www.kaggle.com/artgor/dota-eda-fe-and-models
#  - https://www.kaggle.com/karthur10/data-extraction-from-json-additional-features
#  - https://www.kaggle.com/marketneutral/dota-2-coordinates-eda-animated
# 

# In[ ]:


# Features to compare hero/team efficiency
hero_efficiency_features = [
 'kills',
 'deaths',
 'assists',
 'denies',
 'gold',
 'lh',
 'xp',
 'health',
 'max_health',
 'max_mana',
 'level',
 'creeps_stacked',
 'camps_stacked',
 'rune_pickups',
 'firstblood_claimed',
 'teamfight_participation',
 'towers_killed',
 'roshans_killed',
 'obs_placed',
 'sen_placed',
]    

# Features 
timed_efficiency_features = [
 'kills',
 'deaths', 
 'assists',
 'denies',
 'gold',
 'lh',
 'xp',
 'creeps_stacked',
 'camps_stacked',
 'rune_pickups',
]    

HERO_CLASSES = 121

def make_rel_features(df, rel_type='absolute'):
    relative_features = pd.DataFrame(index=df.index)
    for feature in hero_efficiency_features:
        # Radiant team 
        columns = ['r' + str(i) + '_' + feature for i in range(1,6)]
        radiant_sum = df[columns].sum(axis=1)
        # Dire team
        columns = ['d' + str(i) + '_' + feature for i in range(1,6)]
        dire_sum = df[columns].sum(axis=1)
        if rel_type == 'absolute':
            relative_features['diff_'+feature+'_'+rel_type] = radiant_sum - dire_sum
        elif rel_type == 'ratio':
            relative_features['diff_'+feature+'_'+rel_type] = (radiant_sum + 1) / (dire_sum + 1)
        else:
            raise Exception('Unknown rel_type: ' + rel_type)            
    return relative_features

def make_timed_features(df, team_char):
    timed_features = pd.DataFrame(index=df.index)
    for feature in timed_efficiency_features:
        columns = [team_char + str(i) + '_' + feature for i in range(1,6)]
        feat_sum = df[columns].sum(axis=1)
        timed_features['permin_'+feature+'_'+team_char] = feat_sum / ((df.game_time + 1) / 60) 
    return timed_features

def make_hero_features(df, team_prefix):
    hero_id_features = [team_prefix + str(i) + '_hero_id' for i in range(1, 6)]
    heroes_flat = df[hero_id_features].values.flatten()
    heroes_sparse = scipy.sparse.csr_matrix(([1] * heroes_flat.shape[0],
                                             heroes_flat,
                                             range(0, heroes_flat.shape[0]  + 1, 5)))
    if heroes_sparse.shape[1] < HERO_CLASSES:
        heroes_sparse = scipy.sparse.hstack([heroes_sparse, 
                                             scipy.sparse.csr_matrix((heroes_sparse.shape[0], HERO_CLASSES-heroes_sparse.shape[1]))])
    # An attentive reader here might question: "Why on Earth do we create a sparse matrix
    # and then convert it to a dense DataFrame?" I have to agree with you. However, this
    # (inefficiency) allows me to make the following code more concise.
    heroes = pd.DataFrame(heroes_sparse.todense(), 
                          columns=['hero_%d' % (i,) for i in range(1, HERO_CLASSES+1)], 
                          index=df.index)    
    return heroes


# # Alice: CatBoost Model
# 
# To make the situation more typical, we will create slightly different featuresets for Alice and Bob (after all, they most likely did their feature extraction and feature engineering independently).

# In[ ]:


def make_alice_features(df):
    game_features = df[['game_time', 
                        'game_mode', 
                        'objectives_len', 
                        'chat_len']]
    relative_features = make_rel_features(df, 'ratio')
    timed_features_radiant = make_timed_features(df, 'r')
    timed_features_dire = make_timed_features(df, 'd')
    radiant_heroes = make_hero_features(df, 'r')
    dire_heroes = make_hero_features(df, 'r')
    dire_heroes.columns = dire_heroes.columns + '_dire'
    return pd.concat([game_features,
                      relative_features,
                      timed_features_radiant,
                      timed_features_dire,
                      radiant_heroes,
                      dire_heroes], axis=1)
    
alice_train_features = make_alice_features(train_data)
alice_train_features.shape


# In[ ]:


alice_model = CatBoostClassifier(iterations=1000, 
                                 random_state=RANDOM_SEED, 
                                 silent=True)

alice_scores = cross_val_score(alice_model, 
                               alice_train_features, 
                               y_train, 
                               cv=5, 
                               scoring='roc_auc')

print('Alice (CatBoost) scores:', alice_scores)
print('Alice (CatBoost) mean ROC AUC: %.4f (std dev: %.6f)' % 
      (alice_scores.mean(), alice_scores.std()))


# # Bob: Logistic Regression Model

# In[ ]:


def make_bob_features(df):
    game_features = df[['game_time', 
                        'game_mode', 
                        'objectives_len', 
                        'chat_len']]
    relative_features = make_rel_features(df)
    timed_features_radiant = make_timed_features(df, 'r')
    team_features = make_hero_features(df, 'r') - make_hero_features(df, 'd')
    return pd.concat([game_features,
                      relative_features,
                      timed_features_radiant,
                      team_features], axis=1)
    
bob_train_features = make_bob_features(train_data)
bob_train_features.shape


# For the logistic regression Bob had to do some preprocessing of the data: OHE of the categorical feature `game_mode` and standard scaling of the most features.

# In[ ]:


def make_indexes(df, column_names):
    return [df.columns.get_loc(c) for c in column_names if c in df]


# In[ ]:


# Prepare lists with indexes of columns that require different approaches in transformation.
# Though ColumnSelector accepts feature names (when processing a DataFrame), later we will
# reuse this pipeline in a situation where column names won't be available
game_mode_idx = make_indexes(bob_train_features, ['game_mode'])
hero_idx = make_indexes(bob_train_features, ['hero_%d' % (i,) for i in range(1, HERO_CLASSES+1)])
other_idx = make_indexes(bob_train_features, 
                         list(filter(lambda x: x.startswith('diff_') or x.startswith('permin_'), 
                                     bob_train_features.columns)))
# Categories definition for the game_mode feature
game_mode_categories = [sorted(list(bob_train_features.game_mode.unique()))]

# Data processing. The resulting dataset for the logistic regression is
# collected from the one-hot encoded game_mode, untouched hero features 
# and the rest features scaled
lr_features = FeatureUnion([   
    # OHE for game mode
    ('game_mode', Pipeline([('select_game_mode', 
                                ColumnSelector(cols=game_mode_idx)),
                             ('ohe_game_mode', 
                                OneHotEncoder(categories=game_mode_categories))])),
    # Leave hero features as they are
    ('hero_features', ColumnSelector(cols=hero_idx)),
    # Standard scaling for the rest
    ('other_features', Pipeline([('select_other_features', 
                                      ColumnSelector(cols=other_idx)),
                                 ('standard_scaling', 
                                      StandardScaler())]))
])

bob_pipeline = Pipeline([('prepare_features', lr_features),
                        ('logit', LogisticRegression(C=1, 
                                      random_state=RANDOM_SEED, 
                                      solver='lbfgs', 
                                      max_iter=500))])

bob_scores = cross_val_score(bob_pipeline, 
                             bob_train_features, 
                             y_train, 
                             cv=5, 
                             scoring='roc_auc')

print('Bob (logit) scores:', bob_scores)
print('Bob (logit) mean ROC AUC: %.4f (std dev: %.6f)' % 
          (bob_scores.mean(), bob_scores.std()))


# # Model Stacking
# 
# To implement stacking of the boosting and logit models Alice and Bob decided to use a very handy class `StackingCVClassifier` from `mlxtend` package. It splits the training set into a specified number of folds ($N_F$, by default $N_F=2$), trains each of the first-level classifiers on all the folds except one and then predicts values for that fold. After $N_F$ repetitions of this process a detaset with first-level model predictions (so-called metafeatures) for each training instance is obtained. Then, this metafeature dataset is used to train the meta-classifier. For more details, please refer to the [documentation](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/).
# 
# In terms of scikit-learn, `StackingCVClassifier` is an estimator with an ordinary `fit(X, y)` method, that means that Alice and Bob have to provide only one training dataset to it, but they have two, with different sets of features.
# 
# The simplest solution is to 'horizontally' concatenate two datasets and then use `ColumnSelector` instances for feeding different models with the respective data prepared for them.
# 

# ## Prepare a 'Glued' Dataset (Optional)
# 
# *NOTE: This is an optional step! If you apply several different classification models to the same features, you can account for differences in feature representation required for the models at the preprocessing stages of the respective pipelines. *
# 

# In[ ]:


# Glue input data for different classifiers into one featureset
bob_train_features.columns = bob_train_features.columns + '_bob'
train_features = pd.concat([alice_train_features, bob_train_features], axis=1)
train_features.shape


# ## Define and Train a Stacking Classifier

# In[ ]:


## Define a pipeline for each first-level classifier

# Features for the Alice's model (CatBoost) are 
# the first alice_train_features.shape[1] columns
alice_col_subset = tuple(range(0, alice_train_features.shape[1]))
alice_pipe = Pipeline([('catboost features', 
                             ColumnSelector(cols=alice_col_subset)),
                       ('catboost', alice_model)])

# Features for the Bob's model (logit) are 
# columns from alice_train_features.shape[1] to the end
bob_col_subset = tuple(range(alice_train_features.shape[1], 
                             train_features.shape[1]))
bob_pipe = Pipeline([('logit features', 
                             ColumnSelector(cols=bob_col_subset)),
                     ('logit', bob_pipeline)])

## Compose a stacking classifier
sclf = StackingCVClassifier(
           classifiers=[alice_pipe,
                        bob_pipe], 
           meta_classifier=LogisticRegression(random_state=RANDOM_SEED, 
                                              solver='lbfgs'),
           use_probas=True,  # if False, meta-features will
                             # be formed from plain predictions of the
                             # first-level classifiers
           cv=3, # This is the number of splits used to
                 # build the meta-feature dataset
           random_state=RANDOM_SEED,
           verbose=1)

sclf.fit(train_features, y_train)


# Please note, that training `StackingCVClassifier` for $K$ first-level classifiers requires fitting $N_F * K + 1$ models. If we want to estimate the quality of our stacking solution with `cross_val_score`, this number should be multiplied by the number of folds in our cross-validation scheme. 
# 
# However, if there is no shortage of time and computing resources, `StackingCVClassifier` goes fine with `GridSearchCV`, allowing to experiment with different values of hyperparameters of the meta-classifier as well as first-level classifiers. 
# 

# ## Accessing the parameters of classifers
# 
# After fitting the stacked classifier we can access the fitted first-level models:
# 

# In[ ]:


sclf.clfs_[0]


# As well as the fitted meta-classifier:

# In[ ]:


sclf.meta_clf_


# ## ROC AUC Comparison
# 
# But Alice and Bob are mostly interested in checking if their stacked solution is better than individual ones:
# 

# In[ ]:


sclf = StackingCVClassifier(
           classifiers=[alice_pipe,
                        bob_pipe], 
           meta_classifier=LogisticRegression(random_state=RANDOM_SEED, 
                                              solver='lbfgs'),
           use_probas=True,  # if False, meta-features will
                             # be formed from plain predictions of the
                             # first-level classifiers
           cv=3, # This is the number of splits used to
                 # build the meta-feature dataset
           random_state=RANDOM_SEED,
           verbose=0)

team_scores = cross_val_score(sclf, 
                              train_features, 
                              y_train, 
                              cv=5, 
                              scoring='roc_auc')

print('Team (stacked) scores:', team_scores)
print('Team (stacked) mean ROC AUC: %.4f (std dev: %.6f)' % 
          (team_scores.mean(), team_scores.std()))


# In[ ]:


team_scores > alice_scores


# In[ ]:


team_scores > bob_scores


# ![image.png](attachment:image.png)

# Alice and Bob are happy with their joint solution and ready to submit it (note, however, that it is not quaranteed at all that their stacked solution would be better than both of the individual solutions, especially if the sets of features of the models are almost the same).

# # Extra
# 
# In this toy example, datasets used by each of the teammates intentionally missed some interesting features present in the dataset of the other. But what if Alice and Bob worked more closely and exchanged the features they found useful? Would their 'plain' models trained with more useful features surpass the stacked solution?
# 
# To answer this question we will construct a merged dataset including the most promicing features present either in Alice's or in Bob's original datasets. Then we will adapt the initial models of Alice and Bob to use this dataset.

# In[ ]:


def make_joint_features(df):
    game_features = df[['game_time', 'game_mode', 'objectives_len', 'chat_len']]
    relative_features_abs = make_rel_features(df, 'absolute')
    relative_features_ratio = make_rel_features(df, 'ratio')
    timed_features_radiant = make_timed_features(df, 'r')
    timed_features_dire = make_timed_features(df, 'd')
    team_features = make_hero_features(df, 'r') - make_hero_features(df, 'd')
    return pd.concat([game_features,
                      relative_features_abs,
                      relative_features_ratio,
                      timed_features_radiant,
                      timed_features_dire,
                      team_features], axis=1)
    
joint_train_features = make_joint_features(train_data)
joint_train_features.shape


# ## Alice's Model on the Joint Dataset

# In[ ]:


alice_model_joint = CatBoostClassifier(iterations=1000, 
                                       random_state=RANDOM_SEED, 
                                       silent=True)

alice_scores_joint = cross_val_score(alice_model_joint, 
                                     joint_train_features, 
                                     y_train, 
                                     cv=5, 
                                     scoring='roc_auc')

print('Alice (CatBoost) scores:', alice_scores_joint)
print('Alice (CatBoost) mean ROC AUC: %.4f (std dev: %.6f)' 
          % (alice_scores_joint.mean(), alice_scores_joint.std()))


# ## Bob's Model on the Joint Dataset

# In[ ]:


game_mode_idx = make_indexes(joint_train_features, ['game_mode'])
hero_idx = make_indexes(joint_train_features, ['hero_%d' % (i,) for i in range(1, HERO_CLASSES+1)])
other_idx = make_indexes(joint_train_features, 
                         list(filter(lambda x: x.startswith('diff_') or x.startswith('permin_'), 
                                     joint_train_features.columns)))
# Categories definition for the game_mode feature
game_mode_categories = [sorted(list(joint_train_features.game_mode.unique()))]

# Data processing. The resulting dataset for the logistic regression is
# collected from the one-hot encoded game_mode, untouched hero features 
# and the rest features scaled
lr_features = FeatureUnion([   
    # OHE for game mode
    ('game_mode', Pipeline([('select_game_mode', 
                                ColumnSelector(cols=game_mode_idx)),
                             ('ohe_game_mode', 
                                OneHotEncoder(categories=game_mode_categories))])),
    # Leave hero features as they are
    ('hero_features', ColumnSelector(cols=hero_idx)),
    # Standard scaling for the rest
    ('other_features', Pipeline([('select_other_features', 
                                      ColumnSelector(cols=other_idx)),
                                 ('standard_scaling', 
                                      StandardScaler())]))
])

bob_pipeline_joint = Pipeline([('prepare_features', lr_features),
                               ('logit', LogisticRegression(C=1, 
                                             random_state=RANDOM_SEED, 
                                             solver='lbfgs', 
                                             max_iter=500))])

bob_scores_joint = cross_val_score(bob_pipeline_joint, 
                                   joint_train_features, 
                                   y_train, 
                                   cv=5, 
                                   scoring='roc_auc')

print('Bob (logit) scores:', bob_scores_joint)
print('Bob (logit) mean ROC AUC: %.4f (std dev: %.6f)' % 
          (bob_scores_joint.mean(), bob_scores_joint.std()))


# We can see, that in this particular example, after the information exchange about features each individual model were able to achieve roughly the same prediction quality on cross-validation than the stacked model (a bit surprisingly, logistic regression model even slightly surpassed all other models, but the margin is unlikely significant).
# 
# Certainly, this example shouldn't be viewed as a general rule, but it supports the idea that early teaming and tight collaboration on solution ideas could be beneficial not only socially, but also in terms of solution quality.
# 
# Finally, let's make a stack of the improved models and see what happens. Note, that now we do not need to add `ColumnSelector`s to the stacked pipelines, as both models are based on the same features.
# 

# In[ ]:


sclf2 = StackingCVClassifier(
            classifiers=[alice_model_joint,
                         bob_pipeline_joint], 
            meta_classifier=LogisticRegression(random_state=RANDOM_SEED, 
                                               solver='lbfgs'),
            use_probas=True,  # if False, meta-features will
                              # be formed from plain predictions of the
                              # first-level classifiers
            cv=3, # This is the number of splits used to
                  # build the meta-feature dataset
            random_state=RANDOM_SEED,
            verbose=0)

team_scores_joint = cross_val_score(sclf2, 
                                    joint_train_features, 
                                    y_train, 
                                    cv=5, 
                                    scoring='roc_auc')

print('Team (stacked) scores:', team_scores_joint)
print('Team (stacked) mean ROC AUC: %.4f (std dev: %.6f)' % 
          (team_scores_joint.mean(), team_scores_joint.std()))


# In[ ]:


team_scores_joint > alice_scores_joint


# In[ ]:


team_scores_joint > bob_scores_joint


# In[ ]:


team_scores_joint > team_scores


# Interestingly enough, even after merging their datasets, they could have been able to further improve their results on cross-validation by stacking their models.
# 
# Let's make a table summarizing the results:

# In[ ]:


results = pd.DataFrame([[alice_scores.mean(), alice_scores.std(), 
                         alice_scores_joint.mean(), alice_scores_joint.std()],
                        [bob_scores.mean(), bob_scores.std(),
                         bob_scores_joint.mean(), bob_scores_joint.std()],
                        [team_scores.mean(), team_scores.std(),
                         team_scores_joint.mean(), team_scores_joint.std()]],
                      index=['Alice', 'Bob', 'Team'])
results.columns = ['Mean (Individual)', 'Std (Individual)',
                   'Mean (Merged)', 'Std (Merged)']

results


# # Acknowledgement
# 
# I'd like to thank my teammate Tatiana Glazkova ([@panikads](https://www.kaggle.com/panikads)) for showing me the stacking feature of `mlxtend` and laboriously implementing our joint solution with it during the competition. 
# 

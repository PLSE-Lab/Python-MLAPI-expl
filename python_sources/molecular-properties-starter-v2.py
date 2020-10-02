#!/usr/bin/env python
# coding: utf-8

# # Predicting Molecular Properties
# Can you measure the magnetic interactions between a pair of atoms?
# 
# In this competition, you will develop an algorithm that can predict the magnetic interaction between two atoms in a molecule (i.e., the `scalar_coupling_constant`).
#  https://www.kaggle.com/c/champs-scalar-coupling/overview/description. 
#  
# ## General information
# * based on : https://www.kaggle.com/robikscube/exploring-molecular-properties-data
# * Forked/based on: https://www.kaggle.com/artgor/molecular-properties-eda-and-models
# * https://www.kaggle.com/robikscube/exploring-molecular-properties-data
# * https://www.kaggle.com/tarunpaparaju/champs-competition-chemistry-background-and-eda
# 
# * Feature engineering brute force: https://www.kaggle.com/artgor/brute-force-feature-engineering
# * https://www.kaggle.com/adrianoavelar/bond-calculation-lb-0-82  , Chemical Bond Calculation 
# 
# * https://www.kaggle.com/buchan/a-neural-network-approach  (Includes covalent bond calc - https://www.kaggle.com/adrianoavelar/bond-calculation-lb-0-82 - Important feature!)
# * https://www.kaggle.com/borisdee/predicting-mulliken-charges-with-acsf-descriptors  - Using external libraries + calc the features given for the train
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pylab as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GroupKFold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# plt.style.use('ggplot')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]


# # The Data
# 
# In this competition, you will be predicting the scalar_coupling_constant between atom pairs in molecules, given the two atom types (e.g., C and H), the coupling type (e.g., 2JHC), and any features you are able to create from the molecule structure (xyz) files.
# 
# For this competition, you will not be predicting all the atom pairs in each molecule rather, you will only need to predict the pairs that are explicitly listed in the train and test files. For example, some molecules contain Fluorine (F), but you will not be predicting the scalar coupling constant for any pair that includes F.
# 
# The training and test splits are by molecule, so that no molecule in the training data is found in the test data.

# In[ ]:


# # Show how the files appear in the input folder
# !ls -GFlash --color ../input


# ## train.csv and test.csv
# The training set, where the first column `molecule_name` is the name of the molecule where the coupling constant originates (the corresponding XYZ file is located at ./structures/.xyz), the second `atom_index_0` and third column `atom_index_1` is the atom indices of the atom-pair creating the coupling and the fourth column `scalar_coupling_constant` is the scalar coupling constant that we want to be able to predict.
# 
# #### Important: 
# * Since this is a DEMO we will only read in a sample of the train data
#     * for realistic purposes, better would be to read in all the data, then

# In[ ]:


train_df = pd.read_csv('../input/train.csv').sample(frac=0.04)
print("Train data shape",train_df.shape)
test_df = pd.read_csv('../input/test.csv')
print("Test shape",test_df.shape)


# We can see each observation provides the:
# - molecule_name
# - atom_index_0 - index of the first atom pair
# - atom_index_1 - index of the second atom pair
# - scalar_coupling_constant (target)

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# The training set is larger than the test set.
# 

# In[ ]:


print('The training set has shape {}'.format(train_df.shape))
print('The test set has shape {}'.format(test_df.shape))


# The target distribution is pretty interesting! Spikes near zero, and -20. There is also a good bit around 80. We will return to these files later.

# In[ ]:


# Distribution of the target
train_df['scalar_coupling_constant'].plot(kind='hist', figsize=(20, 5), bins=700, title='Distribution of the target scalar coupling constant')
plt.show()


# In[ ]:


# Number of of atoms in molecule
fig, ax = plt.subplots(1, 2)
train_df.groupby('molecule_name').count().sort_values('id')['id'].plot(kind='hist',
                                                                       bins=25,
                                                                       color=color_pal[6],
                                                                      figsize=(20, 5),
                                                                      title='# of Atoms in Molecule (Train Set)',
                                                                      ax=ax[0])
test_df.groupby('molecule_name').count().sort_values('id')['id'].plot(kind='hist',
                                                                       bins=25,
                                                                       color=color_pal[2],
                                                                      figsize=(20, 5),
                                                                      title='# of Atoms in Molecule (Test Set)',
                                                                     ax=ax[1])
plt.show()


# ### Get list of molecule types
# * We'll see the target distribution is very different for the different types: We may want to make a model for each type seperately! 
# * We also have relatively few types, so this is easily tractionable

# In[ ]:


typelist = train_df['type'].unique()
print(typelist)


# In[ ]:


train_df.groupby("type")['scalar_coupling_constant'].plot(kind='hist', figsize=(18, 5), bins=500, title='Distribution of the target coupling constant, given type');

# plt.figure(figsize=(26, 24))
# for i, col in enumerate(typelist):
#     plt.subplot(4,2, i + 1)
#     sns.distplot(train_df[train_df['type']==col]['scalar_coupling_constant'],color ='indigo')
#     plt.title(col)


# ## structures.zip annd structures csv files.
# folder containing molecular structure (xyz) files, where: 
# - the first line is the number of atoms in the molecule,
# - followed by a blank line
# - and then a line for every atom, where the first column contains the atomic element (H for hydrogen, C for carbon etc.) and the remaining columns contain the X, Y and Z cartesian coordinates (a standard format for chemists and molecular visualization programs)
# 
# 
# ...lets have a look at the first example from the training set!

# In[ ]:


train_df.head(1)


# ## Look at the xyz file for this example
# - The first number `5` is the number of atoms in this molecule.
# - A blank line....
# - Each following line is the element and their cartesian coordinates. So in this examples we have one Carbon atom and four Hydrogen atoms!

# In[ ]:


get_ipython().system(' cat ../input/structures/dsgdb9nsd_000001.xyz')


# ## Structures.csv
# This file contains the same information as the individual xyz structure files, but in a single file.
# 
# This csv is a lot more useable than the `xyz` files (for most usages, but not all!)

# In[ ]:


structures = pd.read_csv('../input/structures.csv')
structures.head()


# In[ ]:


# 3D Plot!
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
example = structures.loc[structures['molecule_name'] == 'dsgdb9nsd_000001']
ax.scatter(xs=example['x'], ys=example['y'], zs=example['z'], s=100)
plt.suptitle('dsgdb9nsd_000001')
plt.show()


# # Additional Data
# *NOTE: additional data is provided for the molecules in Train only!*
# 
# * If we want to use it for our model, we'll need to extract it externally and/or predict it!
# 
# * Good 101 on the pre-provided values: https://www.kaggle.com/tarunpaparaju/champs-competition-chemistry-background-and-eda
# 
# 

# ## dipole_moments.csv
# - contains the molecular electric dipole moments. These are three dimensional vectors that indicate the charge distribution in the molecule. The first column (molecule_name) are the names of the molecule, the second to fourth column are the X, Y and Z components respectively of the dipole moment.

# In[ ]:


dm = pd.read_csv('../input/dipole_moments.csv')
dm.head()


# ## magnetic_shielding_tensors.csv
# - contains the magnetic shielding tensors for all atoms in the molecules. The first column (molecule_name) contains the molecule name, the second column (atom_index) contains the index of the atom in the molecule, the third to eleventh columns contain the XX, YX, ZX, XY, YY, ZY, XZ, YZ and ZZ elements of the tensor/matrix respectively.

# In[ ]:


mst = pd.read_csv('../input/magnetic_shielding_tensors.csv')
mst.head()


# ## mulliken_charges.csv
# - contains the mulliken charges for all atoms in the molecules. The first column (molecule_name) contains the name of the molecule, the second column (atom_index) contains the index of the atom in the molecule, the third column (mulliken_charge) contains the mulliken charge of the atom.

# In[ ]:


mul = pd.read_csv('../input/mulliken_charges.csv')
mul.head()


# In[ ]:


# Plot the distribution of mulliken_charges
mul['mulliken_charge'].plot(kind='hist', figsize=(15, 5), bins=500, title='Distribution of Mulliken Charges')
plt.show()


# ## potential_energy.csv
# - contains the potential energy of the molecules. The first column (molecule_name) contains the name of the molecule, the second column (potential_energy) contains the potential energy of the molecule.

# In[ ]:


pote = pd.read_csv('../input/potential_energy.csv')
pote.head()


# In[ ]:


# Plot the distribution of potential_energy
pote['potential_energy'].plot(kind='hist',
                              figsize=(15, 5),
                              bins=500,
                              title='Distribution of Potential Energy',
                              color='b')
plt.show()


# ## scalar_coupling_contributions.csv
# - The scalar coupling constants in train.csv (or corresponding files) are a sum of four terms. scalar_coupling_contributions.csv contain all these terms.
#     - The first column (molecule_name) are the **name of the molecule**,
#     - the second **(atom_index_0)** and
#     - third column **(atom_index_1)** are the atom indices of the atom-pair,
#     - the fourth column indicates the **type of coupling**,
#     - the fifth column (fc) is the **Fermi Contact contribution**,
#     - the sixth column (sd) is the **Spin-dipolar contribution**,
#     - the seventh column (pso) is the **Paramagnetic spin-orbit contribution** and
#     - the eighth column (dso) is the **Diamagnetic spin-orbit contribution**.

# In[ ]:


scc = pd.read_csv('../input/scalar_coupling_contributions.csv')
scc.head()


# In[ ]:


scc.groupby('type').count()['molecule_name'].sort_values().plot(kind='barh',
                                                                color='grey',
                                                               figsize=(15, 5),
                                                               title='Count of Coupling Type in Train Set')
plt.show()


# In[ ]:


fig, ax = plt.subplots(2, 2, figsize=(20, 10))
scc['fc'].plot(kind='hist', ax=ax.flat[0], bins=500, title='Fermi Contact contribution', color=color_pal[0])
scc['sd'].plot(kind='hist', ax=ax.flat[1], bins=500, title='Spin-dipolar contribution', color=color_pal[1])
scc['pso'].plot(kind='hist', ax=ax.flat[2], bins=500, title='Paramagnetic spin-orbit contribution', color=color_pal[2])
scc['dso'].plot(kind='hist', ax=ax.flat[3], bins=500, title='Diamagnetic spin-orbit contribution', color=color_pal[3])
plt.show()


# ## Relationship between Target and Features
# ** Keep in mind these features are provided for the training data ONLY**

# In[ ]:


scc = scc.merge(train_df)


# These plots are beautiful. It's a shame we don't have this data for the test set.

# In[ ]:


# Downsample to speed up plot time.
sns.pairplot(data=scc.sample(5000), hue='type', vars=['fc','sd','pso','dso','scalar_coupling_constant'])
plt.show()


# ## Target vs. Atom Count

# In[ ]:


atom_count_dict = structures.groupby('molecule_name').count()['atom_index'].to_dict()


# In[ ]:


train_df['atom_count'] = train_df['molecule_name'].map(atom_count_dict)
test_df['atom_count'] = test_df['molecule_name'].map(atom_count_dict)


# When we look at the target `scalar_coupling_constant` in relation to the `atom_count` - there visually appears to be a relationship. We notice the gap in coupling constant values, between ~25 and ~75. It is rare to see a value within this range. Could this be a good case for a classification problem between the two clusters?

# In[ ]:


train_df.sample(600).plot(x='atom_count',
                           y='scalar_coupling_constant',
                           kind='scatter',
                           color=color_pal[0],
                           figsize=(20, 5),
                           alpha=0.5)
plt.show()


# # Super Simple Baseline Model [1.239 Public LB]
# The second simplest thing we can do as a model is predict that the target is the **average** value that we observe for that **type** in the training set!

# In[ ]:


train_df.groupby('type')['scalar_coupling_constant'].mean().plot(kind='barh',
                                                                 figsize=(15, 5),
                                                                title='Average Scalar Coupling Constant by Type')
plt.show()
type_mean_dict = train_df.groupby('type')['scalar_coupling_constant'].mean().to_dict()
test_df['scalar_coupling_constant'] = test_df['type'].map(type_mean_dict)
test_df[['id','scalar_coupling_constant']].to_csv('super_simple_submission.csv', index=False)


# # Evaluation Metric
# 
# Submissions are evaluated on the Log of the Mean Absolute Error, calculated for each scalar coupling type, and then averaged across types, so that a 1% decrease in MAE for one type provides the same improvement in score as a 1% decrease for another type.
# 
# ![Eval Metric](https://i.imgur.com/AK6z3Dn.png)
# 
# Where:
# 
# - `T` is the number of scalar coupling types
# - `nt` is the number of observations of type t
# - `yi` is the actual scalar coupling constant for the observation
# - `yi^` is the predicted scalar coupling constant for the observation
# 
# For this metric, the MAE for any group has a floor of 1e-9, so that the minimum (best) possible score for perfect predictions is approximately -20.7232.

# Evaluation metric is important to understand as it determines how your model will be scored. Ideally we will set the loss function of our machine learning algorithm to use this metric so we can minimize the specific type of error.
# 
# Check out this kernel by `@abhishek` with code for the evaluation metric: https://www.kaggle.com/abhishek/competition-metric
# 

# In[ ]:


def metric(df, preds):
    df["prediction"] = preds
    maes = []
    for t in df.type.unique():
        y_true = df[df.type==t].scalar_coupling_constant.values
        y_pred = df[df.type==t].prediction.values
        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))
        maes.append(mae)
    return np.mean(maes)


# # Distance Feature Creation
# This feature was found from `@inversion` 's kernel here: https://www.kaggle.com/inversion/atomic-distance-benchmark/output
# The code was then made faster by `@seriousran` here: https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark

# In[ ]:


def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train_df = map_atom_info(train_df, 0)
train_df = map_atom_info(train_df, 1)

test_df = map_atom_info(test_df, 0)
test_df = map_atom_info(test_df, 1)


# In[ ]:


# https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark
train_p_0 = train_df[['x_0', 'y_0', 'z_0']].values
train_p_1 = train_df[['x_1', 'y_1', 'z_1']].values
test_p_0 = test_df[['x_0', 'y_0', 'z_0']].values
test_p_1 = test_df[['x_1', 'y_1', 'z_1']].values

train_df['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test_df['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)


# In[ ]:


# old, base from : @artgor's  kernel - https://www.kaggle.com/artgor/molecular-properties-eda-and-models
# train_df['dist_to_type_mean'] = train_df['dist'] / train_df.groupby('type')['dist'].transform('mean')
# test_df['dist_to_type_mean'] = test_df['dist'] / test_df.groupby('type')['dist'].transform('mean')

### I made my own Versions: 
# train_df['dist_to_type_mean'] = train_df['dist'] / train_df.groupby('type')['dist'].transform('mean')
# test_df['dist_to_type_mean'] = test_df['dist'] / test_df.groupby('type')['dist'].transform('mean')

# train_df['dist_to_type_0_mean'] = train_df['dist'] / train_df.groupby('atom_0')['dist'].transform('mean')
# test_df['dist_to_type_0_mean'] = test_df['dist'] / test_df.groupby('atom_0')['dist'].transform('mean')

# train_df['dist_to_type_1_mean'] = train_df['dist'] / train_df.groupby('atom_1')['dist'].transform('mean')
# test_df['dist_to_type_1_mean'] = test_df['dist'] / test_df.groupby('atom_1')['dist'].transform('mean')

# train_df['molecule_type_dist_mean'] = train_df.groupby([ 'type'])['dist'].transform('mean')
# test_df['molecule_type_dist_mean'] = test_df.groupby(['type'])['dist'].transform('mean')


train_df['dist_to_type_0_mean'] = train_df['dist'] / train_df.groupby(['type','atom_0'])['dist'].transform('mean')
test_df['dist_to_type_0_mean'] = test_df['dist'] / test_df.groupby(['type','atom_0'])['dist'].transform('mean')

train_df['dist_to_type_1_mean'] = train_df['dist'] / train_df.groupby(['type','atom_1'])['dist'].transform('mean')
test_df['dist_to_type_1_mean'] = test_df['dist'] / test_df.groupby(['type','atom_1'])['dist'].transform('mean')


# In[ ]:


# make categorical variables
atom_map = {'H': 0,
            'C': 1,
            'N': 2}
train_df['atom_0_cat'] = train_df['atom_0'].map(atom_map).astype('int')
train_df['atom_1_cat'] = train_df['atom_1'].map(atom_map).astype('int')
test_df['atom_0_cat'] = test_df['atom_0'].map(atom_map).astype('int')
test_df['atom_1_cat'] = test_df['atom_1'].map(atom_map).astype('int')


# In[ ]:


# One Hot Encode the Type
train_df = pd.concat([train_df, pd.get_dummies(train_df['type'])], axis=1)
test_df = pd.concat([test_df, pd.get_dummies(test_df['type'])], axis=1)


# In[ ]:


color_index = 0
axes_index = 0
fig, axes = plt.subplots(8, 1, figsize=(20, 20), sharex=True)
for mtype, d in train_df.groupby('type'):
    d['dist'].plot(kind='hist',
                  bins=1000,
                  title='Distribution of Distance Feature for {}'.format(mtype),
                  color=color_pal[color_index],
                  ax=axes[axes_index])
    if color_index == 6:
        color_index = 0
    else:
        color_index += 1
    axes_index += 1
plt.show()


# # Baseline Models
# - using **atom_count** and **type** as categorical features
# - Run this in a notebook to see the interactive plot of training and test error metrics.

# # LightGBM - CV

# In[ ]:


train_df.shape


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


# Configurables
FEATURES = ['atom_index_0', 'atom_index_1',
            'atom_0_cat',
            'x_0', 'y_0', 'z_0',
            'atom_1_cat', 
            'x_1', 'y_1', 'z_1', 'dist', 
#             'dist_to_type_mean',
            'atom_count',
            '1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN'
            ,'dist_to_type_0_mean',
       'dist_to_type_1_mean'
           ]

# # instead of whitelist, blacklist: # broken in feat importance part
# DROP_FEATS = ['id', 'molecule_name', 'type','scalar_coupling_constant', 'atom_0','atom_1',]

TARGET = 'scalar_coupling_constant'
CAT_FEATS = ['atom_0','atom_1']
## ORIG: 
# N_ESTIMATORS = 2000
# VERBOSE = 500
# EARLY_STOPPING_ROUNDS = 200
# RANDOM_STATE = 529

# faster:
N_ESTIMATORS = 100
VERBOSE = 50
EARLY_STOPPING_ROUNDS = 5
RANDOM_STATE = 529

# ## whitelist feats: 
X = train_df[FEATURES]
X_test = test_df[FEATURES]

# ## exclude cols:
# X = train_df.drop(DROP_FEATS,axis=1)
# X_test = test_df.drop(DROP_FEATS,axis=1)

y = train_df[TARGET]


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

lgb_params = {
#     'num_leaves': 128, #  orig
    'num_leaves': 64,
#               'min_child_samples': 64, # orig
              'min_child_samples': 32,
              'objective': 'regression',
#               'max_depth': 6, # ORIG
                            'max_depth': 5,
#               'learning_rate': 0.1, # orig
              'learning_rate': 0.2,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.9,
              "bagging_seed": 11,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.4,
              'colsample_bytree': 0.9
         }

RUN_LGB = True


# In[ ]:


if RUN_LGB:
    n_fold = 3
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=RANDOM_STATE)

    # Setup arrays for storing results
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()

    # Train the model
    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = lgb.LGBMRegressor(**lgb_params, n_estimators = N_ESTIMATORS, n_jobs = -1)
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  eval_metric='mae',
                  verbose=VERBOSE,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)

        y_pred_valid = model.predict(X_valid)
        y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        # feature importance
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = FEATURES
        fold_importance["importance"] = model.feature_importances_
        fold_importance["fold"] = fold_n + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

        prediction /= folds.n_splits
        scores.append(mean_absolute_error(y_valid, y_pred_valid))
        print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
        oof[valid_idx] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))
        prediction += y_pred


# In[ ]:


feature_importance.head()


# ## Save LGB Results, OOF, and Feature Importance
# It's always a good idea to save your OOF, predictions and feature importances. You never know when they will come in handy in the future.
# 
# We'll save the Number of folds and CV score in the filename.

# In[ ]:


if RUN_LGB:
    # Save Prediction and name appropriately
    submission_csv_name = 'submission_lgb_{}folds_{}CV.csv'.format(n_fold, np.mean(scores))
    oof_csv_name = 'oof_lgb_{}folds_{}CV.csv'.format(n_fold, np.mean(scores))
    fi_csv_name = 'fi_lgb_{}folds_{}CV.csv'.format(n_fold, np.mean(scores))

    print('Saving LGB Submission as:')
    print(submission_csv_name)
    ss = pd.read_csv('../input/sample_submission.csv')
    ss['scalar_coupling_constant'] = prediction
    ss.to_csv(submission_csv_name, index=False)
    ss.head()
    
    # OOF
    oof_df = train_df[['id','molecule_name','scalar_coupling_constant']].copy()
    oof_df['oof_pred'] = oof
    oof_df.to_csv(oof_csv_name, index=False)
    
    # Feature Importance
    feature_importance.to_csv(fi_csv_name, index=False)


# In[ ]:


feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:40].plot(kind="bar")


# In[ ]:


if RUN_LGB:
    # Plot feature importance as done in https://www.kaggle.com/artgor/artgor-utils
    feature_importance["importance"] /= folds.n_splits
    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index

    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

    plt.figure(figsize=(15, 20));
    ax = sns.barplot(x="importance",
                y="feature",
                hue='fold',
                data=best_features.sort_values(by="importance", ascending=False));
    plt.title('LGB Features (avg over folds)');


# # Catboost
# * Similar to LGBM and XGBoost. 

# In[ ]:


# from catboost import Pool, cv

# RUN_CATBOOST_CV = False

# if RUN_CATBOOST_CV:
#     labels = train_df['scalar_coupling_constant'].values
#     cat_features = ['type','atom_count','atom_0','atom_1']
#     cv_data = train_df[['type','atom_count','atom_0','atom_1',
#                         'x_0','y_0','z_0','x_1','y_1','z_1','dist']]
#     cv_dataset = Pool(data=cv_data,
#                       label=labels,
#                       cat_features=cat_features)

# ##     ITERATIONS = 100000 # ORIG
#     ITERATIONS = 1234
#     params = {"iterations": ITERATIONS,
#               "learning_rate" : 0.02,
#               "depth": 7,
#               "loss_function": "MAE",
#               "verbose": False,
#               "task_type" : "GPU"}

#     scores = cv(cv_dataset,
#                 params,
#                 fold_count=5, 
#                 plot="True")
    
#     scores['iterations'] = scores['iterations'].astype('int')
#     scores.set_index('iterations')[['test-MAE-mean','train-MAE-mean']].plot(figsize=(15, 5), title='CV (MAE) Score by iteration (5 Folds)')


# In[ ]:


# from catboost import CatBoostRegressor, Pool

# # ITERATIONS = 200000 # Default
# ITERATIONS = 1234 # faster

# FEATURES = [#'atom_index_0',
#             'atom_index_1',
#             'atom_0',
#             'x_0', 'y_0', 'z_0',
#             'atom_1', 
#             'x_1', 'y_1', 'z_1',
#             'dist', 'dist_to_type_mean',
#             'atom_count',
#             'type']
# TARGET = 'scalar_coupling_constant'
# CAT_FEATS = ['atom_0','atom_1','type']

# train_dataset = Pool(data=train_df[FEATURES],
#                   label=train_df['scalar_coupling_constant'].values,
#                   cat_features=CAT_FEATS)

# cb_model = CatBoostRegressor(iterations=ITERATIONS,
#                              learning_rate=0.2,
#                              depth=7,
#                              eval_metric='MAE',
#                              random_seed = 529,
#                              task_type="GPU")

# # Fit the model
# cb_model.fit(train_dataset, verbose=500)

# # Predict
# test_data = test_df[FEATURES]

# test_dataset = Pool(data=test_data,
#                     cat_features=CAT_FEATS)

# ss = pd.read_csv('../input/sample_submission.csv')
# ss['scalar_coupling_constant'] = cb_model.predict(test_dataset)
# ss.to_csv('basline_catboost_submission.csv', index=False)


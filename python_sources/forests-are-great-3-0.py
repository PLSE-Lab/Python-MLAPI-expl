#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# https://www.kaggle.com/c/learn-together
# 
# My attempt at the contest, upvoted kernels, which I found useful.
# 
# **Resources**
# 
# * [How to add table of contents](https://www.kaggle.com/questions-and-answers/69732)
# * [Hyperparameter Tuning](https://towardsdatascience.com/hyperparameter-tuning-c5619e7e6624)
# * [Stacking: Improve model performance](https://dkopczyk.quantee.co.uk/stacking/)

# # Versioning
# 
# * Version: 3.0
# * Steps: 
#   - Feature importance visualization
#   - Manual stacking, with layer one models below and RandomForest stacking model
#     - RandomForestClassifier
#     - XGBClassifier
#     - AdaBoostClassifier
#     - SVC
#     - KNeighborsClassifier
#   - Hyperparameter optimization for layer one using grid search
#   - For identifying stack model and validation, using X_train, y_train; For df_test fitting on X, Y

# # Table of contents <a id="0"></a>
# * [Imports](#imports)
# * [Block selection](#block-selection)
# * [Identify important Features](#identify-important-features)
# * [Feature engineering](#feature-engineering)
# * [Separate features and target](#separate-features-and-target)
# * [Define models](#define-models)
# * [Get initial scores](#get-initial-scores)
# * [Grid search](#grid-search)
# * [Generate output](#generate-output)

# In[ ]:


import os
dir = get_ipython().getoutput('ls -a')
if ('kernel-metadata.json' in dir):
    src = 'Local'
    # Local environment
    data_path = './data/learn-together'
else:
    # Kaggle environment
    src = 'Kaggle'
    data_path = '../input/learn-together'

print('Environment set to [{env}]'.format(env=src))
for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))


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


# # Imports <a id="imports"></a>
# [Go back to top](#0)

# In[ ]:


# Suppress future defaults warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# System imports
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Models
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Utilities
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2


# # Block selection <a id="block-selection"></a>
# [Go back to top](#0)

# In[ ]:


get_feature_importances = 1
drop_low_correlation_features = 1
grid_search = 1
validation = 0
generate_output = 1

low_correlation_features = []
grid_search_n_splits = 5
layer_one_folds = 10


# In[ ]:


df_test = pd.read_csv(data_path + '/test.csv')
df_sample_submission = pd.read_csv(data_path + '/sample_submission.csv')
df = pd.read_csv(data_path + '/train.csv')


# In[ ]:


# Identify columns with only 1 value, these are unlikely to be helpful
col_singular = [col for col in df.columns if df[col].nunique() == 1]
print('Singular columns: {}'.format(col_singular))

# Drop singular columns
df.drop(col_singular, axis=1, inplace=True)
df_test.drop(col_singular, axis=1, inplace=True)


# In[ ]:


# Check if target types are evenly spread
plt.ylabel('frequency')
plt.xlabel('cover type')
plt.bar(df['Cover_Type'].unique(), df['Cover_Type'].value_counts(), color ='green', width=0.2)
plt.rcParams["figure.figsize"] = (5,5)
plt.show()

# Evenly distributed, that's great**


# # Identify important features <a id="identify-important-features"></a>
# [Go back to top](#0)

# In[ ]:


# Print numerical values of important features
if get_feature_importances:
    target = 'Cover_Type'
    features = list(df.columns)
    features.remove(target)

    X = df[features]
    y = df[target]

    bestfeatures = SelectKBest(k=10)
    fit = bestfeatures.fit(X, y)
    
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    
    # Concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score'] 
    print(featureScores.nlargest(20,'Score'))


# In[ ]:


# plot graph of feature importances for better visualization
if get_feature_importances:
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) 
    
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()


# In[ ]:


# Generate heatmap
if get_feature_importances:
    # Only considering non-categorical columns for simplicity
    df_subset = df[['Elevation', 'Aspect', 'Slope',
           'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
           'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
           'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
           'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
           'Wilderness_Area4', 'Cover_Type']]

    corrmat = df_subset.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(10,10))
    g=sns.heatmap(df_subset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# # Feature engineering <a id="feature-engineering"></a>
# [Go back to top](#0)

# In[ ]:


# Drop low correlation feature
if drop_low_correlation_features:
    df.drop(low_correlation_features, axis=1, inplace=True)
    df_test.drop(low_correlation_features, axis=1, inplace=True)


# # Separate features and target <a id="separate-features-and-target"></a>
# [Go back to top](#0)

# In[ ]:


# Separate features and target
target = 'Cover_Type'
features = list(df.columns)
features.remove(target)

X = df[features]
y = df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, train_size=0.8)


# # Define models <a id="define-models"></a>
# [Go back to top](#0)

# In[ ]:


# Define Base (level 0) and Stacking (level 1) estimators
# Commented rows were during Hyperparameter optimization run
base_models = []

model = {'model': LGBMClassifier(
                        num_leaves=128,
                        verbose=-1,
                        random_state=5,
                        n_jobs=1)}
parameters = {'n_estimators': [100, 200, 300, 400]}
model['parameters'] = parameters
model['grid_search'] = 1
base_models.append(model)

model = {'model': ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                              min_samples_split=2,
                              max_depth=50,
                              random_state=5,
                              n_jobs=1)}
parameters = {'n_estimators': [200, 300, 400, 500]}
model['parameters'] = parameters
model['grid_search'] = 0
base_models.append(model)
         
model = {'model': AdaBoostClassifier(n_estimators=200, random_state=5)}
#model = {'model': AdaBoostClassifier(algorithm="SAMME.R"
#                                     , learning_rate=1.0
#                                     , n_estimators=200
#                                     , random_state=5)}
parameters = {'n_estimators': [100, 150, 200, 400]}
model['parameters'] = parameters
model['grid_search'] = 0
base_models.append(model)

model = {'model': SVC(probability=True, random_state=5, gamma='scale')}
#model = {'model': SVC(probability=True, C=10, random_state=5)}
Cs = [0.01, 0.1, 1, 10, 100]
parameters = {'C': Cs}
model['parameters'] = parameters
model['grid_search'] = 0
base_models.append(model)

model = {'model': XGBClassifier(random_state=5)}
parameters = {}
model['parameters'] = parameters
model['grid_search'] = 0
#base_models.append(model)

model = {'model': RandomForestClassifier(n_estimators=400, random_state = 5)}
#model = {'model': RandomForestClassifier(n_estimators=800, random_state = 5)}
parameters = {'n_estimators': [200, 300, 400]}
model['parameters'] = parameters
model['grid_search'] = 0
base_models.append(model)

model = {'model': KNeighborsClassifier()}
#model = {'model': KNeighborsClassifier(n_neighbors=3
#                                      , weights='distance')}
parameters = {'n_neighbors': range(3,12,2), 
              'weights': ['uniform', 'distance']}
model['parameters'] = parameters
model['grid_search'] = 1
#base_models.append(model)

model = {'model': LogisticRegression(random_state=5)}
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
model['parameters'] = parameters
model['grid_search'] = 1
#base_models.append(model)

# Define Stacking estimator
stack_model = RandomForestClassifier(n_estimators=300, random_state=5)


# # Get initial scores <a id="get-initial-scores"></a>
# [Go back to top](#0)

# In[ ]:


# Evaluate Base estimators separately
for base_model in base_models:
    model = copy.deepcopy(base_model['model'])
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_val)
    
    # Calculate accuracy
    acc = accuracy_score(y_val, y_pred)
    print('{} Accuracy: {:.2f}%'.format(model.__class__.__name__, acc * 100))


# # Grid search <a id="grid-search"></a>
# [Go back to top](#0)

# In[ ]:


# Do grid search on each base model
if grid_search:
    for base_model in base_models:
        if base_model['grid_search']:
            print('Model: {model_name}'.format(model_name=base_model['model'].__class__.__name__))
            print('Optimizing parameters: [{params}]'.format(params=base_model['parameters']))
            kfold = KFold(n_splits=grid_search_n_splits, shuffle=True)
            CV = GridSearchCV(base_model['model']
                          , param_grid=base_model['parameters']
                          , scoring = 'accuracy'
                          , n_jobs=-1
                          , cv=kfold)
            CV.fit(X_train, y_train)
            best_model = CV.best_estimator_
            base_model['best_model'] = best_model
            #print('Best score and parameter combination = ')
            #print(CV.best_score_)    
            #print(CV.best_params_) 
            #print('\n')

for base_model in base_models:
    if 'best_model'not in base_model:
        base_model['best_model'] = base_model['model']


# In[ ]:


if grid_search:
    for base_model in base_models:
        if base_model['grid_search']:
            model = copy.deepcopy(base_model['best_model'])
            print('After grid search: ')
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            print('{} Accuracy: {:.2f}%\n'.format(model.__class__.__name__, acc * 100))


# # Stacking intro <a id="stacking-intro"></a>
# [Go back to top](#0)

# ## Layer 1
# 1. Layer 1, loop across folds
#   - Separate X and y into training and validation sets. Keep validation set aside and ignore for now
#   - For the first base model, split the training data into n folds
#   - Fit the base model on X_train, y_train from n-1 folds, and predict on the remaining nth fold. 
#   - Add the predictions into a meta series for the base model
# 2. Repeat (1), n times, each time with new nth fold. This will cover full training data. Will then end up with predictions for all n folds (full training data) in the meta seies (number of rows same as X_train)
# 3. Loop across models
#   - Repeat (1) and (2) for all base models
#   - Combine meta series for each base model into meta df - one column per base model, number of rows same as original training data. 
# 4. Optionally add an original feature from the training data into the meta dataframe
# 5. This dataframe is feature data for next stage
# 
# ## Layer 2
# 6. Fit the stacking model on the meta Dataframe (output of (6)) and y_train. This is our stacked model.
# 
# ## Validation
# 7. Fit first base model on the full input data (X, y). Predict on X_val, this will generate a meta series
# 8. Repeat (8) across each base model, combine output series from each base model to make meta dataframe for next layer
# 9. Add the same feature column (as in (5)), from X_val into the meta dataframe from (9). This is source for next layer
# 10. Predict using stacked model (output of (7)) on the output of (10)
# 11. Compare prediction from (11) with y_val to get score
# 
# ## Testing
# 12. Repeat steps (7), (8), (9), (10), this time using df_test in place of X_val

# In[ ]:


# Create first level predictions (meta-features)
def hold_out_predict(clf, X, y, cv):
        
    """Performing cross validation hold out predictions for stacking"""
    # Initilize
    n_classes = len(np.unique(y)) # Assuming that training data contains all classes
    meta_features = np.zeros((X.shape[0], n_classes)) 
    n_splits = cv.get_n_splits(X, y)
    
    # Loop over folds
    print("Starting hold out prediction with {} splits for {}.".format(n_splits, clf.__class__.__name__))
    for train_idx, hold_out_idx in cv.split(X): 
        
        # Split data
        X_train = X.iloc[train_idx]    
        y_train = y.iloc[train_idx]
        X_hold_out = X.iloc[hold_out_idx]

        # Fit estimator to K-1 parts and predict on hold out part
        est = copy.deepcopy(clf)
        est.fit(X_train, y_train)
        y_hold_out_pred = est.predict_proba(X_hold_out)
        
        # Fill in meta features
        meta_features[hold_out_idx] = y_hold_out_pred

    return meta_features


# In[ ]:


# Create first level predictions (meta-features) from training data

# Define folds
kfold = KFold(n_splits=layer_one_folds, shuffle=True)

# Loop over classifier to produce meta features
meta_train = pd.DataFrame()
for base_model in base_models:
    
    model = copy.deepcopy(base_model['best_model'])
    # Create hold out predictions for a classifier
    meta_train_model = hold_out_predict(model, X_train, y_train, kfold)
    #print(pd.DataFrame(meta_train_model).head())
    
    # Gather meta training data
    meta_train = pd.concat([meta_train, pd.DataFrame(meta_train_model)], axis=1)
    #print(pd.DataFrame(meta_train).head())

stack_model.fit(meta_train, y_train)


# # Validation <a id="validation"></a>
# [Go back to top](#0)

# In[ ]:


if validation:
    # Create meta-features for testing data
    meta_val = pd.DataFrame()
    for base_model in base_models:

        model = copy.deepcopy(base_model['best_model'])
        # Create hold out predictions for a classifier
        model.fit(X, y)
        meta_val_model = model.predict_proba(X_val)

        meta_val = pd.concat([meta_val, pd.DataFrame(meta_val_model)], axis=1)

    y_pred = stack_model.predict(meta_val)
    score = accuracy_score(y_val, y_pred)


# # Get final parameters <a id="get-final-parameters"></a>
# [Go back to top](#0)

# In[ ]:


print('Final params\n')
for base_model in base_models:
    final_model = base_model['best_model']
    model_name = final_model.__class__.__name__
    model_params = final_model.get_params()
    print('Base model: [{model}]'.format(model=model_name))
    print('Model params: {params}'.format(params=json.dumps(model_params, indent = 4)))

final_stack_model = stack_model
model_name = final_stack_model.__class__.__name__
model_params = final_stack_model.get_params()
print('Stack model: [{model}]'.format(model=model_name))
print('Model params: {params}'.format(params=json.dumps(model_params, indent = 4)))


# # Generate output <a id="generate-output"></a>
# [Go back to top](#0)

# In[ ]:


# Final output
if generate_output:
    meta_test = pd.DataFrame()
    for base_model in base_models:
        model = copy.deepcopy(base_model['best_model'])
        
        # Fit model
        model.fit(X, y)
        meta_test_model = model.predict_proba(df_test)
    
        # Gather meta training data
        meta_test = pd.concat([meta_test, pd.DataFrame(meta_test_model)], axis=1)
    
    # Final output
    preds = stack_model.predict(meta_test)

    # Save test predictions to file
    output = pd.DataFrame({'Id': df_sample_submission.Id,
                   'Cover_Type': preds})
    output.head()
    output.to_csv('submission.csv', index=False)


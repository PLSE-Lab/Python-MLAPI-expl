#!/usr/bin/env python
# coding: utf-8

# ![](https://upload.wikimedia.org/wikipedia/commons/d/dd/Ouzellake.jpg)

# ## Contents
# 1. [Intro](#1)
# 1. [Setup](#2)
# 1. [Exploring The Data](#3)
#     1. [The Basics](#31)
#     1. [Target Distribution](#32)
#     1. [Correlation](#33)    
# 1. [Data Prep](#4)
#     1. [Drop Unused Features](#41)
# 1. [Model Building](#5)
#     1. [Random Forest](#51)
#     1. [XGBoost](#52)
#     1. [Initial Scores](#53)
#     1. [First Submission](#54)
#     1. [Initial Results](#55)
# 1. [Model Tuning](#6)
#     1. [Feature Reduction](#61)
#     1. [Hyperparameter Tuning](#62)
#     1. [Second Submission](#63)
#     1. [SVC](#64)
#     1. [Third Submission](#65)
# 1. [Conclusion](#7)    
# 

# <a id='1'></a>
# ## Intro
# I this notebook I combined some good ideas I have seen in other notebooks as well as external sources like official documentation and blogs. It goes through some initial EDA, first model training, further iterative model tuning and competition score submissions. Models are based on RF, XGB and SVC.

# <a id='2'></a>
# ## Setup
# 
# The usual imports and data loading:

# In[ ]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

train_df = pd.read_csv("../input/learn-together/train.csv")
test_df = pd.read_csv("../input/learn-together/test.csv")


# <a id='3'></a>
# ## Exploring The Data
# <a id='31'></a>
# ### The Basics
# To get a feel of the data available:

# In[ ]:


train_df.head()


# In[ ]:


print('Train set size: ', train_df.shape)
print('Test set size: ' , test_df.shape)


# In[ ]:


print("Missing values in train set: ", train_df.isna().any().any())
print("Missing values in test set: ", test_df.isna().any().any())


# One of the noticeable things is that the "ID" column matches the index, and will not provide any useful information to train the model. Another observation is that test set is much larger than training set. Finally, "Soil_Type" is already one-hot-encoded. There are no missing values, so the dataset seems to be already pre-processed and prepared for analysis and modeling.

# In[ ]:


train_df.describe()


# Looking at stats, nothing suspicious as far as I can tell. Without specific domain knowledge it is difficult to judge if these values are reasonable.

# The dataset has many categorical columns: 
# 
# - Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - 
#     Wilderness area designation
# - Soil_Type (40 binary columns, 0 = absence or 1 = presence) - 
#     Soil Type designation
# - Cover_Type (7 types, integers 1 to 7) - 
#     Forest Cover Type designation
# 
# I'm interested to know if unique values (the categories) in these columns in training set match those in the test set (excluding the target col - "Cover_Type", which is missing in test set). This is important, as the model will not be able to effectively interpret values (categories) from test dataset if it did not encounter them before in the training set.

# In[ ]:


cat_cols_filter = train_df.columns.str.startswith(('Soil', 'Wild'))
cat_col_names = train_df.loc[:, cat_cols_filter].columns.values

# Iterate through categorcial columns in both train and tests sets to find differences in unique values
# It is also good to know how many unique values are there to help decide what to do in case mismatch is found
for col in cat_col_names:
    if set(train_df[col].unique()) != set(test_df[col].unique()):
        print(f'Col [{col}] value / count:')
        print('-------------- train -----------')
        print(f'{train_df[col].value_counts().to_string()}')
        print('-------------- test ------------')
        print(f'{test_df[col].value_counts().to_string()}\n')


# Two columns contain more categories in test than in train sets- the values of "ones" are in test and not in train. It is probably safe to drop both of these columns from future model, as their impact is relativey small - in case of "Soil_Type15" there are only 3 values of "1" in test set and 105 values of "1" in test set for "Soil_Type7" (out of 565k).

# <a id='32'></a>
# ### Target Distribution
# 
# Looking at value distribution info, I am interested to see if the target ("Cover_Type") categories are equally represented in training set, so that model can learn to be equally effective at predicting all classes:

# In[ ]:


train_df['Cover_Type'].value_counts().plot.bar();


# It appears Kaggle has thought about it already and provided a training set with perfectly distributed target values.

# <a id='33'></a>
# ### Correlation
# 
# I've seen some kernels showing correlation heat-maps to try to see which feature impacts target variable most. I'm gonna argue here that since target variable is categorical, with numbers (1 through 7) only indicating their class, and have no ordinal meaning, the standard correlation analysis would be wrong (as typically correlation measures how increase or decrease in a particular ordinal numerical feature increases or decreases another numerical ordinal feature).
# 
# Instead, I will look at which non-categorical features correlate with each other, to perhaps find some that are highly correlated and thus interchangeable:

# In[ ]:


num_col_names = [cname for cname in train_df.columns.values if (cname not in cat_col_names) and (cname != 'Cover_Type')]
corr_matrix_df = train_df[num_col_names].corr()

# Since Pandas do not have a built-in heatmap plot, I'm using pyplot and seaborn here instead
fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(corr_matrix_df, annot=True, ax=ax);


# The correlation coefficient has values between -1 to 1:
# * A value closer to 0 implies weaker correlation (exact 0 implying no correlation)
# * A value closer to 1 implies stronger positive correlation
# * A value closer to -1 implies stronger negative correlation
# 
# From what I can see there are some rather strong negative correlations regarding hillshade, which is logical, but other than that I cannot conclude that some features are excludable.

# <a id='4'></a>
# ## Data Prep
# <a id='41'></a>
# ### Drop Unused Features
# Ok, so up till now the following has been decided, based on observations:
# 1. Drop "ID" column from training data (but preserve the ID's, as they will need to be part of results submission)
# 2. Drop "Soil_Type7" and "Soil_Type15" from training and test data
# 3. Leave the remaining columns as is for now, will look at their importance later.

# In[ ]:


train_prep_df = train_df.copy()
test_prep_df = test_df.copy()

train_prep_df.drop(["Id"], axis = 1, inplace=True)
test_ids = test_df["Id"]
test_prep_df.drop(["Id"], axis = 1, inplace=True)

train_prep_df.drop(["Soil_Type7", "Soil_Type15"], axis = 1, inplace=True)
test_prep_df.drop(["Soil_Type7", "Soil_Type15"], axis = 1, inplace=True)

feature_names = [f for f in train_prep_df.columns.values if f != 'Cover_Type']
target_name = 'Cover_Type'

#to make sure I got it right
print(len(feature_names)) #should be 56-3-1 = 52


# <a id='5'></a>
# ## Model Building
# 
# The most straight-forward way to proceed is to train a tree-based model. They do not require data normalization, so without any further prep work I can easily proceed to model training. I will train Random Forest model and an XGBoost model.

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(train_prep_df[feature_names], train_prep_df[target_name], test_size=0.2, random_state=0)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# <a id='51'></a>
# ### Random Forest

# In[ ]:


rf_model = RandomForestClassifier(n_jobs=4, random_state=0)
rf_model.fit(X_train, y_train)


# <a id='52'></a>
# ### XGBoost

# In[ ]:


xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=7, n_jobs=4, seed=0)
xgb_model.fit(X_train, y_train)


# <a id='53'></a>
# ### Initial Scores
# 
# To see how the models did:

# In[ ]:


rf_train_score = rf_model.score(X_train, y_train)
rf_val_score = rf_model.score(X_val, y_val)
print('RF train score: ', rf_train_score)
print('RF val score: ', rf_val_score)

xgb_train_score = xgb_model.score(X_train, y_train)
xgb_val_score = xgb_model.score(X_val, y_val)
print('XGB train score: ', xgb_train_score)
print('XGB val score: ', xgb_val_score)


# RF model seems to fare better, however the training accuracy of close to 100% is an indication of over-fitting. I will submit predictions of both models to see how they compare as of now.

# <a id='54'></a>
# ### First Submission
# 
# Save the predictions of initial RF and XGBoost models:

# In[ ]:


test_preds_rf = rf_model.predict(test_prep_df)
test_preds_xgb = xgb_model.predict(test_prep_df)

output = pd.DataFrame({'Id': test_ids, 'Cover_Type': test_preds_rf})
output.to_csv('initial_rf.csv', index=False);

output = pd.DataFrame({'Id': test_ids, 'Cover_Type': test_preds_xgb})
output.to_csv('initial_xgb.csv', index=False);

print('Done!')


# <a id='55'></a>
# ### Initial Results
# 
# After submission the RF model got a score of **0.70173**, while the XGB model got a disappointing score of **0.58439**. Overall these results are somewhere in the bottom half of all submissions, which is to be expected, given that most participants created almost identical models.
# Time to think how to squeeze some additional juice out of my models to improve performance.

# <a id='6'></a>
# ## Model Tuning
# <a id='61'></a>
# ### Feature Reduction
# I am curious to see if before any further model tuning I can identify features that contribute little to nothing towards target prediction. Luckily, the trained RF model has built-in property "feature_importances_" that illustrates that.
# 

# In[ ]:


# First I will create a new dataframe to hold the feature importance values, so that it is easier to plot them 
feature_df = pd.DataFrame({'feature': feature_names, 'importance': rf_model.feature_importances_})
ax = feature_df.sort_values('importance', ascending=False).plot.bar(x='feature', figsize=(15, 6), fontsize=12)


# From the looks of it, anything east of Soil_Type20 is just noise, so I'll take it out of both training and test data:

# In[ ]:


# What value is the cut-off border? 
print(feature_df[feature_df['feature']=='Soil_Type20'])

#store this value
cutoff = feature_df[feature_df['feature']=='Soil_Type20']['importance'].values[0]
print('\nCut-off val: ', cutoff)


# In[ ]:


cols_to_keep = feature_df[feature_df['importance']>cutoff]['feature']

# Prepare new set of training / validation data
X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_prep_df[cols_to_keep], train_prep_df[target_name], test_size=0.2, random_state=1)
print('New train / val data shape: ', X_train_new.shape, X_val_new.shape, y_train_new.shape, y_val_new.shape)

# Also modify test data set accordingly
test_new_df = test_prep_df[cols_to_keep]
print('New test data shape: ', test_new_df.shape)


# Out of curiosity, I do a quick model training with the reduced data to see if it had any impact on performance.

# In[ ]:


rf_model_new = RandomForestClassifier(n_jobs=4, random_state=0);
rf_model_new.fit(X_train_new, y_train_new);

rf_train_score_new = rf_model_new.score(X_train_new, y_train_new)
rf_val_score_new = rf_model_new.score(X_val_new, y_val_new)
print('RF new train score: ', rf_train_score_new)
print('RF new val score: ', rf_val_score_new)
print('RF old val score: ', rf_val_score)

xgb_model_new = xgb.XGBClassifier(objective='multi:softmax', num_class=7, n_jobs=4, seed=0)
xgb_model_new.fit(X_train_new, y_train_new);

xgb_train_score_new = xgb_model_new.score(X_train_new, y_train_new)
xgb_val_score_new = xgb_model_new.score(X_val_new, y_val_new)
print('\nXGB new train score: ', xgb_train_score_new)
print('XGB new val score: ', xgb_val_score_new)
print('XGB old val score: ', xgb_val_score)


# It appears the performance has even increased a bit!

# <a id='62'></a>
# ### Hyper-parameter Tuning
# 
# For subsequent model training I will do parameter tuning with random and grid search to let the model find it's own best hyper-parameters. This will also utilize cross-validation.
# 
# My approach will be to at first use random search function to try a limited number of combinations to find some potentially good parameter values, and then do a grid search through a reduced list of narrowed-down combinations. This is due to brute-force approach (grid-searching at once all possible combos) would likely result in model training for unfeasibly long time.
# 
# I have performed parameter searching in Google colab notebook while I worked on this kernel and already obtained the results.
# After several experimental runs through random search and grid search, these are the best param values I found:
# * RF: {'bootstrap': False, 'max_depth': 45, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 950}
# * XGB: {'colsample_bytree': 0.8, 'gamma': 0.5, 'learning_rate': 0.035, 'max_depth': 45, 'min_child_weight': 1, 'n_estimators': 500, 'subsample': 1.0}
# 
# In this notebook I will just train a new RF and XGB model with the above best parameters that my grid search found and proceed from there, but the code for grid search is provided below for reference.

# In[ ]:


# check out scikit and xgb docs for more info on which params are best suited for tuning

# RF initial random search through params 
#random_params = {
#    'n_estimators': [100,200,400,800],
#    'max_features': ["sqrt", None, "log2"],
#    'max_depth': [None, 50, 100, 200, 400, 800, 1600],
#    'min_samples_split': [2, 5],
#    'bootstrap': [True, False],
#}
#
# RandomizedSearchCV and GridSearchCV already include cross-validation functionality, and return a model that will be ready for training accordingly
#rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(random_state=0), param_distributions=random_params, scoring='accuracy', n_iter=200, cv=5, n_jobs=4, verbose=10, random_state=0)
#rf_random.fit(X_train_new, y_train_new)
#
# Grid search through reduced list of params
#tune_params = {
#    'n_estimators': [200,400,800],
#    'max_features': ["sqrt", None],
#    'max_depth': [None, 100, 400],
#    'min_samples_split': [2],
#    'bootstrap': [False],
#}
#
#rf_tuned = GridSearchCV(RandomForestClassifier(random_state=0), tune_params, scoring='accuracy', cv=5, scoring='accuracy', n_jobs=4, verbose=10)
#rf_tuned.fit(X_train_new, y_train_new)


# In[ ]:


# XGBoost grid search
#xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=7, n_jobs=4, seed=0)
#
# Grid search through reduced list of params
#random_params = {
#        'n_estimators': [100,200, 400, 800],
#        'min_child_weight': [1, 5, 10],
#        'gamma': [0.5, 1, 1.5, 2, 5],
#        'subsample': [0.6, 0.8, 1.0],
#        'colsample_bytree': [0.6, 0.8, 1.0],
#        'learning_rate': [0.1,0.05,0.001],
#        'max_depth': [3, 4, 5, 10]
#}
#
#xgb_random = RandomizedSearchCV(estimator = xgb_clf, param_distributions=random_params, scoring='accuracy',n_iter=500, cv=5, n_jobs=4, verbose=10, random_state=0)
#xgb_random.fit(X_train_new, y_train_new)


# In[ ]:


#RF results
#rf_random_val_score = rf_random.score(X_val_new, y_val_new)
#rf_tuned_val_score = rf_random.score(X_val_new, y_val_new)
#
#print("Best parameter set found in random search:")
#print()
#print(rf_random.best_params_)
#print()
#print('RF random val score: ', rf_random_val_score)
#print('RF random val score improvement vs previous: ', rf_random_val_score - rf_val_score_new)
#
#print("\nBest parameter set found in grid search:")
#print()
#print(rf_tuned.best_params_)
#print()
#print('RF tunded val score: ', rf_tuned.score(X_val_new, y_val_new))
#print('RF tuned val score improvement vs previous: ', rf_tuned_val_score - rf_val_score_new)

#XGB results
#xgb_random_val_score = xgb_random.score(X_val_new, y_val_new)
#xgb_tuned_val_score = xgb_random.score(X_val_new, y_val_new)
#
#print("Best parameter set found in random search:")
#print()
#print(xgb_random.best_params_)
#print()
#print('XGB random val score: ', xgb_random_val_score)
#print('XGB random val score improvement vs previous: ', xgb_random_val_score - xgb_val_score_new)
#
#print("\nBest parameter set found in grid search:")
#print()
#print(rf_tuned.best_params_)
#print()
#print('XGB tunded val score: ', xgb_tuned.score(X_val_new, y_val_new))
#print('XGB tuned val score improvement vs previous: ', xgb_tuned_val_score - xgb_val_score_new)


# Since I already have a set of optimized parameter values, I will train new RF and XGB models so that I can proceed with this notebook.

# In[ ]:


rf_tuned = RandomForestClassifier(bootstrap=False, max_depth=45, max_features='sqrt', min_samples_split=2, min_samples_leaf=1, n_estimators=950, random_state=0, n_jobs=4);
rf_tuned.fit(X_train_new, y_train_new);

rf_tuned_train_score = rf_tuned.score(X_train_new, y_train_new)
rf_tuned_val_score = rf_tuned.score(X_val_new, y_val_new)
print('RF tuned train score: ', rf_tuned_train_score)
print('RF tuned val score: ', rf_tuned_val_score)
print('RF old val score: ', rf_val_score)

xgb_tuned = xgb.XGBClassifier(objective='multi:softmax', num_class=7, n_estimators=500, max_depth=45, subsample=1.0, learning_rate=0.035, min_child_weight=1, gamma=0.5, colsample_bytree=0.8, n_jobs=4, seed=0)
xgb_tuned.fit(X_train_new, y_train_new);

xgb_tuned_train_score = xgb_tuned.score(X_train_new, y_train_new)
xgb_tuned_val_score = xgb_tuned.score(X_val_new, y_val_new)
print('\nXGB tuned train score: ', xgb_tuned_train_score)
print('XGB tuned val score: ', xgb_tuned_val_score)
print('XGB old val score: ', xgb_val_score)


# Tuning the models helped with validation score improvement, though they appear to still be over-fitting training data.

# <a id='63'></a>
# ### Second Submission
# 
# And now to see how these new tuned models did in the leaderboard:

# In[ ]:


test_preds_rf_tuned = rf_tuned.predict(test_new_df)
test_preds_xgb_tuned = xgb_tuned.predict(test_new_df)

output = pd.DataFrame({'Id': test_ids, 'Cover_Type': test_preds_rf_tuned})
output.to_csv('tuned_rf.csv', index=False);

output = pd.DataFrame({'Id': test_ids, 'Cover_Type': test_preds_xgb_tuned})
output.to_csv('tuned_xgb.csv', index=False);

print('Done!')


# This time the tuned RF model got a score of **0.75903**, while tuned XGB got **0.74485**. This is a significant improvement, but still not on a level of top performers using same algorithms. What can be done to improve RF and XGB even further, without going so far as to start reading scientific papers?

# <a id='64'></a>
# ### SVC
# 
# I wanted to try another classification model- SVC. Similarly as with RF and XGB, I did extensive grid searching for best param values in Google colab, so here I will just proceed to train the model with those param values.
# 
# Just to note - models like SVMs need data to be pre-scaled, so there is one extra step before grid search.

# In[ ]:


# I scale the whole dataset, and then split into training / validation data again specificaly for SVC training
# Target colkumn is niot scaled
scaler = StandardScaler()
train_scaled_data = scaler.fit_transform(train_prep_df[cols_to_keep])
test_scaled_data = scaler.fit_transform(test_prep_df[cols_to_keep])

train_scaled_df = pd.DataFrame(data=train_scaled_data, columns=cols_to_keep)
test_scaled_df = pd.DataFrame(data=test_scaled_data, columns=cols_to_keep)

train_scaled_df[target_name] = train_prep_df[target_name]

# Prepare new set of training / validation data
X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled = train_test_split(train_scaled_df[cols_to_keep], train_scaled_df[target_name], test_size=0.2, random_state=1)
print('Original train / val data shape: ', X_train_new.shape, X_val_new.shape, y_train_new.shape, y_val_new.shape)
print('New scaled train / val data shape: ', X_train_scaled.shape, X_val_scaled.shape, y_train_scaled.shape, y_val_scaled.shape)

print('\nOriginal test data shape: ', test_new_df.shape)
print('New scaled test data shape: ', test_scaled_df.shape)


# In[ ]:


# Training with the best found params
svc_tuned = SVC(C=130, gamma=0.045, kernel='rbf', decision_function_shape='ovo', random_state=0);
svc_tuned.fit(X_train_scaled, y_train_scaled);

svc_tuned_train_score = svc_tuned.score(X_train_scaled, y_train_scaled)
svc_tuned_val_score = svc_tuned.score(X_val_scaled, y_val_scaled)
print('SVC tuned train score: ', svc_tuned_train_score)
print('SVC tuned val score: ', svc_tuned_val_score)


# <a id='65'></a>
# ### Third Submission
# 
# Let's compare SVC against other entries in the leaderboard.

# In[ ]:


test_preds_svc_tuned = svc_tuned.predict(test_scaled_df)

output = pd.DataFrame({'Id': test_ids, 'Cover_Type': test_preds_svc_tuned})
output.to_csv('tuned_svc.csv', index=False);

print('Done!')


# I got a score of **0.51918**, which was both a surprise and a big dissapointment, I did not expect such a difference in validation and test scores.

# <a id='7'></a>
# ## Conclusion
# 
# In this notebook I went from exploring the data to creating and fine-tuning 3 separate classification models. Except SVC, they achieved decent score, but are not on a level of best competition submissions. In the next notebook I intend to play around with the models a bit more, perhaps try even combining them to increase the accuracy of predictions. Ideas / suggestions on improving the models welcome.

# In[ ]:





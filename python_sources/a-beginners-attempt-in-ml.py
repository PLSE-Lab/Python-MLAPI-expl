#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# I'm just a beginner who started using `pandas` one week ago and now trying to do some basic data analysis for the first time.  
# I would very much like to hear your feedback, comments and improvements to my code.
# 
# ---
# 
# This notebook continues from [my last one](https://www.kaggle.com/narimiran/kobe-bryant-shot-selection/beginners-first-time) stopped, so I'll just quickly repeat all modifications I've already done, and then continue with the new stuff.

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')

df = pd.read_csv('../input/data.csv')

not_needed = []

not_needed.extend(['game_event_id', 'game_id'])

not_needed.extend(['lon', 'lat'])

df['time_remaining'] = 60 * df.loc[:, 'minutes_remaining'] + df.loc[:, 'seconds_remaining']
not_needed.extend(['minutes_remaining', 'seconds_remaining'])

df['season'] = df['season'].apply(lambda x: x[:4])
df['season'] = pd.to_numeric(df['season'])

dist = pd.DataFrame({'true_dist': np.sqrt((df['loc_x']/10)**2 + (df['loc_y']/10)**2), 
                     'shot_dist': df['shot_distance']})
df['shot_distance_'] = dist['true_dist']
not_needed.append('shot_distance')

df['3pt_goal'] = df['shot_type'].str.contains('3PT').astype('int')
not_needed.append('shot_type')

not_needed.append('shot_zone_range')

not_needed.extend(['team_id', 'team_name'])

df['game_date'] = pd.to_datetime(df['game_date'])
df['game_year'] = df['game_date'].dt.year
df['game_month'] = df['game_date'].dt.month
df['game_day'] = df['game_date'].dt.dayofweek
not_needed.append('game_date')

df['home_game'] = df['matchup'].str.contains('vs.').astype(int)
not_needed.append('matchup')

df.set_index('shot_id', inplace=True)

df = df.drop(not_needed, axis=1)

random_sample = df.take(np.random.permutation(len(df))[:10])
random_sample.head(10)


# # New stuff
# 
# After we've explored the data [last time](https://www.kaggle.com/narimiran/kobe-bryant-shot-selection/beginners-first-time), it's time to analyze some more.

# ## Action types

# In[ ]:


df['action_type'].value_counts()


# There are too many (57) different action types, and many of them have only few shots, so we'll keep first 25 action types (with most of shot attempts), and all other action types will be under `other` category.

# In[ ]:


rare_action_types = df['action_type'].value_counts()[25:]
rare_actions = rare_action_types.index.values

df.loc[df['action_type'].isin(rare_actions), 'action_type'] = 'other'
df['action_type'].value_counts()


# ## Periods and overtime

# In[ ]:


df['period'].value_counts()


# Under 400 shot attempts (with similar accuracy) were made in overtime periods (periods 5, 6, 7), so we'll combine them in one category: `overtime`.

# In[ ]:


overtime = np.array([5, 6, 7])
df.loc[df['period'].isin(overtime), 'period'] = 'overtime'
df['period'].value_counts()


# ## Playoffs
# 
# As we've seen earlier there's no difference in accuracy between regular season and playoffs, so column `playoffs` won't be needed.

# In[ ]:


df = df.drop('playoffs', axis=1)


# In[ ]:


# Creating dummies for categorical features

We can't use categorical features so we'll convert them to dummies.


# In[ ]:


df.dtypes


# In[ ]:


categorical = ['action_type', 'combined_shot_type', 'shot_zone_area', 'shot_zone_basic', 
               'opponent', 'period', 'season', 'game_year', 'game_month', 'game_day']

for column in categorical:
    dummy = pd.get_dummies(df[column], prefix=column)
    df = df.join(dummy)
    df.drop(column, axis=1, inplace=True)

df.head()


# # Separating the data
# 
# Splitting the data in two parts - one for our learning and other for submission.

# In[ ]:


unknown_shots = df['shot_made_flag'].isnull()

submission_data = df[unknown_shots].drop('shot_made_flag', 1)
data = df[~unknown_shots]

X = data.drop('shot_made_flag', 1)
y = data['shot_made_flag']


# # Feature selection
# 
# We have 146 features, but would like to reduce that number to only most important features.
# 
# ---
# 
# ***Big THANK YOU goes to [Norbert Kozlowski](https://www.kaggle.com/khozzy) and [his script](https://www.kaggle.com/khozzy/kobe-bryant-shot-selection/kobe-shots-show-me-your-best-model/) which helped me a lot to make all of the code from now till the end of the notebook.***
# 
# ---

# In[ ]:


from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ## Variance Threshold

# In[ ]:


threshold = 0.9
vt = VarianceThreshold().fit(X)

feat_var_threshold = X.columns[vt.variances_ > threshold * (1-threshold)].values
feat_var_threshold


# ## Random Forest Classifier

# In[ ]:


model = RandomForestClassifier()
model.fit(X, y)

feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])
feat_RFC = feature_imp.sort_values("importance", ascending=False).head(35)

feat_RFC = feat_RFC.index.values
feat_RFC


# ## Recursive feature elimination (RFE)

# In[ ]:


rfe = RFE(LogisticRegression(), 35)
rfe.fit(X, y)

feature_rfe_scoring = pd.DataFrame({'feature': X.columns, 'score': rfe.ranking_})

feat_rfe = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
feat_rfe


# ## Putting it all together

# In[ ]:


features = np.hstack([feat_var_threshold, feat_RFC, feat_rfe])

features = np.unique(features)
print('Final features set:\n')
for f in features:
    print("-{}".format(f))
    
len(features)


# We'll make new datasets with only those columns.

# In[ ]:


submission_data = submission_data.ix[:, features]
data = data.ix[:, features]
X = X.ix[:, features]


# # Testing different algorithms

# In[ ]:


from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[ ]:


seed = 2016
num_folds = 5
num_instances = len(X)
jobs = -1

scoring = 'log_loss'

kfold = KFold(n=num_instances, n_folds=num_folds, random_state=seed)


# ## Logistic regression

# In[ ]:


model = LogisticRegression()

result = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print("({0:.4f}) +/- ({1:.4f})".format(result.mean(), result.std()))


# ## K-nearest neighbors

# In[ ]:


model = KNeighborsClassifier(n_neighbors=20, n_jobs=jobs)

result = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print("({0:.4f}) +/- ({1:.4f})".format(result.mean(), result.std()))


# ## Random forest

# In[ ]:


model = RandomForestClassifier(n_estimators=200, n_jobs=jobs)

result = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print("({0:.4f}) +/- ({1:.4f})".format(result.mean(), result.std()))


# ## Ada boost

# In[ ]:


model = AdaBoostClassifier(random_state=seed)

results = cross_val_score(model, X, y, cv=kfold, scoring=scoring, n_jobs=jobs)
print("({0:.4f}) +/- ({1:.4f})".format(results.mean(), results.std()))


# ## Gradient Boosting

# In[ ]:


model = GradientBoostingClassifier(random_state=seed)

results = cross_val_score(model, X, y, cv=kfold, scoring=scoring, n_jobs=jobs)
print("({0:.4f}) +/- ({1:.4f})".format(results.mean(), results.std()))


# ## Linear Discriminant Analysis (LDA)

# In[ ]:


model = LinearDiscriminantAnalysis(solver='lsqr')

results = cross_val_score(model, X, y, cv=kfold, scoring=scoring, n_jobs=jobs)
print("({0:.4f}) +/- ({1:.4f})".format(results.mean(), results.std()))


# # Grid search
# 
# We'll use `GridSearchCV` to find the best parameters for each algorithm that we used above.

# ## Logistic regression

# In[ ]:


lr_grid = GridSearchCV(estimator = LogisticRegression(random_state=seed),
                       param_grid = {'penalty': ['l1', 'l2'], 
                                     'C': [0.01, 0.1, 1, 10]}, 
                       cv = kfold, 
                       scoring = scoring)

lr_grid.fit(X, y)

print(lr_grid.best_score_)
print(lr_grid.best_params_)


# ## K-nearest neighbors

# In[ ]:


knn_grid = GridSearchCV(estimator = KNeighborsClassifier(n_jobs=jobs),
                        param_grid = {'n_neighbors': [50, 80],
                                      'weights': ['uniform'],
                                      'algorithm': ['ball_tree'],
                                      'leaf_size': [2, 10], 
                                      'p': [1]}, 
                        cv = kfold, 
                        scoring = scoring)

knn_grid.fit(X, y)

print(knn_grid.best_score_)
print(knn_grid.best_params_)


# ## Random forest

# In[ ]:


rf_grid = GridSearchCV(estimator = RandomForestClassifier(warm_start=True, random_state=seed, n_jobs=jobs), 
                       param_grid = {'n_estimators': [100, 200],
                                     'criterion': ['entropy'], 
                                     'max_features': ['auto', 20], 
                                     'max_depth': [None, 10]}, 
                       cv = kfold, 
                       scoring = scoring)

rf_grid.fit(X, y)

print(rf_grid.best_score_)
print(rf_grid.best_params_)


# ## Ada boost

# In[ ]:


ada_grid = GridSearchCV(estimator = AdaBoostClassifier(random_state=seed), 
                        param_grid = {'n_estimators': [10, 25, 50, 100, 150],
                                      'learning_rate': [1e-3, 1e-2, 1e-1, 1]},
                        cv = kfold, 
                        scoring = scoring, 
                        n_jobs = jobs)

ada_grid.fit(X, y)

print(ada_grid.best_score_)
print(ada_grid.best_params_)


# ## Gradient Boosting

# In[ ]:


gbm_grid = GridSearchCV(estimator = GradientBoostingClassifier(warm_start=True, random_state=seed),
                        param_grid = {'n_estimators': [100, 200],
                                      'max_depth': [5],
                                      'max_features': ['auto', 'log2'],
                                      'learning_rate': [0.1]}, 
                        cv = kfold, 
                        scoring = scoring, 
                        n_jobs = jobs)

gbm_grid.fit(X, y)

print(gbm_grid.best_score_)
print(gbm_grid.best_params_)


# ## LDA

# In[ ]:


lda_grid = GridSearchCV(estimator = LinearDiscriminantAnalysis(),
                        param_grid = {'solver': ['lsqr'], 
                                      'shrinkage': [None, 'auto'],
                                      'n_components': [None, 2, 5, 10]},
                        cv = kfold, 
                        scoring = scoring,
                        n_jobs = jobs)

lda_grid.fit(X, y)

print(lda_grid.best_score_)
print(lda_grid.best_params_)


# ## Grid search summary

# In[ ]:


print('lr', lr_grid.best_score_)
print(lr_grid.best_params_)
print()
print('knn', knn_grid.best_score_)
print(knn_grid.best_params_)
print()
print('rf', rf_grid.best_score_)
print(rf_grid.best_params_)
print()
print('ada', ada_grid.best_score_)
print(ada_grid.best_params_)
print()
print('gbm', gbm_grid.best_score_)
print(gbm_grid.best_params_)
print()
print('lda', lda_grid.best_score_)
print(lda_grid.best_params_)


# # Voting classifier
# 
# After lots of trial-and-error, I decided not to use KNN and ADA (two algorithms with the worst score).
# 
# Weights were chosen based on same principle - trying a lot of values.

# In[ ]:


estimators = [('lr', LogisticRegression(C=1, penalty='l2', random_state=seed)), 
              ('rf', RandomForestClassifier(warm_start=True, max_features=20, n_estimators=200, 
                                            max_depth=5, criterion='entropy', random_state=seed)),
              ('gbm', GradientBoostingClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, max_features='log2')),
              ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None))]


# In[ ]:


voters = VotingClassifier(estimators, voting='soft', weights=[1, 2, 2, 1])

results = cross_val_score(voters, X, y, cv=kfold, scoring=scoring, n_jobs=jobs)
print("({0:.4}) +/- ({1:.4f})".format(results.mean(), results.std()))


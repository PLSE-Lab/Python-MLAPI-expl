#!/usr/bin/env python
# coding: utf-8

# # First Look of the Don't Overfit II Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import seaborn as sns
import random
import os, sys

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

get_ipython().system('ls ../input/')


# In[ ]:


# Read in datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
ss = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


print('Train shape {}'.format(train.shape))
print('Test shape {}'.format(test.shape))


# In[ ]:


train.groupby('target').count()['id'].plot(kind='barh', title='Target Distribution', figsize=(15, 5))
plt.show()


# ## 64-36 split of target in training set - will that hold true for test set?

# In[ ]:


train['target'].mean() * 100


# # Plot distribution of features in train vs test
# - Picked 5 random features
# - There are a lot of features (300)

# In[ ]:


random.seed(5)
for x in range(0, 5):
    random_feature = random.randint(1,299)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15 ,3))
    train[str(random_feature)].plot(kind='hist', ax=ax1, bins=20, title='Feature {}: Train set'.format(random_feature))
    test[str(random_feature)].plot(kind='hist', ax=ax2, bins=20, title='Feature {}: Test set'.format(random_feature))
    plt.show()


# # See correlation between first 4 features and target

# In[ ]:


sns.pairplot(train, vars=['target', '0','1','2','3','4'], hue='target')
plt.show()


# # Walkthrough of SKlearn Classification Algs (Not worrying about overfitting yet)
# - Lets not worry about overfitting yet and try out some classification algs

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

train_X = train.drop(['id','target'], axis=1).as_matrix()
train_y = train['target'].values
test_X = test.drop(['id'], axis=1).as_matrix()


# ## KNeighborsClassifier - Public LB 0.549

# In[ ]:


clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(train_X, train_y)
test['target'] = clf.predict(test_X)
test[['id','target']].to_csv('submission_KNeighborsClassifier.csv', index=False)
test = test.drop('target', axis=1) # drop the target column


# ## DecisionTreeClassifier - Public LB 0.558

# In[ ]:


clf = DecisionTreeClassifier()
clf.fit(train_X, train_y)
# Delete the old prediction
test['target'] = clf.predict(test_X)
test[['id','target']].to_csv('submission_DecisionTreeClassifier.csv', index=False)
test = test.drop('target', axis=1) # drop the target column


# ## RandomForestClassifier - Public LB 0.574

# In[ ]:


clf = RandomForestClassifier()
clf.fit(train_X, train_y)
test['target'] = clf.predict(test_X)
test[['id','target']].to_csv('submission_RandomForestClassifier.csv', index=False)
test = test.drop('target', axis=1) # drop the target column


# ## AdaBoostClassifier - Public LB 0.638

# In[ ]:


clf = AdaBoostClassifier()
clf.fit(train_X, train_y)
test['target'] = clf.predict(test_X)
test[['id','target']].to_csv('submission_AdaBoostClassifier.csv', index=False)
test = test.drop('target', axis=1) # drop the target column


# ## Naive Bayes - Public LB 0.611

# In[ ]:


clf = GaussianNB()
clf.fit(train_X, train_y)
test['target'] = clf.predict(test_X)
test[['id','target']].to_csv('submission_GaussianNB.csv', index=False)
test = test.drop('target', axis=1) # drop the target column


# ## XGBoost - Public LB ???

# In[ ]:


import xgboost as xgb
clf = xgb.XGBClassifier()
clf.fit(train_X, train_y)
test['target'] = clf.predict(test_X)
test[['id','target']].to_csv('submission_XGBClassifier.csv', index=False)
test = test.drop('target', axis=1) # drop the target column


# ## Logistic Regression - Public LB ???

# In[ ]:


clf = LogisticRegression(class_weight='balanced', penalty='l1', C=1.0, solver='liblinear')
clf.fit(train_X, train_y)
test['target'] = clf.predict(test_X)
test[['id','target']].to_csv('submission_LogisticRegression.csv', index=False)
test = test.drop('target', axis=1) # drop the target column


# # Cross Validation
# Now lets use some cross validation techniques to validate our scores for each model type. Cross validation allows us to train multiple models on the training data by splitting differently each time it trains the model.

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

for model_type in [KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier,
                   AdaBoostClassifier, xgb.XGBClassifier, LogisticRegression]:
    clf = model_type()
    kfold = KFold(n_splits=5, shuffle=True)
    cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)
    scores = cross_val_score(clf, train_X, train_y, cv=cv, scoring='roc_auc')
    print("Print {} Accuracy: {} (+/- {})".format(model_type.__name__, scores.mean(), scores.std() * 2))


# # Parameter Tuning using RandomizedSearchCV
# - Next we want to do some parameter tuning on our best model.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

clf = LogisticRegression(class_weight='balanced', solver='liblinear')

# Search through these optino
penalty = ['l1', 'l2']
C = uniform(loc=0, scale=4)
hyperparameters = dict(C=C, penalty=penalty)

rand_cv = RandomizedSearchCV(clf, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1, scoring='roc_auc')
best_model = rand_cv.fit(train_X, train_y)

print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])


# In[ ]:


cv_results = pd.DataFrame()
cv_results['params'] = best_model.cv_results_['params']
cv_results['mean_test_score'] = best_model.cv_results_['mean_test_score']
cv_results['std_test_score'] = best_model.cv_results_['std_test_score']
cv_results['rank_test_score'] = best_model.cv_results_['rank_test_score']
cv_results = pd.concat([cv_results.drop('params', axis=1), pd.DataFrame(cv_results['params'].tolist())], axis=1)
cv_results.sort_values('rank_test_score').head()


# In[ ]:


cv_results['penalty_color'] = cv_results.apply(lambda x: 1 if x['penalty'] == 'l1' else 0, axis=1)
cv_results[['mean_test_score','C']].plot.scatter(x='mean_test_score', y='C', c=cv_results['penalty_color'], colormap='viridis')
plt.show()


# ## Focused Gridsearch
# - Use only l1
# - Smaller C values

# In[ ]:


# Search through these optino
C = [0.001, 0.01, 0.02, 0.05, 0.1, 0.12, 0.13, 0.15, 0.16, 0.17, 0.178, 0.179, 0.175, 0.2, 0.3]
hyperparameters = dict(C=C)

clf = LogisticRegression(solver='liblinear', class_weight='balanced', penalty='l1')
rand_cv = RandomizedSearchCV(clf, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1, scoring='roc_auc')
best_model = rand_cv.fit(train_X, train_y)

print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
print('Best Score: {}'.format(best_model.best_score_))


# In[ ]:


test['target'] = best_model.predict(test_X)
test[['id','target']].to_csv('submission_LogisticRegression_randomCV.csv', index=False)
test = test.drop('target', axis=1) # drop the target column


# In[ ]:


# Use predict proba
test['target'] = best_model.predict_proba(test_X)[:,1]
test[['id','target']].to_csv('submission_LogisticRegression_randomCV_proba.csv', index=False)
test = test.drop('target', axis=1) # drop the target column


# # XGBClassifier with RandomizedSearchCV

# In[ ]:


clf = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic')
# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
rand_cv = RandomizedSearchCV(clf, params, random_state=1, n_iter=20, cv=5, verbose=0, n_jobs=-1, scoring='roc_auc')
best_model = rand_cv.fit(train_X, train_y)


# In[ ]:


best_model.best_score_


# In[ ]:


test['target'] = best_model.predict(test_X)
test[['id','target']].to_csv('submission_XGBClassifier_randomCV.csv', index=False)
test = test.drop('target', axis=1) # drop the target column

# Use predict proba
test['target'] = best_model.predict_proba(test_X)[:,1]
test[['id','target']].to_csv('submission_XGBClassifier_randomCV_proba.csv', index=False)
test = test.drop('target', axis=1) # drop the target column


# # Logistic Regression

# In[ ]:


clf = LogisticRegression(class_weight='balanced', penalty='l1', C=0.01, solver='liblinear').fit(train_X, train_y)


# In[ ]:





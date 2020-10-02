#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


# Configuring visualizations
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 12, 8


# In tree-based algorithms, random state has to be fixed to obtain a deterministic behaviour during fitting. We can change RANDOM_STATE in the shell below to obtain different outputs.

# In[ ]:


# Setting random state
RANDOM_STATE = 123


# We'd like to quickly build a messy model before heading into further process of model refinement. Obviously, features like Age, Sex and PClass are more likely to be linked with survival probabilities of passengers. We also want to include features like SibSp, ParCh, Fare, Embarked into our first model before deciding whether to keep them. Features like Name, Cabin and Ticket contain may contain information to be find, but processing these features my be kind of uncertain and bothering, and Cabin contains lots of missing values. We want to leave these features for our last considerastion to engineer.

# In[ ]:


# Loading datasets
train_set = pd.read_csv('../input/titanic/train.csv')
test_set = pd.read_csv('../input/titanic/test.csv')
X_train = train_set.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
X_test = test_set.iloc[:, [1, 3, 4, 5, 6, 8, 10]].values
y_train = train_set.iloc[:, 1].values


# In[ ]:


# Exploring train set
train_set.info()
train_set.describe(include='all')


# In[ ]:


# Exploring test set
test_set.info()
test_set.describe(include='all')


# One of the advantages of tree-based algorithms is that they need little data preprocessing. We still need to encode categorical data and to take care of missing data, but creating dummy variables or feature scaling are not indispensible any more. 

# In[ ]:


# Taking care of missing data (Age, Embarked, Fare)
from sklearn.impute import SimpleImputer
imputer_age = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train[:, 2:3] = imputer_age.fit_transform(X_train[:, 2:3])
X_test[:, 2:3] = imputer_age.fit_transform(X_test[:, 2:3])
imputer_embarked = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X_train[:, 6:7] = imputer_embarked.fit_transform(X_train[:, 6:7])
imputer_fare = SimpleImputer(missing_values=np.nan, strategy='median')
X_test[:, 5:6] = imputer_fare.fit_transform(X_test[:, 5:6])


# In[ ]:


# Encoding categorical data (PClass, Sex, Embarked)
from sklearn.preprocessing import LabelEncoder
labelencoder_pclass = LabelEncoder()
X_train[:, 0] = labelencoder_pclass.fit_transform(X_train[:, 0])
X_test[:, 0] = labelencoder_pclass.transform(X_test[:, 0])
labelencoder_sex = LabelEncoder()
X_train[:, 1] = labelencoder_sex.fit_transform(X_train[:, 1])
X_test[:, 1] = labelencoder_sex.transform(X_test[:, 1])
labelencoder_embarked = LabelEncoder()
X_train[:, 6] = labelencoder_embarked.fit_transform(X_train[:, 6])
X_test[:, 6] = labelencoder_embarked.transform(X_test[: ,6])


# The main parameters to adjust when using Random Forest algorithm is the number of trees and the size of the random subsets of features to consider when splittinga node. The larger the better for the number of trees, but also the longer it will take to compute. In addition, results will stop getting significantly better beyond a critical number of trees. For the size of the random subsets of features, the lower the greater the reduction of variance, but also the greater the increase in bias. We want to snatch an impression about appropriate choices for the two parameters early, since it may be time-consuming to cross-validate. OOB scores provide us with a helpful tool since they allow the Random Forest classifier to be fit and validated whilst being trained. 

# In[ ]:


# Plotting OOB scores against number of trees and number of features
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
ensemble_clfs = [     
                 ('max_features=2', RandomForestClassifier(max_features=2)), 
#                 ('max_features=3', RandomForestClassifier(max_features=3)), 
#                 ('max_features=4', RandomForestClassifier(max_features=4)), 
#                 ('max_features=5', RandomForestClassifier(max_features=5)), 
#                 ('max_features=6', RandomForestClassifier(max_features=6)), 
                 ('max_features=7', RandomForestClassifier(max_features=7))
                 ]
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
[min_estimators, max_estimators] = [20, 750]
for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1, 10):
        clf.set_params(n_estimators=i, warm_start=True, oob_score=True, 
                       min_impurity_decrease=1e-4, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        error_rate[label].append((i, clf.oob_score_))
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)
plt.xlim(min_estimators, max_estimators)
plt.xlabel('n_estimators')
plt.ylabel('OOB error rate')
plt.legend(loc='best')
plt.grid()
plt.show()


# The plot above shows that OOB scores become stable after the number of trees reaches 400. We shall set n_estimator to be 400, and still use cross validation and grid search to find an optimal max_features.

# Now we are ready to build our first Random Forest model. Both OOB score and cross validation score give us a good indication of how our model works. In addition, we can see that these two scores are fairly close to each other. 

# In[ ]:


# Building a Random Forest model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=400, bootstrap=True, 
                                    criterion='gini', max_depth=None, 
                                    min_samples_split=2, min_samples_leaf=1, 
                                    max_features='auto', max_leaf_nodes=None, 
                                    min_impurity_decrease=1e-3, oob_score=True, 
                                    n_jobs=-1, random_state=RANDOM_STATE)
classifier.fit(X_train, y_train)
print('OOB score:', classifier.oob_score_)
print('feature importances:', classifier.feature_importances_)


# In[ ]:


# Cross validation
from sklearn.model_selection import cross_validate
results = cross_validate(classifier, X_train, y_train, cv=10, scoring='accuracy',
                         return_train_score=True, return_estimator=False, n_jobs=-1)
print('train score:', results['train_score'].mean())
print('test score:', results['test_score'].mean())


# We can still plot learning curves to give our model a quick diagnosis. The training curve and the cross-validation cruve didn't converge in the plot below thus showed a moderate sign of overfitting. We can tune the parameters max_features to reduce the variance. In addition, we can control the sizes of trees to reduce overfitting. Other possible measures includes manually deleting some features. 

# In[ ]:


# Plotting learning curves
from sklearn.model_selection import learning_curve
m_range = np.linspace(.05, 1.0, 20)
train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train, cv=10,  
                                                        train_sizes=m_range, shuffle=False,
                                                        scoring='accuracy', n_jobs=-1)
plt.figure()
plt.title('Learning Curves')
plt.ylim(.6, 1.05)
plt.xlabel('Training examples')
plt.ylabel('Score')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
                 test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.legend(loc='best')
plt.show()


# We can use various arguments to control the sizes of trees, such as max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes and min_impurity_decrease. These arguments have similar effects and we choose the last one over others for no specific reasons. 

# In[ ]:


# Plotting validation curves
from sklearn.model_selection import validation_curve
param_range = np.geomspace(1e-5, 1e-1, 21)
train_scores, test_scores = validation_curve(classifier, X_train, y_train, cv=10,
                                             param_name='min_impurity_decrease', 
                                             param_range=param_range, 
                                             scoring='accuracy', n_jobs=-1)
plt.figure()
plt.title('Validation Curves')
plt.ylim(.7, 1.05)
plt.xlabel('Size of Trees')
plt.ylabel('Score')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.semilogx(param_range, train_scores_mean, label='Training score', color='darkorange', lw=2)
plt.semilogx(param_range, test_scores_mean, label='Cross-validation score', color='navy', lw=2)
plt.fill_between(param_range, train_scores_mean - train_scores_std, 
                 train_scores_mean + train_scores_std, alpha=0.2, color='darkorange', lw=2)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color='navy', lw=2)
plt.legend(loc='best')
plt.show()


# In[ ]:


# Grid search
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [400], 'max_features': ['auto', None], 
              'min_impurity_decrease': list(np.geomspace(1e-4, 1e-3, 6))}
grid_search = GridSearchCV(classifier, param_grid, iid=False, refit=True, cv=10,
                           return_train_score=False, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print('best parameters:', grid_search.best_params_)
print('best score:', grid_search.best_score_)


# We also obtained an attribute named feature importance previously. It seems that features SibSp, ParCh and Embarked are not really significant in our model. Let's create a new feature by combining the former two and drop the last one.

# In[ ]:


# Createing a new feature representing family sizes and conducting preprocessing
family_train = X_train[:, 3] + X_train[:, 4] + 1
family_train = family_train.reshape(len(family_train), 1)
family_test = X_test[:, 3] + X_test[:, 4] + 1
family_test = family_test.reshape(len(family_test), 1)
X_train = np.concatenate((X_train[:, :3], family_train, X_train[:, 5:]), axis=1)
X_test = np.concatenate((X_test[:, :3], family_test, X_test[:, 5:]), axis=1)
del family_train, family_test


# In[ ]:


# Dropping feature Embarked
X_train = X_train[:, :-1]
X_test = X_test[:, :-1]


# In[ ]:


# Building a Random Forest model
from sklearn.ensemble import RandomForestClassifier
best_mid = grid_search.best_params_['min_impurity_decrease']
classifier = RandomForestClassifier(n_estimators=400, bootstrap=True, 
                                    criterion='gini', max_depth=None, 
                                    min_samples_split=2, min_samples_leaf=1, 
                                    max_features='auto', max_leaf_nodes=None, 
                                    min_impurity_decrease=best_mid, oob_score=True, 
                                    n_jobs=-1, random_state=RANDOM_STATE)
classifier.fit(X_train, y_train)
print('OOB score:', classifier.oob_score_)
print('feature importances:', classifier.feature_importances_)


# In[ ]:


# Cross validation
from sklearn.model_selection import cross_validate
results = cross_validate(classifier, X_train, y_train, cv=10, scoring='accuracy',
                         return_train_score=True, return_estimator=False, n_jobs=-1)
print('train score:', results['train_score'].mean())
print('test score:', results['test_score'].mean())


# In[ ]:


# Plotting learning curves
from sklearn.model_selection import learning_curve
m_range = np.linspace(.05, 1.0, 20)
train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train, cv=10,  
                                                        train_sizes=m_range, shuffle=False,
                                                        scoring='accuracy', n_jobs=-1)
plt.figure()
plt.title('Learning Curves')
plt.ylim(.6, 1.05)
plt.xlabel('Training examples')
plt.ylabel('Score')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
                 test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.legend(loc='best')
plt.show()


# In[ ]:


# Plotting validation curves
from sklearn.model_selection import validation_curve
param_range = np.geomspace(1e-5, 1e-1, 21)
train_scores, test_scores = validation_curve(classifier, X_train, y_train, cv=10,
                                             param_name='min_impurity_decrease', 
                                             param_range=param_range, 
                                             scoring='accuracy', n_jobs=-1)
plt.figure()
plt.title('Validation Curves')
plt.ylim(.7, 1.05)
plt.xlabel('Size of Trees')
plt.ylabel('Score')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.semilogx(param_range, train_scores_mean, label='Training score', color='darkorange', lw=2)
plt.semilogx(param_range, test_scores_mean, label='Cross-validation score', color='navy', lw=2)
plt.fill_between(param_range, train_scores_mean - train_scores_std, 
                 train_scores_mean + train_scores_std, alpha=0.2, color='darkorange', lw=2)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color='navy', lw=2)
plt.legend(loc='best')
plt.show()


# In[ ]:


# Grid search
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [400], 'max_features': [2, 3, 4, 5], 
              'min_impurity_decrease': list(np.geomspace(1e-4, 1e-2, 11))}
grid_search = GridSearchCV(classifier, param_grid, iid=False, refit=True, cv=10,
                           return_train_score=False, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print('best parameters:', grid_search.best_params_)
print('best score:', grid_search.best_score_)


# Now we can submit our predictions.

# In[ ]:


# Making predictions and submitting
y_pred = grid_search.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_set.iloc[:, 0].values,
                           'Survived': y_pred})
submission.to_csv('submission_RandomForest.csv', index=False)


# References:
# 
# [Decision Trees](https://scikit-learn.org/stable/modules/tree.html#tree)
# 
# [Forests of randomized trees](https://scikit-learn.org/stable/modules/ensemble.html#forest)
# 
# [OOB Scores for Random Forest](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html#)
# 
# [Plotting Learning Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py)
# 
# [Plottign Validation Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py)

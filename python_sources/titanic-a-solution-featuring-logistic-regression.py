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


# Feature Age in both train set and test set have considerably large number of missing values. Feature Embarked in train set has two missing values. Feature Fare in test set has only one missing value. We need to take care of them before next step.

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


# Features Sex and Embarked are categorical data and need to be transformed into numerical data. Further more, features PClass and Embarked contains more than two classes and we need to create dummy variables for our purpose. After this, we can do the feature scaling to transform features into comparable sizes.

# In[ ]:


# Encoding categorical data (Sex)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X_train[:, 1] = labelencoder.fit_transform(X_train[:, 1])
X_test[:, 1] = labelencoder.transform(X_test[:, 1])


# In[ ]:


# Encoding categorical data (PClass, Embarked)
from sklearn.preprocessing import OneHotEncoder
onehotencoder_pclass = OneHotEncoder(categories='auto', drop='first', sparse=False)
pclass_train = onehotencoder_pclass.fit_transform(X_train[:, 0:1])
pclass_test = onehotencoder_pclass.transform(X_test[:, 0:1])
onehotencoder_embarked = OneHotEncoder(categories='auto', drop='first', sparse=False)
embarked_train = onehotencoder_embarked.fit_transform(X_train[:, 6:7])
embarked_test = onehotencoder_embarked.transform(X_test[:, 6:7])
X_train = np.concatenate((pclass_train, X_train[:, 1:6], embarked_train), axis=1)
X_test = np.concatenate((pclass_test, X_test[:, 1:6], embarked_test), axis=1)
del pclass_train, pclass_test, embarked_train, embarked_test


# In[ ]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=True, with_std=True)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Now we are ready to build our first logistic regression model. We want to use cross validation to quickly check how our model works. Though cross validation gives us a mean test score that is not too bad, we want to give our model a diagnosis to figure out how we can improve our model.

# In[ ]:


# Building a logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True,
                                solver='lbfgs', multi_class='auto')


# In[ ]:


# Cross validation
from sklearn.model_selection import cross_validate
results = cross_validate(classifier, X_train, y_train, cv=10, scoring='accuracy',
                         return_train_score=True, return_estimator=False, n_jobs=-1)
print('train score:', results['train_score'].mean())
print('test score:', results['test_score'].mean())


# Learning plot is a very helpful tool for diagnostic prupose. In the learning plot below, learning curves for train score and test score converged quickly and showed no sign of overfitting. By this, we may want to reduce bias to elevate performence of our model. Possible remedial measures include adding features that are not previously included in our model or polynomial features, and of course tuning the penalty parameter C. 

# The following shell could be run repeatedly for our various models.

# In[ ]:


# Plotting learning curves
from sklearn.model_selection import learning_curve
m_range = np.linspace(.05, 1.0, 20)
train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train, cv=10,  
                                                        train_sizes=m_range, shuffle=False,
                                                        scoring='accuracy', n_jobs=-1)
plt.figure()
plt.title('Learning Curves')
plt.ylim(.7, .9)
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


# To figure out the propriate range for penalty parameter C, we may want to plot validation curves. It turns out that the optimal choice for C is around 1e-2, but default setting C=1.0 performs fairly well too. 

# The following shell could be run for our various models.

# In[ ]:


# Plotting validation curves
from sklearn.model_selection import validation_curve
C_range = np.geomspace(1e-3, 1e+1, 41)
train_scores, test_scores = validation_curve(classifier, X_train, y_train, cv=10,
                                             param_name='C', param_range=C_range,
                                             scoring='accuracy', n_jobs=-1)
plt.figure()
plt.title('Validation Curves')
plt.ylim(.6, .9)
plt.xlabel('C')
plt.ylabel('Score')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.semilogx(C_range, train_scores_mean, label='Training score', color='darkorange', lw=2)
plt.semilogx(C_range, test_scores_mean, label='Cross-validation score', color='navy', lw=2)
plt.fill_between(C_range, train_scores_mean - train_scores_std, 
                 train_scores_mean + train_scores_std, alpha=0.2, color='darkorange', lw=2)
plt.fill_between(C_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color='navy', lw=2)
plt.legend(loc='best')
plt.show()


# We want to create a new feature representing family sizes by combining features SibSp and ParCh. The idea is that this new feature may be more imformative and linked with survival probabilities of passengers.

# In[ ]:


# Createing a new feature representing family sizes
train_set['Family'] = train_set['SibSp'] + train_set['Parch'] + 1
test_set['Family'] = test_set['SibSp'] + test_set['Parch'] +1


# In[ ]:


# Conducting preprocessing of the new feature
family_train = train_set['Family'].values
family_train = family_train.reshape(len(family_train), 1)
family_test = test_set['Family'].values
family_test = family_test.reshape(len(family_test), 1)
scaler_family = StandardScaler(with_mean=True, with_std=True)
family_train = scaler_family.fit_transform(family_train)
family_test = scaler_family.fit_transform(family_test)
X_train = np.concatenate((X_train[:, :4], family_train, X_train[:, 6:]), axis=1)
X_test = np.concatenate((X_test[:, :4], family_test, X_test[: ,6:]), axis=1)
del family_train, family_test


# Now we can fitting a model to our new train set. Then we use cross validation to check out whether this engineering improved our model. The test score is slighty better than before and thus we decide to keep this new feature. We can also use the codes above to plot learning curves and validation curves to get a better understanding of how our model performs.

# In[ ]:


# Building a logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True,
                                solver='lbfgs', multi_class='auto')


# In[ ]:


# Cross validation
from sklearn.model_selection import cross_validate
results = cross_validate(classifier, X_train, y_train, cv=10, scoring='accuracy',
                         return_train_score=True, return_estimator=False, n_jobs=-1)
print('train score:', results['train_score'].mean())
print('test score:', results['test_score'].mean())


# Next we want to try banding the new feature representing family sizes. The idea is that type of a family (single/small/large) may be more informative than the number of family numbers aboard ship. 

# In[ ]:


# Banding family sizes
train_set['FamilyType'] = 0
train_set.loc[train_set['Family'] == 1, 'FamilyType'] = 'Single'
train_set.loc[(train_set['Family'] >= 2) & (train_set['Family'] <= 4), 'FamilyType'] = 'Small'
train_set.loc[(train_set['Family'] >= 5), 'FamilyType'] = 'Large'
test_set['FamilyType'] = 0
test_set.loc[test_set['Family'] == 1, 'FamilyType'] = 'Single'
test_set.loc[(test_set['Family'] >= 2) & (test_set['Family'] <= 4), 'FamilyType'] = 'Small'
test_set.loc[(test_set['Family'] >= 5), 'FamilyType'] = 'Large'


# In[ ]:


# Conducting preprocessing of the new feature
family_train = train_set['FamilyType'].values
family_train = family_train.reshape(len(family_train), 1)
family_test = test_set['FamilyType'].values
family_test = family_test.reshape(len(family_test), 1)
onehotencoder_family = OneHotEncoder(categories='auto', drop='first', sparse=False)
family_train = onehotencoder_family.fit_transform(family_train)
family_test = onehotencoder_family.transform(family_test)
scaler_family = StandardScaler(with_mean=True, with_std=True)
family_train = scaler_family.fit_transform(family_train)
family_test = scaler_family.fit_transform(family_test)
X_train = np.concatenate((X_train[:, :4], family_train, X_train[:, 5:]), axis=1)
X_test = np.concatenate((X_test[:, :4], family_test, X_test[:, 5:]), axis=1)
del family_train, family_test


# Again we fit a new model to our new train set and check the validity. The test score is slighty better than before and thus we decide to keep this new feature. We can also use the codes above to plot learning curves and validation curves to get a better understanding of how our model performs.

# In[ ]:


# Building a logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True,
                                solver='lbfgs', multi_class='auto')


# In[ ]:


# Cross validation
from sklearn.model_selection import cross_validate
results = cross_validate(classifier, X_train, y_train, cv=10, scoring='accuracy',
                         return_train_score=True, return_estimator=False, n_jobs=-1)
print('train score:', results['train_score'].mean())
print('test score:', results['test_score'].mean())


# Next we want to try banding feature Age. Again we fit a new model to our new train set and check the validity. The test score is slighty better than before and thus we decide to keep this new feature. We can also use the codes above to plot learning curves and validation curves to get a better understanding of how our model performs.

# In[ ]:


# Banding age
train_set['AgeBand'] = 0
train_set.loc[train_set['Age'] <= 16, 'AgeBand'] = 1
train_set.loc[(train_set['Age'] > 16) & (train_set['Age'] <= 32), 'AgeBand'] = 2
train_set.loc[(train_set['Age'] > 32) & (train_set['Age'] <= 48), 'AgeBand'] = 3
train_set.loc[(train_set['Age'] > 48) & (train_set['Age'] <= 64), 'AgeBand'] = 4
train_set.loc[(train_set['Age'] > 64), 'AgeBand'] = 5
test_set['AgeBand'] = 0
test_set.loc[test_set['Age'] <= 16, 'AgeBand'] = 1
test_set.loc[(test_set['Age'] > 16) & (test_set['Age'] <= 32), 'AgeBand'] = 2
test_set.loc[(test_set['Age'] > 32) & (test_set['Age'] <= 48), 'AgeBand'] = 3
test_set.loc[(test_set['Age'] > 48) & (test_set['Age'] <= 64), 'AgeBand'] = 4
test_set.loc[(test_set['Age'] > 64), 'AgeBand'] = 5


# In[ ]:


# Conducting preprocessing of the new feature
age_train = train_set['AgeBand'].values
age_train = age_train.reshape(len(age_train), 1)
age_test = test_set['AgeBand'].values
age_test = age_test.reshape(len(age_test), 1)
onehotencoder_age = OneHotEncoder(categories='auto', drop='first', sparse=False)
age_train = onehotencoder_age.fit_transform(age_train)
age_test = onehotencoder_age.transform(age_test)
scaler_age = StandardScaler(with_mean=True, with_std=True)
age_train = scaler_age.fit_transform(age_train)
age_test = scaler_age.fit_transform(age_test)
X_train = np.concatenate((X_train[:, :3], age_train, X_train[:, 4:]), axis=1)
X_test = np.concatenate((X_test[:, :3], age_test, X_test[:, 4:]), axis=1)


# In[ ]:


# Building a logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True,
                                solver='lbfgs', multi_class='auto')


# In[ ]:


# Cross validation
from sklearn.model_selection import cross_validate
results = cross_validate(classifier, X_train, y_train, cv=10, scoring='accuracy',
                         return_train_score=True, return_estimator=False, n_jobs=-1)
print('train score:', results['train_score'].mean())
print('test score:', results['test_score'].mean())


# Next we want to try banding feature Fare. Again we fit a new model to our new train set and check the validity. The test score is slighty better than before and thus we decide to keep this new feature. We can also use the codes above to plot learning curves and validation curves to get a better understanding of how our model performs.

# In[ ]:


# Banding fare
train_set['FareBand'] = 0
train_set.loc[train_set['Fare'] <= 8.5, 'FareBand'] = 1
train_set.loc[(train_set['Fare'] > 8.5) & (train_set['Fare'] <= 16.5), 'FareBand'] = 2
train_set.loc[(train_set['Fare'] > 16.5) & (train_set['Fare'] <= 32.5), 'FareBand'] = 3
train_set.loc[(train_set['Fare'] > 32.5), 'FareBand'] = 4
test_set['FareBand'] = 0
test_set.loc[test_set['Fare'] <= 8.5, 'FareBand'] = 1
test_set.loc[(test_set['Fare'] > 8.5) & (test_set['Fare'] <= 16.5), 'FareBand'] = 2
test_set.loc[(test_set['Fare'] > 16.5) & (test_set['Fare'] <= 32.5), 'FareBand'] = 3
test_set.loc[(test_set['Fare'] > 32.5), 'FareBand'] = 4
test_set.loc[test_set['Fare'].isnull(), 'FareBand'] = 2


# In[ ]:


# Conducting preprocessing of the new feature
fare_train = train_set['FareBand'].values
fare_train = fare_train.reshape(len(fare_train), 1)
fare_test = test_set['FareBand'].values
fare_test = fare_test.reshape(len(fare_test), 1)
onehotencoder_fare = OneHotEncoder(categories='auto', drop='first', sparse=False)
fare_train = onehotencoder_fare.fit_transform(fare_train)
fare_test = onehotencoder_fare.transform(fare_test)
scaler_fare = StandardScaler(with_mean=True, with_std=True)
fare_train = scaler_fare.fit_transform(fare_train)
fare_test = scaler_fare.fit_transform(fare_test)
X_train = np.concatenate((X_train[:, :10], fare_train, X_train[:, 11:]), axis=1)
X_test = np.concatenate((X_test[:, :10], fare_test, X_test[:, 11:]), axis=1)
del fare_train, fare_test


# In[ ]:


# Building a logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True,
                                solver='lbfgs', multi_class='auto')


# In[ ]:


# Cross validation
from sklearn.model_selection import cross_validate
results = cross_validate(classifier, X_train, y_train, cv=10, scoring='accuracy',
                         return_train_score=True, return_estimator=False, n_jobs=-1)
print('train score:', results['train_score'].mean())
print('test score:', results['test_score'].mean())


# Now we want to fit a polynomial logistic regression model to see what happens. Note that we have to add polynomial features before conducting feature scaling, which is definitely necessary under this circumstance. 

# In[ ]:


# Loading datasets
train_set = pd.read_csv('../input/titanic/train.csv')
test_set = pd.read_csv('../input/titanic/test.csv')
X_train = train_set.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
X_test = test_set.iloc[:, [1, 3, 4, 5, 6, 8, 10]].values
y_train = train_set.iloc[:, 1].values


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


# Encoding categorical data (Sex)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X_train[:, 1] = labelencoder.fit_transform(X_train[:, 1])
X_test[:, 1] = labelencoder.transform(X_test[:, 1])


# In[ ]:


# Encoding categorical data (PClass, Embarked)
from sklearn.preprocessing import OneHotEncoder
onehotencoder_pclass = OneHotEncoder(categories='auto', drop='first', sparse=False)
pclass_train = onehotencoder_pclass.fit_transform(X_train[:, 0:1])
pclass_test = onehotencoder_pclass.transform(X_test[:, 0:1])
onehotencoder_embarked = OneHotEncoder(categories='auto', drop='first', sparse=False)
embarked_train = onehotencoder_embarked.fit_transform(X_train[:, 6:7])
embarked_test = onehotencoder_embarked.transform(X_test[:, 6:7])
X_train = np.concatenate((pclass_train, X_train[:, 1:6], embarked_train), axis=1)
X_test = np.concatenate((pclass_test, X_test[:, 1:6], embarked_test), axis=1)
del pclass_train, pclass_test, embarked_train, embarked_test


# Feature mapping enables us to add polynomial features into the origianl dataset. We can also try different degrees to obtian the optimal model. Note that degrees greater than three are not appropriate considering the relative sizes between number of samples and number of features.

# In[ ]:


# Feature mapping for polynomial regression and feature scaling
from sklearn.preprocessing import PolynomialFeatures
features = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
X_train = features.fit_transform(X_train)
X_test = features.fit_transform(X_test)


# In[ ]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=True, with_std=True)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[ ]:


# Building a logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True,
                                solver='lbfgs', multi_class='auto')


# In[ ]:


# Cross validation
from sklearn.model_selection import cross_validate
results = cross_validate(classifier, X_train, y_train, cv=10, scoring='accuracy',
                         return_train_score=True, return_estimator=False, n_jobs=-1)
print('train score:', results['train_score'].mean())
print('test score:', results['test_score'].mean())


# The test score is somewhat higher than those we obtained previously. We thus use this model to make our final predictions.  

# In[ ]:


# Plotting learning curves
from sklearn.model_selection import learning_curve
m_range = np.linspace(.05, 1.0, 20)
train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train, cv=10,  
                                                        train_sizes=m_range, shuffle=False,
                                                        scoring='accuracy', n_jobs=-1)
plt.figure()
plt.title('Learning Curves')
plt.ylim(.7, .96)
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
C_range = np.geomspace(1e-4, 1e+2, 31)
train_scores, test_scores = validation_curve(classifier, X_train, y_train, cv=10,
                                             param_name='C', param_range=C_range,
                                             scoring='accuracy', n_jobs=-1)
plt.figure()
plt.title('Validation Curves')
plt.ylim(.6, .9)
plt.xlabel('C')
plt.ylabel('Score')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.semilogx(C_range, train_scores_mean, label='Training score', color='darkorange', lw=2)
plt.semilogx(C_range, test_scores_mean, label='Cross-validation score', color='navy', lw=2)
plt.fill_between(C_range, train_scores_mean - train_scores_std, 
                 train_scores_mean + train_scores_std, alpha=0.2, color='darkorange', lw=2)
plt.fill_between(C_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color='navy', lw=2)
plt.legend(loc='best')
plt.show()


# We can use grid search to choose the optimal choice of C. 

# In[ ]:


# Grid search
from sklearn.model_selection import GridSearchCV
param_grid = {'C': list(np.geomspace(1e-2, 1e-0, 21))}
grid_search = GridSearchCV(classifier, param_grid, iid=False, refit=True, cv=10, 
                           return_train_score=True, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print('best parameters:', grid_search.best_params_)
print('best score:', grid_search.best_score_)


# Now we can submit our predictions.

# In[ ]:


# Making predictions and submitting outputs
y_pred = grid_search.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_set.iloc[:, 0].values,
                           'Survived': y_pred})
submission.to_csv('submission_logistic.csv', index=False)


# References:
# 
# [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
# 
# [Plotting Learning Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py)
# 
# [Plotting Validation Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py)

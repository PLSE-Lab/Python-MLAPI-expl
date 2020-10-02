#!/usr/bin/env python
# coding: utf-8

# <hr/>
# # Comparing various regression models (R^2 = 0.89)
# [Rodrigo Pontes](https://rodrigodlpontes.github.io/website/)
# <hr/>
# 
# In this notebook, we will compare the performance of various regression models. We will also explore the usage of different base estimators with Bagging. Finally, we will tune hyperparameters for Bagging and XGBoost.
# 
# We will focus only on models, since there are already many great kernels that explore data analysis and feature selection. The only feature transformation that will be performed is transforming categorical features into one-hot encoding, which will help some of the models.
# 
# The models that will be compared are:
# * Linear Regression w/ Ridge Regularization
# * Linear Regression** w/ Lasso Regularization
# * Polynomial Regression w/ Ridge Regularization
# * k-NN
# * Bagging w/ Decision Trees
# * Random Forests
# * Extremely Randomized Trees
# * AdaBoost w/ Decision Stumps
# * Gradient Boosting
# * XGBoost
# 
# We will then look at using the following base estimators with Bagging:
# * Support Vector Regression
# * Polynomial Regression w/ Ridge Reguilarization
# * k-NN
# * Decision Trees
# 
# Short descriptions of the models are provided, but if you are already familiar with them they can be ignored.
# 
# If you find this notebook useful, please don't forget to <b><font color="blue">like it</font></b>! *P.S.: This is my very first notebook and I do not have extensive experience with ML, so any feedback would be greatly appreciated :)*

# # Setup
# 
# We start by importing the necessary libraries and classes

# In[ ]:


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from scipy.stats import geom, uniform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Dataset
# 
# Let's import the dataset

# In[ ]:


df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head()


# Get our raw training samples

# In[ ]:


raw_X = df.loc[:,'bedrooms':]
raw_X.head()


# And transform some of the categorical features into one-hot encoding

# In[ ]:


X = raw_X.copy()
for col in ['waterfront', 'view', 'condition', 'grade']:
    one_hot = pd.get_dummies(X[col], prefix=col)
    X = X.drop(col, axis=1)
    X = X.join(one_hot)
X.head()


# Finally, let's get our target outputs

# In[ ]:


y = df['price']
y.head()


# ## Setup
# 
# We create our CV folds, and a helper method to print the mean R^2 score from CV. We also create the list which will store our results.

# In[ ]:


inner_cv = KFold(n_splits=2, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=0)
def print_score(s): print(f'Score mean: {s.mean()}')
results = []


# # Comparing models
# 
# To compare models, we will use a small amount of hyperparameters which will be tuned using Nested CV. We will then use the results of the outer CV (i.e. of each winning hyperparameter combination) for comparisons.
# 
# ### Linear Regression w/ Ridge
# 
# Also frequently just called Ridge Regression, performs Linear Regression with L^2 regularization.

# In[ ]:


params = {
    'alpha': [0.1, 1, 10]
}
lr_gs = GridSearchCV(Ridge(), params, cv=inner_cv)
score = cross_val_score(lr_gs, X, y, cv=outer_cv, n_jobs=4)
results.append(('Ridge Regression', score))
print_score(score)


# ### Linear Regression w/ Lasso
# 
# Also frequently just called Lasso Regression, performs Linear Regression with L^1 regularization.

# In[ ]:


params = {
    'alpha': [0.1, 1, 10]
}
lr_gs = GridSearchCV(Lasso(), params, cv=inner_cv)
score = cross_val_score(lr_gs, X, y, cv=outer_cv, n_jobs=4)
results.append(('Lasso Regression', score))
print_score(score)


# ### Polynomial Regression w/ Ridge
# 
# Fits a polynomial against the data (instead of a hyperplane like in linear regression). To accomplish this, we transform the feature space using a polynomial transformation (thus, we are finding the best fit hyperplane in this new space, or equivalently, the best fit polynomial in the original feature space). We will use Ridge (L^2) regularization.

# In[ ]:


params = {
    'p__degree': [2, 3],
    'r__alpha': [0.1, 1, 10]
}
pr = Pipeline(steps=[('p', PolynomialFeatures()), ('r', Ridge())])
pr_gs = GridSearchCV(pr, param_grid=params, cv=inner_cv)
score = cross_val_score(pr_gs, raw_X, y, cv=outer_cv, n_jobs=4)
results.append(('Polynomial Regression', score))
print_score(score)


# ### k-NN
# 
# Regular k-Nearest Neighbors regression.

# In[ ]:


params = {
    'n_neighbors': [3, 5, 10, 15],
    'weights': ['uniform', 'distance']
}
knn_gs = GridSearchCV(KNeighborsRegressor(), params, cv=inner_cv)
score = cross_val_score(knn_gs, X, y, cv=outer_cv, n_jobs=4)
results.append(('k-NN', score))
print_score(score)


# ### Bagging w/ Decision Trees
# 
# Bagging is a technique that builds an ensemble of models with the main goal of reducing overfitting (variance) of said model without having to change its underlying algorithm. Each model is built by using random samples of the dataset drawn with replacement, and sometimes also a random subset of features (when that is the case, the method is technically known as Random Patches as opposed to Bagging). Both are explored here through the parameters provided to GridSearchCV. Note that using all samples and all features would be redundant as we would simply be building many identical models and averaging those (obtaining the same result as one individual model).
# 
# Decision trees are a very common option for the underlying model, and are in fact the default parameter in sklearn (thus no estimator is specified in the code below). As mentioned, other estimators will be explored in the next section.

# In[ ]:


params = {
    'n_estimators': [10, 50, 100, 250],
    'max_samples': [0.5, 0.75],
    'max_features': [0.5, 0.75, 1.0]
}
rf_gs = GridSearchCV(BaggingRegressor(), params, cv=inner_cv)
score = cross_val_score(rf_gs, X, y, cv=outer_cv, n_jobs=4)
results.append(('Bagging', score))
print_score(score)


# ### Random Forests
# 
# *Note: sklearn's documentation seems to indicate that the below is the case, but feedback would be appreciated as the wording is not 100% clear.*
# 
# Random Forests build on the idea of Bagging and take it a step further by slightly changing how the trees are built (note that with regular Bagging, no modifications are made to how the underlying model works which is why any model can be used). Instead of using a subset of features for each individual model, subsets are obtained when splitting a node. Just like before, using all samples and all features would be redundant, so this is not considered.

# In[ ]:


params = {
    'n_estimators': [10, 50, 100, 250],
    'max_samples': [0.5, 0.75],
    'max_features': [0.5, 0.75, 1.0]
}
rf_gs = GridSearchCV(RandomForestRegressor(), params, cv=inner_cv)
score = cross_val_score(rf_gs, X, y, cv=outer_cv, n_jobs=4)
results.append(('Random Forest', score))
print_score(score)


# ### Extremely Randomized Trees
# 
# Extremely Random Trees takes the idea of Random Forests even further by introducing more randomness in an effort to reduce variance. Instead of choosing the most optimal thresholds for splits, random thresholds for each feature currently being considered are calculated and the best of the most discrimanative amongst these is chosen.

# In[ ]:


params = {
    'n_estimators': [10, 50, 100, 250],
    'max_samples': [0.5, 0.75, 1.0],
    'max_features': [0.5, 0.75, 1.0]
}
rf_gs = GridSearchCV(ExtraTreesRegressor(bootstrap=True), params, cv=inner_cv)
score = cross_val_score(rf_gs, X, y, cv=outer_cv, n_jobs=4)
results.append(('ERT', score))
print_score(score)


# ### AdaBoost w/ Decision Stumps
# 
# AdaBoost is a very popular Boosting algorithm, and was the first practical and useful Boosting algorithm to be developed. Boosting in itself is the idea of combining various weak learners (models that do better than chance) to create a powerful ensemble model. AdaBoost does that by iteratively training models while prioritizing samples that were misclassified in previous iterations. In the context of regression, "misclassification" becomes how far away a prediction was from the true output value. The final estimator is a weighted sum of the weak learners, weighted by how well each performs.
# 
# Like in Bagging, Decision Trees are a common choice for the underlying base estimators. However, Bagging performs best with fully developed and complex models, while Boosting prefers simple models (i.e. weak learners). Thus, short Decision Trees are used, also referred to as "Decision Stumps". This is the default option in sklearn (thus no estimator is specified here either).

# In[ ]:


params = {
    'n_estimators': [10, 50, 100, 250],
    'learning_rate': [0.1, 1.0, 10.0],
    'loss': ['linear', 'square', 'exponential']
}
rf_gs = GridSearchCV(AdaBoostRegressor(), params, cv=inner_cv)
score = cross_val_score(rf_gs, X, y, cv=outer_cv, n_jobs=4)
results.append(('AdaBoost', score))
print_score(score)


# ### Gradient Boosting
# 
# Gradient Boosting is based on the idea that Boosting can be viewed as an optimization algorithm, with each iteration following the negative gradient with respect to the outputs of the model of some differentiable loss function. Although *(I believe)* like Bagging, different base estimators can be used in theory, in sklearn only Decision Trees are implemented (like in AdaBoost, the default parameters make it so that the trees can't become too complex). 

# In[ ]:


params = {
    'n_estimators': [10, 50, 100, 250],
    'learning_rate': [0.1, 1.0, 10.0],
    'loss': ['ls', 'lad', 'huber', 'quantile']
}
rf_gs = GridSearchCV(GradientBoostingRegressor(), params, cv=inner_cv)
score = cross_val_score(rf_gs, X, y, cv=outer_cv, n_jobs=4)
results.append(('Gradient Boosting', score))
print_score(score)


# ### XGBoost
# 
# XGBoost is a very popular gradient tree boosting algorithm, with various features which makes it faster and more powerful.

# In[ ]:


params = {
    'n_estimators': [10, 50, 100, 250],
    'learning_rate': [0.1, 0.5, 1.0]
}
rf_gs = GridSearchCV(XGBRegressor(), params, cv=inner_cv)
score = cross_val_score(rf_gs, X, y, cv=outer_cv, n_jobs=4)
results.append(('XGBoost', score))
print_score(score)


# ### Comparisons
# 
# From the boxplot, we see that the ensemble models performed very well as usual, with the exception of AdaBoost. XGBoost was the best performing algorithm.
# 
# *Note: I found the poor performance on AdaBoost to be quite surprising, if anyone has any insight as to why please share below!*

# In[ ]:


names = list(map(lambda x: x[0], results))
data = list(map(lambda x: x[1], results))
fig, ax = plt.subplots(figsize=(15, 5))
plt.setp(ax.get_xticklabels(), rotation=45)
sns.boxplot(x=names, y=data)
plt.show()


# # Exploring Bagging
# 
# Here we test various regression models as the base estimator for Bagging, since it is the only ensemble approach that performed well and that allows for the underlying model to be changed.

# In[ ]:


inner_cv = KFold(n_splits=2, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=0)
estimator_results = []


# ### SVR as Base Estimator
# 
# Support Vector Machines can be used for regression (usually called "Support Vector Regression" in this context). The approach is similar to regular SVMs, except this time the error is calculated as the absolute difference between the real output and predicted output, and constraints involve making these differences smaller than some small error value (though slack variables are added to allow for some misclassification, similar to "soft margins" in regular SVMs).

# In[ ]:


params = {
    'base_estimator__kernel': ['poly', 'rbf', 'sigmoid'],
    'n_estimators': [10, 25],
    'max_samples': [0.5, 0.75],
    'max_features': [0.5, 0.75, 1.0]
}
rf_svm_gs = GridSearchCV(BaggingRegressor(base_estimator=SVR()), params, cv=inner_cv)
score = cross_val_score(rf_svm_gs, raw_X, y, cv=outer_cv, n_jobs=4)
estimator_results.append(('SVM', score))
print_score(score)


# ### Polynomial Regression as Base Estimator

# In[ ]:


params = {
    'base_estimator__p__degree': [2, 3],
    'base_estimator__r__alpha': [0.1, 1, 10],
    'n_estimators': [10, 25],
    'max_samples': [0.5, 0.75],
    'max_features': [0.5, 0.75, 1.0]
}
pr = Pipeline(steps=[('p', PolynomialFeatures()), ('r', Ridge())])
rf_pr_gs = GridSearchCV(BaggingRegressor(base_estimator=pr), param_grid=params, cv=inner_cv)
score = cross_val_score(rf_pr_gs, raw_X, y, cv=outer_cv, n_jobs=4)
estimator_results.append(('Polynomial Regression', score))
print_score(score)


# ### k-NN as Base Estimator

# In[ ]:


params = {
    'base_estimator__n_neighbors': [3, 5, 10, 15],
    'base_estimator__weights': ['uniform', 'distance'],
    'n_estimators': [10, 25],
    'max_samples': [0.5, 0.75],
    'max_features': [0.5, 0.75, 1.0]
}
rf_knn_gs = GridSearchCV(BaggingRegressor(base_estimator=KNeighborsRegressor()), params, cv=inner_cv)
score = cross_val_score(rf_knn_gs, X, y, cv=outer_cv, n_jobs=4)
estimator_results.append(('k-NN', score))
print_score(score)


# ### Decision Tree as Base Estimator
# 
# *Note: Although I am 99% sure DecisionTreeRegressor is the default estimator as mentioned previously, I decided to explictly set it just to be sure.*

# In[ ]:


params = {
    'base_estimator__criterion': ['mse', 'friedman_mse', 'mae'],
    'base_estimator__splitter': ['best', 'random'],
    'n_estimators': [10, 25],
    'max_samples': [0.5, 0.75],
    'max_features': [0.5, 0.75, 1.0]
}
rf_dt_gs = GridSearchCV(BaggingRegressor(base_estimator=DecisionTreeRegressor()), params, cv=inner_cv)
score = cross_val_score(rf_dt_gs, X, y, cv=outer_cv, n_jobs=4)
estimator_results.append(('Decision Tree', score))
print_score(score)


# ### Comparisons
# 
# We see that Decision Trees outperform all other estimators (although Polynomial Regression is not far off), which further demonstrates why they are usually the default choice for Bagging and Boosting. Polynomial Regression's performance was basically the same, while k-NN showed a slight improvement over not using Bagging. SVR performed very poorly.
# 
# *Note: I am not sure if SVR's terrible performance was due to some mistake on my end, if you know what went wrong please share below!*

# In[ ]:


names = list(map(lambda x: x[0], estimator_results))
data = list(map(lambda x: x[1], estimator_results))
fig, ax = plt.subplots(figsize=(10, 5))
plt.setp(ax.get_xticklabels(), rotation=45)
sns.boxplot(x=names, y=data)
plt.show()


# # Tuning Hyperparameters
# 
# For hyperparameter tuning, I decided to move away from GridSearchCV and use RandomizedSearchCV, which randomly samples from different distributions that are supplied to each hyperparameter (if a list is used, elements from it are sampled uniformly). Only geometric and uniform distributions are used.
# 
# The thought process behind using a geometric distribution for the number of estimators was that we want to smoothly make the usage of more and more estimators less likely. The geometric distribution also makes it so that we pay "more attention" to lower numbers (since it might make a difference to pick 100 or 125 estimators) than higher numbers (since it might not make much difference if we pick 500 or 525 estimators).

# In[ ]:


gs_cv = KFold(n_splits=10, shuffle=True, random_state=0)


# In[ ]:


params = {
    'n_estimators': geom(p=0.01, loc=50 - 1), # Geometric distribution starting at 50 w/ p=0.01
    'max_samples': uniform(loc=0.5, scale=0.9 - 0.5), # [loc, loc + scale]
    'max_features': uniform(loc=0.5, scale=1.0 - 0.5), # [loc, loc + scale]
    'criterion': ['mse', 'mae']
}
rf_gs = RandomizedSearchCV(ExtraTreesRegressor(bootstrap=True), params, cv=gs_cv, n_iter=20, n_jobs=4)
rf_results = rf_gs.fit(X, y).cv_results_
rf_df = pd.DataFrame.from_dict(rf_results)
rf_df.sort_values('rank_test_score').head()


# In[ ]:


params = {
    'n_estimators': geom(p=0.01, loc=50 - 1),
    'learning_rate': uniform(loc=0.1, scale=1.0 - 0.1),
    'booster': ['gbtree', 'gblinear', 'dart']
}
xgb_gs = RandomizedSearchCV(XGBRegressor(), params, cv=gs_cv, n_iter=50, n_jobs=4)
xgb_results = xgb_gs.fit(X, y).cv_results_
xgb_df = pd.DataFrame.from_dict(xgb_results)
xgb_df.sort_values('rank_test_score').head()


# Although RandomSearchCV may not give us nice round numbers like GridSearchCV, it may help give us some idea of a good hyperparameter combination. Since it is faster than GridSearchCV, it can explore the hyperparameter space more in the same amount of time, or give us a fast approximation. The results for the best hyperparameters are listed above.

# # Conclusion
# 
# Unsurprisingly, XGBoost was the best performing algorithm. I am still interested in some more in-depth analysis on tuning its hyperparameters, and also maybe looking at other gradient tree boosting algorithms such as LightGBM and CatBoost, so I may make another notebook on that.
# 
# It was also interesting to see how other algorithms would perform with Bagging, since I was always curious to see if any could beat using Decision Trees.
# 
# Once again, if you found this notebook useful, please don't forget to <b><font color="blue">like it</font></b>! I'm very new to Kaggle so I don't know what people find to be helpful, and that would be a good metric to learn. And please leave any and all feedback!

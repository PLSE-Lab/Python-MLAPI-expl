#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import sklearn
import os
print(os.listdir("../input"))


# In[ ]:


for i in [np,pd,sns,xgb, sklearn]:
    print(i.__version__)


# In[ ]:


import subprocess
from IPython.display import Image
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

seed = 104


# ## Prepare Data
# In all examples we will be dealing with binary classification. Generate 20 dimensional artificial dataset with 1000 samples, where 8 features holding information, 3 are redundant and 2 repeated.

# In[ ]:


X,y= make_classification(n_samples=1000, n_features=20, n_informative=8, n_redundant=3, n_repeated=2, random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=seed)


# In[ ]:


pd.DataFrame(X_train).head(3)


# In[ ]:


pd.Series(y_train).head(3)


# In[ ]:


print("Train label distribution:")
print(Counter(y_train))

print("\nTest label distribution:")
print(Counter(y_test))


# # Single Decision Tree
# This code will create a single decision tree, fit it using training data and evaluate the results using test sample.

# In[ ]:


decision_tree = DecisionTreeClassifier(random_state=seed).fit(X_train, y_train)                # TRAINING THE CLASSIFIER MODEL

decision_tree_y_pred = decision_tree.predict(X_test)                                           # PREDICT THE OUTPUT
decision_tree_y_pred_prob = decision_tree.predict_proba(X_test)

decision_tree_accuracy = accuracy_score(y_test, decision_tree_y_pred)
decision_tree_logloss = log_loss(y_test, decision_tree_y_pred_prob)

print("== Decision Tree ==")
print("Accuracy: {0:.2f}".format(decision_tree_accuracy))
print("Log loss: {0:.2f}".format(decision_tree_logloss))
print("Number of nodes created: {}".format(decision_tree.tree_.node_count))


# We can see two things:
# 
# 1. the log loss score is not very promising (due to the fact that leaves in decision tree outputs either 0 or 1 as probability which is heaviliy penalized in case of errors, but the accuracy score is quite decent,
# 2. the tree is complicated (large number of nodes)
# 
# You can inspect first few predicted outputs, and see that only 2 instances out of 5 were classified correctly.

# In[ ]:



print('True labels:')
print(y_test[:35,])
print('\nPredicted labels:')
print(decision_tree_y_pred[:35,])
print('\nPredicted probabilities:')
print(decision_tree_y_pred_prob[:5,])


# # Adaboost
# Below we are creating a AdaBoost classifier running on 1000 iterations (1000 trees created). Also we are growing decision node up to first split (they are called decision stumps). We are also going to use SAMME algorithm which is inteneded to work with discrete data (output from base_estimator is 0 or 1)

# In[ ]:


adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), algorithm='SAMME', n_estimators=1000, random_state=seed)

adaboost.fit(X_train, y_train)

adaboost_y_pred = adaboost.predict(X_test)
adaboost_y_pred_proba = adaboost.predict_proba(X_test)

adaboost_accuracy = accuracy_score(y_test, adaboost_y_pred)
adaboost_logloss = log_loss(y_test, adaboost_y_pred_proba)

print("== AdaBoost ==")
print("Accuracy: {0:.2f}".format(adaboost_accuracy))
print("Log loss: {0:.2f}".format(adaboost_logloss))


# In[ ]:


print('True labels:')
print(y_test[:25,])
print('\nPredicted labels:')
print(adaboost_y_pred[:25,])
print('\nPredicted probabilities:')
print(adaboost_y_pred_proba[:5,])


# In[ ]:


print("Error: {0:.2f}".format(adaboost.estimator_errors_[0]))
print("Tree importance: {0:.2f}".format(adaboost.estimator_weights_[0]))


# ## Gradient Boosted Trees
# Let's construct a gradient boosted tree consiting of 1000 trees where each successive one will be created with gradient optimization. Again we are going to leave most parameters with their default values, specifiy only maximum depth of the tree to 1 (again decision stumps), and setting warm start for more intelligent computations.

# In[ ]:


gbc = GradientBoostingClassifier(max_depth=1, n_estimators=1000, warm_start=True, random_state=seed)

gbc.fit(X_train, y_train)

gbc_y_pred = gbc.predict(X_test)
gbc_y_pred_proba = gbc.predict_proba(X_test)

gbc_accuracy = accuracy_score(y_test, gbc_y_pred)
gbc_logloss = log_loss(y_test, gbc_y_pred_proba)

print("== Gradient Boosting ==")
print("Accuracy: {0:.2f}".format(gbc_accuracy))
print("Log loss: {0:.2f}".format(gbc_logloss))


# The obtained results are obviously the best of all presented algorithm. We have obtained most accurate algorithm giving more sensible predictions about class probabilities.
# 
# 

# In[ ]:


print('True labels:')
print(y_test[:25,])
print('\nPredicted labels:')
print(gbc_y_pred[:25,])
print('\nPredicted probabilities:')
print(gbc_y_pred_proba[:5,])


# ## Using XGBoost

# In[ ]:


dtrain = xgb.DMatrix('../input/agaricus_train.txt')
dtest = xgb.DMatrix('../input/agaricus_test.txt')


# In[ ]:


print([w for w in dir(xgb) if not w.startswith( "_")])


# In[ ]:


print("Train dataset contains {0} rows and {1} columns".format(dtrain.num_row(), dtrain.num_col()))
print("Test dataset contains {0} rows and {1} columns".format(dtest.num_row(), dtest.num_col()))


# In[ ]:


print("Train possible labels: ", np.unique(dtrain.get_label()))
print("\nTest possible labels: ", np.unique(dtest.get_label()))


# ## Specify training parameters
# Let's make the following assuptions and adjust algorithm parameters to it:
# 
# <ol>we are dealing with binary classification problem ('objective':'binary:logistic'),</ol>
# <ol>we want shallow single trees with no more than 2 levels ('max_depth':2),</ol>
# <ol>we don't any oupout ('silent':1),</ol>
# <ol>we want algorithm to learn fast and aggressively ('eta':1),</ol>
# <ol>we want to iterate only 5 rounds</ol>

# In[ ]:


params = {
    'objective':'binary:logistic',
    'max_depth':2,
    'silent':1,
    'eta':1
}

num_rounds = 5


# In[ ]:


bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds)


# In[ ]:


watchlist = [(dtest,'test'), (dtrain,'train')]
bst = xgb.train(params, dtrain, num_rounds, watchlist)


# ## Make Predicitons

# In[ ]:


preds_prob = bst.predict(dtest)
preds_prob


# In[ ]:


labels = dtest.get_label()
preds = preds_prob > 0.5 # threshold
correct = 0

for i in range(len(preds)):
    if (labels[i] == preds[i]):
        correct += 1

print('Predicted correctly: {0}/{1}'.format(correct, len(preds)))
print('Error: {0:.4f}'.format(1-correct/len(preds)))


# ## Spotting most important features

# In[ ]:


dtrain = xgb.DMatrix('../input/agaricus_train.txt')
dtest = xgb.DMatrix('../input/agaricus_test.txt')


# In[ ]:


params = {
    'objective':'binary:logistic',
    'max_depth':1,
    'silent':1,
    'eta':0.5
}

num_rounds = 5


# In[ ]:


watchlist  = [(dtest,'test'), (dtrain,'train')] # native interface only
bst = xgb.train(params, dtrain, num_rounds, watchlist)


# In[ ]:


trees_dump = bst.get_dump(fmap='../input/featmap.txt', with_stats=True)

for tree in trees_dump:
    print(tree)


# For each split we are getting the following details:
# 
# <ol>which feature was used to make split,</ol>
# <ol>possible choices to make (branches)</ol>
# <ol>gain which is the actual improvement in accuracy brough by that feature. The idea is that before adding a new split on a feature X to the branch there was some wrongly classified elements, after adding the split on this feature, there are two new branches, and each of these branch is more accurate (one branch saying if your observation is on this branch then it should be classified as 1, and the other branch saying the exact opposite),</ol>
# <ol>cover measuring the relative quantity of observations concerned by that feature</ol>
# 
# ## Plotting
# Hopefully there are better ways to figure out which features really matter. We can use built-in function plot_importance that will create a plot presenting most important features due to some criterias. We will analyze the impact of each feature for all splits and all trees and visualize results.
# 
# See which feature provided the most gain:

# In[ ]:


xgb.plot_importance(bst, importance_type='gain', xlabel='Gain')


# In[ ]:


xgb.plot_importance(bst)


# In[ ]:


importances = bst.get_fscore()
importances


# In[ ]:


# create df
importance_df = pd.DataFrame({'Splits': list(importances.values()),'Feature': list(importances.keys())})
importance_df.sort_values(by='Splits', inplace=True)
importance_df.plot(kind='barh', x='Feature', figsize=(8,6), color='orange')


# ## Bias-variance Trade-off

# In[ ]:


from sklearn.model_selection import validation_curve
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from xgboost.sklearn import XGBClassifier
from scipy.sparse import vstack

seed = 123
np.random.seed(seed)


# In[ ]:


print([w for w in dir(xgb.sklearn) if not w.startswith('_')])


# In[ ]:


#X, y = make_classification(n_samples=1000, n_features=20, n_informative=8, n_redundant=3, n_repeated=2, random_state=seed)
seed = 2024
X, y = make_classification(n_samples=10000, n_features=30, n_informative=10, n_redundant=5, n_repeated=3, random_state=seed)


# We will divide into 10 stratified folds (the same distibution of labels in each fold) for testing

# In[ ]:


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cv = skf.split(X,y)
cv 


# In[ ]:


default_params = {
    'objective': 'binary:logistic',
    'max_depth': 1,
    'learning_rate': 0.3,
    'silent': 1.0
}

n_estimators_range = np.linspace(1, 200, 10).astype('int')

train_scores, test_scores = validation_curve(
    XGBClassifier(**default_params),
    X, y,
    param_name = 'n_estimators',
    param_range = n_estimators_range,
    cv=cv,
    scoring='accuracy'
)


# Show the validation curve plot
# 
# 

# In[ ]:


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig = plt.figure(figsize=(10, 6), dpi=100)

plt.title("Validation Curve with XGBoost (eta = 0.3)")
plt.xlabel("number of trees")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.1)

plt.plot(n_estimators_range,
             train_scores_mean,
             label="Training score",
             color="r")

plt.plot(n_estimators_range,
             test_scores_mean, 
             label="Cross-validation score",
             color="g")

plt.fill_between(n_estimators_range, 
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, 
                 alpha=0.2, color="r")

plt.fill_between(n_estimators_range,
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std,
                 alpha=0.2, color="g")

plt.axhline(y=1, color='k', ls='dashed')

plt.legend(loc="best")
plt.show()

i = np.argmax(test_scores_mean)
print("Best cross-validation result ({0:.2f}) obtained for {1} trees".format(test_scores_mean[i], n_estimators_range[i]))


# In[ ]:


default_params = {
    'objective': 'binary:logistic',
    'max_depth': 2, # changed
    'learning_rate': 0.3,
    'silent': 1.0,
    'colsample_bytree': 0.6, # added
    'subsample': 0.7 # added
}

n_estimators_range = np.linspace(1, 200, 10).astype('int')

train_scores, test_scores = validation_curve(
    XGBClassifier(**default_params),
    X, y,
    param_name = 'n_estimators',
    param_range = n_estimators_range,
    cv=cv,
    scoring='accuracy'
)


# In[ ]:


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig = plt.figure(figsize=(10, 6), dpi=100)

plt.title("Validation Curve with XGBoost (eta = 0.3)")
plt.xlabel("number of trees")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.1)

plt.plot(n_estimators_range,
             train_scores_mean,
             label="Training score",
             color="r")

plt.plot(n_estimators_range,
             test_scores_mean, 
             label="Cross-validation score",
             color="g")

plt.fill_between(n_estimators_range, 
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, 
                 alpha=0.2, color="r")

plt.fill_between(n_estimators_range,
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std,
                 alpha=0.2, color="g")

plt.axhline(y=1, color='k', ls='dashed')

plt.legend(loc="best")
plt.show()

i = np.argmax(test_scores_mean)
print("Best cross-validation result ({0:.2f}) obtained for {1} trees".format(test_scores_mean[i], n_estimators_range[i]))


# ## Hyper-parameter Tuning

# In[ ]:


from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from scipy.stats import randint, uniform

# reproducibility
seed = 342
np.random.seed(seed)


# In[ ]:


seed = 2024
X, y = make_classification(n_samples=1000, n_features=30, n_informative=10,
                           n_redundant=5, n_repeated=3, random_state=seed)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cv = skf.split(X,y)


# In[ ]:


print(X.shape)
print(y.shape)
print(X[:2,:])


# In[ ]:


type(train_index)


# ## Grid Search 
# In grid-search we start by defining a dictionary holding possible parameter values we want to test. All combinations will be evaluted.

# In[ ]:


params_grid = {
    'max_depth': [1, 2, 3],
    'n_estimators': [5, 10, 25, 50],
    'learning_rate': np.linspace(1e-16, 1, 3)
}


# Add a dictionary to fixed parameters

# In[ ]:


params_fixed = {
    'objective': 'binary:logistic',
    'silent': 1
}


# Create a GridSearchCV estimator. We will be looking for combination giving the best accuracy.

# In[ ]:


bst_grid = GridSearchCV(estimator=XGBClassifier(**params_fixed, seed=seed),param_grid=params_grid,cv=cv,scoring='accuracy')


# Before running the calculations notice that $3*4*3*10=360$ models will be created to test all combinations. You should always have rough estimations about what is going to happen

# In[ ]:


bst_grid.fit(X, y)


# Now, we can look at all obtained scores, and try to manually see what matters and what not. A quick glance looks that the largeer n_estimators then the accuracy is higher.

# In[ ]:


bst_grid.cv_results_


# ### Filter them manually to get the best combination

# In[ ]:


print("Best accuracy obtained: {0}".format(bst_grid.best_score_))
print("Parameters:")
for key, value in bst_grid.best_params_.items():
    print("\t{}: {}".format(key, value))


# Looking for best parameters is an iterative process. You should start with coarsed-granularity and move to to more detailed values.
# 
# ## Randomized Grid-Search
# When the number of parameters and their values is getting big traditional grid-search approach quickly becomes ineffective. A possible solution might be to randomly pick certain parameters from their distribution. While it's not an exhaustive solution, it's worth giving a shot.
# 
# Create a parameters distribution dictionary:

# In[ ]:


params_dist_grid = {
    'max_depth': [1, 2, 3, 4],
    'gamma': [0, 0.5, 1],
    'n_estimators': randint(1, 1001), # uniform discrete random distribution
    'learning_rate': uniform(), # gaussian distribution
    'subsample': uniform(), # gaussian distribution
    'colsample_bytree': uniform() # gaussian distribution
}


# Initialize RandomizedSearchCV **to randomly pick 10 combinations of parameters**. With this approach you can easily control the number of tested models.

# In[ ]:


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cv = skf.split(X,y)

rs_grid = RandomizedSearchCV(
    estimator=XGBClassifier(**params_fixed, seed=seed),
    param_distributions=params_dist_grid,
    n_iter=10,
    cv=cv,
    scoring='accuracy',
    random_state=seed
)


# In[ ]:


rs_grid.fit(X, y)


# In[ ]:


print([w for w in dir(rs_grid) if not w.startswith('_')])


# In[ ]:


print(rs_grid.best_estimator_)
print(rs_grid.best_params_)
print(rs_grid.best_score_)
#print(rs_grid.score)


#  ## Evaluation Metric
#  What is already available?
# There are already some predefined metrics availabe. You can use them as the input for the eval_metric parameter while training the model.
# 
# <ol>rmse - root mean square error,</ol>
# <ol>mae - mean absolute error,</ol>
# <ol>logloss - negative log-likelihood</ol>
# <ol>error - binary classification error rate. It is calculated as #(wrong cases)/#(all cases). Treat predicted values with probability $p &gt; 0.5$ as positive,</ol>
# <ol>merror - multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases),</ol>
# <ol>auc - area under curve,</ol>
# <ol>ndcg - normalized discounted cumulative gain,</ol>
# <ol>map - mean average precision</ol>
# By default an error metric will be used.

# In[ ]:


params = {
    'objective':'binary:logistic',
    'max_depth':1,
    'silent':1,
    'eta':0.5
}

num_rounds = 5
watchlist  = [(dtest,'test'), (dtrain,'train')]

bst = xgb.train(params, dtrain, num_rounds, watchlist)


# In[ ]:


params['eval_metric'] = 'logloss'
bst = xgb.train(params, dtrain, num_rounds, watchlist)


# You can also use multiple evaluation metrics at one time

# In[ ]:


params['eval_metric'] = ['logloss', 'auc']
bst = xgb.train(params, dtrain, num_rounds, watchlist)


# ## Creating custom evaluation metric
# In order to create our own evaluation metric, the only thing needed to do is to create a method taking two arguments - predicted probabilities and DMatrix object holding training data.
# 
# In this example our classification metric will simply count the number of misclassified examples assuming that classes are positive. You can change this threshold if you want more certainty.
# 
# The algorithm is getting better when the number of misclassified examples is getting lower. Remember to also set the argument maximize=False while training.

# In[ ]:


# custom evaluation metric
def misclassified(pred_probs, dtrain):
    labels = dtrain.get_label() # obtain true labels
    preds = pred_probs > 0.5 # obtain predicted values
    return 'misclassified', np.sum(labels != preds)


# In[ ]:


bst = xgb.train(params, dtrain, num_rounds, watchlist, feval=misclassified, maximize=False)


# You can see that even though the params dictionary is holding eval_metric key these values are being ignored and overwritten by feval.
# 
# ## Extracting the evaluation results
# You can get evaluation scores by declaring a dictionary for holding values and passing it as a parameter for evals_result argument.

# In[ ]:


evals_result = {}
bst = xgb.train(params, dtrain, num_rounds, watchlist, feval=misclassified, maximize=False, evals_result=evals_result)


# In[ ]:


from pprint import pprint
pprint(evals_result)


# ## Early stopping
# There is a nice optimization trick when fitting multiple trees.
# 
# You can train the model until the validation score stops improving. Validation error needs to decrease at least every early_stopping_rounds to continue training. This approach results in simpler model, because the lowest number of trees will be found (simplicity).
# 
# In the following example a total number of 1500 trees is to be created, but we are telling it to stop if the validation score does not improve for last ten iterations.

# In[ ]:


params['eval_metric'] = 'error'
num_rounds = 1500

bst = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=10)


# When using early_stopping_rounds parameter resulting model will have 3 additional fields - bst.best_score, bst.best_iteration and bst.best_ntree_limit.

# In[ ]:


print("Booster best train score: {}".format(bst.best_score))
print("Booster best iteration: {}".format(bst.best_iteration))
print("Booster best number of trees limit: {}".format(bst.best_ntree_limit))


# Also keep in mind that train() will return a model from the last iteration, not the best one.
# 
# 

# ## Cross validating results
# Native package provides an option for cross-validating results (but not as sophisticated as Sklearn package). The next input shows a basic execution. Notice that we are passing only single DMatrix, so it would be good to merge train and test into one object to have more training samples.

# In[ ]:


num_rounds = 10 # how many estimators
hist = xgb.cv(params, dtrain, num_rounds, nfold=10, metrics={'error'}, seed=seed)
hist


# Notice that:
# 
# <ol>by default we get a pandas data frame object (can be changed with as_pandas param),</ol>
# <ol>metrics are passed as an argument (muliple values are allowed),</ol>
# <ol>we can use own evaluation metrics (param feval and maximize),</ol>
# <ol>we can use early stopping feature (param early_stopping_rounds)</ol>

# ## Dealing with missing values

# In[ ]:


# create valid dataset
np.random.seed(seed)

data_v = np.random.rand(10,5) # 10 entities, each contains 5 features
data_v


# In[ ]:


# add some missing values
data_m = np.copy(data_v)

data_m[2, 3] = np.nan
data_m[0, 1] = np.nan
data_m[0, 2] = np.nan
data_m[1, 0] = np.nan
data_m[4, 4] = np.nan
data_m[7, 2] = np.nan
data_m[9, 1] = np.nan

data_m


# In[ ]:


np.random.seed(seed)

label = np.random.randint(2, size=10) # binary target
label


# ### Native interface
# In this case we will check how does the native interface handles missing data. Begin with specifing default parameters

# In[ ]:


# specify general training parameters
params = {
    'objective':'binary:logistic',
    'max_depth':1,
    'silent':1,
    'eta':0.5
}

num_rounds = 5


# In the experiment first we will create a valid DMatrix (with all values), see if it works ok, and then repeat the process with lacking one.
# 
# 

# In[ ]:


dtrain_v = xgb.DMatrix(data_v, label=label)

xgb.cv(params, dtrain_v, num_rounds, seed=seed)


# The output obviously doesn't make sense, because the data is completely random.
# 
# When creating DMatrix holding missing values we have to explicitly tell what denotes that it's missing. Sometimes it might be 0, 999 or others. In our case it's Numpy's NAN. Add missing argument to DMatrix constructor to handle it.

# In[ ]:


dtrain_m = xgb.DMatrix(data_m, label=label, missing=np.nan)

xgb.cv(params, dtrain_m, num_rounds, seed=seed)


# It looks like the algorithm works also with missing values.
# 
# In XGBoost chooses a soft way to handle missing values.
# 
# When using a feature with missing values to do splitting, XGBoost will assign a direction to the missing values instead of a numerical value.
# 
# Specifically, XGBoost guides all the data points with missing values to the left and right respectively, then choose the direction with a higher gain with regard to the objective.
# 
# ### Sklearn wrapper
# The following section shows how to validate the same behaviour using Sklearn interface.
# 
# Begin with defining parameters and creating an estimator object.

# In[ ]:


params = {
    'objective': 'binary:logistic',
    'max_depth': 1,
    'learning_rate': 0.5,
    'silent': 1.0,
    'n_estimators': 5
}

clf = XGBClassifier(**params)
clf


# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score(clf, data_v, label, cv=2, scoring='accuracy')


# In[ ]:


cross_val_score(clf, data_m, label, cv=2, scoring='accuracy')


# ## Handling Imbalanced Datasets

# In[ ]:


from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# reproducibility
seed = 123


# In[ ]:


X, y = make_classification(
    n_samples=200,
    n_features=5,
    n_informative=3,
    n_classes=2,
    weights=[.9, .1],
    shuffle=True,
    random_state=seed
)

print('There are {} positive instances.'.format(y.sum()))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=seed)

print('Total number of postivie train instances: {}'.format(y_train.sum()))
print('Total number of positive test instances: {}'.format(y_test.sum()))


# ## Baseline model
# In this approach try to completely ignore the fact that classed are imbalanced and see how it will perform. Create DMatrix for train and test data.

# In[ ]:


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)


# Assume that we will create 15 decision tree stumps, solving binary classification problem, where each next one will be train very aggressively.
# 
# These parameters will also be used in consecutive examples.

# In[ ]:


params = {
    'objective':'binary:logistic',
    'max_depth':1,
    'silent':1,
    'eta':1
}

num_rounds = 15


# Train the booster and make predictions.

# In[ ]:


bst = xgb.train(params, dtrain, num_rounds)
y_test_preds = (bst.predict(dtest) > 0.5).astype('int')


# In[ ]:


pd.crosstab(
    pd.Series(y_test, name='Actual'),
    pd.Series(y_test_preds, name='Predicted'),
    margins=True
)


# In[ ]:


print('Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_test_preds)))
print('Precision: {0:.2f}'.format(precision_score(y_test, y_test_preds)))
print('Recall: {0:.2f}'.format(recall_score(y_test, y_test_preds)))


# Intuitively we know that the foucs should be on finding positive samples. First results are very promising (94% accuracy - wow), but deeper analysis show that the results are biased towards majority class - we are very poor at predicting the actual label of positive instances. That is called an accuracy paradox.
# 
# ## Custom weights
# Try to explicitly tell the algorithm what important using relative instance weights. Let's specify that positive instances have 5x more weight and add this information while creating DMatrix.

# In[ ]:


weights = np.zeros(len(y_train))
weights[y_train == 0] = 1
weights[y_train == 1] = 5

dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights) # weights added
dtest = xgb.DMatrix(X_test)


# In[ ]:


bst = xgb.train(params, dtrain, num_rounds)
y_test_preds = (bst.predict(dtest) > 0.5).astype('int')

pd.crosstab(
    pd.Series(y_test, name='Actual'),
    pd.Series(y_test_preds, name='Predicted'),
    margins=True
)


# In[ ]:


print('Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_test_preds)))
print('Precision: {0:.2f}'.format(precision_score(y_test, y_test_preds)))
print('Recall: {0:.2f}'.format(recall_score(y_test, y_test_preds)))


# You see that we made a trade-off here. We are now able to better classify the minority class, but the overall accuracy and precision decreased. Test multiple weights combinations and see which one works best.
# 
# ## Use scale_pos_weight parameter
# You can automate the process of assigning weights manually by calculating the proportion between negative and positive instances and setting it to scale_pos_weight parameter.
# 
# Let's reinitialize datasets.

# In[ ]:


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)


# Calculate the ratio between both classes and assign it to a parameter.

# In[ ]:


train_labels = dtrain.get_label()

ratio = float(np.sum(train_labels == 0)) / np.sum(train_labels == 1)
params['scale_pos_weight'] = ratio


# In[ ]:


bst = xgb.train(params, dtrain, num_rounds)
y_test_preds = (bst.predict(dtest) > 0.5).astype('int')

pd.crosstab(
    pd.Series(y_test, name='Actual'),
    pd.Series(y_test_preds, name='Predicted'),
    margins=True
)


# In[ ]:


print('Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_test_preds)))
print('Precision: {0:.2f}'.format(precision_score(y_test, y_test_preds)))
print('Recall: {0:.2f}'.format(recall_score(y_test, y_test_preds)))


# You can see that scalling weight by using scale_pos_weights in this case gives better results that doing it manually. We are now able to perfectly classify all posivie classes (focusing on the real problem). On the other hand the classifier sometimes makes a mistake by wrongly classifing the negative case into positive (producing so called false positives).

# In[ ]:





# In[ ]:





# In[ ]:





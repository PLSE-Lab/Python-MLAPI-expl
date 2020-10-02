#!/usr/bin/env python
# coding: utf-8

# # Ensemble Machine Learning Algorithms in Python with scikit-learn
# 
# > Ensembles can give you a boost in accuracy on your dataset.
# ![0_yMbDVA-mPWvzFXCM.png](attachment:0_yMbDVA-mPWvzFXCM.png)
# > In this notebook you will discover how you can create some of the most powerful types of ensembles in Python using scikit-learn.
# 
# # Combine Model Predictions Into Ensemble Predictions
# 
# The three most popular methods for combining the predictions from different models are:
# 
# - **Bagging**. Building multiple models (typically of the same type) from different subsamples of the training dataset.
# - **Boosting**. Building multiple models (typically of the same type) each of which learns to fix the prediction errors of a prior model in the chain.
# - **Voting**. Building multiple models (typically of differing types) and simple statistics (like calculating the mean) are used to combine predictions.
# 
# ***
# 
# A standard classification problem used to demonstrate each ensemble algorithm is the Pima Indians onset of diabetes dataset. It is a binary classification problem where all of the input variables are numeric and have differing scales.
# ***

# # Pima Indians Diabetes Database
# 
# ## 1. Problem Definition
# 
# In a statement,
# > Given clinical parameters about a patient, can we predict whether or not they have diabetes?
# 
# ## 2. Features
# 
# > This is where you'll get different information about each of the features in your data. You can do this via doing your own research (such as looking at the links above) or by talking to a subject matter expert (someone who knows about the dataset).
# 
# **Create data dictionary**
# 
# > The datasets consist of several medical predictor (independent) variables and one target (dependent) variable, Outcome. Independent variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
# 
# [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
# ***
# ### **I hope you find this kernel useful and your <font color="red"><b>UPVOTES</b></font> would be highly appreciated**
# ***

# ## Importing the libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# ## Loading the data

# In[ ]:


df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


pd.set_option('display.float_format', '{:.2f}'.format)
df.describe()


# In[ ]:


categorical_val = []
continous_val = []
for column in df.columns:
    print('==============================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)


# # 1. Data visualization

# In[ ]:


# Visulazing the distibution of the data for every feature
plt.figure(figsize=(20, 20))

for i, column in enumerate(df.columns, 1):
    plt.subplot(3, 3, i)
    df[df["Outcome"] == 0][column].hist(bins=35, color='blue', label='Have Diabetes = NO', alpha=0.6)
    df[df["Outcome"] == 1][column].hist(bins=35, color='red', label='Have Diabetes = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[ ]:


plt.figure(figsize=(30, 30))
sns.pairplot(df, hue='Outcome', height=3, diag_kind='hist')


# In[ ]:


# Let's first check gender
sns.catplot('Outcome', data=df, kind='count')


# In[ ]:


# Another way to visualize the data is to use FacetGrid to plot multiple kedplots on one plot

fig = sns.FacetGrid(df, hue="Outcome", aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = df['Age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()


# # 2. Data Pre-Processing

# In[ ]:


df.columns


# In[ ]:


# How many missing zeros are mising in each feature
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for column in feature_columns:
    print("============================================")
    print(f"{column} ==> Missing zeros : {len(df.loc[df[column] == 0])}")


# In[ ]:


from sklearn.impute import SimpleImputer

fill_values = SimpleImputer(missing_values=0, strategy="mean", copy=False)

df[feature_columns] = fill_values.fit_transform(df[feature_columns])


# In[ ]:


for column in feature_columns:
    print("============================================")
    print(f"{column} ==> Missing zeros : {len(df.loc[df[column] == 0])}")


# In[ ]:


# Visulazing the distibution of the data for every feature
plt.figure(figsize=(20, 20))

for i, column in enumerate(df.columns, 1):
    plt.subplot(3, 3, i)
    df[df["Outcome"] == 0][column].hist(bins=35, color='blue', label='Have Diabetes = NO', alpha=0.6)
    df[df["Outcome"] == 1][column].hist(bins=35, color='red', label='Have Diabetes = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[ ]:


from sklearn.model_selection import train_test_split


X = df[feature_columns]
y = df.Outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n===========================================")
        print(f"accuracy score: {accuracy_score(y_train, pred):.4f}\n")
        print(f"Classification Report: \n \tPrecision: {precision_score(y_train, pred)}\n\tRecall Score: {recall_score(y_train, pred)}\n\tF1 score: {f1_score(y_train, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, clf.predict(X_train))}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        print("Test Result:\n===========================================")        
        print(f"accuracy score: {accuracy_score(y_test, pred)}\n")
        print(f"Classification Report: \n \tPrecision: {precision_score(y_test, pred)}\n\tRecall Score: {recall_score(y_test, pred)}\n\tF1 score: {f1_score(y_test, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# # 3. Bagging Algorithms
# Bootstrap Aggregation or bagging involves taking multiple samples from your training dataset (with replacement) and training a model for each sample.
# 
# The final output prediction is averaged across the predictions of all of the sub-models.
# 
# The three bagging models covered in this section are as follows:
# 
# 1. Bagged Decision Trees
# 2. Random Forest
# 3. Extra Trees

# ## 3. 1. Bagged Decision Trees
# Bagging performs best with algorithms that have high variance. A popular example are decision trees, often constructed without pruning.
# 
# **BaggingClassifier**:
# 
# A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.
# 
# This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting. If samples are drawn with replacement, then the method is known as Bagging. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches.
# 
# **BaggingClassifier Parameters:**
# - `base_estimator` : The base estimator to fit on random subsets of the dataset. If None, then the base estimator is a decision tree.
# ***
# - `n_estimators` : The number of base estimators in the ensemble.
# ***
# - `max_samples` : The number of samples to draw from X to train each base estimator.
# ***
# - `max_features` : The number of features to draw from X to train each base estimator.
# ***
# - `bootstrap` : Whether samples are drawn with replacement. If False, sampling without replacement is performed.
# ***
# - `bootstrap_features` : Whether features are drawn with replacement.
# ***
# - `oob_score` : Whether to use out-of-bag samples to estimate the generalization error.
# ***
# - `warm_start` : When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble.

# In[ ]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
bagging_clf = BaggingClassifier(base_estimator=tree, n_estimators=1500, random_state=42)
bagging_clf.fit(X_train, y_train)


# In[ ]:


print_score(bagging_clf, X_train, y_train, X_test, y_test, train=True)
print_score(bagging_clf, X_train, y_train, X_test, y_test, train=False)


# ## 3. 2. Random Forest
# 
# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
# 
# The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if `bootstrap=True` (default).
# 
# - **Random forest algorithm parameters:**
# - `n_estimators`: The number of trees in the forest.
# *** 
# - `criterion`: The function to measure the quality of a split. Supported criteria are "`gini`" for the Gini impurity and "`entropy`" for the information gain.
# ***
# - `max_depth`: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than `min_samples_split` samples.
# ***
# - `min_samples_split`: The minimum number of samples required to split an internal node.
# ***
# - `min_samples_leaf`: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches.  This may have the effect of smoothing the model, especially in regression.
# ***
# - `min_weight_fraction_leaf`: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
# ***
# - `max_features`: The number of features to consider when looking for the best split.
# ***
# - `max_leaf_nodes`: Grow a tree with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
# ***
# - `min_impurity_decrease`: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
# ***
# - `min_impurity_split`: Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
# ***
# - `bootstrap`: Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
# ***
# - `oob_score`: Whether to use out-of-bag samples to estimate the generalization accuracy.
# ***
# - `warm_start` : When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rand_forest = RandomForestClassifier(random_state=42, n_estimators=1000)
rand_forest.fit(X_train, y_train)


# In[ ]:


print_score(rand_forest, X_train, y_train, X_test, y_test, train=True)
print_score(rand_forest, X_train, y_train, X_test, y_test, train=False)


# ## 3. 3. Extra Trees
# Extra Trees are another modification of bagging where random trees are constructed from samples of the training dataset.
# 
# You can construct an Extra Trees model for classification using the ExtraTreesClassifier class.
# 
# **ExtraTreeClassifier**:
# 
# This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
# 
# **ExtraTreeClassifier Parameters**:
# - `n_estimators`: The number of trees in the forest.
# *** 
# - `criterion`: The function to measure the quality of a split. Supported criteria are "`gini`" for the Gini impurity and "`entropy`" for the information gain.
# ***
# - `max_depth`: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than `min_samples_split` samples.
# ***
# - `min_samples_split`: The minimum number of samples required to split an internal node.
# ***
# - `min_samples_leaf`: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches.  This may have the effect of smoothing the model, especially in regression.
# ***
# - `min_weight_fraction_leaf`: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
# ***
# - `max_features`: The number of features to consider when looking for the best split.
# ***
# - `max_leaf_nodes`: Grow a tree with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
# ***
# - `min_impurity_decrease`: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
# ***
# - `min_impurity_split`: Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
# ***
# - `bootstrap`: Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
# ***
# - `oob_score`: Whether to use out-of-bag samples to estimate the generalization accuracy.
# ***
# - `warm_start` : When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble.

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier

extra_tree_clf = ExtraTreesClassifier(n_estimators=1000, max_features=7, random_state=42)
extra_tree_clf.fit(X_train, y_train)


# In[ ]:


print_score(extra_tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(extra_tree_clf, X_train, y_train, X_test, y_test, train=False)


# # 4. Boosting Algorithms
# Boosting ensemble algorithms creates a sequence of models that attempt to correct the mistakes of the models before them in the sequence.
# 
# Once created, the models make predictions which may be weighted by their demonstrated accuracy and the results are combined to create a final output prediction.
# 
# The two most common boosting ensemble machine learning algorithms are:
# 
# 1. AdaBoost
# 2. Stochastic Gradient Boosting
# ***

# ## 4. 1. AdaBoost
# AdaBoost was perhaps the first successful boosting ensemble algorithm. It generally works by weighting instances in the dataset by how easy or difficult they are to classify, allowing the algorithm to pay or or less attention to them in the construction of subsequent models.
# 
# You can construct an AdaBoost model for classification using the AdaBoostClassifier class.
# 
# **AdaBoostClassifier**:
# 
# An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.
# 
# **AdaBoostClassifier Params**:
# - `base_estimator` : The base estimator from which the boosted ensemble is built.
# ***
# - `n_estimators` : The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.
# ***
# - `learning_rate` : Learning rate shrinks the contribution of each classifier by ``learning_rate``. There is a trade-off between ``learning_rate`` and ``n_estimators``.
# ***
# - `algorithm` : If 'SAMME.R' then use the SAMME.R real boosting algorithm. ``base_estimator`` must support calculation of class probabilities. If 'SAMME' then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

ada_boost_clf = AdaBoostClassifier(n_estimators=30)
ada_boost_clf.fit(X_train, y_train)


# In[ ]:


print_score(ada_boost_clf, X_train, y_train, X_test, y_test, train=True)
print_score(ada_boost_clf, X_train, y_train, X_test, y_test, train=False)


# ## 4. 2. Stochastic Gradient Boosting
# Stochastic Gradient Boosting (also called Gradient Boosting Machines) are one of the most sophisticated ensemble techniques. It is also a technique that is proving to be perhaps of the the best techniques available for improving performance via ensembles.
# 
# **GradientBoostingClassifier**:
# 
# GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage ``n_classes_`` regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. Binary classification is a special case where only a single regression tree is induced.
# 
# **GradientBoostingClassifier Parameters**:
# 
# - `loss` : loss function to be optimized. 'deviance' refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss 'exponential' gradient boosting recovers the AdaBoost algorithm.
# ***
# - `learning_rate` : learning rate shrinks the contribution of each tree by `learning_rate`. There is a trade-off between learning_rate and n_estimators.
# ***
# - `n_estimators` : The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
# ***
# - `subsample` : The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. `subsample` interacts with the parameter `n_estimators`. Choosing `subsample < 1.0` leads to a reduction of variance and an increase in bias.
# ***
# - `criterion` : The function to measure the quality of a split. Supported criteria are "friedman_mse" for the mean squared error with improvement score by Friedman, "mse" for mean squared error, and "mae" for the mean absolute error. The default value of "friedman_mse" is generally the best as it can provide a better approximation in some cases.
# ***
# - `min_samples_split`: The minimum number of samples required to split an internal node.
# ***
# - `min_samples_leaf`: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches.  This may have the effect of smoothing the model, especially in regression.
# ***
# - `min_weight_fraction_leaf`: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
# ***
# - `max_depth`: maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.
# ***
# - `min_impurity_decrease`: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
# ***
# - `min_impurity_split`: Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
# ***
# - `max_features`: The number of features to consider when looking for the best split.
# ***
# - `max_leaf_nodes`: Grow trees with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
# ***
# - `warm_start`: When set to ``True``, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just erase the previous solution.
# ***
# - `validation_fraction`: The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if ``n_iter_no_change`` is set to an integer.
# ***
# - `n_iter_no_change`: used to decide if early stopping will be used to terminate training when validation score is not improving. By default it is set to None to disable early stopping. If set to a number, it will set aside ``validation_fraction`` size of the training data as validation and terminate training when validation score is not improving in all of the previous ``n_iter_no_change`` numbers of iterations. The split is stratified.
# ***
# - `tol`: Tolerance for the early stopping. When the loss is not improving by at least tol for ``n_iter_no_change`` iterations (if set to a number), the training stops.
# ***
# - `ccp_alpha`: Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ``ccp_alpha`` will be chosen.
# ***

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

grad_boost_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
grad_boost_clf.fit(X_train, y_train)


# In[ ]:


print_score(grad_boost_clf, X_train, y_train, X_test, y_test, train=True)
print_score(grad_boost_clf, X_train, y_train, X_test, y_test, train=False)


# # 5. Voting Ensemble
# 
# Voting is one of the simplest ways of combining the predictions from multiple machine learning algorithms.
# 
# It works by first creating two or more standalone models from your training dataset. A Voting Classifier can then be used to wrap your models and average the predictions of the sub-models when asked to make predictions for new data.
# 
# The predictions of the sub-models can be weighted, but specifying the weights for classifiers manually or even heuristically is difficult. More advanced methods can learn how to best weight the predictions from submodels, but this is called stacking (stacked generalization) and is currently not provided in scikit-learn.
# 
# **VotingClassifier** : 
# - `estimators` : Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones of those original estimators that will be stored in the class attribute ``self.estimators_``.
# ***
# - `voting` : If 'hard', uses predicted class labels for majority rule voting. Else if 'soft', predicts the class label based on the argmax of the sums of the predicted probabilities, which is recommended for an ensemble of well-calibrated classifiers.
# ***

# In[ ]:


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

estimators = []
log_reg = LogisticRegression(solver='liblinear')
estimators.append(('Logistic', log_reg))

tree = DecisionTreeClassifier()
estimators.append(('Tree', tree))

svm_clf = SVC(gamma='scale')
estimators.append(('SVM', svm_clf))

voting = VotingClassifier(estimators=estimators)
voting.fit(X_train, y_train)


# In[ ]:


print_score(voting, X_train, y_train, X_test, y_test, train=True)
print_score(voting, X_train, y_train, X_test, y_test, train=False)


# ## Summary
# In this notebook we discovered ensemble machine learning algorithms for improving the performance of models on your problems.
# 
# You learned about:
# 
# 1. Bagging Ensembles including Bagged Decision Trees, Random Forest and Extra Trees.
# 2. Boosting Ensembles including AdaBoost and Stochastic Gradient Boosting.
# 3. Voting Ensembles for averaging the predictions for any arbitrary models.
# 
# ## References:
# - [Bagging and Random Forest Ensemble Algorithms for Machine Learning](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/)
# - [Ensemble Machine Learning Algorithms in Python with scikit-learn](https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/)
# [Scikit-learn Library](https://scikit-learn.org/stable/)

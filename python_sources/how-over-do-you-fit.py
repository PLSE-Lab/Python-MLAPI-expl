#!/usr/bin/env python
# coding: utf-8

# # How Over Do You Fit?
# *Abstract*.  We create validation curves for some basic models trained on the data of the Don't Overfit! II competition.
# 
# ## Introduction
# The [Dont't Overfit! II competition](https://www.kaggle.com/c/dont-overfit-ii) challenges us to model a binary target depending on 200 continuous variables without overfitting, using only 250 training samples.  For low-dimensional problems, whether a model underfits or overfits can be easily decided by visual inspection, as is nicely explained in the scikit-learn example [Underfitting vs. Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html).
# ![image](https://scikit-learn.org/stable/_images/sphx_glr_plot_underfitting_overfitting_001.png)
# Quoting from there, "the plot shows the function that we want to approximate, which is a part of the cosine function.  In addition, the samples from the real function and the approximations of different models are displayed.  The models have polynomial features of different degrees. We can see that a linear function (polynomial with degree 1) is not sufficient to fit the training samples. This is called underfitting. A polynomial of degree 4 approximates the true function almost perfectly. However, for higher degrees the model will overfit the training data, i.e. it learns the noise of the training data."
# 
# However, as stated in the scikit-learn [User Guide](https://scikit-learn.org/stable/modules/learning_curve.html), "in the simple one-dimensional problem that we have seen in the example it is easy to see whether the estimator suffers from bias or variance.  However, in high-dimensional spaces, models can become very difficult to visualize.''  Let's have a look at the scikit-learn example [Plotting Validation Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html):
# ![image](https://scikit-learn.org/stable/_images/sphx_glr_plot_validation_curve_001.png)
# Quoting from there, "in this plot you can see the training scores and validation scores of an SVM for different values of the kernel parameter gamma.  For very low values of gamma, you can see that both the training score and the validation score are low. This is called underfitting. Medium values of gamma will result in high values for both scores, i.e. the classifier is performing fairly well.  If gamma is too high, the classifier will overfit, which means that the training score is good but the validation score is poor."
# 
# So let's apply this idea to the Don't Overfit! II competition.
# 
# As usual, we start by loading some libraries (input hidden).

# In[ ]:


import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

from scipy.stats                   import randint, lognorm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy                 import DummyClassifier
from sklearn.ensemble              import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection     import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model          import LogisticRegression, SGDClassifier
from sklearn.metrics               import roc_auc_score
from sklearn.model_selection       import cross_val_score, LeaveOneOut, ParameterGrid, GridSearchCV, RandomizedSearchCV, validation_curve, StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.pipeline              import Pipeline, make_pipeline
from sklearn.preprocessing         import StandardScaler
from sklearn.svm                   import SVC


# Then we load the training data and convert it to scikit-learn's (X, y) format.  The output is hidden since everybody is probably already familiar with it.

# In[ ]:


train = pd.read_csv('../input/train.csv')
X = train.drop(['id', 'target'], axis=1)
y = train['target']
train


# Let's use the area under the ROC curve as the scoring variable as in the competition, and use stratified 5 times cross validation repeated 10 times as the cross validation strategy.

# In[ ]:


scoring='roc_auc'
cv=RepeatedStratifiedKFold(5, 10, random_state=0)


# The following function is a wrapper around [validation_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html) that acually plots the output, slightly adapted from the scikit-learn example [Plotting Validation Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html):

# In[ ]:


def plot_validation_curve(estimator, param_name, param_range, X=X, y=y, groups=None, cv=cv, scoring=scoring, n_jobs=None, pre_dispatch='all', verbose=0, error_score='raise-deprecating'):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, groups, cv, scoring, n_jobs, pre_dispatch, verbose, error_score)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std (train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores,  axis=1)
    test_scores_std   = np.std (test_scores,  axis=1)
    
    plt.suptitle(estimator.__class__.__name__)
    plt.xlabel(param_name)
    if scoring is not None: plt.ylabel(scoring) 
    lw = 2
    param_range = [str(x) for x in param_range] # make axis categorical so we don't have to think about scaling
    plt.plot(param_range, train_scores_mean, label='Training score', color='darkorange', lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label='Cross-validation score', color='navy', lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
    plt.legend(loc='best')
    plt.show()


# So let's have a look at some examples.
# 
# ## Dummy classifiers
# scikit-learn contains some [dummy classifiers](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators) that implement simple rules of thumb.  These should certainly not overfit, but let's verify:

# In[ ]:


plot_validation_curve(DummyClassifier(random_state=0),
                      'strategy', ['stratified', 'most_frequent', 'prior', 'uniform'])


# ## $k$ nearest neighbors
# For uniformly weighted [$k$ nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html#classification), overfitting reduces for large $k$, and vanishes around $k = 120$.  Note that for $k = 200$ for stratified 5 fold cross validation the classifier predicts simply the class probabilities in the entire test set, to the area under the ROC curve drops to $0$.

# In[ ]:


plot_validation_curve(KNeighborsClassifier(),
                      'n_neighbors', [20, 40, 60, 80, 100, 120, 140, 160, 180, 200])


# Standardization does not substantially change this, but seems to add a bit of overfitting since the standardization adds more parameters to be fitted.  (Most public kernels in the competition that do standardization do this on the entire training set outside the cross validation loop, so they fail to detect this effect.)

# In[ ]:


plot_validation_curve(Pipeline([('std', StandardScaler()),
                                ('clf', KNeighborsClassifier())]),
                      'clf__n_neighbors', [20, 40, 60, 80, 100, 120, 140, 160, 180, 200])


# For distance weighted $k$ nearest neighbors, the model overfits for all $k$ as to be expected, but the cross-validation score increases for larger $k$.

# In[ ]:


plot_validation_curve(KNeighborsClassifier(weights='distance'),
                      'n_neighbors', [20, 40, 60, 80, 100, 120, 140, 160, 180, 200])


# ## Logistic regression
# [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) overfits for the $l_2$ norm.

# In[ ]:


plot_validation_curve(LogisticRegression(penalty='l2',
                                         solver='liblinear'),
                      "C", [0.001, 0.01, 0.1, 1, 10, 100, 1000])


# For the $l_1$ norm, overfitting starts for $C > 0.04$.

# In[ ]:


plot_validation_curve(LogisticRegression(penalty='l1',
                                         solver='liblinear'),
                      'C', [.01, .02, .03, .04, .05, .06, .07, .08, .09, .1])


# Using balanced class weights does improve performance for $C = 0.03$ but still overfits for $C > 0.04$.

# In[ ]:


plot_validation_curve(LogisticRegression(penalty='l1',
                                         solver='liblinear',
                                         class_weight='balanced'),
                      'C', [.01, .02, .03, .04, .05, .06, .07, .08, .09, .1])


# Standardization does not change the situation.

# In[ ]:


plot_validation_curve(Pipeline([('std', StandardScaler()),
                                ('clf', LogisticRegression(penalty='l1',
                                                           solver='liblinear',
                                                           class_weight='balanced'))]),
                      'clf__C', [.01, .02, .03, .04, .05, .06, .07, .08, .09, .1])


# If $C$ is too large, overfitting starts as soon as we take more than the two most significant features into account.  For $C = 0.1$ it becomes pretty severe as $k$ increses:

# In[ ]:


plot_validation_curve(Pipeline([('sel', SelectKBest()),
                                ('clf', LogisticRegression(penalty="l1",
                                                           solver='liblinear',
                                                           class_weight='balanced',
                                                           C=0.1))]),
                      'sel__k', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50])


# If we keep $C = 0.05$, which is only slightly too large, training and cross-validation score still start to diverge when taking more than two features into account, but the model does not overfit as extremely:

# In[ ]:


plot_validation_curve(Pipeline([('sel', SelectKBest()),
                                ('clf', LogisticRegression(penalty="l1",
                                                           solver='liblinear',
                                                           class_weight='balanced',
                                                           C=0.05))]),
                      'sel__k', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50])


# For $C = 0.04$, since there was no severe overfitting even on the complete dataset, this also holds if we restrict to the $k$ most significant features.  However, the performance does not improve once we go beyond the two most significant features.

# In[ ]:


plot_validation_curve(Pipeline([('sel', SelectKBest()),
                                ('clf', LogisticRegression(penalty="l1",
                                                           solver="liblinear",
                                                           class_weight="balanced",
                                                           C=0.04))]),
                      'sel__k', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50])


# Instead of selecting the featues upfront, in the public kernes in the competition it's pupular to use [recursive feature elimination](https://scikit-learn.org/stable/modules/feature_selection.html#rfe). However, this does not reduce overfitting.

# In[ ]:


plot_validation_curve(RFE(LogisticRegression(penalty="l1",
                                             solver='liblinear',
                                             class_weight='balanced',
                                             C=0.1)),
                      'n_features_to_select', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50])


# ## Linear discriminant analysis
# For [linear discriminant analysis](https://scikit-learn.org/stable/modules/lda_qda.html#lda-qda), overfitting seems to occur for all shrinkages.

# In[ ]:


plot_validation_curve(LinearDiscriminantAnalysis(solver='lsqr'),
                      'shrinkage', [0.001, 0.001, .01, .1, 1])


# ## Support vector machines
# For [support vector machines](https://scikit-learn.org/stable/modules/svm.html#svm-classification), it seems to be very difficult to find parameter combinatins that do not overfit.
# Let's first have a look at the different kernels with the default parameters:

# In[ ]:


plot_validation_curve(SVC(gamma='scale',
                          probability=True),
                      'kernel', ['linear', 'poly', 'rbf', 'sigmoid'])


# ### Linear kernel
# We start by investigating the linear kernel, which has only one parameter $C$.  It overfits for all values of $C$.

# In[ ]:


plot_validation_curve(SVC(gamma="auto", # gamma is not really a parameter
                                        # of the linear kernel, this is
                                        # just to silence some warnings.
                          probability=True), 
                      'C', [1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3])


# ### Sigmoid kernel
# Now let's have a look at the sigmoid kernel which has parameters $\gamma$, $C$ and $\mathrm{coef}_0$.
# For $C = 10^{-4}$, overfitting reduces a bit.

# In[ ]:


plot_validation_curve(SVC(gamma="scale",
                          kernel="sigmoid",
                          probability=True),
                      "C", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3])


# If we fix $C$, the choice of $\mathrm{coef}_0 = 10$ seems to show some reduced overfitting:

# In[ ]:


plot_validation_curve(SVC(gamma='scale',
                          kernel='sigmoid',
                          C=1e-4,
                          probability=True),
                      'coef0', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3])


# Again, if we keep $C$ and $\mathrm{coef}_0$, the choice of $\gamma = 0.01$ shows some reduction in overfitting:

# In[ ]:


plot_validation_curve(SVC(kernel="sigmoid",
                          coef0=10,
                          C=1e-4,
                          probability=True),
                      "gamma", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])


# ### Polynomial kernel
# Now let's have a look at polynomial kernels.  All degrees overfit with the default parameters:

# In[ ]:


plot_validation_curve(SVC(gamma='scale',
                          kernel='poly',
                          probability=True),
                      'degree', [1, 2, 3, 4, 5])


# Let's investigate the quadratic case.  For $\mathrm{coef}_0 = 0.05$ there is still overfitting but some improvement in cross-validation score:

# In[ ]:


plot_validation_curve(SVC(gamma="scale",
                          kernel="poly",
                          degree=2,
                          probability=True),
                      "coef0", [.001, .002, .005, .01, .02, 0.05, .1, .2, .5, 1])


# So we fix $\mathrm{coeff}_0 = 0.05$ and investigate $C$.  Unfortunately, overfitting occurs for all values of $C$:

# In[ ]:


plot_validation_curve(SVC(gamma="scale",
                          kernel="poly",
                          degree=2,
                          coef0=0.05,
                          probability=True),
                      "C", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1,10,100])


# Similarly, in the cubic case, for $\mathrm{coef}_0 = 1$ there is still overfitting but some improvement in cross-validation score:

# In[ ]:


plot_validation_curve(SVC(gamma='scale',
                          kernel='poly',
                          degree=3,
                          probability=True),
                      'coef0', [.01, .02, 0.05, .1, .2, .5, 1, 2, 5, 10, 20, 50])


# Here, overfitting reduces slightly for $\gamma < 10^{-6}$, but is still massive for all $\gamma$:

# In[ ]:


plot_validation_curve(SVC(gamma="scale",
                          kernel="poly",
                          degree=3,
                          coef0=1,
                          probability=True),
                      "gamma", [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])


# ## Stochastic gradient descent
# 
# [Stochastic gradient descent](https://scikit-learn.org/stable/modules/sgd.html#sgd) overfits for all $\alpha$.

# In[ ]:


plot_validation_curve(SGDClassifier(loss='modified_huber',
                                    max_iter=1000,
                                    tol=1e-3),
                      'alpha', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])


# Standardization does not change anything:

# In[ ]:


plot_validation_curve(Pipeline([('std', StandardScaler()),
                                ('clf', SGDClassifier(loss='modified_huber',
                                                      max_iter=1000,
                                                      tol=1e-3))]),
                      'clf__alpha', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])


# ## Random forest
# How about [random forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)?  Even if we severely restict the maximal depth of trees, and look at the minimal number of samples per leaf, overfitting only stops when the AUC drops to $0.5$.

# In[ ]:


plot_validation_curve(RandomForestClassifier(max_depth=2, n_estimators=10),
                      'min_samples_leaf', [10, 20, 30, 40, 50, 60, 70, 80])


# ## Bagging
# Let's have a look at [bagging](https://scikit-learn.org/stable/modules/ensemble.html#bagging).  This does still overfit:

# In[ ]:


plot_validation_curve(BaggingClassifier(LogisticRegression(C=0.1,
                                                           penalty='l1',
                                                           class_weight='balanced',
                                                           solver='liblinear',
                                                           random_state=0),
                                        max_samples=0.8,
                                        bootstrap=True,
                                        random_state=0),
                      'n_estimators', [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])


# 

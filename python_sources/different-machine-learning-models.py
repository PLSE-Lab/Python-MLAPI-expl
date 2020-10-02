#!/usr/bin/env python
# coding: utf-8

# # Welcome to my Notebook!
# * **I recently joined kaggel and couldn't keep a track of models that i can use for the competitions so I have compiled a list of different models. I thought of making it public so anyone else having the same problem can also use it**
# 
# * **This notebook contains a simple implementation of each model**
# 
# * **You can use them as a reference**
# 
# * **Below is the table of contents**
# 
# 
# 
# * **I have used below mentioned resource;  You can read about different models in detail there**
# source:https://scikit-learn.org/stable/supervised_learning.html
# 
# 
# * ***Do upvote it, if you like it!!***
# 
# 
# *Ps. I am new to data science, do correct me if I made any mistake in this kernel*

# <a id = "table_of_contents"></a>
# # Table of Contents
# 
# [LinearRegression](#1)
# 
# [RidgeRegression](#2)
# 
# [RidgeCV](#3)
# 
# [Lasso](#4)
# 
# [MultiTaskLasso](#5)
# 
# [MultiTaskElasticNet](#6)
# 
# [Lars](#7)
# 
# [OMP](#8)
# 
# [BayesianRidgeRegression](#9)
# 
# [LogisticRegressor](#10)
# 
# [TweedieRegressor](#11)
# 
# [SVM](#12)
# 
# [SGD](#13)
# 
# [KNeighbourClassifier](#14)
# 
# [KNeighborRegressor](#15)
# 
# [RadiusNighborRegressor](#16)
# 
# [GaussianProcessClassifier](#17)
# 
# [GaussianNB](#18)
# 
# [DecisionTree Classifier and Regressor](#19)
# 
# [BaggingClassifier](#20)
# 
# [RandomForestClassifier](#21)
# 
# [ExtraTreeClassifier](#22)
# 
# [AdaBoost](#23)
# 
# [GradientBoosting](#24)
# 
# [HistGradientBoosting](#25)
# 
# [VotingClassifier](#26)
# 
# [Using the VotingClassifier with GridSearchCV](#27)
# 
# [XGBoost](#28)
# 
# [GridSearchCV](#29)
# 
# 
# 
# 

# <a id = "1"></a>
# > # **LinearRegression**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit(X, y)


# <a id = "2"></a>
# > # **Ridge regression**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn import linear_model

reg = linear_model.Ridge(alpha=.5)

reg.fit(X, y)


# <a id = "3"></a>
# > # **RidgeCV**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


import numpy as np

from sklearn import linear_model

reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
 
reg.fit(X, y)


# <a id = "4"></a>
# # **Lasso**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn import linear_model

reg = linear_model.Lasso(alpha=0.1)

reg.fit(X, y)


# <a id = "5"></a>
# # **Multi task lasso**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn import linear_model

clf = linear_model.MultiTaskLasso(alpha=0.1)

clf.fit(X,y)


# <a id = "6"></a>
# > # **MultiTaskElasticNet**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn import linear_model

clf = linear_model.MultiTaskElasticNet(alpha=0.1)

clf.fit(X,y)


# <a id = "7"></a>
# > # **least angle regression (Lars)**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn import linear_model

reg = linear_model.Lars(n_nonzero_coefs=1)

reg.fit(X, y)


# <a id = "8"></a>
# > # **OrthogonalMatchingPursuit(OMP)**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn.linear_model import OrthogonalMatchingPursuit

omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)

omp.fit(X, y)


# <a id = "9"></a>
# > # **BayesianRidge Regression**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn import linear_model

X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]

Y = [0., 1., 2., 3.]

reg = linear_model.BayesianRidge()

reg.fit(X, Y)


# <a id = "10"></a>
# > # **LogisticRegression**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn.linear_model import LogisticRegression

clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01, solver='saga')

clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01, solver='saga')

clf_en_LR = LogisticRegression(C=C, penalty='elasticnet', solver='saga', l1_ratio=l1_ratio, tol=0.01)

clf_l1_LR.fit(X, y) clf_l2_LR.fit(X, y) clf_en_LR.fit(X, y)


# <a id = "11"></a>
# > # **TweedieRegressor**
# [Go back to the Table of Contents](#table_of_contents)
#   
#   The choice of the distribution depends on the problem at hand:
# 
#     If the target values 
# 
#     are counts (non-negative integer valued) or relative frequencies (non-negative), you might use a Poisson deviance with log-link.
# 
#     If the target values are positive valued and skewed, you might try a Gamma deviance with log-link.
# 
#     If the target values seem to be heavier tailed than a Gamma distribution, you might try an Inverse Gaussian deviance (or even higher variance powers of the Tweedie family).
# 
# Examples of use cases include:
# 
#     Agriculture / weather modeling: number of rain events per year (Poisson), amount of rainfall per event (Gamma), total rainfall per year (Tweedie / Compound Poisson Gamma).
# 
#     Risk modeling / insurance policy pricing: number of claim events / policyholder per year (Poisson), cost per event (Gamma), total cost per policyholder per year (Tweedie / Compound Poisson Gamma).
# 
#     Predictive maintenance: number of production interruption events per year: Poisson, duration of interruption: Gamma, total interruption time per year (Tweedie / Compound Poisson Gamma).
# > # 

# In[ ]:


from sklearn.linear_model import TweedieRegressor

reg = TweedieRegressor(power=1, alpha=0.5, link='log')

reg.fit(X, y)


# <a id = "12"></a>
# > # **SVM**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


X = [[0, 0], [1, 1]]

y = [0, 1]

clf = svm.SVC()

clf.fit(X, y)


# <a id = "13"></a>
# > # **SGD Classifier and Regressor**
# [Go back to the Table of Contents](#table_of_contents)
# 
# The advantages of Stochastic Gradient Descent are:
# 
#         Efficiency.
# 
#         Ease of implementation (lots of opportunities for code tuning).
# 
# The disadvantages of Stochastic Gradient Descent include:
# 
#         SGD requires a number of hyperparameters such as the regularization parameter and the number of iterations.
# 
#         SGD is sensitive to feature scaling.
# 
# ****

# In[ ]:


from sklearn.linear_model import SGDClassifier

X = [[0., 0.], [1., 1.]]

y = [0, 1]

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

clf.fit(X, y)


# In[ ]:


import numpy as np

from sklearn.linear_model import SGDRegressor

n_samples, n_features = 10, 5

rng = np.random.RandomState(0)

y = rng.randn(n_samples)

X = rng.randn(n_samples, n_features)

reg = SGDRegressor(max_iter=1000, tol=1e-3)

reg.fit(X, y)


# <a id = "14"></a>
# > # **KNeighborsClassifier**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn import neighbors, datasets

n_neighbors = 15

clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

clf.fit(X, y)


# <a id = "15"></a>
# > #  **KNeighborsRegressor**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


X = [[0], [1], [2], [3]]

y = [0, 0, 1, 1]

from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=2)

neigh.fit(X, y)


# <a id = "16"></a>
# > # **RadiusNeighborsRegressor**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


X = [[0], [1], [2], [3]]

y = [0, 0, 1, 1]

from sklearn.neighbors import RadiusNeighborsRegressor

neigh = RadiusNeighborsRegressor(radius=1.0)

neigh.fit(X, y)


# <a id = "17"></a>
# > # **GaussianProcessClassifier**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn.datasets import load_iris

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

X, y = load_iris(return_X_y=True)

kernel = 1.0 * RBF(1.0)

gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X, y)


# <a id = "18"></a>
# > # **GaussianNB**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)


# <a id = "19"></a>
# > # **DecisionTreeClassifier and regressor**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn import tree

X = [[0, 0], [1, 1]]

Y = [0, 1]

clf = tree.DecisionTreeClassifier() / clf = tree.DecisionTreeREgressor()

clf = clf.fit(X, Y)


# # **Ensemble models**

# <a id = "20"></a>
# > # **BaggingClassifier**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn.ensemble import BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier

bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5) #can use other classifier also instead of KNC


# <a id = "21"></a>
# > # **RandomForestClassifier**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

X = [[0, 0], [1, 1]]

Y = [0, 1]

clf = RandomForestClassifier(n_estimators=10)

clf = clf.fit(X, Y)


# <a id = "22"></a>
# > # **ExtraTreesClassifier and its comparison**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn.model_selection import cross_val_score

from sklearn.datasets import make_blobs

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)

scores = cross_val_score(clf, X, y, cv=5)

scores.mean()
>0.98...

clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)

scores = cross_val_score(clf, X, y, cv=5)

scores.mean()
>0.999...

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)

scores = cross_val_score(clf, X, y, cv=5)

scores.mean() 
# >0.999 True


# <a id = "23"></a>
# > # **AdaBoostClassifier**
# [Go back to the Table of Contents](#table_of_contents)
# 
# 
# 
# The number of weak learners is controlled by the parameter n_estimators. The learning_rate parameter controls the contribution of the weak learners in the final combination. By default, weak learners are decision stumps. Different weak learners can be specified through the base_estimator parameter. The main parameters to tune to obtain good results are n_estimators and the complexity of the base estimators (e.g., its depth max_depth or minimum required number of samples to consider a split min_samples_split).

# In[ ]:


from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_iris

from sklearn.ensemble import AdaBoostClassifier

X, y = load_iris(return_X_y=True)

clf = AdaBoostClassifier(n_estimators=100)

scores = cross_val_score(clf, X, y, cv=5)

scores.mean() 
# >0.9...


# <a id = "24"></a>
# # **GradientBoostingClassifier and regressor**
# [Go back to the Table of Contents](#table_of_contents)
# 
# At each iteration the base classifier is trained on a fraction subsample of the available training data. The subsample is drawn without replacement. A typical value of subsample is 0.5.

# In[ ]:


from sklearn.datasets import make_hastie_10_2

from sklearn.ensemble import GradientBoostingClassifier

X, y = make_hastie_10_2(random_state=0)

X_train, X_test = X[:2000], X[2000:]

y_train, y_test = y[:2000], y[2000:]

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)

clf.score(X_test, y_test)

# >0.913...


# In[ ]:


import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.datasets import make_friedman1

from sklearn.ensemble import GradientBoostingRegressor

X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)

X_train, X_test = X[:200], X[200:]

y_train, y_test = y[:200], y[200:]

est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)

mean_squared_error(y_test, est.predict(X_test))

# >5.00...


# ***Additional fit***

# In[ ]:


_ = est.set_params(n_estimators=200, warm_start=True) # set warm_start and new nr of trees

_ = est.fit(X_train, y_train) # fit additional 100 trees to est

mean_squared_error(y_test, est.predict(X_test))

# >3.84...


# # Experimental 
# **Use when data size is in tens of thousands no need of imputation ; still in development**

# <a id = "25"></a>
# # **HistGradientBoostingClassifier and regressor**
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.datasets import make_hastie_10_2

X, y = make_hastie_10_2(random_state=0)

X_train, X_test = X[:2000], X[2000:]

y_train, y_test = y[:2000], y[2000:]

clf = HistGradientBoostingClassifier(min_samples_leaf=1, max_depth=2, learning_rate=1, max_iter=1).fit(X_train, y_train)

clf.score(X_test, y_test)

# >0.8965


# <a id = "26"></a>
# # VotingClassifier
# [Go back to the Table of Contents](#table_of_contents)
# 
# The idea behind the VotingClassifier is to combine conceptually different machine learning classifiers and use a majority vote or the average predicted probabilities (soft vote) to predict the class labels. Such a classifier can be useful for a set of equally well performing model in order to balance out their individual weaknesses.

# In[ ]:


from sklearn import datasets

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

iris = datasets.load_iris()

X, y = iris.data[:, 1:3], iris.target

clf1 = LogisticRegression(random_state=1)

clf2 = RandomForestClassifier(n_estimators=50, random_state=1)

clf3 = GaussianNB()

eclf = VotingClassifier( estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
   scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)

   print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

   
   
   
# >Accuracy: 0.95 (+/- 0.04) [Logistic Regression]

# >Accuracy: 0.94 (+/- 0.04) [Random Forest]

# >Accuracy: 0.91 (+/- 0.04) [naive Bayes]

# >Accuracy: 0.95 (+/- 0.04) [Ensemble]


# <a id = "27"></a>
# # Using the VotingClassifier with GridSearchCV
# [Go back to the Table of Contents](#table_of_contents)
# 
# The VotingClassifier can also be used together with GridSearchCV in order to tune the hyperparameters of the individual estimators:

# In[ ]:


from sklearn.model_selection import GridSearchCV

clf1 = LogisticRegression(random_state=1)

clf2 = RandomForestClassifier(random_state=1)

clf3 = GaussianNB()

eclf = VotingClassifier( estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft' )

params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200]}

grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)

grid = grid.fit(iris.data, iris.target)


# <a id = "28"></a>
# # XGBoost
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:


import xgboost as xgb

#read in data

dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')

dtest = xgb.DMatrix('demo/data/agaricus.txt.test')

#specify parameters via map

param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }

num_round = 2

bst = xgb.train(param, dtrain, num_round)

#make prediction

preds = bst.predict(dtest)


# <a id = "29"></a>
# # GridsearchCV
# [Go back to the Table of Contents](#table_of_contents)

# In[ ]:




from sklearn import svm, datasets

from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

svc = svm.SVC()

clf = GridSearchCV(svc, parameters)

clf.fit(iris.data, iris.target)

GridSearchCV(estimator=SVC(), param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})


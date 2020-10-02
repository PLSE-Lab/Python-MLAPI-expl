#!/usr/bin/env python
# coding: utf-8

# # Non-parametric regression
# 
# ## Discussion
# 
# A machine learning estimator or technique is **parametric** if it relies on the underlying distribution of the data in some way. Most of the tools in classical statistics are parametrics: they fundamentally rely on assumptions about the underlying distribution of the data being studied (is it normally distributed? is it poisson distributed? etcetera). Techniques like least squares regression (which assumes normally distributed residuals) are parametric. The alternative is tools that are **non-parametric**. Non-parametric machine learning algorithms do not rely on assumptions about the shape of the underlying data to work. Non-parametric techniques include decision trees and support vector machines.
# 
# Being non-parametric does away with a whole class of data problems ("does this data have the properties I need it to have for my algorithm to make sense?") and, in doing so, adds a lot of flexibility to the machine learner's toolbelt. However, this does not come without its drawbacks. Non-parametric algorithms tend not to perform as well as parametric models when faced with small amounts of data; for sufficiently small data, regression is the only option that makes sense. This is the primary reason why regression and similar tools have been the mainstay of statistics during the "era of small data". Similarly, non-parametric algorithms also tend to be more computationally intensive, and to have less numerically stable solutions. Creating a two-class SVM, for example, involves building hyperplane partitions trained by iterating over the space with a stochastic gradient descent algorithm ([link](https://www.kaggle.com/residentmario/support-vector-machines-and-stoch-gradient-descent)). Doing the same work with logistic regression, by contrast, is incredibly cheap&mdash;you "only" have to invert a feature matrix and multiply it across.
# 
# Nevertheless, with large enough datasets, complicated enough data, and enough training time on the part of the machine learner, non-parametric tools tend to eclipse parametric ones in overall accuracy.
# 
# Non-parametric techniques tend to be natively set in the classification context. However, they can all be extended relatively easily to regression, in one way or another. These techniques provide a useful alternative to parametric techniques like ridge regression for doing this kind of work (though I suspect ridge, lasso, and elastic net regression are more popular overall).
# 
# In this notebook I demo a handful of the different non-parametric regression techniques available in `sklearn`. For the sake of demonstration I will show an application on both a synthetic dataset provided in `sklearn` and real (albeit low-signal data): the number of wins or draws Brazil records in its FIFA World Cup matches:

# In[99]:


import pandas as pd
df = pd.read_csv("../input/WorldCupMatches.csv")
df = (df
          .loc[df.apply(lambda srs: ((srs['Home Team Name'] == "Brazil") & (srs['Home Team Goals'] >= srs['Away Team Goals'])) | 
                        ((srs['Away Team Name'] == "Brazil") & (srs['Away Team Goals'] >= srs['Home Team Goals'])), axis='columns')]
          .Datetime
          .dropna()
          .map(lambda dt: dt.split("-")[0].strip())
          .map(lambda dt: dt.split(" ")[-1])
          .value_counts()
          .sort_index()
)

import numpy as np
X = np.asarray(list(range(len(df))))[:, np.newaxis]
y = df.values

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
df.plot.line(marker="o", linewidth=0, color='black', title='Brazil Wins or Ties Per World Cup, 1930-2016', figsize=(12, 6))


# In[98]:


import numpy as np
rng = np.random.RandomState(0)

X_p = 5 * rng.rand(100, 1)
y_p = np.sin(X_p).ravel()

y_p[::5] += 3 * (0.5 - rng.rand(X_p.shape[0] // 5))

fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(X_p[:, 0], y_p, marker='o', color='black', linewidth=0)
plt.suptitle("Synthetic Polynomial Data")


# ## Kernel ridge regression
# 
# Kernel ridge regression is an extension of ridge regression that uses the kernel trick. In other words, instead of training a ridge regression algorithm on data in its original linear setting, kernel ridge regression has us transform that data to some other complex vector space, learn the features in that space, and then transform back into the space we started in using the kernel trick. Since it corresponds with a parametric method, ridge regression, I'm not sure this classifier is strictly classifiable as non-parametric, but it's nevertheless suitable for inclusion here.
# 
# If you're not familiar with ridge regression check out [this notebook](https://www.kaggle.com/residentmario/ridge-regression-with-video-game-sales-prediction). To learn more about the kernel trick check out [this notebook](https://www.kaggle.com/residentmario/kernels-and-support-vector-machine-regularization).

# In[174]:


from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))


clf = KernelRidge(kernel='rbf', gamma=0.1)
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0])
pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')
axarr[0].set_title("Brazil")

clf = Ridge()
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0], color='gray')


axarr[1].plot(X_p[:, 0], y_p, marker='o', color='black', linewidth=0)

clf = KernelRidge(kernel='rbf', gamma=0.8)
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
sort = np.argsort(X_p[:, 0])
axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0)

clf = Ridge()
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
axarr[1].plot(X_p[:, 0], y_pred, color='gray')
axarr[1].set_title("Sine Wave")
pass


# In these plots the original ridge regressor is in gray and the kernel regressor is in blue. You can clearly see how the non-parametric nature of the RBF kernel allowed us to capture the polynomial nature of the inputted sine wave (though to be fair you could probably have achieved the same with polynomial ridge regression).
# 
# ## Decision tree regression
# 
# Decision trees learn to classify things by finding informative splits along records in the dataset (employing Gini impurity minimization as its loss function). They can be extended to regression by asking the decision tree to output not a class, but a floating point number. The most informative splits therefore become the record values *and* the choice of floating point number which is most informative for the model. This adds a little bit of work of course, but not that much, since every interval in the dataset will logically have an easily computed "inefficiency decrease maximizing value".

# In[186]:


from sklearn.tree import DecisionTreeRegressor

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))


clf = DecisionTreeRegressor(max_depth=5)
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0], color='lightsteelblue')
pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')

clf = DecisionTreeRegressor(max_depth=2)
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0], color='steelblue')
pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')

axarr[0].set_title("Brazil")

axarr[1].plot(X_p[:, 0], y_p, marker='o', color='black', linewidth=0)

clf = DecisionTreeRegressor(max_depth=5)
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
sort = np.argsort(X_p[:, 0])
axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0, color='lightsteelblue')

clf = DecisionTreeRegressor(max_depth=2)
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
sort = np.argsort(X_p[:, 0])
axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0, color='steelblue')
axarr[1].set_title("Sine Wave")
pass


# A distinctive feature of decision tree regression is the stepwise nature of the regression result. Notice that in both cases a depth-2 tree in dark blue performs significantly better than a depth-5 tree in light blue, with the latter overfitting by quite a lot in both cases. You can read more about decision tress in [this notebook](https://www.kaggle.com/residentmario/decision-trees-with-animal-shelter-outcomes).

# ## Nearest-neighbor regression
# 
# Nearest-neighbor regression is an adaptation of nearest-neighbor algorithms to regression. In nearest-neighbor algoriths each point in the dataset determines its class by looking at the classes of a certain number points nearest to it (and perhaps aggregating their votes somehow). The extension to the regression case couldn't be simpler: instead of just taking the majority class, average the values of the contributing points (perhaps with some kind of weighing, perhaps not).

# In[209]:


from sklearn.neighbors import KNeighborsRegressor

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))


clf = KNeighborsRegressor(n_neighbors=5, weights='uniform')
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0], color='steelblue')
pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')

# clf = KNeighborsRegressor(n_neighbors=5, weights='distance')
# clf.fit(X, y)
# y_pred = clf.predict(X)
# pd.Series(y_pred).plot(ax=axarr[0], color='red')
# pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')

axarr[0].set_title("Brazil")

axarr[1].plot(X_p[:, 0], y_p, marker='o', color='black', linewidth=0)

clf = KNeighborsRegressor(n_neighbors=10, weights='uniform')
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
sort = np.argsort(X_p[:, 0])
axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0, color='steelblue')

# clf = KNeighborsRegressor(n_neighbors=5, weights='distance')
# clf.fit(X_p, y_p)
# y_pred = clf.predict(X_p)
# sort = np.argsort(X_p[:, 0])
# axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0, color='steelblue')
axarr[1].set_title("Sine Wave")
pass


# This is the default implementation, which weights each neighbor equally. There is also a version that is weighted by distance (`weights='distance'`) and a separate top-level classifier, `RadiusNeighborsRegressor`, for radius-based point selection (as opposed to simply specifying many $n$ nearest neighbors contribute).

# ## Support Vector Regression
# 
# Support Vector Regression is an extension of the support vector machine algorithm (SVM) to the regression case. In SVMs we divide the space by constructing separating hyperplanes which maximize the distance between point clusters in some complex space built using the kernel trick. To learn more about SVMs read [this notebook](https://www.kaggle.com/residentmario/support-vector-machines-and-stoch-gradient-descent). To learn more about the kernel trick check out [this notebook](https://www.kaggle.com/residentmario/kernels-and-support-vector-machine-regularization).
# 
# It's non-obvious to me how the support vector machine may be extended to regression. The `sklearn` documentation skims on the details and points to [this paper](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.114.4288), which might be worth reading.

# In[214]:


from sklearn.svm import SVR

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))


clf = SVR()
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0], color='steelblue')
pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')

axarr[0].set_title("Brazil")

axarr[1].plot(X_p[:, 0], y_p, marker='o', color='black', linewidth=0)

clf = SVR()
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
sort = np.argsort(X_p[:, 0])
axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0, color='steelblue')

axarr[1].set_title("Sine Wave")
pass


# The linear support vector regressor does an admirable job fitting our sine wave data. Notice that just as with kernel ridge regression, the regression outputted by SVR will be smooth, not intrinsically jagged like in nearest-neighbor regression or stepwise as in decision tree regression.

# ## Stochastic gradient descent regression
# 
# `SGDRegressor` is an extension of stochastic gradient descent classification to the regression case. Stochastic gradient descent is less a machine learning algorithm than it is an *approach to implementing* a machine learning technique, and hence this classifier may be used to re-implement `SVR` (albeit with better scalability and sampling), to re-implement `LinearRegression`,  and a few other things besides. As with support vector regression, I'm unclear on how the extensions work; something to investigate later.

# In[223]:


from sklearn.linear_model import SGDRegressor

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))


clf = SGDRegressor()
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0], color='steelblue')
pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')

axarr[0].set_title("Brazil")

axarr[1].plot(X_p[:, 0], y_p, marker='o', color='black', linewidth=0)

clf = SGDRegressor()
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
sort = np.argsort(X_p[:, 0])
axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0, color='steelblue')

axarr[1].set_title("Sine Wave")
pass


# The default loss function is `squared_loss`, thus resulting in an implementation of ordinary least squares regression (but stochastic!). Try forking this notebook and looking at some of the other loss functions.
# 
# ## Conclusion
# 
# Thanks for reading!

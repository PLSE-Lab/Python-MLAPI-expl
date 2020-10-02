#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In [the previous notebook](https://www.kaggle.com/residentmario/gaming-cross-validation-and-hyperparameter-search/notebook) I introduced the concepts of cross validation and hyperparameter search (if you don't know what those are, or need a refresher, start there). I implemented grid search by hand in [a separate notebook](https://www.kaggle.com/residentmario/nyc-buildings-part-2-feature-scales-grid-search/), but the technique that we implemented in that notebook are built into `scikit-learn` as the `GridSearchCV` (of course!).
# 
# In this notebook we'll look at how this work can be managed by a `scikit-learn` [pipeline](http://scikit-learn.org/stable/modules/pipeline.html).

# In[ ]:


import pandas as pd
survey = pd.read_csv(
    "../input/BoilingSteam_com_LinuxGamersSurvey_Q1_2016_Public_Sharing_Only.csv"
).loc[1:]

import numpy as np

spend = survey.loc[:, ['LinuxGamingHoursPerWeek', 'LinuxGamingSpendingPerMonth']].dropna()
spend = spend.assign(
    LinuxGamingHoursPerWeek=spend.LinuxGamingHoursPerWeek.map(lambda v: int(v) if str.isdigit(v) else np.nan),
    LinuxGamingSpendingPerMonth=spend.LinuxGamingSpendingPerMonth.map(lambda v: float(v) if str.isdecimal(v) else np.nan)
).dropna()

X = spend['LinuxGamingHoursPerWeek'].values[:, np.newaxis]
y = spend['LinuxGamingSpendingPerMonth'].values[:, np.newaxis]


# # Pipe-in'
# 
# You can declare a pipeline directly as an object, using the `Pipeline` constructor, or use the `make_pipeline` convenience function (which we will use). Recall that for our model in the previous notebook we used polynomial regression and performed hyperparameter search over the degrees of our variables.
# 
# Organizing that into a simple pipeline is easy.

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

pipe = make_pipeline(PolynomialFeatures(), LinearRegression())


# You can get the steps you are using with `steps` or `named_steps` utility methods.

# In[ ]:


pipe.steps


# In[ ]:


pipe.named_steps


# Setting a parameter is weird, though. You have to take the name of the step, add a `__`, and then specify what you want to mutate that step's parameter to.

# In[ ]:


pipe.set_params(polynomialfeatures__degree=3)


# We can use `fit` and `predict` on the pipeline to get a result, as usual.

# In[ ]:


pipe.fit(X, y)
y_hat = pipe.predict(X)

# Plot the result.
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

sortorder = np.argsort(spend['LinuxGamingHoursPerWeek'].values)
s = spend['LinuxGamingHoursPerWeek'].values[sortorder]
y_hat = y_hat[sortorder]
plt.plot(s, y_hat)
plt.scatter(spend['LinuxGamingHoursPerWeek'], 
            spend['LinuxGamingSpendingPerMonth'], 
            color='black')


# A pipeline technically needs to consist of a string of n transformations (classes that implement a callable `transform` method) and then a single model at the end (implementing `fit` and `predict`). That means that it is very easy to write our own functions for insertion in a pipeline, if we are so inclined.
# 
# Running a grid search occurs outside of the pipeline, *on* the pipeline. Getting the result is as simple as throwing the pipe into `GridSearchCV`.

# In[ ]:


from sklearn.grid_search import GridSearchCV

pipe = make_pipeline(PolynomialFeatures(), LinearRegression())
param_grid = dict(polynomialfeatures__degree=list(range(1, 5)))
grid_search = GridSearchCV(pipe, param_grid=param_grid)


# In[ ]:


grid_search.fit(X, y)


# The resulting model is stored in the `estimator` parameter.

# In[ ]:


grid_search.estimator


# ## Discussion
# 
# For simple models like this one the benefit of using a pipeline is relatively small. However, for increasingly complex models, the organizational "struts" that the pipeline provides are increasingly useful.
# 
# The `scikit-learn` documentation lists these three benefits to using pipelines:
# 
# > **Convenience and encapsulation**
# >
# > You only have to call fit and predict once on your data to fit a whole sequence of estimators.
# >
# > **Joint parameter selection**
# >
# > You can grid search over parameters of all estimators in the pipeline at once.
# >
# > **Safety**
# >
# > Pipelines help avoid leaking statistics from your test data into the trained model in cross-validation, by ensuring that the same samples are used to train the transformers and predictors.
# 
# Convenience and encapsulation is an obvious one. The purported benefit of joint parameter selection is interesting; using a pipeline unlocks the ability to do more complicated grid searches, like, for example, mixing and matching what kinds of transforms we apply to the dataset *before* running (and potentially searching through) the model.
# 
# The point on safety is hardest to explain. Basically, using a pipeline helps defend against a difficult-to-escape form of model leakage known as "knowledge leakage". This warrents another notebook: for more on that, [read this notebook](https://www.kaggle.com/residentmario/leakage-especially-knowledge-leakage/).
# 
# ## Feature Union
# 
# It's also worth briefly mentioning the `FeatureUnion` class. This class provides a way of combining transforms without having to place an estimator at the end (e.g. it's a `pipeline[:-1]`).

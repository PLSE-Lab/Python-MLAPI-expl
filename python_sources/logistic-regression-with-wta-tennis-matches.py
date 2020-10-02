#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression with WTA Tennis Matches
# 
# ## Discussion
# 
# Logistic regression is an adaptation of linear regression to binary classification&mdash;predicting 0-or-1 target variables.
# 
# Linear regression can technically be used for this task, but many of the fundamental assumptions built into linear regression are violated when you do this:
# 
# 1. The distribution is non-normal (it can only take on two values after all).
# 2. The error terms will be heteroskedastic by definition.
# 
# Logistic regression returns a result that's interpretable as an exact probability because it's in the range $(0, 1)$. This form of regression returns a result based on the [logistic function](https://en.wikipedia.org/wiki/Logistic_function) (hence the name):
# 
# $$\sigma(t) = \frac{1}{1 + e^{-t}}$$
# 
# Logistic regression is premised around the squeezing that this function applies to data towards 0 or 1:
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)
# 
# Here is the full logistic regression expression, expressed in terms of its fundamental unit, the logit:
# 
# $$\ln{\frac{P}{1-P}} = \exp{(\beta_0 + \beta X + \varepsilon)}$$
# 
# Where $P$ is the prediction (the probability), $\beta_0$ is the constant coefficient, $\beta$ is the vector of non-constant coefficients, and $\varepsilon$ is the error term. The term on the left is known as the "logit", and it does all of the heavy lifting.
# 
# Notice that the term inside of the $\exp{\cdot}$ term is the linear regression formula! The logistic regression model is simply a non-linear transformation on linear regression. To get the result $P$ (or equivalently $y$), we do this to it:
# 
# $$P = \frac{1}{(1 + \exp{(-\beta_0 - \beta X)})}$$
# 
# What we've achieved here is, via the application of the logistic function to the linear regression result formula, a transformation from a line to logistic curve:
# 
# ![](http://www.appstate.edu/~whiteheadjc/service/logit/logit.gif)

# ## Application
# 
# Let's use the `scikit-learn` logistic regression function to predict which of two players will win a match, based on what each player's rank is.

# In[ ]:


import pandas as pd
matches = pd.read_csv("../input/wta_matches_2015.csv")
matches.head()


# In[ ]:


import numpy as np

point_diff = (matches.winner_rank_points - matches.loser_rank_points).dropna()
X = point_diff.values[:, np.newaxis]
y = (point_diff > 0).values.astype(int).reshape(-1, 1)

sort_order = np.argsort(X[:, 0])
X = X[sort_order, :]
y = y[sort_order, :]


# In[ ]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X, y.ravel())
y_hat = clf.predict(y[:, 0].reshape(-1, 1))
y_hat = y_hat[sort_order]


# Here is the raw result in the record, which shows that the player with more ranking points wins somewhat less than two-thirds of the time.

# In[ ]:


pd.Series(y[:, 0]).value_counts()


# But our classifier classifies every single record as a greater-ranking-points-wins outcome!

# In[ ]:


pd.Series(y_hat).value_counts()


# This results in the following propensity for error:

# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

pd.Series(y_hat == y[:, 0]).value_counts().plot.bar()


# We can understand this better by looking at `predict_proba`, which returns the probabilities that the result will be a 0 or 1 response, respectively. This shows that logistic regression can't find any response in the data; the most likely winners-by-ranking points are just 5% more likely to win than the most likely losers-by-ranking points.

# In[ ]:


clf.predict_proba(y[:, 0].reshape(-1, 1))


# This is interesting, it shows that this model needs further diagnosing and problem-shooting.

# ## Interpretation
# 
# Looking at the coefficients (just one, since this is a univariate regression):

# In[ ]:


clf.coef_


# This coefficient is the log-odds: the effect that a point in the difference between the two players has on the *logarithm* of the odds that the winner will be in class 1 (the higher-ranked players).
# 
# To get the effect that one point has on the odds, we need to reverse the logarithm, by taking:
# 
# $$\Delta \text{Odds} = e^{\beta_0 + \beta_1 + \ldots + \beta_k}$$
#  
#  Applying this to our result, we find:

# In[ ]:


import math

math.exp(1)**clf.coef_


# According to our model each point of advantage in the rankings results in one more point in odds favoring that player!
# 
# Since the difference in points is distributed in the hundreds, or even thousands, this is more evidence that the model is simply taking the mean prediction, e.g. splitting the data at 0.

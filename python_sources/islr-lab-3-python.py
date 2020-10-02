#!/usr/bin/env python
# coding: utf-8

# # Lab: Logistic Regression, LDA, QDA, and KNN

# In this lab we will go over how to perform logistic regression, linear and quadratic discriminant analysis, and k-nearest neighbors in order to predict categorical responses in Python. As with Lab 2, we will use both StatsModels (for [logistic regression](http://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html#statsmodels.discrete.discrete_model.Logit)) and scikit-learn (for [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression); [linear and quadratic discriminant analysis](https://scikit-learn.org/stable/modules/lda_qda.html); and [KNN classification](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification)). Before starting, we will import the various Python packages we'll use in this lab. This will also include some of the [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) from scikit-learn.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Import classes from scikit-learn for logistic regression, LDA, QDA, and KNN classification
# Also import convenience function for computing confusion matrices
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# Load StatsModels API
# Note that if we wish to use R-style formulas, then we would load statsmodels.formula.api instead
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ## The Stock Market Data

# First, we'll look at some numerical and graphical summaries of the `Smarket` data set from the `ISLR` library. It consists of the percentage returns for the S&P 500 stock index over 1,250 days, from the beginning of 2001 until the end of 2005. For each date, there are the following pieces of information:
# 
# - `Lag1` through `Lag5`: the percentage returns for each of the five previous trading days
# - `Volume`: the number of shares traded on the previous day, in billions
# - `Today`: the percentage return on the date in question
# - `Direction`: whether the market was `Up` (positive percentage return) or `Down` (negative percentage return) on this date.

# In[ ]:


Smarket_filepath = "../input/Smarket.csv"
Smarket = pd.read_csv(Smarket_filepath, index_col = "Unnamed: 0")
Smarket.head()


# In[ ]:


# Check for missing values
Smarket.isna().any()


# In[ ]:


Smarket.shape


# In[ ]:


Smarket.describe()


# In[ ]:


Smarket["Direction"].value_counts()


# Lastly, before moving forward, we'll produce a scatterplot matrix of all the variables in the set. To visualize the difference between "Up" days and "Down" days, we'll use the `Direction` column to determine the hue.

# In[ ]:


sns.pairplot(Smarket, hue = "Direction")


# Next, let's produce a matrix of all the pairwise correlations among the variables in the data set. Note that unlike the `cor()` function in R, the `corr()` function in Pandas will automatically ignore qualitative variables.

# In[ ]:


Smarket.corr()


# Looking at the last row, we see that the correlations between the lag variables and today's returns are close to zero, which means that there appears to be little correlation between today's returns and the returns from previous days. The only substantial correlation is between `Year` and `Volume`, with a value of 0.539. We can plot the data to see that `Volume` is increasing over time. In other words, the average number of shares traded daily increased from 2001 to 2005.

# In[ ]:


fig = plt.figure(figsize = (10, 8))
ax = plt.axes()
ax.scatter(x = Smarket.index, y = Smarket["Volume"], alpha = 0.5)
ax.set(xlabel = "Day", ylabel = "Shares traded (in billions)");


# ## Logistic Regression

# We'll start off by fitting a logistic regression model in order to predict `Direction` using `Lag1` through `Lag5` and `Volume`. For the first way of doing this, we'll use the [Logit class](http://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html#statsmodels.discrete.discrete_model.Logit) from StatsModels. Alternatively, with StatsModels, we have the option of using the [GLM class](http://www.statsmodels.org/stable/glm.html) with the family set to [binomial](http://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.Binomial.html#statsmodels.genmod.families.family.Binomial). 

# In[ ]:


# Using the Logit class from StatsModels
# First encode the response numerically
endog = (Smarket["Direction"] == "Up").astype("int64")
exog = sm.add_constant(Smarket.drop(columns = ["Direction", "Year", "Today"]))
logit_mod = sm.Logit(endog, exog)
logit_res = logit_mod.fit()
print(logit_res.summary())


# In[ ]:


# Using the GLM class from StatsModels
# First encode the response numerically
endog = (Smarket["Direction"] == "Up").astype("int64")
exog = sm.add_constant(Smarket.drop(columns = ["Direction", "Year", "Today"]))
glm_mod = sm.GLM(endog, exog, family = sm.families.Binomial())
glm_res = glm_mod.fit()
print(glm_res.summary())


# While `Lag1` has the smallest p-value among all of the coefficient estimates, it still has a value of 0.145, which is still relatively large. Thus, even though the negative coefficient for `Lag1` suggests that if the market had a positive return yesterday, then it is less likely to go up today, there isn't any clear evidence of a real association between `Lag1` and `Direction`.
# 
# The coefficients of the fitted model are stored in the `params` attribute of the fitted model. To see what other properties are stored in the results class for each fitted model, as well as what functions they have, we can either refer to the documentation ([LogitResults](http://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.LogitResults.html#statsmodels.discrete.discrete_model.LogitResults), [GLMResults](http://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLMResults.html#statsmodels.genmod.generalized_linear_model.GLMResults)) or use the `dir` command on a results class object.

# In[ ]:


logit_res.params


# In[ ]:


pd.DataFrame({"Estimate":logit_res.params, "Std. Error":logit_res.bse, "z value":logit_res.tvalues,
             "Pr(>|z|)":logit_res.pvalues})


# Before moving on to making predictions with the model, we'll also use scikit-learn's [LogisticRegression class](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) to fit the above logistic regression model. For the purposes of this lab, we won't worry about extending the LogisticRegression class to include all of the statistical parameters and analyses that StatsModels and R can do.

# In[ ]:


X = Smarket.drop(columns = ["Direction", "Year", "Today"])
y = Smarket["Direction"]

# By default scikit-learn uses an l2 regularization penalty, which we don't want for
# vanilla logistic regression as described in the book
log_reg = LogisticRegression(penalty = "none", solver = "lbfgs")
log_reg.fit(X, y)
params = np.concatenate((log_reg.intercept_, log_reg.coef_.flatten()))
pd.DataFrame(data = {"Coef. Est.":params}, index = X.columns.insert(0, "intercept"))


# In order to make a prediction, we use the `predict()` function to use the model to compute the probability that the market will go up for a set of given values of the predictors. When computing the predicted values, by default StatsModels will compute the probabilities $P(Y = 1 | X)$ for each observation. If we don't supply a data set to the `exog` parameter of the `predict()` function, then it will compute the probabilties for the training data that was used to fit the model. Here are the first ten probabilities that `predict()` computed.

# In[ ]:


# Predictions from using Logit
logit_probs = logit_res.predict()
logit_probs[0:10]


# In[ ]:


# Predictions from using GLM with the Binomial family
glm_probs = glm_res.predict()
glm_probs[0:10]


# These values correspond to the probability of the market going up, rather than down, since we specifically coded 1 for `Up` and 0 for `Down` when passing the `Direction` column to StatsModels.
# 
# When computing the predicted values using scikit-learn, we use the `predict_proba()` function, which produces an array with a column for the probability of each class label for each observation.

# In[ ]:


log_reg_probs = log_reg.predict_proba(X)
pd.DataFrame(log_reg_probs, columns = log_reg.classes_).head(10)


# While we can convert the predicted probabilities into class labels `Up` or `Down`, the `predict()` function for LogisticRegression in scikit-learn actually does this automatically by assigning each prediction to the class for which it has the highest probability (which in the case of binary classification is the class with probability greater than 0.5). 

# In[ ]:


log_reg_pred = log_reg.predict(X)
pd.DataFrame(np.hstack((log_reg_probs, log_reg_pred.reshape(-1,1))), 
             columns = np.concatenate((log_reg.classes_, ["Prediction"]))).head(10)


# In addition, both StatsModels and scikit-learn have convenient methods for generating confusion matrices in order to determine how many observations were correctly or incorrectly labeled. For the Logit class in StatsModels, we use the [`pred_table()` function](http://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.LogitResults.pred_table.html#statsmodels.discrete.discrete_model.LogitResults.pred_table), which can take an optional `threshold` argument to adjust the threshold probability for assigning an observation to a class. By default the threshold value is 0.5. Also note that `pred_table[i,j]` refers to the number of times `i` was observed and the model predicted `j`. Correct predictions are along the diagonal.

# In[ ]:


pd.DataFrame(logit_res.pred_table(), columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])


# For scikit-learn, we can use the `confusion_matrix()` function, which is part of the [metrics](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix) subclass. Scikit-learn uses the same axes convention as StatsModels (entry $C_{i, j}$ of the confusion matrix $C$ is the number of observations known to be in group $i$ and predicted to be in group $j$). Note confusion matrices produced in R using the `table()` function have the opposite axes convention, and are therefore the transpose of the StatsModels/scikit-learn confusion matrices.

# In[ ]:


pd.DataFrame(confusion_matrix(y, log_reg_pred), 
             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])


# The diagonal of the confusion matrix represents correct predictions, while the off-diagonal represents incorrect predictions. Thus, the model correctly predicted that the market would go up on 507 days and go down on 145 days, for a total of 652 correct predictions. This means that logistic regression correctly predicted the movement of the market $652/1250 = 52.16\%$ of the time. In StatsModels, another way we can compute this is to use the `mean()` function.

# In[ ]:


logit_preds = pd.Series(logit_probs).apply(lambda x: "Up" if (x > 0.5) else "Down")
(logit_preds == Smarket["Direction"].values).mean()


# This works because the `logit_preds == Smarket["Direction"].values` is a boolean vector, and in the context of the `mean()` function the value `TRUE` is treated as 1 and `FALSE` is treated as 0. With scikit-learn, we can also use the `score()` function to conveniently perform this computation.

# In[ ]:


log_reg.score(X, y)


# Note that this correct prediction rate of 52.16% comes from testing our logistic regression model on the same data set we used for training. In other words, our *training* error rate is 47.84. Since the training error rate tends to underestimate the test error rate, the correct prediction rate isn't as good as it might initially seem. One way we can better assess the accuracy of the logistic regression model in this setting is to fif the model using part of the data, and then examine how well it performs when predicting the *held out* data. Since in practice we are more interested with how well our model performs on days in the future for which the market's movements are unknown, this will yield a more realistic error rate. In order to implement this strategy, we'll again use a boolean mask to separate out the observations from 2001 through 2004 from the observations from 2005.

# In[ ]:


train_mask = (Smarket["Year"] < 2005)


# To start with, we create a boolean mask where each element is either `TRUE` if it corresponds to an observation that occurred before 2005 or `FALSE` if it corresponds to an observation that occurred in 2005.
# 
# We can then use the mask to pick out the submatrix of the stock market data set which consists of all columns, and only the rows for which the elements of `train` are false. In other words, `Smarket_2005` is a copy of the submatrix of `Smarket` consisting only of all observations from 2005. Note that we didn't need to create the mask `train` separately, but doing so is good practice to allow us to easily manipulate our training set in a single place.

# In[ ]:


# Recall that to to element-wise logical operators for boolean indexing in Pandas, we need to use
# | for or, & for and, ~ for not
Smarket_2005 = Smarket[~train_mask]
Smarket_2005.head()


# Now we'll train our logistic regression model on the training set (consisting of the observations from the years 2000-2004) before testing the model on our held-out test set.

# In[ ]:


# Using the Logit class from StatsModels, and training only on the training set
# First encode the response numerically
train_endog = (Smarket.loc[train_mask, "Direction"] == "Up").astype("int64")
train_exog = sm.add_constant(Smarket[train_mask].drop(columns = ["Direction", "Year", "Today"]))
logit_mod = sm.Logit(train_endog, train_exog)
logit_res = logit_mod.fit()


# Again we emphasize that we are training and testing our model on two completely separate data sets: we used only the dates from before 2005 for training, and only used dates from 2005 for testing. Again let's compute the predictions for 2005 and compare them to the actual movements of the market over that time period.

# In[ ]:


test_exog = sm.add_constant(Smarket[~train_mask].drop(columns = ["Direction", "Year", "Today"]))
test_endog = Smarket.loc[~train_mask, "Direction"]
logit_test_probs = logit_res.predict(test_exog)
logit_test_preds = pd.Series(logit_test_probs).apply(lambda x: "Up" if (x > 0.5) else "Down")
pd.DataFrame(confusion_matrix(test_endog, logit_test_preds), 
             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])


# In[ ]:


(logit_test_preds == Smarket_2005["Direction"].values).mean()


# In[ ]:


(logit_test_preds != Smarket_2005["Direction"].values).mean()


# Now we repeat the process using scikit-learn.

# In[ ]:


# Using LogisticRegression from scikit-learn
# First separate out the training and test sets
X_train = Smarket[train_mask].drop(columns = ["Direction", "Year", "Today"])
y_train = Smarket.loc[train_mask, "Direction"]
X_test = Smarket[~train_mask].drop(columns = ["Direction", "Year", "Today"])
y_test = Smarket.loc[~train_mask, "Direction"]

# Fit the model using the training set
log_reg = LogisticRegression(penalty = "none", solver = "lbfgs")
log_reg.fit(X_train, y_train)

#Test the model using the held-out test set
log_reg_pred = log_reg.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, log_reg_pred), 
             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])


# In[ ]:


log_reg.score(X_test, y_test)


# In[ ]:


1 - log_reg.score(X_test, y_test)


# Here, we see that when we used separate training and testing sets the test error rate is about 52%, which is worse than random guessing! This suggests that we should try to experiment some more to see if we can improve our model. One thing we can try is removing the variables that didn't seem to be helpful in predicting `Direction` and refit the logistic regression model. As we recall, none of the p-values for the predictors were particularly small, even the smallest one which corresponded to `Lag1`. Since using predictors that have no relationship with the response increase the variance of a model without a corresponding decrease in bias, they therefore tend to negatively impact the test error rate. Let's try refitting the logistic regression using just `Lag1` and `Lag2`, which seemed to have the highest predictive power in the original logistic regression model. Again we start by doing this with StatsModels.

# In[ ]:


# Using the Logit class from StatsModels, and training only on the training set
# Separate out the training and test sets
# Then encode the response numerically
train_endog = (Smarket.loc[train_mask, "Direction"] == "Up").astype("int64")
train_exog = sm.add_constant(Smarket.loc[train_mask, ["Lag1", "Lag2"]])
test_exog = sm.add_constant(Smarket.loc[~train_mask, ["Lag1", "Lag2"]])
test_endog = Smarket.loc[~train_mask, "Direction"]

# Fit logistic regression model using the training set
logit_mod = sm.Logit(train_endog, train_exog)
logit_res = logit_mod.fit()

# Test the model using the held-out test set
logit_test_probs = logit_res.predict(test_exog)
logit_test_preds = pd.Series(logit_test_probs).apply(lambda x: "Up" if (x > 0.5) else "Down")
pd.DataFrame(confusion_matrix(test_endog, logit_test_preds), 
             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])


# In[ ]:


(logit_test_preds == Smarket_2005["Direction"].values).mean()


# Now we repeat the process using scikit-learn.

# In[ ]:


# Using LogisticRegression from scikit-learn
# First separate out the training and test sets
X_train = Smarket.loc[train_mask, ["Lag1", "Lag2"]]
y_train = Smarket.loc[train_mask, "Direction"]
X_test = Smarket.loc[~train_mask, ["Lag1", "Lag2"]]
y_test = Smarket.loc[~train_mask, "Direction"]

# Fit the model using the training set
log_reg = LogisticRegression(penalty = "none", solver = "lbfgs")
log_reg.fit(X_train, y_train)

#Test the model using the held-out test set
log_reg_pred = log_reg.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, log_reg_pred), 
             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])


# In[ ]:


log_reg.score(X_test, y_test)


# Here things look a little better, since about 56% of the daily movements in 2005 were correctly predicted. However, if we predicted that the market would increase every day we would also have the same error rate.

# In[ ]:


(Smarket_2005["Direction"] == "Up").mean()


# Thus, in terms of the overall error rate, logistic regression is no better than naively predicting that the market would increase every day. However, when we examine the confusion matrix a little more closely we can see that on days when logistic regression predicts an increase in the maket, it has a slightly better accuracy rate of $106 / (106 + 76) = 0.582$. This suggests a possible strategy of buying on days when the model predicts an increasing market, and avoiding trades when the model predicts a decreasing market. Without further investigation, though, we do not know if this small improvement is real or just due to random chance.
# 
# Lastly, if we wish to predict the returns associated with particular values of `Lag1` and `Lag2`, such as on a day when `Lag1 = 1.2` and `Lag2 = 1.1`, or on a day when `Lag1 = 1.5` and `Lag2 = -0.8`, we can do this using the `predict()` function in StatsModels or `predict_proba()` function in scikit-learn if we want the raw probabilities. As already mentioned, the `predict()` function in scikit-learn will give the predicted class labels.

# In[ ]:


df = pd.DataFrame({"Lag1": [1.2, 1.5], "Lag2":[1.1, -0.8]})


# In[ ]:


# Using StatsModels to generate predicted probabilities for particular values of Lag1 and Lag2
logit_res.predict(sm.add_constant(df))


# In[ ]:


# Using scikit-learn to generate predicted probabilities for particular values of Lag1 and Lag2
pd.DataFrame(log_reg.predict_proba(df), columns = log_reg.classes_)


# In[ ]:


# Using scikit-learn to directly computed predicted class labels
log_reg.predict(df)


# ## Linear Discriminant Analysis

# Next, we'll perform linear discriminant analysis (LDA) on the `Smarket` data. Since StatsModels does not support LDA, we'll only be using the [LinearDiscriminantAnalysis class](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis) from scikit-learn. Just as we did with the second attempt at using logistic regression, we'll only use the observations from before 2005 to fit the model, and we'll only use `Lag1` and `Lag2` as the predictors.
# 
# Note that there are three solvers that the LinearDiscriminantAnalysis class can use:
# 
# - Singular value decomposition `svd` (default): Does not compute covariance matrix, which can be advantageous when there are a large number of features. It can be used for both classification and transformation, though it does not support shrinkage.
# - Least squares `lsqr`: An efficient algorithm, though it can only be used for classification. It does support shrinkage.
# - Eigenvalue decomposition `eigen`: Based on optimizing the "between class scatter" to "within class scatter" ratio. It can be used for both classification and transformation, and also supports shrinkage. Since it needs to compute the covariance matrix, it might not be suitable for situations when there are a large number of features. 
# 
# The coefficients computed using least squares are slightly different from those obtained when using singular value decomposition or eigenvalue decomposition, but overall classification performance should be essentially the same.

# In[ ]:


# First separate out the training and test sets
X_train = Smarket.loc[train_mask, ["Lag1", "Lag2"]]
y_train = Smarket.loc[train_mask, "Direction"]
X_test = Smarket.loc[~train_mask, ["Lag1", "Lag2"]]
y_test = Smarket.loc[~train_mask, "Direction"]

# Fit the LDA model using the training set
lda_clf = LinearDiscriminantAnalysis(store_covariance = True)
lda_clf.fit(X_train, y_train)


# In[ ]:


# Prior probabilities for the classes
pd.Series(lda_clf.priors_, index = lda_clf.classes_)


# In[ ]:


# Group means for each predictor within each class
pd.DataFrame(lda_clf.means_, index = lda_clf.classes_, columns = X_train.columns)


# In[ ]:


# Coefficients for linear discriminants
pd.Series(lda_clf.coef_.flatten(), index = X_train.columns)


# First, we note that `lda_clf.priors_` indicates that the estimated prior probabilities of the groups are $\hat{\pi}_1 = 0.492$ and $\hat{\pi}_2 = 0.508$. This means that 49.2% of the training observations correspond to days during which the market went down and 50.8% of the training observations correspond to days during which the market went up. In addition, the output also provides the averages of each predictor within each class (e.g. the average value of `Lag1` for days in which the market went down is 0.0428) with `lda_clf.means_`, which are used in LDA as estimates of the actual group means $\mu_k$. These suggest that there is a tendency for the previous two days' returns to be negative on the days when the market increases, and a tendency for the previous two days' returns to be positive on the days when the market decreases. Lastly, `lda_clf.coef_` provides the coefficients of `Lag1` and `Lag2` that are used to form the LDA decision rule. In other words, we can think of these as the multipliers of the elements of $X = x$ in the formula
# 
# \begin{equation}
#     \delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log(\pi_k).
# \end{equation}
# 
# Recall that in the fomula $\Sigma$ is the $2 \times 2$ covariance matrix of `Lag1` and `Lag2`; $\mu_k$ is the vector of the group means of `Lag1` and `Lag2` for either the group `Down` or the group `Up`; $\pi_k$ is the estimated prior probability of either `Down` or `Up`. Strictly speaking, since we are using estimates in the model, $\Sigma$, $\mu_k$, and $\pi_k$ should be replaced with their hatted versions. Also recall that this formula is the vector/matrix version of
# 
# \begin{equation}
#     \delta_k = x\frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log(\pi_k).
# \end{equation}
# 
# When working with just two classes $k$ and $l$, the coefficients are actually from the log-ratio description of LDA, which is discussed in Section 4.3 (Formula 4.9) of *Elements of Statistical Learning*, as explained in the [user guide page for linear and quadratic discriminant analysis](https://scikit-learn.org/stable/modules/lda_qda.html).
# 
# \begin{equation}
#     \log \frac{\text{Pr}(G = k | X = x)}{\text{Pr}(G = l | X = x)} = 
#     x^T\Sigma^{-1}(\mu_k - \mu_l) - \frac{1}{2}(\mu_k + \mu_l)^T\Sigma^{-1}(\mu_k - \mu_l) + \log \frac{\pi_k}{\pi_l}
# \end{equation}
# 
# 
# More concretely, if $-0.055441 \times$ `Lag1` $-0.044345 \times$ `Lag2` is large, then the LDA classifier will predict a market increase; if it is small, then the LDA classifier will predict a market decline.

# In[ ]:


# Scalings for linear discriminants
pd.Series(lda_clf.scalings_.flatten(), index = X_train.columns)


# In addition to the classifier coefficients, the LinearDiscriminantAnalysis class also computes scaled coefficients, stored as `lda_clf.scalings_`, which are the coefficients printed out by `lda()` in R. One way of thinking about the scalings is that they are the LDA coefficients one would obtain after normalizing the data so that the within groups covariance matrix is spherical (also known as [whitening](https://theclevermachine.wordpress.com/2013/03/30/the-statistical-whitening-transform/) the data to have an identity covariance matrix), as discussed in the [R documentation for `lda()`](https://www.rdocumentation.org/packages/MASS/versions/7.3-51.4/topics/lda). More of the underlying mathematics of the scaled coefficients is discussed in detail in [this stats.stackexchange post](https://stats.stackexchange.com/a/48859) and can also be found in the source code for the LinearDiscriminantAnalysis class, as well as the section of the user guide decribing how to use [LDA for dimensionality reduction](https://scikit-learn.org/stable/modules/lda_qda.html#mathematical-formulation-of-lda-dimensionality-reduction). One last mathematical source is Section 4.3.2 in *Elements of Statistical Learning*. Looking under the hood at the source code for the `_solve_eigen()` function, which is used to compute the model via an eigenvalue decomposition, we can more clearly see the relationship between `scalings_`, which are eigenvectors in the decomposition used for solving, and the original values of `coef_`.
# 
# ```
# self.scalings_ = evecs
# self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
# ```
# 
# To confirm those relationships, we compute the by hand below.

# In[ ]:


# Computing the coefficients for Lag1 and Lag2 using the log-ratio formula above
# These coefficients match with the coefficient values obtained when using the least-squares solver
np.linalg.inv(lda_clf.covariance_) @ (lda_clf.means_[1] - lda_clf.means_[0])


# In[ ]:


# Relating the coefficients with the scaling values
# These are the coefficients when using the singular value decomposition
# or eigenvalue decomposition solver
orig_coefs = np.dot(lda_clf.means_, lda_clf.scalings_).dot(lda_clf.scalings_.T)
log_ratio_coefs = orig_coefs[1, :] - orig_coefs[0, :]
log_ratio_coefs


# Before moving forward, we can compute the LDA scores using the `decision_function()` method. When performing binary classification, as we are doing in this lab, the scores computed will be the log-ratio scores. In this case, a value greater than zero means that the class corresponding to `lda_clf.classes_[1]` will be predicted.

# In[ ]:


lda_scores = lda_clf.decision_function(X_train)
sns.distplot(lda_scores, kde = False, axlabel = "log-ratio score")


# There are four methods that the LinearDiscriminantAnalysis class can use for prediction-related computations. 
# 
# - `predict()` contains LDA's predictions about the movement of the market.
# - `predict_proba()` is a matrix whose $k$th column contains the posterior probability that the corresponding observation belongs to the $k$th class. This is computed using Bayes' theorem.
# - `decision_function()` contains the linear discriminant scores (log-ratio scores in binary classification), as described earlier.
# - `score()` computes the accuracy on the given test data by computing the mean accuracy of `predict()` with respect to the true class labels.
# 
# Recall that Bayes' theorem states that
# 
# \begin{equation}
#     \text{Pr}(Y = k | X = x) = \frac{\pi_k f_k(x)}{\sum_{l = 1}^K \pi_l f_l(x)},
# \end{equation}
# 
# where $K$ is the total number of classes for $Y$; $\pi_k$ is the overall, or *prior* probability that a randomly chosen observation comes from the $k$th class; and $f_k(x)$ is defined to be $\text{Pr}(X = x | Y = k)$, the probability that $X = x$ for an observation $X$ that comes from the $k$th class.

# In[ ]:


lda_pred = lda_clf.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, lda_pred), 
             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])


# Recall that in Section 4.5 of the book, we saw that LDA and logistic regression often have similar performance. We see that this is the case for our stock market data. In fact, it just so happens that we have an identical confusion matrix and test accuracy number compared to the logistic regression model.

# In[ ]:


lda_clf.score(X_test, y_test)


# Notice that since there are two classes, LDA uses a posterior probability threshold of 50% in order to make the predictions that go into `class`.

# In[ ]:


lda_pred_probs = pd.DataFrame(lda_clf.predict_proba(X_test), columns = lda_clf.classes_)
lda_pred_probs.head()


# In[ ]:


pd.Series(lda_pred).head()


# If we wanted to use a posterior probability threshold other than 50% in order to make predictions, we can use a boolean mask with the matrix of posterior probabilities. For example, let's suppose we wish to predict a market decrease only if we are very certain that it will indeed decrease on that day -- say, if the posterior probability of a decrease is at least 90%.

# In[ ]:


(lda_pred_probs["Down"] > 0.9).sum()


# No days in 2005 meet that threshold. In fact, the greatest posterior probability of decrease in all of 2005 was 52.02%.

# In[ ]:


lda_pred_probs["Down"].max()


# Moreover, the overall greatest posterior probability for all of 2005 is only 54.22%.

# In[ ]:


lda_pred_probs.max()


# ## Quadratic Discriminant Analysis

# Next, let's fit a quadratic discriminant analysis (QDA) model to the Smarket data. QDA is also part of `sklearn.discriminant_analysis` as the `QuadraticDiscriminantAnalysis()` class. It uses the same syntax as that of `LinearDiscriminantAnalysis`, though it doesn't have the option to choose a solver aside from singular value decomposition.

# In[ ]:


# First separate out the training and test sets
X_train = Smarket.loc[train_mask, ["Lag1", "Lag2"]]
y_train = Smarket.loc[train_mask, "Direction"]
X_test = Smarket.loc[~train_mask, ["Lag1", "Lag2"]]
y_test = Smarket.loc[~train_mask, "Direction"]

# Fit the QDA model using the training set
qda_clf = QuadraticDiscriminantAnalysis()
qda_clf.fit(X_train, y_train)


# In[ ]:


# Prior probabilities for the classes
pd.Series(qda_clf.priors_, index = qda_clf.classes_)


# In[ ]:


# Group means for each predictor within each class
pd.DataFrame(qda_clf.means_, index = qda_clf.classes_, columns = X_train.columns)


# While the QuadraticDiscriminantAnalysis class stores many of the same parameters as the LinearDiscriminantAnalysis class, such as the prior probabilities of the groups (`qda_clf.priors_`) as well as the group means (`qda_clf.means_`), it does not have a parameter storing coefficients of the linear discriminants. This is because the QDA classifier involves a quadratic, not linear, function of the predictors. Recall that in QDA the formula $\delta_k(x)$, the posterior probability that the observation $X = x$ is a member of the $k$th class, is
# 
# \begin{align}
#     \delta_k(x) &= -\frac{1}{2}(x - \mu_k)^T\Sigma_k^{-1}(x - \mu_k) - \frac{1}{2}\log|\Sigma_k| + \log(\pi_k) \\
#     &= -\frac{1}{2}x^T\Sigma_K^{-1}x + x^T\Sigma_k^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma_k^{-1}\mu_k - \frac{1}{2}\log|\Sigma_k| + \log(\pi_k),
# \end{align}
# 
# where observations from the $k$th class are normally distributed with (population) mean $\mu_k$ and (population) covariance matrix $\Sigma_k$. This comes from the assumption in QDA that each class has its own covariance matrix, which differs from the assumption in LDA that the classes had the same covariance matrix.
# 
# Even though the QuadraticDiscriminantAnalysis class is slightly different in the parameters it stores, it has the same four prediction-related methods as LinearDiscriminantAnalysis (`predict()`, `predict_proba()`, `decision_function()`, and `score()`.

# In[ ]:


qda_pred = qda_clf.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, qda_pred), 
             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])


# In[ ]:


qda_clf.score(X_test, y_test)


# Here, we see that the QDA predictions are accurate almost 60% of the time for the 2005 data, which wasn't used to fit the model. This suggests that the quadratic form assumed by QDA may capture the true relationship more accurately than the linear forms assumed by LDA and logistic regression. However, this model should be evaluated on a larger test set to get a better sense of its predictive strength.
# 
# As a side note, we can again use a boolean mask with the matrix of posterior probabilities to make predictions with a probability threshold that is different from 50%. Let's see what QDA says for the cautious scenario of using a 90% probability threshold to predict a decrease.

# In[ ]:


qda_pred_probs = pd.DataFrame(qda_clf.predict_proba(X_test), columns = lda_clf.classes_)
(qda_pred_probs["Down"] > 0.9).sum()


# Again no days in 2005 meet this threshold, since the maximum posterior probability of a decrease in all of 2005 was just 52.24%, and the overall maximum posterior probability in all of 2005 was 54.62%.

# In[ ]:


qda_pred_probs.max()


# ## K-Nearest Neighbors

# The last method of working with the `Smarket` data we'll use is K-nearest neighbors. To do this, we'll use the `KNeighborsClassifier` class from the `sklearn.neighbors` module. For basic documentation, we refer to the [user guide section on nearest neighbors classification](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification), as well as the [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) for the `KNeighborsClassifier` class. Unlike the KNN classifier in R, one benefit of using scikit-learn is the fact that the process for fitting the model and then using it to make predictions is consistent with the previous classifiers we have used in this lab.
# 
# Note that the `KNeighborsClassifier` takes a few arguments that are worth pointing out.
# 
# - `n_neighbors`: This is the value of $K$. By default it is 5.
# - `weights`: A weight function which allows us to weight the neighbors when performing prediction. This can either be uniform weighting (the default); inverse-distance weighting to give closer neighbors higher weight values than further ones; or a user-defined weight function.
# - `algorithm`: The algorithm used to compute the nearest neighbors. The options are [BallTree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree), [KDTree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree), brute-force, and attempting to automatically choose one based on the values passed to the `fit()` method.
# - `metric`: The distance metric used to measure the distance between neighbors. There are a number of options described in the [DistanceMetric class](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric). By default Euclidean distance (i.e. Minkowski distance with $p = 2$) is used.
# 
# For the purposes of this lab, the only parameter which we will modify is `n_neighbors`.

# In[ ]:


# First separate out the training and test sets
X_train = Smarket.loc[train_mask, ["Lag1", "Lag2"]]
y_train = Smarket.loc[train_mask, "Direction"]
X_test = Smarket.loc[~train_mask, ["Lag1", "Lag2"]]
y_test = Smarket.loc[~train_mask, "Direction"]

# Set NumPy random seed for consistency and reproducibility for our results
np.random.seed(1)

# Fit the KNN model using the training set and K = 1
knn_1_clf = KNeighborsClassifier(n_neighbors = 1)
knn_1_clf.fit(X_train, y_train)


# Note that in order to have consistency and have reproducibility for our results, we set a random seed before performing `KNeighborsClassifier.fit()` since Python randomly breaks any ties if several observations are tied as nearest neighbors.

# In[ ]:


knn_1_pred = knn_1_clf.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, knn_1_pred), 
             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])


# In[ ]:


knn_1_clf.score(X_test, y_test)


# Using $K = 1$, our results aren't very good, since only $(43 + 83)/252 = 50\%$ of the observations are correctly predicted. Of course, using $K=1$ is a very flexible fit, so perhaps it was too much flexibility. Let's try again using $K = 3$.

# In[ ]:


# Fit the KNN model using the training set and K = 3
knn_3_clf = KNeighborsClassifier(n_neighbors = 3)
knn_3_clf.fit(X_train, y_train)
knn_3_pred = knn_3_clf.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, knn_3_pred), 
             columns = ["PredDown", "PredUp"], index = ["ActualDown", "ActualUp"])


# In[ ]:


knn_3_clf.score(X_test, y_test)


# In[ ]:


knn_scores = []
for k in range(1, 16, 2):
    knn_clf = KNeighborsClassifier(n_neighbors = k)
    knn_clf.fit(X_train, y_train)
    knn_scores.append(knn_clf.score(X_test, y_test))
fig = plt.figure()
ax = plt.axes()
ax.plot(range(1, 16, 2), knn_scores)
ax.set(xlabel = "n_neighbors", ylabel = "test accuracy");


# The results improved slightly to about 53.2% test accuracy, though further increasing $K$ does not provide any additional improvements. For this data, it appears that QDA provides the best results of the methods we have examined thus far.

# ## An Application to Caravan Insurance Data

# Lastly, we'll use KNN with the `Caravan` data set from the `ISLR` library. This set includes 85 predictors that measure demographic characteristics for 5,822 individuals. The response variable is `Purchase` which indicates whether or not a given individual purchases a caravan insurance policy. After loading the data into a Pandas dataframe, let's explore it a little bit.

# In[ ]:


caravan_filepath = "../input/Caravan.csv"
Caravan = pd.read_csv(caravan_filepath, index_col = "Unnamed: 0")
Caravan.head()


# In this set, only about 6% of people purchased caravan insurance.

# In[ ]:


Caravan["Purchase"].value_counts()


# In[ ]:


(Caravan["Purchase"] == "Yes").mean()


# Before proceeding, we note that since the KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, the scale of the variables matters. Variables with a large scale will have a much larger effect on the *distance* between the observations, and hence on the KNN classifier, than variables that are on a small scale. For example, if we have a data set with the variables `salary` (measured in dollars) and `age` (measured in years), a difference of \\$1000 in salary is quite large compared to a difference of 50 years of age when it comes to computing distances in KNN. As a result, `salary` will dominate the KNN classification results while `age` will have almost no effect. Moreover, the importance of scale to the KNN classfier means that changing the units (such as measuring `salary` in Japanese yen, or measuring `age` in minutes) of the predictors could result in vastly different classification results compared to what we would get when using dollars and years.
# 
# One way to handle this problem is to *standardize* the data so that all variables are given a mean of zero and a standard deviation of one. Doing this will put all variables on a comparable scale. While we could do this by hand, it is much more convenient to use the [`transform()` function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.transform.html#pandas.DataFrame.transform) as described in the section on [split-apply-combine in the Pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#transformation). When doing this with the `Caravan` data, we exclude the qualitative `Purchase` variable.

# In[ ]:


standardized_X = Caravan.drop(columns = ["Purchase"]).transform(lambda x: (x - x.mean())/x.std())


# We can check the mean and variance of first two columns of `Caravan` and compare those with that of the standardized columns to confirm that the standardized ones have mean zero and standard deviation one.

# In[ ]:


Caravan.mean().head()


# In[ ]:


standardized_X.mean().head()


# Note that the standardized means aren't exactly equal to zero due to slight floating point precision errors.

# In[ ]:


Caravan.var().head()


# In[ ]:


standardized_X.var().head()


# Now we split the observations into a test set, containing the first 1,000 observations, and a training set, containing the remaining observations.

# In[ ]:


test_mask = range(1, 1001)
X_train = standardized_X.drop(index = test_mask)
X_test = standardized_X.loc[test_mask, ]
y_train = Caravan.drop(index = test_mask)["Purchase"]
y_test = Caravan.loc[test_mask, "Purchase"]


# Once we have separated out the training and test observations, we start off by fitting a KNN model on the training data using $K = 1$ and then evaluate is performance on the test data.

# In[ ]:


# Set NumPy random seed for consistency and reproducibility for our results
np.random.seed(1)

# Fit the KNN model using the training set and K = 1
knn_1_clf = KNeighborsClassifier(n_neighbors = 1)
knn_1_clf.fit(X_train, y_train)


# In[ ]:


1 - knn_1_clf.score(X_test, y_test)


# The KNN error rate with $K = 1$ is just under 12%, which appears to be pretty good at a first glance. However, since only 6% of customers purchased insurance, we could get the error rate down to 6% just by always predicting `No` regardless of the values of the predictors.

# In[ ]:


(y_test != "No").mean()


# While the overall error rate is rather low, depending on the situation we may be interested in other methods of measuring the quality of the model. For example, if there is some non-trivial cost to trying to sell insurance to a given individual, then the company would like to try and sell insurance only to customers who are likely to buy it. In that case, the company is more interested in the fraction of individuals that are correctly predicted to buy insurance.

# In[ ]:


knn_1_pred = knn_1_clf.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, knn_1_pred), 
             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])


# Looking at the fraction of individuals that are correctly predicted to buy insurance, the KNN classifier with $K = 1$ does far better than random guessing among the customers that are predicted to buy insurance. Among the 77 customers it predicted to buy insurance, 9 (11.7%) of them actually do purchase insurance. This is double the rate one would obtain from random guessing.
# 
# We can also explore other values of $K$, such as $K = 3$ and $K = 5$, to see how performance changes.

# In[ ]:


# Using K = 3
knn_3_clf = KNeighborsClassifier(n_neighbors = 3)
knn_3_clf.fit(X_train, y_train)
knn_3_pred = knn_3_clf.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, knn_3_pred), 
             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])


# In[ ]:


# Using K = 5
knn_5_clf = KNeighborsClassifier(n_neighbors = 5)
knn_5_clf.fit(X_train, y_train)
knn_5_pred = knn_5_clf.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, knn_5_pred), 
             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])


# With $K = 3$, the success rate of correctly predicting customers who buy insurance increases to $5/25 = 20\%$, and with $K = 5$ the rate is $4/15 = 26.7\%$. That is over four times the rate that we get from random guessing! This suggests that KNN is finding some real patterns in a difficult data set.
# 
# To compare, we can also fit a logistic regression model to the data.

# In[ ]:


# Using the Logit class from StatsModels, and training only on the training set
# Separate out the training and test sets
# Then encode the response numerically
train_endog = (y_train == "Yes").astype("int64")
train_exog = sm.add_constant(X_train)
# Need to use has_constant = "add" to force StatsModels to include a constant term;
# X_test coincidentally happens to have a column which is already constant
test_exog = sm.add_constant(X_test, has_constant = "add")
test_endog = y_test

# Fit logistic regression model using the training set
logit_mod = sm.Logit(train_endog, train_exog)
logit_res = logit_mod.fit()

# Test the model using the held-out test set
logit_test_probs = logit_res.predict(test_exog)
logit_test_preds = pd.Series(logit_test_probs).apply(lambda x: "Yes" if (x > 0.5) else "No")
pd.DataFrame(confusion_matrix(test_endog, logit_test_preds), 
             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])


# As we can see, if we use 0.5 as the predicted probability cut-off for logistic regression, only seven of the test observations are predicted to purchase insurance. Even worse, all seven of these predictions are incorrect!

# In[ ]:


logit_test_preds = pd.Series(logit_test_probs).apply(lambda x: "Yes" if (x > 0.25) else "No")
pd.DataFrame(confusion_matrix(test_endog, logit_test_preds), 
             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])


# We aren't required to use a cut-off of 0.5, though. If we instead use a cut-off of 0.25, we get much better results: we predict that 33 people will purchase insurance, and we are correct for 33% of these people. This is over five times better than random guessing!
# 
# To wrap up this lab, let's repeat the logistic regression computations we just did using scikit-learn. Note that since `log_reg.predict()` uses a cut-off of 0.5 by default for binary classification, we need to work with the probabilities generated by `log_reg.predict_proba()` by hand in order to use different cut-offs.

# In[ ]:


# Fit the model using the training set
log_reg = LogisticRegression(penalty = "none", solver = "newton-cg")
log_reg.fit(X_train, y_train)

#Test the model using the held-out test set
log_reg_pred = log_reg.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, log_reg_pred), 
             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])


# In[ ]:


log_reg_prob = pd.DataFrame(log_reg.predict_proba(X_test), columns = log_reg.classes_)
log_reg_pred = log_reg_prob["Yes"].apply(lambda x: "Yes" if (x > 0.25) else "No")
pd.DataFrame(confusion_matrix(y_test, log_reg_pred), 
             columns = ["PredNo", "PredYes"], index = ["ActualNo", "ActualYes"])


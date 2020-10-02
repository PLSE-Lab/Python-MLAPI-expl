#!/usr/bin/env python
# coding: utf-8

# # Lab: Linear Regression

# In this lab, we will go over how to do linear regression using Python. There are a few ways we can go about this, depending on how much we are interested in investigating ancillary statistics for the coefficients in the model (p-values, residual standard error, etc.). If our main goal is to generate the model and focus on the basic statistics for evaluating the model, then [scikit-learn](https://scikit-learn.org/) will work well. If we want to analyze detailed ancillary statistics associated with the model, then [StatsModels](http://www.statsmodels.org/) will generate those statistics without requiring us to code functions to compute them from scratch. Also, StatsModels allows for the usage of [R-style formulas](http://www.statsmodels.org/stable/example_formulas.html) when fitting models for those who are already comfortable with that syntax. For completeness and to get practice using both packages, I will complete this lab both ways.

# ## Loading the necessary packages and data sets

# This lab involves the `Boston` and `Carseats` data sets, so before getting started we should make sure we have them available on our computer along with our usual Python data science packages. Recall that since we are analyzing these data sets in Python instead of `R`, we may should make sure to download the corresponding CSV files for each set from the book's [website](http://www.statlearning.com) under the "Data Sets and Figures" link. The corrected Boston housing data set which I am using can be downloaded from the [CMU StatLib archive](http://lib.stat.cmu.edu/datasets/boston_corrected.txt). The `Carseats` data set wasn't available directly as a CSV file from the book's website, so I needed to load it from the ISLR library in R and then export it.

# In[ ]:


# Load the standard Python data science packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load the LinearRegression class from scikit-learn's linear_model module
from sklearn.linear_model import LinearRegression

# Load the stats module from scipy so we can code the functions to compute model statistics
from scipy import stats

# Load StatsModels API
# Note that if we wish to use R-style formulas, then we would load statsmodels.formula.api instead
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ## Simple linear regression

# As we saw in the third applied exercise from Chapter 2, one of the factors in the `Boston` data set is `cmedv`, which records the median house value for 506 tracts of land around Boston. To start with, we'll try to predict `cmedv` using the other factors, such as `rm` (average number of rooms per house), `age` (percent of owner-occupied homes built prior to 1940), and `lstat` (percent of low socioeconomic households). First, we'll take a look at the first few rows of data and then check again for missing values.

# In[ ]:


# Load the corrected Boston housing data set
# Create a multi-index on TOWN and TRACT columns
# Exclude the TOWNNO, LON, LAT, and MEDV columns when loading the set
boston_filepath = "../input/corrected-boston-housing/boston_corrected.csv"
boston = pd.read_csv(boston_filepath, index_col = ["TOWN", "TRACT"], 
                     usecols = ["TOWN", "TRACT", "CMEDV", "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])


# In[ ]:


boston.head()


# In[ ]:


boston.isnull().any()


# First, we'll use scikit-learn to fit a simple linear regression model using `CMEDV` as the response and `LSTAT` as the predictor. To do so, we create a `LinearRegression` object and then use the `LinearRegression.fit(X, y)` function to fit a linear model with `X` as the predictor and `y` as the response. For more details, we can look at the [official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) and a [basic example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html). Note that the `fit(X, y)` function requires the training `X` and `y` data to have the correct shape. In particular, when working with a single predictor, we cannot just pass the corresponding column of the dataframe as an argument, as `df["Col_name"].shape` has the value `(num_rows, )`, meaning that the underlying values are stored in a one-dimensional array. Instead, we need a two-dimensional array with shape `(num_rows, 1)`, which means that we have `num_rows` observations, and a single feature value for each observation. As soon as we start using multiple predictors, `df.loc[:, ["Pred_1", ..., "Pred_p"]]` will have the correct shape of `(num_rows, p)`. The same is true as soon as we start using multiple targets.

# In[ ]:


# Create a LinearRegression object
reg = LinearRegression()
# Need to extract the two columns we are using and then reshape them so that
# X has the shape (n_samples, n_features),
# Y has the shape (n_samples, n_targets)
# When using -1 as one of the reshape arguments, NumPy will infer the value
# from the length of the array and the values for the other dimensions
X = boston["LSTAT"].values.reshape(-1, 1)
y = boston["CMEDV"].values.reshape(-1, 1)
# Fit linear regression model using X = LSTAT, y = CMEDV
reg.fit(X, y);


# Once we have fitted a linear regression model, the `LinearRegression` object stores some basic info that we can access.

# In[ ]:


print("Model coefficients:", reg.coef_)
print("Model intercept:", reg.intercept_)


# This tells us that the regression line takes the form $y = -0.95x + 34.58$, where $y$ is the response `CMEDV` and $x$ is the predictor `LSTAT`. We can also access the value for $R^2$, the coefficient of determination, using the `score()` function.

# In[ ]:


# In the score function, X is an array of the test samples
# It needs to have shape = (num_samples, num_features)
# y is an array of the true values for the given X
reg.score(X, y)


# If we want any more information beyond this, such as p-values and standard errors for the coefficients, or residual standard error and F-statistic values for the model, we'll need to write some additional code of our own using SciPy and some linear algebra knowledge. The underlying math which goes into the calculation of the variance for each coefficient in a least-squares model with p predictors can be found at the beginning of Chapter 3 in the book [*Elements of Satistical Learning*](http://www-stat.stanford.edu/ElemStatLearn). This [Stackoverflow post](https://stackoverflow.com/a/42677750) serves as the basis for my implementation.

# In[ ]:


def detailed_linear_regression(X, y):
    """
    Assume X is array-like with shape (num_samples, num_features)
    Assume y is array-like with shape (num_samples, num_targets)
    Computes the least-squares regression model and returns a dictionary consisting of
    the fitted linear regression object; a series with the residual standard error,
    R^2 value, and the overall F-statistic with corresponding p-value; and a dataframe
    with columns for the parameters, and their corresponding standard errors,
    t-statistics, and p-values.
    """
    # Create a linear regression object and fit it using x and y
    reg = LinearRegression()
    reg.fit(X, y)
    
    # Store the parameters (regression intercept and coefficients) and predictions
    params = np.append(reg.intercept_, reg.coef_)
    predictions = reg.predict(X)
    
    # Create matrix with shape (num_samples, num_features + 1)
    # Where the first column is all 1s and then there is one column for the values
    # of each feature/predictor
    X_mat = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
    
    # Compute residual sum of squares
    RSS = np.sum((y - predictions)**2)
    
    # Compute total sum of squares
    TSS = np.sum((np.mean(y) - y)**2)
    
    # Estimate the variance of the y-values
    obs_var = RSS/(X_mat.shape[0] - X_mat.shape[1])
    
    # Residual standard error is square root of variance of y-values
    RSE = obs_var**0.5
    
    # Variances of the parameter estimates are on the diagonal of the 
    # variance-covariance matrix of the parameter estimates
    var_beta = obs_var*(np.linalg.inv(np.matmul(X_mat.T, X_mat)).diagonal())
    
    # Standard error is square root of variance
    se_beta = np.sqrt(var_beta)
    
    # t-statistic for beta_i is beta_i/se_i, 
    # where se_i is the standard error for beta_i
    t_stats_beta = params/se_beta
    
    # Compute p-values for each parameter using a t-distribution with
    # (num_samples - 1) degrees of freedom
    beta_p_values = [2 * (1 - stats.t.cdf(np.abs(t_i), X_mat.shape[0] - 1))
                    for t_i in t_stats_beta]
    
    # Compute value of overall F-statistic, to measure how likely our
    # coefficient estimate are, assuming there is no relationship between
    # the predictors and the response
    F_overall = ((TSS - RSS)/(X_mat.shape[1] - 1))/(RSS/(X_mat.shape[0] - X_mat.shape[1]))
    F_p_value = stats.f.sf(F_overall, X_mat.shape[1] - 1, X_mat.shape[0] - X_mat.shape[1])
    
    # Construct dataframe for the overall model statistics:
    # RSE, R^2, F-statistic, p-value for F-statistic
    oa_model_stats = pd.Series({"Residual standard error": RSE, "R-squared": reg.score(X, y),
                                "F-statistic": F_overall, "F-test p-value": F_p_value})
    
    # Construct dataframe for parameter statistics:
    # coefficients, standard errors, t-statistic, p-values for t-statistics
    param_stats = pd.DataFrame({"Coefficient": params, "Standard Error": se_beta,
                                "t-value": t_stats_beta, "Prob(>|t|)": beta_p_values})
    return {"model": reg, "param_stats": param_stats, "oa_stats": oa_model_stats}


# In[ ]:


detailed_reg = detailed_linear_regression(X, y)


# In[ ]:


np.round(detailed_reg["param_stats"], 4)


# In[ ]:


np.round(detailed_reg["oa_stats"], 4)


# This was a lot of work! It took me a solid 1.5 days of working through the Stackoverflow post and section of *Elements of Statistical Learning* to feel like I had a decently confident grasp on the underlying mathematics and how to translate that math into Python code, but I think it was a really good process to go through. Now, let's see how we can generate the model and access the summary statistics using [StatsModels](http://www.statsmodels.org/stable/regression.html#references).

# In[ ]:


# Use the terms exog (exogenous) and endog (endogenous) for X and y,
# respectively to match the language used in the StatsModels documentation
# Need to manually add a column for the intercept, as StatsModels does not
# include it by default when performing ordinary least-squares regression
exog = sm.add_constant(boston["LSTAT"])
endog = boston["CMEDV"]

# Generate the model
mod = sm.OLS(endog, exog)

# Fit the model
res = mod.fit()

#Print out model summary
print(res.summary())


# Note that there are a lot of other attributes of the fitted model that we can analyze and use. See the documentation for the [OLSResults object](http://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.html#statsmodels.regression.linear_model.OLSResults). As we can see, that was a lot simpler! This reflects one of the big differences between the philosophies behind scikit-learn and StatsModels -- scikit-learn is more focused on applying the results of machine learning strategies, while StatsModels has a larger emphasis on being able to easily analyze the detailed underlying statistics that go into generating a model.
# 
# Next, let's compute some confidence and prediction intervals for various aspects of our least squares regression model. Once again we'll start off with doing things by hand using SciPy, though thankfully this is a bit easier to do as we can use the `interval(alpha, df, loc = 0, scale = 1)` function associated with SciPy's [Student t-distribution](https://docs.scipy.org/doc/scipy-1.3.0/reference/generated/scipy.stats.t.html) object. First off, we'll produce 95% confidence intervals for the parameter estimates.

# In[ ]:


def param_conf_int(X, y, level = 0.95):
    """
    Assume X is array-like with shape (num_samples, num_features)
    Assume y is array-like with shape (num_samples, num_targets)
    Assume level, if given, is a float with 0 < level < 1
    Computes confidence intervals at the given confidence level for each parameter
    in the linear regression model relating the predictors X to the response y
    Returns a dataframe with the endpoints of the confidence interval for each parameter
    """
    # Store parameters and corresponding stats for easy access
    detailed_reg = detailed_linear_regression(X, y)
    param_stats = detailed_reg["param_stats"]
    conf_intervals = pd.DataFrame()
    # Degrees of freedom = num_samples - (num_features + 1)
    df = X.shape[0] - (X.shape[1] + 1)
    a, b = str(round((1 - level)*100/2, 2)) + "%", str(round((1 + level)*100/2, 2)) + "%"
    # Loop through each parameter
    for index in param_stats.index:
        coeff = param_stats.loc[index, "Coefficient"]
        std_err = param_stats.loc[index, "Standard Error"]
        # alpha = level of confidence
        # df = degrees of freedom = num_samples - number of parameters
        # loc = center of t-interval = estimated coefficient value
        # scale = standard error in coefficient estimate
        conf_intervals = conf_intervals.append(pd.DataFrame([stats.t.interval(level, df, loc = coeff, scale = std_err)],
                                                            columns = [a, b]), ignore_index = True)
    return conf_intervals


# In[ ]:


param_conf_int(X, y)


# When using a model generated by StatsModels, we can use the `conf_int(alpha = 0.05)` method of the RegressionResults object. Note that in StatsModels, the argument `alpha` refers to the *significance level*, which is equivalent to $1 - l$, where $l$ is the *confidence level*.

# In[ ]:


res.conf_int()


# By adjusting the `level` parameter in the `param_conf_int()` function I wrote, or by adjusting the `alpha` parameter in the `conf_int()` function from StatsModels, we can also produce other confidence intervals for the parameters. For example, we can produce 99% confidence intervals.

# In[ ]:


param_conf_int(X, y, level = 0.99)


# In[ ]:


res.conf_int(alpha = 0.01)


# Next, we compute confidence and prediction intervals for the prediction of `CMEDV` for a given value of `LSTAT`. Once again, we'll start off by coding a function by hand to do this, referring to [these notes from Stanford](http://statweb.stanford.edu/~susan/courses/s141/horegconf.pdf) for the underlying mathematics of what values of the error to use when computing the intervals. Note that the Stanford notes only cover the situation of a single predictor and a single response.

# In[ ]:


def predict_intervals(X, y, X_star, level = 0.95, kind = "confidence"):
    """
    Assume X is array-like with shape (num_samples, num_features)
    Assume y is array-like with shape (num_samples, num_targets)
    Assume X_star is array-like with shape (num_predictions, num_features) with x-values for which we want predictions
    Assume level, if given, is a float with 0 < level < 1
    Assume kind, if given is either the string "confidence" or "prediction" for the kind of interval
    Computes confidence intervals at the given confidence level for each parameter
    in the linear regression model relating the predictors X to the response y
    Returns a dataframe with the endpoints of the confidence interval for each parameter
    """
    # Store parameters and corresponding stats for easy access
    detailed_reg = detailed_linear_regression(X, y)
    predictions = detailed_reg["model"].predict(X_star)
    RSE = detailed_reg["oa_stats"]["Residual standard error"]
    intervals = pd.DataFrame()
    # Degrees of freedom = num_samples - (num_features + 1)
    df = X.shape[0] - (X.shape[1] + 1)
    a, b = str(round((1 - level)*100/2, 2)) + "%", str(round((1 + level)*100/2, 2)) + "%"
    x_bar = X.mean()
    x_tss = np.sum((X - x_bar)**2)
    # Loop through each x-value being used for prediction
    for i in range(len(predictions)) :
        prediction = predictions[i, 0]
        x_star = X_star[i, 0]
        conf_error = RSE * (1/X.shape[0] + (x_star - x_bar)**2/x_tss)**0.5
        predict_error = (RSE**2 + conf_error**2)**0.5
        # alpha = level of confidence
        # df = degrees of freedom = num_samples - number of parameters
        # loc = center of t-interval = predicted value from linear regression model
        # scale = standard error in predicted value estimate
        if (kind == "confidence"):
            lower, upper = stats.t.interval(level, df, loc = prediction, scale = conf_error)
            intervals = intervals.append(pd.Series({"prediction": prediction, a: lower, b: upper}),
                                         ignore_index = True)
        elif (kind == "prediction"):
            lower, upper = stats.t.interval(level, df, loc = prediction, scale = predict_error)
            intervals = intervals.append(pd.Series({"prediction": prediction, a: lower, b: upper}),
                                         ignore_index = True)
    return intervals


# FIrst we produce 95% confidence intervals for the predicted `CMEDV` value using `LSTAT` values of 5, 10, and 15.

# In[ ]:


predict_intervals(X, y, np.array([5, 10, 15]).reshape((-1, 1)), level = 0.95, kind = "confidence")


# Similarly, we can produce 99% confidence intervals for the predicted `CMEDV` values, again using `LSTAT` values of 5, 10, and 15.

# In[ ]:


predict_intervals(X, y, np.array([5, 10, 15]).reshape((-1, 1)), level = 0.99, kind = "confidence")


# Now we produce 95%, and then 99%, prediction intervals for the predicted `CMEDV` values, once again using `LSTAT` values of 5, 10, and 15.

# In[ ]:


predict_intervals(X, y, np.array([5, 10, 15]).reshape((-1, 1)), level = 0.95, kind = "prediction")


# In[ ]:


predict_intervals(X, y, np.array([5, 10, 15]).reshape((-1, 1)), level = 0.99, kind = "prediction")


# Looking at the intervals, we can see that a 95% confidence interval for the predicted value of `CMEDV` using an `LSTAT` value of 10 is (24.481, 25.631), while the corresponding 95% prediction interval is (12.913, 37.199). Both intervals are centered around the same point, the predicted value for `CMEDV` of 25.053, but the prediction interval is much wider. Now, let's compare with the intervals we get when using StatsModels.

# In[ ]:


reg_predictions = res.get_prediction(np.array([[1, 5], [1, 10], [1, 15]]))


# In[ ]:


pd.DataFrame(reg_predictions.conf_int(alpha = 0.05), columns = ["2.5%", "97.5%"])


# By default, the `conf_int()` method gives 95% confidence intervals for the predicted values, though by adjusting the `level` parameter we can change the confidence/prediction level, and `obs` parameter allows us to change from confidence intervals to prediction intervals.

# In[ ]:


# Produce 99% prediction intervals for the predicted values of CMEDV
pd.DataFrame(reg_predictions.conf_int(obs = True, alpha = 0.01), columns = ["0.5%", "99.5%"])


# As we can see, these results match up with what we computed by hand. Once again, in this context using StatsModels proves to be a lot more convenient! The functions we created for this section are pretty handy, but since we did them one at a time in a somewhat piecemeal fashion, they aren't as cohesive and nicely implemented as they could be, so lets combine them all into a class that extends the scikit-learn LinearRegression class. In addition, I'll rewrite my `predict_intervals()` function to properly work for the situation of multiple linear regression. To do so, I will follow the math from [these notes from the University of Minnesota](http://users.stat.umn.edu/~helwig/notes/mlr-Notes.pdf). 

# In[ ]:


class ExtendedLinearRegression(LinearRegression):
    
    def detailed_linear_regression(self, X, y):
        """
        Assume X is array-like with shape (num_samples, num_features)
        Assume y is array-like with shape (num_samples, num_targets)
        include_intercept is a boolean where True means X does not already have a column
        for the intercept
        Computes the least-squares regression model and returns a dictionary consisting of
        the fitted linear regression object; a series with the residual standard error,
        R^2 value, and the overall F-statistic with corresponding p-value; and a dataframe
        with columns for the parameters, and their corresponding standard errors,
        t-statistics, and p-values.
        """
        # Create a linear regression object and fit it using x and y
        self.training_X, self.training_y = X, y
        self.fit(X, y)
    
        # Store the parameters (regression intercept and coefficients) and predictions
        self.params = np.append(self.intercept_, self.coef_)
        predictions = self.predict(X)
    
        # Create matrix with shape (num_samples, num_features + 1)
        # Where the first column is all 1s and then there is one column for the values
        # of each feature/predictor
        X_mat = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
    
        # Compute residual sum of squares
        self.RSS = np.sum((y - predictions)**2)
    
        # Compute total sum of squares
        self.TSS = np.sum((np.mean(y) - y)**2)
    
        # Estimate the variance of the y-values
        obs_var = self.RSS/(X_mat.shape[0] - X_mat.shape[1])
    
        # Residual standard error is square root of variance of y-values
        self.RSE = obs_var**0.5
    
        # Variances of the parameter estimates are on the diagonal of the 
        # variance-covariance matrix of the parameter estimates
        self.var_beta_mat = obs_var*(np.linalg.inv(np.matmul(X_mat.T, X_mat)))
        self.var_beta = self.var_beta_mat.diagonal()
    
        # Standard error is square root of variance
        self.se_beta = np.sqrt(self.var_beta)
    
        # t-statistic for beta_i is beta_i/se_i, 
        # where se_i is the standard error for beta_i
        t_stats_beta = self.params/self.se_beta
    
        # Compute p-values for each parameter using a t-distribution with
        # (num_samples - 1) degrees of freedom
        beta_p_values = [2 * (1 - stats.t.cdf(np.abs(t_i), X_mat.shape[0] - 1)) for t_i in t_stats_beta]
    
        # Compute value of overall F-statistic, to measure how likely our
        # coefficient estimate are, assuming there is no relationship between
        # the predictors and the response
        self.F_overall = ((self.TSS - self.RSS)/(X_mat.shape[1] - 1))/(self.RSS/(X_mat.shape[0] - X_mat.shape[1]))
        self.F_p_value = stats.f.sf(self.F_overall, X_mat.shape[1] - 1, X_mat.shape[0] - X_mat.shape[1])
    
        # Construct dataframe for the overall model statistics:
        # RSE, R^2, F-statistic, p-value for F-statistic
        oa_model_stats = pd.Series({"Residual standard error": self.RSE, "R-squared": self.score(X, y), "F-statistic": self.F_overall, "F-test p-value": self.F_p_value})
    
        # Construct dataframe for parameter statistics:
        # coefficients, standard errors, t-statistic, p-values for t-statistics
        param_stats = pd.DataFrame({"Coefficient": self.params, "Standard Error": self.se_beta, "t-value": t_stats_beta, "Prob(>|t|)": beta_p_values})
        return {"model": self, "param_stats": param_stats, "oa_stats": oa_model_stats}
    
    def param_conf_int(self, level = 0.95):
        """
        Assume level, if given, is a float with 0 < level < 1
        Computes confidence intervals at the given confidence level for each parameter
        in the linear regression model relating the predictors X to the response y
        Returns a dataframe with the endpoints of the confidence interval for each parameter
        """
        conf_intervals = pd.DataFrame()
        # Degrees of freedom = num_samples - (num_features + 1)
        df = self.training_X.shape[0] - (self.training_X.shape[1] + 1)
        a, b = str(round((1 - level)*100/2, 2)) + "%", str(round((1 + level)*100/2, 2)) + "%"
        # Loop through each parameter
        for i in range(len(self.params)):
            coeff = self.params[i]
            std_err = self.se_beta[i]
            # alpha = level of confidence
            # df = degrees of freedom = num_samples - number of parameters
            # loc = center of t-interval = estimated coefficient value
            # scale = standard error in coefficient estimate
            conf_intervals = conf_intervals.append(pd.DataFrame([stats.t.interval(level, df, loc = coeff, scale = std_err)], columns = [a, b]), ignore_index = True)
        return conf_intervals
    
    def predict_intervals(self, X_pred, level = 0.95, kind = "confidence"):
        """
        Assume X_pred is array-like with shape (num_predictions, num_features) with x-values for which we want predictions
        Assume level, if given, is a float with 0 < level < 1
        Assume kind, if given is either the string "confidence" or "prediction" for the kind of interval
        Computes confidence intervals at the given confidence level for each parameter
        in the linear regression model relating the predictors X to the response y
        Returns a dataframe with the endpoints of the confidence interval for each parameter
        """
        # Store predictions for easy access
        predictions = self.predict(X_pred)
        intervals = pd.DataFrame()
        # Degrees of freedom = num_samples - (num_features + 1)
        df = self.training_X.shape[0] - (self.training_X.shape[1] + 1)
        a, b = str(round((1 - level)*100/2, 2)) + "%", str(round((1 + level)*100/2, 2)) + "%"
        # Loop through each x-value being used for prediction
        for i in range(len(predictions)):
            prediction = predictions[i]
            # Need to append the leading 1 since our matrix of regression parameter
            # Estimates has first row the estimate for the constant
            x_star = np.append(np.ones(1), X_pred[i])
            conf_error = np.matmul(np.matmul(x_star.T, self.var_beta_mat), x_star)**0.5
            predict_error = (self.RSE**2 + conf_error**2)**0.5
            # alpha = level of confidence
            # df = degrees of freedom = num_samples - number of parameters
            # loc = center of t-interval = predicted value from linear regression model
            # scale = standard error in predicted value estimate
            if (kind == "confidence"):
                lower, upper = stats.t.interval(level, df, loc = prediction, scale = conf_error)
                intervals = intervals.append(pd.Series({"prediction": prediction[0], a: lower[0], b: upper[0]}), ignore_index = True) 
            elif(kind == "prediction"):
                lower, upper = stats.t.interval(level, df, loc = prediction, scale = predict_error)
                intervals = intervals.append(pd.Series({"prediction": prediction[0], a: lower[0], b: upper[0]}), ignore_index = True)
        return intervals


# Having a class like this makes the act of accessing the various regression statistics more like the way things are done with StatsModels. For example, we can go back and redo some of the computations we did in a way that extends better to the further examples later on in this lab.

# In[ ]:


extended_reg = ExtendedLinearRegression()
detailed_regression_stats = extended_reg.detailed_linear_regression(X, y)
np.round(detailed_regression_stats["param_stats"], 4)


# In[ ]:


np.round(detailed_regression_stats["oa_stats"], 4)


# In[ ]:


extended_reg.predict_intervals(np.array([5, 10, 15]).reshape((-1, 1)), level = 0.99, kind = "prediction")


# There are a few different options for plotting a least squares regression line alongside the scatterplot of `CMEDV` versus `LSTAT`. The first, which would be best for exploring whether or not least squares regression would be appropriate, as well as explore some simple variable transforms beyond linear regression (e.g. using polynomial regression of some degree bigger than 1), would be to use Seaborn's `lmplot()` or `regplot()` functions. The `lmplot()` function can be especially handy if we want to quickly visualize regression models that reflect different facets of the data. One example given in the [tutorial for visualizing linear relationships in Seaborn](http://seaborn.pydata.org/tutorial/regression.html) is exploring the difference between the tipping habits smokers and non-smokers as it relates to the total bill. Note that by default, Seaborn will include translucent bands around the regression line to indicate a 95% confidence interval for the regression estimate. This behavior can be altered by setting the `ci` parameter with an integer between 0 and 100 for the confidence level, or be turned off by setting `ci = None`.

# In[ ]:


# Plot scatterplot with regression line and default 95% confidence interval for regression estimate
# Set the marker transparancy to 0.25 in order to more clearly see the regression line
# Make regression line orange so it is more visible
sns.regplot(x = "LSTAT", y = "CMEDV", data = boston, scatter_kws = {"alpha":0.25}, line_kws = {"color":"orange"})


# In[ ]:


# Plot scatterplot with regression line and 99% confidence interval for regression estimate
# Set the marker transparancy to 0.25 in order to more clearly see the regression line
# Make regression line orange so it is more visible
sns.regplot(x = "LSTAT", y = "CMEDV", data = boston, ci = 99, scatter_kws = {"alpha":0.25}, line_kws = {"color":"orange"})


# Notice that we can alter some options for how the points of the scatter plot and the regression line are displayed by using the `scatter_kws` and `line_kws` arguments, respectively. For further information about the options that can be tweaked, refer to the Matplotlib documentation for [plot](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html) for the options for `line_kws` and for [scatter](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html) for `scatter_kws`.

# We can also quickly plot the residuals with Seaborn's `residplot()` function.

# In[ ]:


# Set the marker transparency to 0.25 in order to improve visibility
sns.residplot(x = "LSTAT", y = "CMEDV", data = boston, scatter_kws = {"alpha":0.25})


# While plotting with Seaborn is pretty convenient, it is limited in the types of regression models it can produce and the transformations we can make to the predictor variable. In addition, Seaborn can only produce one type of residual plot. If we want to produce other plots, we'll need to do things a little more directly using Matplotlib.pyplot. First, let's demonstrate how to produce a scatter plot of `LSTAT` versus `CMEDV` with the least squares regression line on it. Here we'll be using the object-oriented syntax for using Matplotlib.pyplot.

# In[ ]:


# Generate a range of x-values to feed into the regression model for producing the
# line that gets passed to the plot function
x = np.linspace(0, 40, num = 100).reshape(-1, 1)
predictions = reg.predict(x)
fig = plt.figure()
ax = plt.axes()
# Plot the regression line in orange with a line width of 3 to increase visibility
ax.plot(x, predictions, color = "orange", linewidth = 3)
# Plot the scatter plot using an alpha value of 0.25 for the markers to reduce clutter
ax.scatter(boston["LSTAT"], boston["CMEDV"], alpha = 0.25)
# Give the scatterplot some labels that are more descriptive
ax.set(xlabel = "LSTAT", ylabel = "CMEDV", xlim = (0, 40))


# From the scatter plot, we can see that there is some evidence that the relationship between `lstat` and `medv` is non-linear. We'll take a look at this a little later in the lab.

# While Seaborn makes it really convenient to produce the residuals plot, it isn't quite as convenient to generate the other diagnostic plots which R can generate when applyint the `plot()` function to directly to the output from `lm()`. This is especially the case if we are using just scikit-learn, as we would have to calculate quantities such as the residuals or leverage values a little more directly. Things are a little easier when using StatsModels, as the `OLSResults` class has a number of built-in functions to calculate these quantities. To start off, let's compare the process for making a residuals plot by hand using scikit-learn versus StatsModels.

# In[ ]:


# Generating residual plot by hand using scikit-learn
# Compute predicted values
predicted_cmedv = reg.predict(X)
# Compute residuals
residuals = y - predicted_cmedv
fig = plt.figure()
ax = plt.axes()
# Plot residuals versus fitted values
ax.scatter(predicted_cmedv, residuals, alpha = 0.25)
# Plot orange dashed horizontal line y = 0
ax.axhline(y = 0, color = "orange", linestyle = "--")
# Give the plot some descriptive axis labels
ax.set(xlabel = "Fitted value of CMEDV", ylabel = "Residual value")


# In[ ]:


# Generating residual plot by hand using StatsModels
# Compute predicted values
predicted_cmedv = res.predict()
# Compute residuals
residuals = res.resid
fig = plt.figure()
ax = plt.axes()
# Plot residuals versus fitted values
ax.scatter(predicted_cmedv, residuals, alpha = 0.25)
# Plot orange dashed horizontal line y = 0
ax.axhline(y = 0, color = "orange", linestyle = "--")
# Give the plot some descriptive axis labels
ax.set(xlabel = "Fitted value of CMEDV", ylabel = "Residual value")


# For now we won't worry about producing the Q-Q plot or Scale-Location plot in Python, but for future reference one option is to use some of the [built-in plotting methods in StatsModels](https://www.statsmodels.org/stable/graphics.html) for this if we don't want to code things by hand. We will, however, still go over how to compute leverage statistics and (internally) studentized residuals and then plot them. We will refer to the [Wikipedia page on studentized residuals](https://en.wikipedia.org/wiki/Studentized_residual) for the underlying math.

# In[ ]:


# Appened leading column of ones to the matrix of predictors
design_mat = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
# Compute hat matrix
hat_mat = design_mat @ np.linalg.inv(design_mat.T @ design_mat) @ design_mat.T
# Leverage values are the diagonal of the hat matrix
leverage_vals = hat_mat.diagonal()
residuals = (y - reg.predict(X)).flatten()
residual_standard_error = (np.sum(residuals**2) / (design_mat.shape[0] - design_mat.shape[1]))**0.5
# Compute studentized residuals
studentized_residuals = residuals/(residual_standard_error*(1 - leverage_vals)**0.5)
fig = plt.figure()
ax = plt.axes()
# Plot studentized residuals versus fitted values
ax.scatter(predicted_cmedv, studentized_residuals, alpha = 0.25)
# Plot orange dashed horizontal line y = 0
ax.axhline(y = 0, color = "orange", linestyle = "--")
# Give the plot some descriptive axis labels
ax.set(xlabel = "Fitted value of CMEDV", ylabel = "Studentized residual value")


# In[ ]:


fig = plt.figure()
ax = plt.axes()
# Plot leverage values for each observation
ax.scatter(np.arange(design_mat.shape[0]), leverage_vals, alpha = 0.25)
# Plot orange dashed horizontal line y = (p + 1)/n, the average leverage for all observations
ax.axhline(y = design_mat.shape[1]/design_mat.shape[0], color = "orange", linestyle = "--")
# Give the plot some descriptive axis labels
ax.set(xlabel = "Index", ylabel = "Leverage value", ylim = (0, leverage_vals.max()*1.1))


# Since we stored the leverage values in a NumPy array, we can use the `argmax()` function to find the index of the maximum leverage value. In other words, we can use it in this case to find out which observation has the largest leverage value.

# In[ ]:


leverage_vals.argmax()


# Recall that NumPy arrays are zero-indexed, so this tells us that the 375th observation has the largest leverage value.

# If we want to use StatsModels to compute leverage values and studentized residuals, we use the `OLSResults.get_influence()` to generate an [OLSInfluence](http://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.OLSInfluence.html) object. From there, we can use the `OLSInfluence.hat_matrix_diagonal()` function to compute leverage values and `OLSInfluence.resid_studentized()` to compute (internally) studentized residuals.

# ## Multiple Linear Regression

# To perform multiple linear regression using least squares, we again use the LinearRegression class from scikit-learn (or our ExtendedLinearRegression class, which extends the scikit-learn LinearRegression class), or we use the OLS class from StatsModels. In either case, the only change in syntax is in selecting the columns we use for regression. This will give a linear regression function of the form $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n$. To start off with, we perform multiple linear regression using `LSTAT` and `AGE` to predict `CMEDV`.

# In[ ]:


# Create an ExtendedLinearRegression object
reg = ExtendedLinearRegression()
# Need to extract the columns we are using and then reshape them so that
# X has the shape (n_samples, n_features),
# Y has the shape (n_samples, n_targets)
# When using -1 as one of the reshape arguments, NumPy will infer the value
# from the length of the array and the values for the other dimensions
X = boston.loc[:, ["LSTAT", "AGE"]].values
y = boston["CMEDV"].values.reshape(-1, 1)
# Fit linear regression model using X = LSTAT, y = CMEDV
detailed_regression_stats = reg.detailed_linear_regression(X, y)


# In[ ]:


np.round(detailed_regression_stats["param_stats"], 4)


# In[ ]:


np.round(detailed_regression_stats["oa_stats"], 4)


# In[ ]:


# Use the terms exog (exogenous) and endog (endogenous) for X and y,
# respectively to match the language used in the StatsModels documentation
# Need to manually add a column for the intercept, as StatsModels does not
# include it by default when performing ordinary least-squares regression
exog = sm.add_constant(boston.loc[:, ["LSTAT", "AGE"]])
endog = boston["CMEDV"]

# Generate the model
mod = sm.OLS(endog, exog)

# Fit the model
res = mod.fit()

#Print out model summary
print(res.summary())


# A quick way to peform a regression using all of the predictors in a data set is to use the `drop()` function in Pandas to drop the column containing the response variable.

# In[ ]:


# Create an ExtendedLinearRegression object
reg = ExtendedLinearRegression()
# Need to extract the columns we are using and then reshape them so that
# X has the shape (n_samples, n_features),
# Y has the shape (n_samples, n_targets)
# When using -1 as one of the reshape arguments, NumPy will infer the value
# from the length of the array and the values for the other dimensions
X = boston.drop(columns = ["CMEDV"]).values
y = boston["CMEDV"].values.reshape(-1, 1)
# Fit linear regression model using X = LSTAT, y = CMEDV
detailed_regression_stats = reg.detailed_linear_regression(X, y)


# In[ ]:


np.round(detailed_regression_stats["param_stats"], 4)


# In[ ]:


np.round(detailed_regression_stats["oa_stats"], 4)


# In[ ]:


# Use the terms exog (exogenous) and endog (endogenous) for X and y,
# respectively to match the language used in the StatsModels documentation
# Need to manually add a column for the intercept, as StatsModels does not
# include it by default when performing ordinary least-squares regression
exog = sm.add_constant(boston.drop(columns = ["CMEDV"]))
endog = boston["CMEDV"]

# Generate the model
mod = sm.OLS(endog, exog)

# Fit the model
res = mod.fit()

#Print out model summary
print(res.summary())


# If we wish to access individual components of a scikit-learn LinearRegression object or StatsModels OLSResults object, we can access they by name. The command `dir` shows what class variables and functions are available to access. For more details, we can also look at the documentation for [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) or [StatsModels](http://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.html).

# Some fancier statistics require coding additional functions by hand. For example, while StatsModels does have a function to [compute variance inflation factors](http://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html), its current implementation can only do so for one predictor variable at a time. Thus, we will write our own function to compute the variance inflation factors for all of our predictors at once.

# Recall that the variance inflation factor for a predictor $X_j$ is the ratio of the variance of its coefficient $\hat{\beta}_j$ in the full model using divided by the variance of $\hat{\beta}_j$ in the model just using $X_j$. Another way to compute the variance inflation factor for each variable is to use the formula
# 
# \begin{equation}
# \text{VIF}(\hat{\beta}_j) = 
# \frac{1}{1 - R^2_{X_j | X_{-j}}},
# \end{equation}
# 
# where $R^2_{X_j | X_{-j}}$ is the $R^2$ value from a regression using $X_j$ as the response and the remaining variables as the predictors.

# In[ ]:


def vif(predictors):
    """
    Assumes predictors is a Pandas dataframe with at least two columns
    Returns a Pandas series containing the variance inflation factor for each column variable
    """
    columns = predictors.columns
    vif_series = pd.Series()
    for col_name in columns:
        X = predictors.drop(columns = [col_name]).values
        y = predictors[col_name].values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        r_sq = reg.score(X, y)
        vif_series[col_name] = 1/(1 - r_sq)
    return vif_series


# In[ ]:


vif(boston.drop(columns = ["CMEDV"]))


# Variance inflation factors close to the minimum value of 1 indicate a small amount of collinearity, while values exceeding 5 or 10 are generally considered to indicate a problematic amount of collinearity. For this data, most of the variance inflation factors are low to moderate, though `RAD` and `TAX` stand out as having values on the high end.

# In the multiple linear regression output, we see that `AGE` has a high p-value of 0.958, so we might want to exclude it. If we wish to perform regression with a small number of excluded variables the simplest thing to do is to re-run the regression after dropping the variable(s) we wish to exclude.

# In[ ]:


# Create an ExtendedLinearRegression object
reg = ExtendedLinearRegression()
# Need to extract the columns we are using and then reshape them so that
# X has the shape (n_samples, n_features),
# Y has the shape (n_samples, n_targets)
# When using -1 as one of the reshape arguments, NumPy will infer the value
# from the length of the array and the values for the other dimensions
X = boston.drop(columns = ["CMEDV", "AGE"]).values
y = boston["CMEDV"].values.reshape(-1, 1)
# Fit linear regression model using X = LSTAT, y = CMEDV
detailed_regression_stats = reg.detailed_linear_regression(X, y)


# In[ ]:


np.round(detailed_regression_stats["param_stats"], 4)


# In[ ]:


np.round(detailed_regression_stats["oa_stats"], 4)


# In[ ]:


# Use the terms exog (exogenous) and endog (endogenous) for X and y,
# respectively to match the language used in the StatsModels documentation
# Need to manually add a column for the intercept, as StatsModels does not
# include it by default when performing ordinary least-squares regression
exog = sm.add_constant(boston.drop(columns = ["CMEDV", "AGE"]))
endog = boston["CMEDV"]

# Generate the model
mod = sm.OLS(endog, exog)

# Fit the model
res = mod.fit()

#Print out model summary
print(res.summary())


# ## Interaction Terms

# If we wish to include interaction terms in a linear regression model [using StatsModels](http://www.statsmodels.org/stable/example_formulas.html), we can use the syntax `statsmodels.formula.api.ols(formula = "y~x1:x2", data = df)`, which uses [patsy](https://patsy.readthedocs.io/en/latest/) to implement `R`-style formula syntax which tells StatsModels to include an interaction term between `x1` and `x2`. This would give us the regression function $y = \beta_0 + \beta_{12}x_1x_2$. If we want to include the individual variables themselves, as well as the interaction term between them, we can use `x1*x2`, which is a shorthand for `x1 + x2 + x1:x2`. Calling `ols(formula = "y~x1*x2", data = df)` would give us the regression function $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \beta_{12}x_1x_2$. 

# In[ ]:


# Using patsy to include interaction terms via R-style formulas

# Generate a linear regression model with LSTAT, AGE, and an interaction term between
# them to predict CMEDV
mod = smf.ols(formula = "CMEDV ~ LSTAT*AGE", data = boston)
res = mod.fit()
print(res.summary())


# Alternatively, if for some reason we do not wish to use `R`-style formulas when using StatsModels, we can create the columns for the interaction terms by hand.

# In[ ]:


# Creating a column forthe interaction terms by hand

# Use the terms exog (exogenous) and endog (endogenous) for X and y,
# respectively to match the language used in the StatsModels documentation
# Need to manually add a column for the intercept, as StatsModels does not
# include it by default when performing ordinary least-squares regression
exog = sm.add_constant(boston.loc[:, ["LSTAT", "AGE"]].assign(LSTAT_AGE = boston["LSTAT"] * boston["AGE"]))
endog = boston["CMEDV"]

# Generate the model
mod = sm.OLS(endog, exog)

# Fit the model
res = mod.fit()

#Print out model summary
print(res.summary())


# When using scikit-learn, while creating the columns for interaction terms by hand is still an option, a better way is to use [PolynomialFeatures transformer](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions) as part of a model pipeline.

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create a pipeline which first transforms the data to include up to second degree terms
# Setting interaction_only to True indicates that we only want interaction terms
# and excludes higher powers of the individual features
# Since the transformed data includes the 0-degree (i.e. constant = 1) feature
# an intercept is not necessary in the linear regression
model = Pipeline([("poly", PolynomialFeatures(degree = 2, interaction_only = True)),
                 ("linear", LinearRegression(fit_intercept = False))])
X = boston.loc[:, ["LSTAT", "AGE"]].values
y = boston["CMEDV"].values.reshape((-1, 1))
model = model.fit(X, y)
print(model.named_steps["linear"].coef_)


# Using a pipeline like this doesn't play nicely with the ExtendedLinearRegression class I created, so if we want to use that for access to more of the detailed model statistics, we'll need to do things a little more by hand. However, in a production setting, pipeline usage is very useful for convenience and encapsulation, joint parameter selection, and safety.

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
# Create an ExtendedLinearRegression object
reg = ExtendedLinearRegression()
# Create a 2nd degree Polynomial features transformer which only includes interaction terms
poly = PolynomialFeatures(degree = 2, interaction_only = True)
# Need to extract the columns we are using and then reshape them so that
# X has the shape (n_samples, n_features),
# Y has the shape (n_samples, n_targets)
# When using -1 as one of the reshape arguments, NumPy will infer the value
# from the length of the array and the values for the other dimensions
# Transform X using the PolynomialFeatures transformer
# Exclude the intercept column so it plays nicely with how I've written the ExtendedLinearRegression class
X = poly.fit_transform(boston.loc[:,["LSTAT", "AGE"]].values)[:, 1:]
y = boston["CMEDV"].values.reshape(-1, 1)
# Fit linear regression model using X = LSTAT, y = CMEDV
detailed_regression_stats = reg.detailed_linear_regression(X, y)


# In[ ]:


np.round(detailed_regression_stats["param_stats"], 4)


# In[ ]:


np.round(detailed_regression_stats["oa_stats"], 4)


# ## Non-linear transformations of the predictors

# We can also use additional non-linear transformations of our variables when generating least-squares regression models. For example, if we wish to create a predictor $X^2$ from the predictor $X$, we have a few different options depending on which packages we are using. Some of the options include
# 
# - Creating a column for $X^2$ by hand, which works for both scikit-learn and StatsModels
# - Using `np.square(X)` or `X**2` as part of an `R`-style formula in StatsModels
# - Using the PolynomialFeatures transformer with scikit-learn
# 
# Let's use each of these three strategies to perform a regression of `CMEDV` onto `LSTAT` and `LSTAT**2`.

# In[ ]:


# Creating a column for LSTAT**2 by hand

# Use the terms exog (exogenous) and endog (endogenous) for X and y,
# respectively to match the language used in the StatsModels documentation
# Need to manually add a column for the intercept, as StatsModels does not
# include it by default when performing ordinary least-squares regression
exog = sm.add_constant(boston.loc[:, ["LSTAT"]].assign(LSTAT_sq = np.square(boston["LSTAT"])))
endog = boston["CMEDV"]

# Generate the model
mod_square = sm.OLS(endog, exog)

# Fit the model
res_square = mod_square.fit()
#Print out model summary
print(res_square.summary())


# In[ ]:


# Using patsy to include the term LSTAT**2 via R-style formulas

mod_square = smf.ols(formula = "CMEDV ~ LSTAT + np.square(LSTAT)", data = boston)
res_square = mod_square.fit()
print(res_square.summary())


# In[ ]:


# Using PolynomialFeatures transformer with scikit-learn to include the term LSTAT**2

from sklearn.preprocessing import PolynomialFeatures
# Create an ExtendedLinearRegression object
reg = ExtendedLinearRegression()
# Create a 2nd degree Polynomial features transformer
poly = PolynomialFeatures(degree = 2)
# Need to extract the columns we are using and then reshape them so that
# X has the shape (n_samples, n_features),
# Y has the shape (n_samples, n_targets)
# When using -1 as one of the reshape arguments, NumPy will infer the value
# from the length of the array and the values for the other dimensions
# Transform X using the PolynomialFeatures transformer
# Exclude the intercept column so it plays nicely with how I've written the ExtendedLinearRegression class
X = poly.fit_transform(boston.loc[:,["LSTAT"]].values)[:, 1:]
y = boston["CMEDV"].values.reshape(-1, 1)
# Fit linear regression model using X = LSTAT, y = CMEDV
detailed_regression_stats = reg.detailed_linear_regression(X, y)


# In[ ]:


np.round(detailed_regression_stats["param_stats"], 4)


# In[ ]:


np.round(detailed_regression_stats["oa_stats"], 4)


# Since the quadratic term `LSTAT**2` has an extremely small p-value, we have evidence to believe that its inclusion leads to an improved model. To further quantify the extent to which the quadratic fit is superior to the linear fit, we can use the `anova_lm()` function from StatsModels. More details can be found in [the documentation](http://www.statsmodels.org/stable/anova.html).

# In[ ]:


mod_square = smf.ols(formula = "CMEDV ~ LSTAT + np.square(LSTAT)", data = boston).fit()
mod = smf.ols(formula = "CMEDV ~ LSTAT", data = boston).fit()
anova_table = sm.stats.anova_lm(mod, mod_square)
anova_table


# The `anova_lm()` function performs a hypothesis test comparing the two models. The null hypothesis is that both models fit the data equally well, while the alternative hypothesis is that the second model (in our case the model including the quadratic term) performs better. Here, we have an F-statistic of about 138 and an p-value that is essentially zero, which provides strong evidence that the model containing the predictors `LSTAT` and `LSTAT**2` is a better fit than the one containing only `LSTAT`. This further confirms our initial suspicions based on the non-linearity we saw in the scatter plot for `CMEDV` and `LSTAT`.

# As discussed earlier, there are a few different ways of generating residual plots for us to have visual evidence that a model containing the predictors `LSTAT` and `LSTAT**2` is a better fit than the one containing only `LSTAT`. First, we can set the `order` parameter in Seaborn's `residplot()` function to generate residual plots for higher-order polynomial regression. For completeness, we first use Seaborn to plot the quadratic fit on a scatterplot of the data as well.

# In[ ]:


# Plot scatterplot with regression line and default 95% confidence interval for regression estimate
# Set the marker transparancy to 0.25 in order to more clearly see the regression line
# Make regression line orange so it is more visible
sns.regplot(x = "LSTAT", y = "CMEDV", data = boston, order = 2, scatter_kws = {"alpha":0.25}, line_kws = {"color":"orange"})


# In[ ]:


# Set the marker transparency to 0.25 in order to improve visibility
sns.residplot(x = "LSTAT", y = "CMEDV", data = boston, order = 2, scatter_kws = {"alpha":0.25})


# Next, we use scikit-learn and Matplotlib directly to produce the scatterplot with quadratic fit and residuals plot.

# In[ ]:


poly = PolynomialFeatures(degree = 2)

# Generate a range of x-values to feed into the regression model for producing the
# line that gets passed to the plot function
x = np.linspace(0, 40, num = 100).reshape(-1, 1)
# Need to transform the x array to properly feed into the predict function
transformed = poly.fit_transform(x)[:, 1:]
predictions = reg.predict(transformed)
fig, axes = plt.subplots(nrows = 2, figsize = (10, 10), gridspec_kw = {})
# Plot the regression line in orange with a line width of 3 to increase visibility
axes[0].plot(x, predictions, color = "orange", linewidth = 3)
# Plot the scatter plot using an alpha value of 0.25 for the markers to reduce clutter
axes[0].scatter(boston["LSTAT"], boston["CMEDV"], alpha = 0.25)
# Give the scatterplot some labels that are more descriptive
axes[0].set(xlabel = "LSTAT", ylabel = "CMEDV", xlim = (0, 40))

# Generating residual plot by hand using scikit-learn
# Compute predicted values
predicted_cmedv = reg.predict(X)
# Compute residuals
residuals = y - predicted_cmedv
# Plot residuals versus fitted values
axes[1].scatter(predicted_cmedv, residuals, alpha = 0.25)
# Plot orange dashed horizontal line y = 0
axes[1].axhline(y = 0, color = "orange", linestyle = "--")
# Give the plot some descriptive axis labels
axes[1].set(xlabel = "Fitted value of CMEDV", ylabel = "Residual value")


# In addition, as we can see above, there isn't a discernible pattern in the residuals for the model which includes the `LSTAT**2` term.

# If we wish to include higher-order predictors of the form $X^k$, the strategy of adding those columns by hand becomes less convenient. It is still doable by appropriately adjusting the how we call the `assign()` function to add columns to our dataframe. Let's demonstrate this by producing a fifth-order polynomial fit.

# In[ ]:


# Creating a fifth-order polynomial fit by hand

# Use the terms exog (exogenous) and endog (endogenous) for X and y,
# respectively to match the language used in the StatsModels documentation
# Need to manually add a column for the intercept, as StatsModels does not
# include it by default when performing ordinary least-squares regression
exog = sm.add_constant(boston.loc[:, ["LSTAT"]].assign(LSTAT_2 = boston["LSTAT"]**2,
                                                      LSTAT_3 = boston["LSTAT"]**3,
                                                      LSTAT_4 = boston["LSTAT"]**4,
                                                      LSTAT_5 = boston["LSTAT"]**5))
endog = boston["CMEDV"]

# Generate the model
mod_quint = sm.OLS(endog, exog)

# Fit the model
res_quint = mod_quint.fit()

#Print out model summary
print(res_quint.summary())


# It is much more convenient to use the PolynomialFeatures transformer from scikit-learn.

# In[ ]:


# Using PolynomialFeatures transformer with scikit-learn to create fifth-order polynomial fit

from sklearn.preprocessing import PolynomialFeatures
# Create an ExtendedLinearRegression object
reg = ExtendedLinearRegression()
# Create a 2nd degree Polynomial features transformer
poly = PolynomialFeatures(degree = 5)
# Need to extract the columns we are using and then reshape them so that
# X has the shape (n_samples, n_features),
# Y has the shape (n_samples, n_targets)
# When using -1 as one of the reshape arguments, NumPy will infer the value
# from the length of the array and the values for the other dimensions
# Transform X using the PolynomialFeatures transformer
# Exclude the intercept column so it plays nicely with how I've written the ExtendedLinearRegression class
X = poly.fit_transform(boston.loc[:,["LSTAT"]].values)[:, 1:]
y = boston["CMEDV"].values.reshape(-1, 1)
# Fit linear regression model using X = LSTAT, y = CMEDV
detailed_regression_stats = reg.detailed_linear_regression(X, y)


# In[ ]:


np.round(detailed_regression_stats["param_stats"], 6)


# In[ ]:


np.round(detailed_regression_stats["oa_stats"], 6)


# In[ ]:


# Using PolynomialFeatures transformer to create fifth-order polynomial fit with StatsModels

poly = PolynomialFeatures(degree = 5)
# Use the terms exog (exogenous) and endog (endogenous) for X and y,
# respectively to match the language used in the StatsModels documentation
# No need to include a column for intercept in this case, since it is
# included when applying the fit_transform function
# Make sure to reshape the LSTAT column values to play nicely with fit_transform
exog = poly.fit_transform(boston["LSTAT"].values.reshape((-1, 1)))
endog = boston["CMEDV"]

# Generate the model
mod_quint = sm.OLS(endog, exog)

# Fit the model
res_quint = mod_quint.fit()

#Print out model summary
print(res_quint.summary())


# This summary suggests that including additional polynomial terms further improves the model fit. Remember that we *do not* care about the coefficient values when evaluating model fit. We care about statistical indicators such as the $R^2$ value for each model. Further investigation of the data indicates that polynomial terms beyond fifth order do not have significant p-values in a regression fit. Before moving on to other non-linear transformations of the predictors, note that we can use Python's string manipulation capabilities as another way of more conveniently performing higher order polynomial fits when using `R`-style formulas in StatsModels.

# In[ ]:


# Using Python string manipulation alongside patsy to create fifth-order polynomial fit

# Create string for the higher-order polynomial terms
poly_terms = "+".join(["I(LSTAT**{0})".format(i) for i in range(2, 6)])
# Join this string with the rest of the formula I wish to use
my_formula = "CMEDV ~ LSTAT + " + poly_terms
mod_quint = smf.ols(formula = my_formula, data = boston)
res_quint = mod_quint.fit()
print(res_quint.summary())


# One last thing to note is that we can use other non-linear transformations of the predictors. For example, we can do a logarithmic model where the predictor is $\ln(X)$, and $X$ is `RM`, the average number of rooms.

# In[ ]:


# Creating a column for log(RM) by hand, using StatsModels
# Here log refers to the natural logarithm

# Use the terms exog (exogenous) and endog (endogenous) for X and y,
# respectively to match the language used in the StatsModels documentation
# Need to manually add a column for the intercept, as StatsModels does not
# include it by default when performing ordinary least-squares regression
exog = sm.add_constant(np.log(boston["RM"].rename("log(RM)")))
endog = boston["CMEDV"]

# Generate the model
mod_log = sm.OLS(endog, exog)

# Fit the model
res_log = mod_log.fit()
#Print out model summary
print(res_log.summary())


# In[ ]:


# Using patsy to include the term log(RM) via R-style formulas

mod_log = smf.ols(formula = "CMEDV ~ np.log(RM)", data = boston)
res_log = mod_log.fit()
print(res_log.summary())


# In[ ]:


# Creating a column for log(RM) by hand, using scikit-learn

# Create an ExtendedLinearRegression object
reg = ExtendedLinearRegression()
# Need to extract the columns we are using and then reshape them so that
# X has the shape (n_samples, n_features),
# Y has the shape (n_samples, n_targets)
# When using -1 as one of the reshape arguments, NumPy will infer the value
# from the length of the array and the values for the other dimensions
# Transform the RM column using np.log
X = np.log(boston["RM"]).values.reshape(-1, 1)
y = boston["CMEDV"].values.reshape(-1, 1)
# Fit linear regression model using X = LSTAT, y = CMEDV
detailed_regression_stats = reg.detailed_linear_regression(X, y)


# In[ ]:


np.round(detailed_regression_stats["param_stats"], 4)


# In[ ]:


np.round(detailed_regression_stats["oa_stats"], 4)


# ## Qualitative Predictors

# Next we'll examine the `Carseats` data set from the `ISLR` library and attempt to predict `Sales` (child car seat sales) in 400 locations based on a number of predictors. To start with, we'll do our usual overview inspection of the first few rows of the data set and then check for missing values.

# In[ ]:


# Load the Carseats data set
# Use the unnamed zeroth column as the index
carseats_filepath = "../input/islr-carseats/Carseats.csv"
carseats = pd.read_csv(carseats_filepath, index_col = ["Unnamed: 0"])


# In[ ]:


carseats.head()


# In[ ]:


carseats.isnull().any()


# Looking at the data, we see that there are a number of qualitative predictors. For example, the `ShelveLoc` predictor is an indicator of the quality of the shelving location, or the space within a store in which the car seat is displayed, at each location. This predictor takes on three possible values: *Bad*, *Medium*, and *Good*. If we include a qualitative variable in `statsmodels.formula.api.ols()`, `Patsy` will automatically generate dummy variables for the possible values of that variable. For more details on how this is done, we can refer to a tutorial on [contrast coding systems for categorical variables with Patsy](http://www.statsmodels.org/stable/contrasts.html) on the StatsModels documentation site, as well as the official [Patsy documentation for coding categorical data](https://patsy.readthedocs.io/en/latest/categorical-coding.html). Now we'll create a multiple regression model that also includes some interaction terms. Note that as far as I can tell (as of November 2019), Patsy currently does not have an equivalent to `.` to include all columns in the way `R` can do this. However, following [this StackExchange post](https://stackoverflow.com/a/22388673), we can use Python string manipulation for this purpose.

# In[ ]:


# Using patsy to include the perform multiple regression using the Carseats data
# Include interaction terms for Income:Advertising and Price:Age
# Note that there are some qualitative predictors

# Create string for the names of all of the columns
all_columns = "+".join(carseats.columns.drop("Sales"))
# Join this string with the rest of the formula I wish to use
my_formula = "Sales ~" + all_columns + "+ Income:Advertising + Price:Age"
mod = smf.ols(formula = my_formula, data = carseats)
res = mod.fit()
print(res.summary())


# By default, Patsy uses [treatment (dummy) coding](https://patsy.readthedocs.io/en/latest/API-reference.html#patsy.Treatment), though there are lots of other contrast coding options we can explore in the linked documentation. In addition, by default the reference is the first level (using alphabetical order), though we have the option to specify this explicitly if we wish. Here we see that Patsy created a dummy variable `ShelveLoc[T.Good]` which is equal to 1 if the shelving location is good and 0 otherwise, along with a dummy variable `ShelveLoc[T.Medium]` which is equal to 1 if the shelving location is medium and 0 otherwise. With this encoding, a bad shelving location corresponds to both dummy variables having a value of 0. The positive coefficients for `ShelveLoc[T.Good]` and `ShelveLoc[T.Medium]` in the regression output indicates that good or medium shelving locations contribute to higher sales compared to a bad location. The higher value of the coefficient for `ShelveLoc[T.Good]` indicates that a good shelving location has leads to a bigger increase in sales (over a bad location) than a medium shelving location.

# To perform this regression with scikit-learn, we'll use the [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) to code the categorical predictors and again use the PolynomialFeatures transformer to add the interaction terms. We can look at the preprocessing section of the user guide for more information on how to use scikit-learn for [encoding categorical data](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features).

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
# Create an ExtendedLinearRegression object
reg = ExtendedLinearRegression()
# Create a 2nd degree Polynomial features transformer which only includes interaction terms
poly = PolynomialFeatures(degree = 2, interaction_only = True)
# Create columns for interaction terms Income:Advertising and Price:Age
income_advert = pd.Series(poly.fit_transform(carseats.loc[:, ["Income", "Advertising"]])[:, -1], name = "Income:Advertising")
price_age = pd.Series(poly.fit_transform(carseats.loc[:, ["Price", "Age"]])[:, -1], name = "Price:Age")
# Encode categorical predictors using OneHotEncoder
# Set the categories and drop the first category when encoding to use reduced-rank coding
# This then replicates the default behavior of how Patsy and R do categorical encoding
enc = OneHotEncoder(categories = [["Bad", "Medium", "Good"], ["No", "Yes"], ["No", "Yes"]], drop = "first")
cat_pred = enc.fit_transform(carseats.loc[:, ["ShelveLoc", "Urban", "US"]]).toarray()
cat_pred = pd.DataFrame(cat_pred, columns = ["ShelveLocMedium", "ShelveLocGood", "UrbanYes", "USYes"])
quant_pred = carseats.loc[:, ["CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education"]].reset_index(drop = True)

# Combine all of the columns into a single dataframe of predictors
# Note that we needed to reset the index for quant_pred in order to have it align with the indices
# for the other columns when joining
# We could avoid this if we worked purely with the underlying NumPy arrays
X = cat_pred.join([quant_pred, income_advert, price_age])
y = carseats["Sales"].values.reshape(-1, 1)
detailed_regression_stats = reg.detailed_linear_regression(X, y)


# In[ ]:


np.round(detailed_regression_stats["param_stats"], 4)


# In[ ]:


np.round(detailed_regression_stats["oa_stats"], 4)


# In[ ]:


# List of columns to remember which column corresponds to each coefficient
X.columns


#!/usr/bin/env python
# coding: utf-8

# # An Exploration of L1 and L2 Regularization
# 
# Author: Bryson J. Banks (@bjbanks, GitHub)

# # What is Regularization?
# 
# [Regularization](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a) is a modified form of regression with the purpose to minimize the risk of overfitting, particularly when [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity) exists within the feature set of the data. High levels of multicollinearity wihtin the feature set leads to increased variance of the coefficient estimates in a typical linear regression model, resulting in estimates that can be very sensitive to minor changes in the model.
# 
# By constraining, shrinking, or "regularizing" the regression coefficient estimates toward zero, this technique discourages our model from taking a more complex or flexible fit, in favor of a more stable fit with less variance in the coefficient estimates. In terms of a typical linear regression model using ordinary least squares, this is done by modifying our typcial loss function ([Residual Sum of Squares, RSS](https://en.wikipedia.org/wiki/Residual_sum_of_squares)) by adding a penalty for higher magnitude coefficient values.
# 
# As with any model, there are tradeoffs to consider when using regularization. We must carefully balance bias vs. variance, by tuning the hyper parameter that scales the magnitude of the added regularization penalty. The more we "regularize" the data, the more we will reduce variance, but only at the expense of introducing more bias. In this notebook we will be focusing on linear regularization practices, specifically looking at the two most common linear regularization methods, Lasso (L1) and Ridge (L2), how they compare to ordinary least squares linear regression, and methods for tuning each of their inherent hyper parameters using cross validation.
# 
# For a more detailed and mathematical explanation of regularization, I found this article particularly informative and easy to follow: http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/

# # Getting started
# 
# To help build our understanding of these methods, we will be working with the [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) data set, and will first create a regular ordinary least squares linear regression model with no regularization. We will then attempt to improve upon this model by adding a regularization penalty by means of the Ridge and Lasso Regression models.
# 
# To start, we of course need to first import all our data and some needed libraries.

# In[78]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings

# ignore certain warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# set seaborn defaults
sns.set()

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png' #set 'png' here when working in notebook")
get_ipython().run_line_magic('matplotlib', 'inline')

# identify data sets
trainData = '../input/train.csv'
testData = '../input/test.csv'

# import data sets
train = pd.read_csv(trainData, header=0)
test = pd.read_csv(testData, header=0)

# combine all data (ignoring Id and SalePrice features)
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']))


# ## A quick look at the data
# 
# After importing both the training and testing data sets, let's first check to see that the data was imported properly, and get a feel for what the data looks like. In particular, notice the combination of both numerical and categorical features, as well as the missing values.

# In[79]:


# view training data
train.head()


# In[80]:


# view testing data
test.head()


# In[81]:


# view combined data
all_data.head()


# # Data Preprocessing
# 
# Our initial data preprocessing steps are adapted from [Alexandru](https://www.kaggle.com/apapiu/regularized-linear-models) with a few additional steps. We'll choose simplicity here over more complicated steps as our focus is on the models themselves, and not advanced preprocessing techniques. We'll do just enough to be able to use and get reliable results from our regression models. Our steps will include:
# 
# 1. Drop clear outliers
# 2. Convert numerical features to strings for features that should really be categorical (e.g. years, months, etc.)
# 3. Encode all categorical labels with value between 0 and n_classes-1
# 4. Normalize heavily skewed numerical features
# 5. Create dummy/indicator features from categorical features
# 6. Replace missing values with feature means
# 7. Set up matrices for regression models

# ### Outliers
# 
# We must be careful when choosing to drop outliers of the risk of losing valuable information, but we see here in the plot below 2 clear outliers toward the bottom right of the plot representing "bad" deals for sellers (low price for large area).

# In[82]:


rcParams['figure.figsize'] = (6.0, 6.0) # define size of figure
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
plt.show()


# As these two observations clearly don't align with the rest of the data, we choose here to drop these as we don't want these clearly "bad" deals introducing extra bias in our prediction models.

# In[83]:


train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index).reset_index(drop=True)

# reset combined data set with new training set
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']))


# ### Numerical to categorical conversions
# 
# `MSSubClass`, `OverallCond`, `YrSold`, and `MoSold` while numerical, are really categorical type features, thus we will convert them to strings so that we can encode them next.

# In[84]:


all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# ### Encode Categorical Labels
# 
# We'll now encode all categorical feature labels with values between 0 and n_classes-1.

# In[85]:


from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))


# ### Normalize

# By plotting the distribution of our target feature, we quickly notice that the distribution appears to be righlty skewed. (Note that we use the skew function from scipy.stats to determine the "skewness" of the feature.)

# In[86]:


from scipy.stats import skew

# plot histogram of "SalePrice"
rcParams['figure.figsize'] = (12.0, 6.0) # define size of figure
g = sns.distplot(train["SalePrice"], label="Skewness: %.2f"%(train["SalePrice"].skew()))
g = g.legend(loc="best")
plt.show()


# Typically, our regression models will perform best with normally distributed data. Thus for best results, let's attempt to normalize the feature with a log transform. (For rightly skewed data, a log transform has the effect of shifting the distribution to appear more "normal", while for leftly skewed data, a log transform will only make the distribution even more leftly skewed.)

# In[87]:


normalizedSalePrice = np.log1p(train["SalePrice"])

# plot histogram of log transformed "SalePrice"
rcParams['figure.figsize'] = (12.0, 6.0) # define size of figure
g = sns.distplot(normalizedSalePrice, label="Skewness: %.2f"%(normalizedSalePrice.skew()))
g = g.legend(loc="best")
plt.show()


# Cool, we see our log transform did surprisingly well, and had the intended effect - the new distribution looks much more "normal". Let's go ahead and apply this log transformation of "SalePrice" to our training data.

# In[88]:


# apply log transform to target
train["SalePrice"] = np.log1p(train["SalePrice"])


# As we'll see, several of the non-target numerical features are also heavily skewed, both rightly and leftly. For each of these, this time we'll choose to use a blanket "yeo-johnson" power transform to attempt to "normalize" each of them, since this tranform "normalizes" both righlty and leftly skewed data. (Here we consider all features with a "skewness" magnitude above 0.75 as "heavily" skewed.)

# In[89]:


# determine features that are heavily skewed
def get_skewed_features():
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) # computes "skewness"
    skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]
    return skewed_feats.index


# In[90]:


from sklearn.preprocessing import power_transform

# find heavily skewed numerical features
skewed_feats = get_skewed_features()
print("{} heavily skewed features.".format(len(skewed_feats)))

# apply power transform to all heavily skewed numeric features
all_data[skewed_feats] = power_transform(all_data[skewed_feats], method='yeo-johnson')
print("Applied power transform.")


# **Q. Why are we "normalizing" the numerical features?**
# 
# In general, standardized or normally distributed data is nice to have, and provides various benefits in different situations. All the specific benefits and situations goes beyond the scope of this notebook, but typically normalizing your data is a good idea in the absence of any other information against the case. In our situation where we plan to use regularization methods, the more extreme observation values in the highly skewed features create a bias that can cause different explanatory variables to be treated not so equally by the regularization penalty term. By normalizing these skewed distributions, it is believed the regularization penalty will then treat different explanatory variables on a more equal footing. Ideally, we want all observations and variables to be treated perfectly equally by our models.

# ### Indicator variables
# 
# Next, we need to create dummy/indicator variables for all of the categorical features so that they can be reasonably used in our regression models.

# In[91]:


# create dummy variables
all_data = pd.get_dummies(all_data)
all_data.shape # we now have 219 features columns compared to original 79


# ### Missing values
# 
# Let's now check for missing values, and replace them with the mean of the corresponding feature.

# In[92]:


# check for any missing values
all_data.isnull().any().any()


# In[93]:


# replace NA's with the mean of the feature
all_data = all_data.fillna(all_data.mean())

# check again for any missing values
all_data.isnull().any().any()


# ### Model matrices
# 
# Lastly, let's setup the matrices needed for sklearn, and then we can begin with the regular ordinary least squares linear regression model. This wraps up our preprocessing steps.

# In[94]:


# create matrices for sklearn
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


# # Linear Regression
# 
# The typcial Ordinary Least Squares Linear Regression model aims to optimize the residual sum of squares (RSS), which is defined as:
# 
# ![Residual Sum of Squares](https://wikimedia.org/api/rest_v1/media/math/render/svg/2f6526aa487b4dc460792bf1eeee79b2bba77709)
# 
# To analyze how well this model performs for this data set, we will fit the model using the training data, and then estimate the model's average root mean square error (RMSE) using k-fold cross validation. Note that we choose RMSE here to analyze the model's accuracy since RMSE is used by the [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition for evaluation and scoring. There are many other metrics that could have been used instead.
# 
# First, we define a function adapted from [Alexandru](https://www.kaggle.com/apapiu/regularized-linear-models) to calculate average RMSE using k-fold cross validation and our training data, so that we can reliably estimate the RMSE produced by each of our models.

# In[95]:


from sklearn.model_selection import cross_val_score

# determine average root mean square error (RMSE) using k-fold cross validation
def rmse_cv(model, cv=5):
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = cv))
    return rmse


# Let's now estimate the RMSE produced by a default linear regression model with the given training data.

# In[96]:


from sklearn.linear_model import LinearRegression

# estimate RMSE for linear regression model
linearModel = LinearRegression()
rmse = rmse_cv(linearModel)
print("RMSE estimate: {}, std: {}".format(rmse.mean(), rmse.std()))


# Cool, so we got a number to compare future models against, RMSE = 0.12178.
# 
# If we now fit this model, we can also look at the largest magnitude coefficient values produced. We'll later compare these against those produced by our regularization models.

# In[97]:


# fit linear model
linearModel.fit(X_train, y)

# get largest magnitude coefficients
coef = pd.Series(linearModel.coef_, index = X_train.columns)
imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])

rcParams['figure.figsize'] = (8.0, 10.0) # define size of figure
imp_coef.plot(kind = "barh")
plt.title("Most Important Coefficients Selected by Ridge")
plt.show()


# Note we don't see any really high coefficient values chosen here because we did a fairly good job preprocessing our data. Had we not removed outliers and normalized the skewed numerical features for example, there would have been higher variance and a high chance of the model picking some noticably high coefficient values in comparison to these. Even with these values though, we'll still be able to see them get compressed toward zero by the regularization models.

# # Ridge Regression (L2-Regularization) 
# 
# Both L1 and L2 regularization aims to optimize the residual sum of squares (RSS) plus a regularization term. For ridge regression (L2), this regularization term is the **sum of the squared coefficients** times a non-negative scaling factor lambda (or alpha in our sklearn model). 
# 
# As we did for the typcial linear regression model, we will again estimate this model's average RMSE in the same way for comparison. First, we will do this for alpha = 0.1, and then we will use cross validation to estimate the optimal alpha that produces the minimum RMSE. Note that 0.1 was chosen at random here, with no particular motivation.

# In[98]:


from sklearn.linear_model import Ridge

# determine RMSE for ridge regression model with alpha = 0.1
ridgeModel = Ridge(alpha = 0.1)
rmse = rmse_cv(ridgeModel)
print("RMSE estimate: {}, std: {}".format(rmse.mean(), rmse.std()))


# Ah ha, we already see an improvement upon the ordinary least squares linear regression model. We now have RMSE = 0.12046 for ridge regression with alpha = 0.1. Remember though we chose 0.1 randomly, and thus most likely isn't the optimal value. Hence, we can probably improve our RMSE even further by tuning alpha. 
# 
# Let's plot the RMSE as alpha scales to get an idea of how RMSE is affected by the value of alpha.

# In[99]:


rcParams['figure.figsize'] = (12.0, 6.0) # define size of figure

# calculate RMSE over several alphas
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

# plot RMSE vs alpha
cv_ridge.plot(title = "RMSE of Ridge Regression as Alpha Scales")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()


# Notice the U shape. We see from the plot the minimum RMSE occurs somewhere around when alpha is in the 10-15 range. Just to be a little more precise, we'll zoom in with values for alpha closer around this range.

# In[100]:


rcParams['figure.figsize'] = (12.0, 6.0) # define size of figure

# calculate RMSE over several alphas
alphas = np.linspace(9.8, 15.2, 541)
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

# plot RMSE vs alpha
cv_ridge.plot(title = "RMSE of Ridge Regression as Alpha Scales")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()


# In[101]:


optimalRidgeAlpha = cv_ridge[cv_ridge == cv_ridge.min()].index.values[0]
print("Optimal ridge alpha: {}".format(optimalRidgeAlpha))


# RMSE appears to be minimal around alpha = 10.62. This looks to be good enough for our purposes, so let's find our updated RMSE estimate using this newly found optimal alpha value.

# In[102]:


# determine RMSE for ridge regression model with optimal alpha
ridgeModel = Ridge(alpha = optimalRidgeAlpha)
rmse = rmse_cv(ridgeModel)
print("RMSE estimate: {}, std: {}".format(rmse.mean(), rmse.std()))


# Again, we have improved our RMSE. We now have RMSE = 0.11320 for a ridge regression model with an optimal alpha of about 10.62, about a 7.04 % improvement on that of the linear regression model. This looks to be about the best RMSE we can hope to get using this training data and a single ridge regression without anymore advanced preprocessing or feature engineering.
# 
# Prior to moving on to Lasso Regression, let's again revisit the largest magnitudes of the selected coefficients and compare them against those selected by the linear regression model.

# In[103]:


# fit ridge model
ridgeModel.fit(X_train, y)

# get largest magnitude coefficients
ridge_coef = pd.Series(ridgeModel.coef_, index = X_train.columns)
ridge_imp_coef = pd.concat([ridge_coef.sort_values().head(10), ridge_coef.sort_values().tail(10)])

rcParams['figure.figsize'] = (8.0, 10.0) # define size of figure
df = pd.DataFrame({ "RidgeRegression" : ridge_imp_coef, "LinearRegression" : imp_coef })
df.plot(kind = "barh")
plt.title("Most Important Coefficients Selected by Ridge")
plt.show()


# As expected, the regularization method has noticeably constrained the largest magnitude coefficient values toward zero when compared to those of the orginal linear regression model.

# # Lasso Regression (L1-Regularization)
# 
# Both L1 and L2 regularization aims to optimize the residual sum of squares (RSS) plus a regularization term. For lasso regression (L2), this regularization term is the **sum of the squared coefficients** times a non-negative scaling factor lambda (or alpha in our sklearn model).
# 
# As we did for the ridge regression model, we will again estimate this model's average RMSE in the same way for comparison. First, we will do this for alpha = 0.1 just as before, and then we will use k-fold cross validation to estimate the optimal alpha that produces the minimum RMSE.

# In[104]:


from sklearn.linear_model import Lasso

# determine RMSE for lasso regression model with alpha = 0.1
lassoModel = Lasso(alpha = 0.1)
rmse = rmse_cv(lassoModel)
print("RMSE estimate: {}, std: {}".format(rmse.mean(), rmse.std()))


# Here we see that a lasso regression model with alpha = 0.1 actually made the least accurate model yet when evaluating by RMSE. Before we give up on lasso regression though, let's use cross validation to tune alpha. Perhaps our 0.1 value was way off.
# 
# Let's try to do this the same way we did so for ridge regression.

# In[105]:


from sklearn.linear_model import Lasso

rcParams['figure.figsize'] = (12.0, 6.0) # define size of figure

# calculate RMSE over several alphas
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]
cv_lasso = pd.Series(cv_lasso, index = alphas)

# plot RMSE vs alpha
cv_lasso.plot(title = "RMSE of Lasso Regression as Alpha Scales")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()


# Hmm, okay so perhaps maybe not the plot we were expecting. The optimum alpha seems to be quite small, but we know it has to be greater than 0, so let's use sklearn's builtin LassoCV function which will use cross validation to select the best alpha from a list of potential options for the fit. (Note there also exists a RidgeCV function that works in a similar way which we could have used for the Ridge model earlier.)

# In[106]:


from sklearn.linear_model import LassoCV

# use built in LassoCV function to select best model for data
lassoModel = LassoCV(alphas = np.linspace(0.0002, 0.0022, 21), cv = 5).fit(X_train, y)
lassoModel.alpha_

optimalLassoAlpha = lassoModel.alpha_
print("Optimal lasso alpha: {}".format(optimalLassoAlpha))


# Alpha = 0.0004. This looks to be close enough to optimal for our purposes, so let's find our updated RMSE estimate with this newly found optimal alpha.

# In[107]:


lassoModel = Lasso(alpha = optimalLassoAlpha)
rmse = rmse_cv(lassoModel)
print("RMSE estimate: {}, std: {}".format(rmse.mean(), rmse.std()))


# Cool, so we see at an optimal alpha of about 0.0004, the lasso regression model seems to perform even better than the optimal ridge regression model for this data set. We now have RMSE = 0.11182, which is about a 8.17 % improvement from that of our linear regression model. This looks to be about the best RMSE we can hope to get using this training data and a single lasso regression without anymore advanced preprocessing or feature engineering.
# 
# Let's now briefly take a look at the features the lasso regression model deems important. Note that the Lasso method will actually do feature selection for you - setting coefficients of features it deems unimportant to zero.

# In[108]:


# fit lasso model
lassoModel.fit(X_train, y)

# get largest magnitude coefficients
lasso_coef = pd.Series(lassoModel.coef_, index = X_train.columns)
lasso_imp_coef = pd.concat([lasso_coef.sort_values().head(10), lasso_coef.sort_values().tail(10)])

rcParams['figure.figsize'] = (8.0, 10.0) # define size of figure
df = pd.DataFrame({ "LassoRegression" : lasso_imp_coef, "LinearRegression" : imp_coef })
df.plot(kind = "barh")
plt.title("Most Important Coefficients Selected by Lasso")
plt.show()


# Again as expected, the values seem to have been compressed toward 0 when compared to those chosen by the original linear regression model.
# 
# As noted above, the Lasso method will actually perform feature selection, so let's now look at the number of coefficients equal and not equal to zero.
# 
# **This is an important difference to take note of between ridge regression and lasso regression. While ridge regression punishes high coefficient values, it will not get rid of irrelevant features by enforcing their coefficients to zero. It will only try to minimize their impact. Lasso regression on the other hand will both punish high coefficient values, and get rid of irrelevant features by setting their coefficiants to zero. Thus, when training data sets with many irrelevant features, the lasso model can be useful in feature selection.** 

# In[109]:


lasso_coef = pd.Series(lassoModel.coef_, index = X_train.columns)
print(sum(lasso_coef != 0))
print(sum(lasso_coef == 0))


# It appears this Lasso model selected 107 of the features in this instance, the most important of which are included in the plot above, while zeroing out the other 112. We won't go into any more detail with regards to the specifc features at this time, but know that the selected features are not always the "correct" features, and should be considered, especially when multicollinearity exists within the feature set.
# 
# ## L0-Norm
# 
# Lastly, to get an idea of how the number of features chosen is impacted by the strength of alpha, let's plot the number of non-zero coefficients that lasso produces as you vary the strength of the regularization parameter alpha. (This is also called the L0-Norm of the coefficients.)

# In[110]:


# scale alpha
alphas = np.linspace(0.0002, 0.4002, 2001)
nonZeros = []

# for each alpha, fit model to training data
for alpha in alphas:
    lassoModel = Lasso(alpha = alpha).fit(X_train, y)
    coef = pd.Series(lassoModel.coef_, index = X_train.columns)
    # append the number of non-zero coefficients
    nonZeros = np.append(nonZeros, sum(coef != 0))

# plot number of non-zeros (L0-Norm) vs alpha
rcParams['figure.figsize'] = (12.0, 6.0) # define size of figure
lzeroNorm = pd.Series(nonZeros, index = alphas)
lzeroNorm.plot(title = "L0-Norm of Lasso Regression Model as Alpha Scales")
plt.xlabel("alpha")
plt.ylabel("number of non-zeros")
plt.show()


# In[111]:


lzeroNorm.max()


# In[112]:


lzeroNorm.min()


# From the plot above, we see as the strength of the regularization parameter alpha grows, the number of selected features drops extremely quick from a max of 134 before eventually leveling off at 4 features when alpha is slightly greater than 0.25. It seems that in general, the higher the strength of alpha, the more restrictive the lasso model becomes with regards to the number of selected features. Keep this in mind when dealing with data sets that contain a large number of irrelevant features.

# # Predictions

# We'll now create our submission predictions. Remember we log transformed the target values so we will need to exponentiate our predictions.

# In[113]:


linearModel = LinearRegression().fit(X_train, y)
lr_submission = pd.DataFrame()
lr_submission['Id'] = test['Id']
lr_submission['SalePrice'] = np.expm1(linearModel.predict(X_test))
lr_submission.to_csv('linear-regression.csv', index=False)


# In[114]:


ridgeModel = Ridge(alpha = optimalRidgeAlpha).fit(X_train, y)
ridge_submission = pd.DataFrame()
ridge_submission['Id'] = test['Id']
ridge_submission['SalePrice'] = np.expm1(ridgeModel.predict(X_test))
ridge_submission.to_csv('ridge.csv', index=False)


# In[115]:


lassoModel = Lasso(alpha = optimalLassoAlpha).fit(X_train, y)
lasso_submission = pd.DataFrame()
lasso_submission['Id'] = test['Id']
lasso_submission['SalePrice'] = np.expm1(lassoModel.predict(X_test))
lasso_submission.to_csv('lasso.csv', index=False)


# # Credits
# 
# Please note that some of the ideas and code in this notebook come from, or are at least inspired by, the work of:
# * Alexandru Papiu: https://www.kaggle.com/apapiu/regularized-linear-models
# 
# If you use parts of this notebook in your own scripts, please give some sort of credit (for example a link back to this or the above original notebook) and upvote. Thanks in advance!
# 
# ## **Thanks for reading. Please feel free to comment, and remember to upvote if you found this notebook helpful or interesting!**

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Predicting Diamond Prices 
# 
# In this notebook we will go through a project, from analyzing data to testing different regression models. The objective is to predict the price of a diamond based on different attributes of the diamond.
# 
# 1. Data Overview
# 2. Split train / test + EDA
# 3. Normalization / Modeling

# In[ ]:


# Data Analysis #
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Model stuff #
from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("seaborn")
plt.rcParams['figure.figsize'] = (12,5)


# # 1. Data Overview

# In[ ]:


# Import data #
dpath = '../input/'
diamonddf = pd.read_csv(dpath + "diamonds.csv")


# In[ ]:


diamonddf.head()


# In[ ]:


diamonddf.info()


# There are roughly 54,000 examples of diamonds, each with 11 qualities. One of the columns looks like an extra index. We'll drop this column.

# In[ ]:


diamonddf.drop("Unnamed: 0", axis = 1, inplace = True) # drop weird column


# In[ ]:


diamonddf.isnull().sum()


# ## Column descriptions
# 
# - **carat:** The weight of the diamond, equivalent to 200mg (should be a good indicator)
# - **cut:** Quality of the cut
# - **color:** Color of the diamond from J to D (worst to best)
# - **clarity:** How clear the diamond is; I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)
# - **depth:** Total depth percentage (relative to x and y). **Will likely be collinear.**
# - **table:** Width of top of diamond relative to widest point (43--95)
# - **price:** In US dollars ($)
# - **x, y, z:** Dimensions of the diamond

# ## Format data types and clean up data
# 
# Before going into the analysis, we want to remove / fill null entries, and change the data type of misspecified columns. We don't go into any feature engineering or anything here yet.

# In[ ]:


diamonddf.dtypes


# Looks good.

# In[ ]:


# Are there any weird values? #
diamonddf.describe(include=['O'])


# In[ ]:


# Quantitative description #
diamonddf.describe()


# In[ ]:


numcols = diamonddf.select_dtypes(include = ['float64','int64']).columns.tolist()


# In[ ]:


colors = sns.color_palette("deep")
fig,axes = plt.subplots(3,3, figsize = (12,8)) # up to 9 quant vars
sns.distplot(diamonddf["carat"], color = colors[0], ax = axes[0,0])
sns.distplot(diamonddf["depth"], color = colors[1], ax = axes[0,1])
sns.distplot(diamonddf["table"], color = colors[2], ax = axes[0,2])
sns.distplot(diamonddf["price"], color = colors[3], ax = axes[1,0])
sns.distplot(diamonddf["x"], color = colors[4], ax = axes[1,1])
sns.distplot(diamonddf["y"], color = colors[0], ax = axes[1,2])
sns.distplot(diamonddf["z"], color = colors[1], ax = axes[2,0])
plt.suptitle("Distribution of Quantitative Data", size = 16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[ ]:


colors = sns.color_palette("deep")
fig,axes = plt.subplots(3,3, figsize = (12,8)) # up to 9 quant vars
sns.boxplot(y= diamonddf["carat"], color = colors[0], ax = axes[0,0])
sns.boxplot(y = diamonddf["depth"], color = colors[1], ax = axes[0,1])
sns.boxplot(y = diamonddf["table"], color = colors[2], ax = axes[0,2])
sns.boxplot(y = diamonddf["price"], color = colors[3], ax = axes[1,0])
sns.boxplot(y = diamonddf["x"], color = colors[4], ax = axes[1,1])
sns.boxplot(y = diamonddf["y"], color = colors[0], ax = axes[1,2])
sns.boxplot(y = diamonddf["z"], color = colors[1], ax = axes[2,0])
plt.suptitle("Distribution of Quantitative Data (boxplots)", size = 16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# The distribution of depth, table, y, and z all have long tails. There is a particular value in $z$ that looks like an error or extreme outlier, and there are other outliers we can explore too.
# 
# x, y, and z shouldn't have 0 as a value, since that wouldn't make physical sense. We can remove all rows where any of these variables are zero, or we can impute them.

# In[ ]:


# diamonds that are probably errors
zero_df = diamonddf[(diamonddf['x'] == 0) |
           (diamonddf['y'] == 0) |
           (diamonddf['z'] == 0)]
zero_df.head()


# In[ ]:


zero_df.shape


# In[ ]:


# Drop the rows with zero as any x, y, or z
diamonddf.drop(zero_df.index, inplace = True)


# * We will stray as much as possible from dropping any outliers since we lack context. That wonky z value could actually be completely valid. Next we will briefly check the categorical variables for any blatant errors.

# ## Categorical Variables

# In[ ]:


cat_vars = diamonddf.select_dtypes(include = 'object').columns.tolist()
fig, axes = plt.subplots(1,3, figsize = (12,5))
i = 0
for var_name in cat_vars:
    diamonddf[var_name].value_counts().sort_values().plot(kind = 'barh', color = 'C0', ax = axes[i])
    axes[i].set_title(var_name)
    i += 1
plt.tight_layout()
plt.show()


# It looks like there aren't any extra categories as the result of spelling or fat finger errors. 

# # 2. Create a test set
# 
# Before going any deeper into the analysis we will create a test set for model evaluation. We don't want to add any bias to the way we create models.

# In[ ]:


train_df, test_df = train_test_split(diamonddf, test_size=0.2, random_state=12)


# In[ ]:


Y_test = test_df['price']
X_test = test_df.drop('price', axis = 1)


# In[ ]:


print("Total dataset size: {}".format(diamonddf.shape))
print("Training set size (80%): {}".format(train_df.shape))
print("Test set size (20%): {}".format(test_df.shape))


# # EDA
# 
# - Which predictors are correlated to the price of a diamond?
# - Are high quality diamonds worth more than low quality diamonds?
# - Are there any immediate interaction effects between a categorical predictor, numerical predictor, and the response?
# - Is there collinearity/multicollinearity in the dataset?
# - Are there any clear outliers that we should investigate?
# 

# In[ ]:


diamonds = train_df.copy()


# In[ ]:


# Pair plot#
sns.pairplot(diamonds)
plt.show()


# It looks like price is skewed right, so we should log transform it for better predictions. There aren't any clear predictors here, so we will have to experiment to find best combinations. There's also a pretty obvious outlier that we should investigate further.
# 
# It also looks like $x$ is related to the carat of the diamond, so these might cause a collinearity issue. We will check the correlation matrix just to be certain.

# ## Investigate strange outlier
# 
# We tried ignoring the outlier but it looks like an obvious error. Let's take a look at the weird outlier that shows up in the y and z plots. 

# In[ ]:


ol1 = diamonds[diamonds['z'] > 20].index
ol2 = diamonds[diamonds['y'] > 20].index

fig, axes = plt.subplots(1,3, figsize = (12,4))
sns.scatterplot(x = diamonds['carat'], y = diamonds['z'], ax = axes[0]) 
axes[0].annotate(ol1[0], (diamonds['carat'].loc[ol1], diamonds['z'].loc[ol1]), size = 12)

sns.scatterplot(x = diamonds['x'], y = diamonds['y'], ax = axes[1])
axes[1].annotate(ol2[0], (diamonds['x'].loc[ol2], diamonds['y'].loc[ol2]), size = 12)

sns.scatterplot(x = diamonds['y'], y = diamonds['z'], ax = axes[2])
axes[2].annotate(ol1[0], (diamonds['y'].loc[ol1], diamonds['z'].loc[ol1]), size = 12)
axes[2].annotate(ol2[0], (diamonds['y'].loc[ol2] - 4, diamonds['z'].loc[ol2] + 1), size = 12)

plt.suptitle("Outliers in 3 sample plots", size = 14)
plt.show()


# In[ ]:


diamonds[diamonds['z'] > 20]


# In[ ]:


diamonds[diamonds['y'] > 20]


# In[ ]:


diamonds['z'].describe()


# The two outliers might be 3.18 instead of 31.8, this is an assumption but it seems reasonable.
# 
# These are two values out of 50,000 that look really weird, and will definitely impact regression models that aren't robust to outliers. Iterative methods like random forest and gradient boosting will be able to handle them, but I will assume that these are errors and that the chances that they will occur naturally in the real world are slim. 
# 
# It's important to note that this isn't the way to handle outliers every time though; imagine if it was a type of observation that wasn't recorded because someone lost all of the diamonds of this type (or something like that). 

# ## Drop two outliers

# In[ ]:


cond = (diamonds['y'] > 20) | (diamonds['z'] > 20) 
diamonds.drop(diamonds[cond].index, inplace = True)


# In[ ]:


fig, axes = plt.subplots(1,3, figsize = (12,4))
sns.scatterplot(x = diamonds['carat'], y = diamonds['z'], ax = axes[0]) 

sns.scatterplot(x = diamonds['x'], y = diamonds['y'], ax = axes[1])

sns.scatterplot(x = diamonds['y'], y = diamonds['z'], ax = axes[2])

plt.suptitle("3 Sample Plots without Outliers", size = 16)
plt.show()


# Looks great. 

# In[ ]:


sns.heatmap(diamonds.corr(), cmap = "RdBu_r", square = True, annot=True, cbar=True)
plt.title("Correlation Between Variables")
plt.show()


# There is very apparent collinearity here. x,z, and z are all correlated with each other (and should either be combined, or one should be used to model the price). There is also a very strong relationship between carat and x, y, and z. **This is probably because carat is a unit of weight**. 
# 
# While not always the case, it does appear that carat is a function of the dimensions with some density coefficient [(source)](https://www.jewelrynotes.com/how-to-calculate-a-diamonds-weight-in-carats/).
# 
# So we will drop x, y, and z.

# In[ ]:


# Drop x, y, and z #
diamonds.drop(['x','y','z'], axis = 1, inplace = True)


# # Correlation categorical predictors and price
# 
# We will now investigate the relationship between a diamond's attributes and its price using categorical variables, and a combination of continuous/categorical. First, we order the categories.

# In[ ]:


clar_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = sorted(diamonds['color'].unique().tolist(), reverse = True)


# In[ ]:


fig, axes = plt.subplots(1,2, figsize = (12,5))
sns.boxplot(x = "cut", y = "price", data = diamonds, order = cut_order, ax = axes[0], palette = 'Blues')
sns.boxplot(x = 'clarity', y = 'price', data = diamonds, order = clar_order, ax = axes[1], palette = 'Blues')
plt.suptitle("Diamond Price by Cut and Clarity", size = 14)
plt.show()


# In[ ]:


fig, axes = plt.subplots(1,2, figsize = (12,5))
sns.scatterplot(x = 'carat', y = 'price', hue = "cut", palette = 'Blues', hue_order = cut_order,
                size = 10, data = diamonds, ax = axes[0])
sns.scatterplot(x = 'carat', y = 'price', hue = "clarity", palette = 'Blues', hue_order = clar_order,
                size = 10, data = diamonds, ax = axes[1])
plt.suptitle("Diamond Price vs. 2 predictors", size = 14)
plt.show()


# In[ ]:


fig, axes = plt.subplots(1,2, figsize = (12,5))
sns.boxplot(x = "color", y = "price", data = diamonds, order = color_order, palette = 'Blues', ax = axes[0])
sns.scatterplot(x = 'carat', y = 'price', hue = "color", palette = 'Blues', hue_order = color_order,
                size = 10, data = diamonds, ax = axes[1])
plt.suptitle("Diamond Price by Color", size = 14)
plt.show()


# There is an interaction between carat, price, and color. Same goes with carat, price, and clarity. Carat and cut don't interact as distinctly.

# # Transformations on Price
# 
# When looking at a summary of our data, it appeared that the response was skewed. While this isn't crucial, we can visualize the effect of different transformations (square root, log, cube root, etc.) on the response. We'll also experiment with this in our modeling stage.

# In[ ]:


fig, axes = plt.subplots(2, 3, figsize = (12,6))
sns.kdeplot(np.log(diamonds['price']), shade=True , color='r', ax = axes[0,0])
axes[0,0].set_title("Log transform")
sns.kdeplot(np.sqrt(diamonds['price']), shade=True , color='b', ax = axes[0,1])
axes[0,1].set_title("Square root transform")
sns.kdeplot((diamonds['price']**(1/3)), shade=True , color='coral', ax = axes[0,2])
axes[0,2].set_title("Cube root transform")
sns.boxplot(y = np.log(diamonds['price']), ax = axes[1,0], color = 'coral')
sns.boxplot(y = np.sqrt(diamonds['price']), ax = axes[1,1], color = 'coral')
sns.boxplot(y = (diamonds['price']**(1/3)), ax = axes[1,2], color = 'coral')
plt.tight_layout()
plt.show()


# # 3. Modeling
# 
# We will consider a few regression models:
# 
# - Ridge Regression
# - LASSO Regression
# - ElasticNet
# - Random Forest Regression
# 
# XGBoost, Support Vector Regression and a stacked ensemble were considered, but SVR was too slow due to the number of features and simple models were performing well which indicated that more complex models weren't necessary.
# 
# 
# We have split our training and test set already, so we can start by one-hot encoding our categorical variables, and then normalizing our numerical variables.
# 
# We provide an interpretable model, as well as a model that (likely) performs better at the loss of some interpretability.

# Error metrics:
# 
# - Mean Absolute Error (how far away are my predictions from ground truth?)
# - Mean Squared Error (is my model making large errors?)
# - $R^2$ Score (goodness-of-fit)
# 
# Since all of our models contain the same amount of predictors $p$, we won't have to refer to the $R^2_{adj}$ score.

# In[ ]:


def error_metrics(y_true, y_pred):
    mean_abs = "Mean Absolute Error: {}".format(mean_absolute_error(y_true, y_pred))
    mean_squared = "Mean Square Error: {}".format(mean_squared_error(y_true, y_pred))
    r2 = "r2 score: {}".format(r2_score(y_true, y_pred))
    return mean_abs, mean_squared, r2


# In[ ]:


# Remove the label #
X_train = diamonds.drop('price', axis = 1)
Y_train = diamonds['price'].copy()


# We're also going to have to remove the outlier from out X_train. The models will learn more useful information this way.
# 
# **Missing Values:**
# 
# We removed the observations where x, y, or z were zero. There aren't any other missing values.

# ### Encoding
# 
# We use encoding since our categorical variables are actually ordinal and not purely categorical. We will use an ordered version of the categories and then map those with corresponding integers for our new columns.

# In[ ]:


def cat_mapper(categories):
    "create a dictionary that maps integers to the ordered categories"
    i = 0
    mapped = {}
    for cat in categories:
        mapped[cat] = i
        i += 1
    return mapped


# In[ ]:


cat_mapper(color_order)


# In[ ]:


cat_mapper(cut_order)


# In[ ]:


cat_mapper(clar_order)


# In[ ]:


X_train[cat_vars].head()


# In[ ]:


X_train_mapped = X_train.copy()
X_train_mapped['cut'] = X_train_mapped['cut'].map(cat_mapper(cut_order))
X_train_mapped['color'] = X_train_mapped['color'].map(cat_mapper(color_order))
X_train_mapped['clarity'] = X_train_mapped['clarity'].map(cat_mapper(clar_order))


# ### Scaling Features
# 
# We will use MinMax scaling, with the possibility of trying StandardScaler in case we want all of the predictors to be roughly normal. We don't have many outliers so robust transformers won't be necessary.

# In[ ]:


minmaxscaler = MinMaxScaler()
numcols = ['carat','depth','table']


# In[ ]:


X_train_mapped[numcols] = minmaxscaler.fit_transform(X_train_mapped[numcols])


# In[ ]:


X_train_mapped.head()


# In[ ]:


# Adjusting the test datasets #
X_test.drop(['x','y','z'], axis = 1, inplace = True)
X_test['cut'] = X_test['cut'].map(cat_mapper(cut_order))
X_test['color'] = X_test['color'].map(cat_mapper(color_order))
X_test['clarity'] = X_test['clarity'].map(cat_mapper(clar_order))


# In[ ]:


X_test[numcols] = minmaxscaler.transform(X_test[numcols])


# ## Ridge Regression

# In[ ]:


alphas = [.01,.1,1,10,100,1000,10000]


# In[ ]:


ridge = RidgeCV(alphas = alphas, cv = 5)
ridge_fit = ridge.fit(X_train_mapped, Y_train)


# In[ ]:


yhat_ridge = ridge_fit.predict(X_test)


# In[ ]:


sns.distplot(Y_test - yhat_ridge)
plt.title("Distribution of Errors (Ridge Regression)")
plt.show()


# In[ ]:


x = np.linspace(0, 30000, 1000)
sns.scatterplot(x = Y_test, y = yhat_ridge)
plt.plot(x,x, color = 'red', linestyle = 'dashed')
plt.xlim(-100, 36000)
plt.ylim(-100, 36000)
plt.title("Actual vs. Predicted (Ridge Regression)")
plt.show()


# In[ ]:


# Ridge error metrics #
error_metrics(Y_test, yhat_ridge)


# ## LASSO Regression

# In[ ]:


lasso = LassoCV(cv=5, random_state=12, alphas = alphas)
lasso_fit = lasso.fit(X_train_mapped, Y_train)
yhat_lasso = lasso_fit.predict(X_test)


# In[ ]:


error_metrics(Y_test, yhat_lasso)


# In[ ]:


sns.distplot(Y_test - yhat_lasso)
plt.title("Distribution of Errors (LASSO Regression)")
plt.show()


# In[ ]:


sns.scatterplot(x = Y_test, y = yhat_lasso)
plt.plot(x,x, color = 'red', linestyle = 'dashed')
plt.xlim(-100, 36000)
plt.ylim(-100, 36000)
plt.title("Actual vs. Predicted (LASSO Regression)")
plt.show()


# It looks like the model underpredicts for diamonds with a price of 10,000, similar to the ridge regression model. This is evident in the histogram (mean is less than zero) and in the scatterplot (the trend is a curve under $y = x$ line)

# ## ElasticNet Regression
# 
# The results of both models are similar, but let's see if a mix of both types of penalties ($L_1$ and $L_2$)improve the prediction accuracy.

# In[ ]:


elasticnet = ElasticNetCV(cv=5, random_state=12,
                          l1_ratio = 0.9,
                          alphas = alphas)
elastic_fit = elasticnet.fit(X_train_mapped, Y_train)
yhat_elastic = elastic_fit.predict(X_test)


# In[ ]:


error_metrics(Y_test, yhat_elastic)


# After a couple of tests with `l1_ratio` equaling 0.5, 0.3, 0.8, and 0.9 we see that using an L1 penalty reduces our errors and increases our $R^2$ score. However having a bit of the $L_2$ penalty decreased our errors.

# ## Random Forest Regression
# 
# We use random forest regression to test a non-linear model on the data. Support Vector Regression was initially considered, but was very slow due to the number of observations (see the [SVR documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) on sklearn for other approaches).
# 
# First we will train a model with 1000 estimators and then use grid search if it looks like the model can be improved

# In[ ]:


randomforest = RandomForestRegressor(max_depth=5, 
                                     random_state=12, 
                                     n_estimators = 1000)
                                     


# In[ ]:


rf_fit = randomforest.fit(X_train_mapped, Y_train)
yhat_rf = rf_fit.predict(X_test)


# In[ ]:


error_metrics(Y_test, yhat_rf)


# The model was tuned with max_depth = 2 and max_depth = 5 with a large improvement when max_depth = 5. This is the best model in terms of our three error metrics.

# In[ ]:


sns.distplot(Y_test - yhat_rf)
plt.title("Distribution of Errors (Random Forest Regression)")
plt.show()


# In[ ]:


sns.scatterplot(x = Y_test, y = yhat_rf)
plt.plot(x,x, color = 'red', linestyle = 'dashed')
plt.xlim(-100,20000)
plt.ylim(-100, 20000)
plt.title("Actual vs. Predicted (LASSO Regression)", size = 14)
plt.show()


# # Final Models
# 
# The best models in terms of mean absolute error are:
# 
# 1. Random Forest Regression: **\$538.72** 
# 2. ElasticNet (with 0.9 L1 penalty): **\$854.20**
# 3. Ridge: **\$860.88**
# 4. LASSO: **\$861.01**
# 

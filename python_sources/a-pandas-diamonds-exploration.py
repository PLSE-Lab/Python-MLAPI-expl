#!/usr/bin/env python
# coding: utf-8

# # Exploring the diamonds data set.
# 
# This IPython notebook takes a look at the diamonds data set. We first do some exploratory data analysis followed by some visualization and modeling. This notebook is also available at my GitHub repository [sweetpy](https://www.github.com/Nelson-Gon/sweetpy).
# 
# * **Contents**
#    - [Modules Used](#Modules-Used)
#    - [Exploratory Data Analysis](#Exploratory-Data-Analysis)
#       * [Data Attributes](#Data-Attributes)
#       * [Basic Data Stats](#Basic-Data-Stats)
#          * [Mean Prices by Cut](#Mean-Prices-by-Cut)
#          * [Mean Prices by Cut and Color](#Mean-Prices-by-Cut-and-Color)
#          * [What about clarity?](#What-about-clarity?)
#          * [Where's the missing data?](#Where's-the-missing-data?)
#    - [Data Visualization](#Data-Visualization)
#       * [Histogram of Prices](#Histogram-of-Prices)
#       * [Skewness Test](#Skewness-Test)
#       * [Prices by cut](#Prices-by-cut)
#       * [Prices by carat](#Prices-by-carat)
#       * [Outliers by carat](#Outliers-by-carat)
#       * [Box Plots](#Box-Plots)
#          * [Zooming In](#Zooming-In)
#       * [Prrices by cut and clarity](#Prices-by-cut-and-clarity)
#       * [Are some diamonds rare?](#Are-some-diamonds-rare?)
#       * [Violin Plots](#Violin-Plots)
#       * [Flawless but cheap?](#Flawless-but-cheap?)
#    - Modeling
#       * [Linear regression](#Linear-regression-in-the-diamond-mines)
#         * [Train-Test-Split](#Data-Split)
#         * [Label Encoding](#Label-Encoding)
#         * [The Model](#The-Model)
#         * [Model Evaluation](#Model-Evaluation)
#           * [What's in the R<sup>2</sup>?!](#What's-in-the-R-squared?!)
#         * [Approaching Reality](#Towards-Reality)
#             * [Feature Selection](#Feature-Importance-and-Selection)
#          
#     - [Future Steps and Feedback](#Future-Steps)

# ## Modules Used
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from seaborn import load_dataset
import seaborn as sns


# ## Exploratory Data Analysis
# 
# To begin our analysis, we need to first get a general idea of what our  dataset looks like. To do this, we shall first take a look at a few common procedures in exploratory data analysis.
# 

# In[ ]:


# Loading the data set
#diamonds = sns.load_dataset("diamonds")
# issues with Kaggle, use their data sets
diamonds = pd.read_csv("../input/diamonds/diamonds.csv")
diamonds.head()


# Having taken a peek at our data set, there are a few things that need to be done. 
# First, we need to have a general understanding of what the columns stand for. For instance, it would be hard to understand what x, y and z stood for unless you had prior knowledge about the data set.[](http://)

# ### Data Attributes
# 
# There are a couple of ways to find out what the column mean: One could simply guess that x, y and z refer to the lengths, widths and heights of the diamonds. One could also do a simple web search. 
# 
# Preferably, one can take a look at the dataset's documentation as shown next. Regrettably, the data set is not well documented by the package authors.

# In[ ]:


#?diamonds


# However, a web [search](https://rstudio-pubs-static.s3.amazonaws.com/316651_5c92e58ef8a343e4b3f618a7b415e2ad.html) does identify what the columns mean and x, y, z are as guessed above.
# 

# ## Basic Data Stats
# 
# Before, taking a look at the data's basic statistics, it might be useful to give the data more meanignful names. To do that, we shall use a data frame(`DataFrame`)'s `rename` method with a `dictionary` mapping.  Since the default is to set `inplace` to `False`, we can set that to `True`.

# In[ ]:


diamonds.rename(columns = {'x': 'length', 'y': 'width', 'z': 'height'}, inplace = True)
diamonds.head()


#  Since we are now more comfortable with the dataset's column names since they are more informative, we can proceed to look at the basic stats.

# In[ ]:


diamonds.describe()


# From the above, we can see that the mean price was high. However, looking at these statistics is less informative since these may vary by cut, color, clarity or depth. Therefore looking at these statistics by group could be more informative.
# 
# ## Mean Prices by Cut

# In[ ]:


diamonds.groupby("cut").mean()


# From the above, it is clearer that the mean prices for instance as one might have expected vary by cut. There is an almost linear relationship between cut and price. It is also interesting to note that the mean depth does not vary that much across the different cut(s).
# 
# We can take the above a step further by looking at how prices for instance vary when the data is grouped by both cut and color.
# 
# ## Mean Prices by Cut and Color

# In[ ]:


# use vectorised cython functions instead.
diamonds.groupby(["cut", "color"]).mean()


# The nature of the display does not allow for a great interpretation of the data. However, from the few fields shown, we can seee a clear difference in prices with respect to color. 
# ## What about clarity?

# In[ ]:


diamonds.groupby("clarity").mean()


# The focus of our analysis has upto now been on the mean prices. It is worth noting what the clarity actually [means](https://ggplot2.tidyverse.org/reference/diamonds.html): **a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))**. With that in mind, the mean prices appear to follow a non linear relationship with respect to clarity. We will verify this as the notebook progresses statistically. 
# 
# Having looked at a few basic stats about the data(remember we focused entirely on the mean), we now take a different route. We need to visualize our data which might reveal more trends and perhaps provide a more informative picture. However, important for modeling purposes is whether we have missing data.
# 
# ## Where's the missing data?

# There are several ways to find missing data, but we will represent it as the percentage of each column that is missing.

# In[ ]:


# unnamed must be removed, leftover from rownames ie index
#diamonds.apply(lambda x: sum(x.notnull())/len(x) * 100)
# rewrite with transform
#diamonds.dtypes
# I conclude that there is no(as far as I know and have researched) more concise alternative to using apply
#diamonds[diamonds.notnull()].groupby(["cut", "color", "clarity"]).transform("sum")
diamonds.notnull().sum() / len(diamonds) * 100


# From the above, we can see that all our columns in fact have 100% of the data which is great. We can then proceed with data visualization.
# 
# ## Data Visualization

# ## Histogram of Prices
# 
# To visualize the prices, we can either use `matplotlib`, `pandas` or `seaborn` amongst other libraries. We shall use `matplotlib`'s `hist`.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
plt.hist(x = "price", data = diamonds, color = "indianred", alpha = 0.8)
plt.title("Histogram of Prices", fontsize = 15)
plt.ylabel("Frequency", fontsize = 15)
plt.xlabel("Prices(USD)", fontsize = 15)


# From the above we it appears(as one might expect), the distribution of prices is **skewed to the right** ie as the prices increase, the number of diamonds decrease. We can carry out a skewness test using `scipy`'s `stats` module to verify this.
# 
# ## Skewness Test
# 

# In[ ]:


from scipy.stats import skew
skew(diamonds["price"])


# From the above, our skew is a positive value(greater than 1) indicating a positive skewness which means that our mean is greater than the median. Let us look at our data's median and mean prices to verify.

# In[ ]:


from IPython.display import display
display(np.mean(diamonds["price"]))
display(np.median(diamonds["price"]))


# ## Prices by cut
# 
# For this part, we shall uses `seaborn`'s awesome functions that deal with categorical plots.
# 

# In[ ]:


sns.catplot(x="cut", y="price",kind="bar", data = diamonds)
plt.title("Prices by cut", fontsize = 15)


# However, to make the above plot more "attractive"/informative/intuitive, we could go a step further and sort the values before plotting them.

# In[ ]:


sorted_data = diamonds.sort_values(by = ["price"] , ascending = False)
#sorted_data.head()
# probably a better way exists that leverages col_order
# This sorts the x-axis in alphabetical order which is a bit less informative
sns.catplot(x = "cut", y="price", data = sorted_data, kind = "bar")


# As shown in [Mean Prices by Cut](#Mean-Prices-by-Cut), the prices do vary by cut and as already stated the relatioship appears to be linear ie as the quality(cut) of the diamond increases, so does its price. There will however always be outliers.  Let us take a look at the prices by carat.
# 
# ## Prices by carat

# In[ ]:


plt.scatter(x = "carat", y = "price", data = diamonds, color = "indianred")
plt.title("Prices by carat", fontsize = 15)
plt.xlabel("Carat", fontsize = 14)
plt.ylabel("Price", fontsize = 14)


# In general, the distribution of the prices is such that most of the diamonds fall in the lower half. On the issue of outliers, there are visible outliers for instance 5 carat diamonds which is a bit unusual. To get a better understanding of these outliers, we shall filter our data set for these outliers and get more info about them.
# 
# ## Outliers by carat
# We see that there is only a single diamond with a carat greater than or equal to 5. Interesting is the fact that the price of this diamond is comparatively high especially given it is of just fair cut. 

# In[ ]:


diamonds[diamonds["carat"] >=5]


# For completeness, we shall show a more pandas like way to achieve the above result. This achieves the same purpose in arguably more user-friendly code. This is especially useful if one is used to querying data bases.

# In[ ]:


diamonds.query("carat >=5")


# ## Box Plots
# Having looked at a few plots, perhaps looking at box and whiskers plots might provide more insight about the data set. What we are interested in is the distirbution of our data across various categories. Let's look at the clarity.

# In[ ]:


sns.boxplot(x="cut", y="price", data = diamonds, palette = "RdBu")
plt.title("Price Distribution by Cut", fontsize = 15, fontweight = "bold")
plt.xlabel("cut",size=14)
plt.ylabel("price", size = 15)


# The above box plot does give a general idea of how the prices are distributed across the different cuts. As noted previously, there appears to be a linear relationship between cut and price. We however, see a few points that do not exactly fall within the range ie they are above the upper quartile(outliers).

#  ## Zooming In
#  
#  We therefore take a look at these by "zooming in" on the data set as follows. It is interesting to note that the mean prices for this data set do not appear to follow a linear relationship. 

# In[ ]:


filtered = diamonds.query("price > 10000")

sns.boxplot(x = "cut", y = "price", data = filtered, palette = "RdBu")


# We have seen(visually) how price is influence by cut but what happens when we add clarity to the story?

# ## Prices by cut and clarity

# In[ ]:


facets = sns.FacetGrid(col = "cut", row = "clarity", data = diamonds)
facets.map(plt.hist, "price")
plt.title("Price Distribution Plot by Cut and Clarity")


# The above plot gives a more detailed account of the price distribution with respect to cut and clarity. The majority of the diamonds appear to fall in the ideal and VS2 category.
# 
# Interesting are the very low numbers of diamonds that fit I1 and good cut criterion. It is also surprising to see that clarity IF and fair cut almost has no diamonds. 
# 
# This is surprising because IF(Internally flawless) diamonds would be expected to be more even with a fair cut. This might however imply that these are extremely rare diamonds.
# 
# To get a clearer picture, we could visualize the distribution by clarity.
# 
# ## Are some diamonds rare?

# In[ ]:


column_order = sorted(diamonds.clarity.unique())
sns.countplot(x="clarity", data = diamonds.sort_values(["price", "depth"]), order = column_order)


# Indeed, we see that internally flawless(IF) diamonds occupy a relatively small portion of the data set. This could perhaps be due to mining techniques. Indeed, [these are extremely rare diamonds](https://beyond4cs.com/clarity/if-and-fl/).
# 
# ## Violin Plots
# We can take this a step further to visualize price distribution by clarity. In this example, we are essentially zooming in on our data to get the distribution of diamonds whose price is greater than or equal to 10000 USD.

# In[ ]:


sns.violinplot(x="clarity", y="price", data = diamonds.query("price > 10000").sort_values("price"),
              order = column_order)


# When I first made this plot, one could see that the internally flawless diamonds had a lower mean price than other levels of clarity. It is interesting however and as one might probably have expected to see that if we zoom in on the dataset to include higher prices, the mean price of the innternally flawless is higher than that of included(I1) and almost equal to that of the slightly included. 
# 
# This therefore suggests that the price of a diamond is(logically) determined by more than just its clarity. 
# 
# ## Flawless but cheap?
# 

# In[ ]:


diamonds.groupby("clarity").mean()


# To take the above a step further, we can sort the clarity in increasing or decreasing order as per the industry standards. 

# In[ ]:


#diamonds.clarity, order = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1","SI2","I1"])
# convert to categorical
#pd.Categorical(diamonds["clarity"], categories= ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1","SI2","I1"] )
# This is probbaly computationally expensive
categorize_data = pd.CategoricalDtype(categories= ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1","SI2","I1"])
diamonds["clairty"] = pd.Series(diamonds.clarity, dtype = categorize_data)
diamonds.groupby("clairty").mean()


# The above is probably a better approach since we now have our data sorted as per an expert in the diamond mining business would expect. Let's see what a plot of this data would look like.

# In[ ]:


# Ok, seaborn's x axis really needs an upgrade. It is currently in my opinion less flexible.
sns.catplot(x = "clarity", y="price", data =  diamonds, kind = "bar",
          col_order = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1","SI2","I1"])


# # Linear regression in the diamond mines
# 
# In order to carry out(build a model) regression analysis on our data, we need to import a popular Machine Learning library(package) known as [scikit-learn](https://scikit-learn.org/stable/index.html)(**sklearn**). In our case, we are more interested in ordinary least squares regression which is [documented here](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares). 
# 
# ## Data Split

# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# Next, we need to split our dataset into a training and test/validation set. We shall then split these into a y(dependent variable) and x(predictor variable(s)). In this case, our y will be the price of a diamond while for starters, we'll build a model that uses all our other attributes for predicting the price of a diamond.

# In[ ]:


# Make dependent and predictor variables
dependent_variable = diamonds["price"]
predictors = diamonds.drop("price", axis = 1)
display(dependent_variable.describe())
#display(predictors.columns)
display(predictors.head(5))


# Next, we need to split our data into a training and testing/validation dataset  as shown below. This will enable us to test our model on "unseen" data and evaluate how well it performs. 
# 
# To do the split, we can use `sklearn`'s `train_test_split` method that makes it quite easy to split the data.

# In[ ]:


# split our data into test/train
x_train, x_test, y_train, y_test = train_test_split(predictors, 
                                                    dependent_variable, test_size = 0.2, random_state = 101)
print(len(x_train),len(x_test), len(y_train), len(y_test) )


# Having split our data, we next fit our model. The approach we have taken is to predict the price using all features in the dataset. Such a model may or may not best explain our data. We shall see how to take a more informed "guess" later. 
# 
# Before we can proceed with building our model, we need to solve one more isssue. Some of our columns are of type `object` which is not suitable for regression. We therefore need to encode this data. Again, we take advantage of `sklearn`'s prepocessing features to help us in doing so.
# 
# ## Label Encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
x_train.dtypes


# In[ ]:


# apply the encoder on cut, color and clarity
# Instantiate the Encoder
encoder = LabelEncoder()
# select_if(is.object, col)
to_encode = x_train.select_dtypes(exclude = 'float').columns.values
x_train[list(to_encode)] = x_train[to_encode].apply(encoder.fit_transform)
x_test[list(to_encode)] = x_test[to_encode].apply(encoder.fit_transform)
x_train.head()


# Alternatively, we can simply convert our columns to panda's Categorical data type.

# In[ ]:


diamonds[to_encode].astype("category").apply(lambda x: x.cat.codes).head()


# ## The Model
# Having encoded our data, we can proceed with building our model. There are a number of libraries that have been written to achieve this but as alluded to earlier, we shall use `sklearn`. First we need to instantiate a regressor that we shall then use to fit our model. 

# In[ ]:


# make a model object
regressor = linear_model.LinearRegression()
# Use the regressor to fit our model
regressor.fit(x_train, y_train)


# We can now proceed to use our model for predictions.

# In[ ]:


# predict on x_test
predicted_price = regressor.predict(x_test)


# ## Model Evaluation
# 
# To understand why we need to evaluate our models and why metrics are important, one ought to go back in time to a rather famous [quote](https://stats.stackexchange.com/questions/57407/what-is-the-meaning-of-all-models-are-wrong-but-some-are-useful/57414):
# 
# >
# 
#     "Essentially, all models are wrong, but some are useful."
# 
#     --- Box, George E. P.; Norman R. Draper (1987). Empirical Model-Building and Response Surfaces, p. 424, Wiley. ISBN 0471810339.
#  
#  
#  Having said that, let us take a look at our model's R<sup>2</sup> on both the train and test datasets. This is known as the `score` in `sklearn`'s linear_model module. 

# In[ ]:


display("R Squared value(Train): ", regressor.score(x_train, y_train) * 100)
display("R Squared value(Test): ", regressor.score(x_test, y_test) * 100)


# # What's in the R squared?!
# However, [what does the R<sup>2</sup> tell us](http://people.duke.edu/%7Ernau/rsquared.htm)?! This value while not conclusive gives us an understanding of how much variance can be explained by our model. It is often assumed that the higher the R<sup>2</sup> value, the better the model. However, as you probably now know from the above link, this is often not the case.  A better metric is the adjusted R squared which we can calculate as shown here:

# In[ ]:


y_hat = predicted_price
# sum squared residuals(sse)
ssr = sum((y_test - y_hat)**2)
# sum squared total
sst = sum((y_test - np.mean(y_test)) **2 )
r_squared = 1-(ssr/sst)
# compute the adjusted r squared
adj_r_squared =    1 - (1-regressor.score(x_test, y_test))*(len(y_test)-1)/(len(y_test)- x_test.shape[1]-1)
print(r_squared, adj_r_squared)


# We can take the above a step further by using `sklearn`'s builtin metrics.

# In[ ]:


from sklearn import metrics


# In[ ]:


display("MSE: ", metrics.mean_squared_error(predicted_price, y_test))
display("MAE: ", metrics.mean_absolute_error(predicted_price, y_test))


# ## Towards Reality
# 
# From the above, one can easily see a major flaw in this approach. Surely, not all the dataset's features will have a major impact on the model. We can turn to looking at correlations between the variables to see which are more correlated and use these to build more realistic models.

# In[ ]:


corrs = diamonds.corr()

corrs


# We can plot the correlations based on one to one or one to many correlation.

# In[ ]:


sns.heatmap(data = corrs, square = True, cmap = ["dodgerblue", "lightgreen", "gray"], annot = True)


# ## Feature Importance and Selection
# 
# Having obtained these correlations, we can use them to decide which features are more highly correlated and therefore build our models based on these. We can utilise `sklearn`'s [`feature_selection`](https://scikit-learn.org/stable/modules/feature_selection.html) module to help us better get feature importance. This is "nice" since it will provide us with p values and their respective significance. From the documentation, `f_regression` attempts to:
# ```
# Univariate linear regression tests.
# 
# Linear model for testing the individual effect of each of many regressors.
# This is a scoring function to be used in a feature selection procedure, not
# a free standing feature selection procedure.
# 
# 
# 1. The correlation between each regressor and the target is computed,
#    that is, ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) *
#    std(y)).
# 2. It is converted to an F score then to a p-value.
# 
# ```

# In[ ]:


from sklearn.feature_selection import f_regression


# In[ ]:


f_value, p_value = f_regression(x_train, y_train)
data = pd.DataFrame([f_value, x_train.columns.values, p_value]).T
data.columns = ["f_value", "predictor", "p_value"]
sorted_data = data.sort_values(by = "p_value", ascending = False)
sorted_data


# Having sorted our f_values and p_values, we need to make a choice on what our cut off point is for a statistically significant p_value. Due to convention(Bayesians will likely disagree), I will set the cut off point to 0.05 and choose only those predictors with a p_value less than 0.05 as being statistically significant. We can make this fancy as follows.

# In[ ]:


sorted_data["signif" ] = np.where(sorted_data["p_value"] > 0.05 , "ns", "sf")
sorted_data[np.where(sorted_data["signif"] == "sf", True, False)].drop([10,0])


# From the above, it appears that the most likely(highly correlated) variables with price are cut,clarity, table and color. We shall therefore fit a new model with only these predictors and see how well it works, but first how does this result relate to the correlations we obtained above? 

# In[ ]:


to_index = list(sorted_data["predictor"].values)
# cannot use drop_dupes for some reason
del to_index[2]
# display correlations with price
corrs[[x for x in to_index if x in corrs.columns]].loc["price"]


# Looking at the above table, we can  see that carat, length, width and height happen to have a high correlation with  price. Therefore, we can build our models around these features and see how well it performs. Let us visualize these and see. 

# In[ ]:


# Points clustered together. Might be best to reduce this clustering
# Decided to filter for only highly priced diamonds
# perhaps add some noise(jitter?)

sns.lmplot(x = "carat", y = "price", data = diamonds.query("price >= 10000"), hue = "clarity")


# From the above plot, we see a generally highly linear relationship between price and clarity/carat. It is therefore possible that a model with these two features alone would likely provide strong predictive power for a dimaond's price. 
# 
# **It is worth mentioning however that linear regression does not mean a linear relationship between the target and the variables but rather the target and the coefficients of the variables ie in the equation y = mx + c, y is linearly correlated to m and not necessarily x.**
# 
# 
# ## Future Steps
# 
# It would be close to impossible if not impossible to fully analyse and cover all important parts of linear regression. I therefore conclude this kernel(notebook) by thanking you for your time. As a bonus, here are a few interesting future steps that lack of space unfortunately prevents us from exploring.
# 
# 1. Build Models based on the most highly correlated variables
# 
# 2. Compare the above models with a model that uses all variables in the dataset.
# 
# 3. Use a Bayesian approach to statistical modeling to see how well this would compare to what we have.
# 
# 4. Improve the visualizations.
# 
# 5. Experiment with more "advanced models"(knns, svms, random forests, xgboost, adaboost, etc)
# 
# 6. Deploy  a real world model and see how this works on an actual real world problem.
# 
# 
# **Thank you for reading and do provide feeback on what works and what doesn't.** You can contact me via my Github [repo](https://github.com/Nelson-Gon/sweetpy) or by email(via [Kaggle](https://www.kaggle.com/gonnel)'s) contact me .
# 

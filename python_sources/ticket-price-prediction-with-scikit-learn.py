#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The goal of this kernel is to go through the workflow of a simple ML task: predicting the price of a train ticket based on all the information surrounding it. We will cover:
# 
# * **Preprocessing**: How to clean parse dates, remove missing values, handle correlations, encoding categorical data, and splitting the dataset.
# * **Linear Regression**: A simple baseline model
# * **Linear SVM**: A slightly more sophisticated model that has been used extensively in the 90's.
# * **Light GBM**: An advanced model often used in Kaggle competitions, based on ensembling.

# In[ ]:


from lightgbm import LGBMRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.svm import LinearSVR


# # Preprocessing

# ## Load Data

# In[ ]:


df = pd.read_csv('../input/renfe.csv', index_col=0)

print(df.shape)
df.head()


# ## Expand Dates
# 
# Since the date is given in a string format, we will have to expand it into different columns: year, month, day, and day of the week.****

# In[ ]:


for col in ['insert_date', 'start_date', 'end_date']:
    date_col = pd.to_datetime(df[col])
    df[col + '_hour'] = date_col.dt.hour
    df[col + '_minute'] = date_col.dt.minute
    df[col + '_second'] = date_col.dt.second
    df[col + '_weekday'] = date_col.dt.weekday_name
    df[col + '_day'] = date_col.dt.day
    df[col + '_month'] = date_col.dt.month
    df[col + '_year'] = date_col.dt.year
    
    del df[col]


# In[ ]:


df.head()


# ## Removing Missing Values
# 
# Let's take a look at all the columns with missing values, and decide whether to remove any row based on missing values.

# In[ ]:


df.isnull().sum()


# The missing values are pretty consistent and pretty isolated (only 300k samples out of 2M). Since we are doing price prediction, we can simply drop them.

# In[ ]:


df.dropna(inplace=True)


# ## Finding unique columns
# 
# If a certain column only contains one category of value, then we will drop it. This will also tell us either we have continuous or categorical data.

# In[ ]:


for col in df.columns:
    print(col, ":", df[col].unique().shape[0])


# We see that all the data is categorical in this case. We can one-hot-encode them afterwards.
# 
# Also, it seems like there is only one year in the dataset. We can safely drop that column.

# In[ ]:


columns_to_drop = [col for col in df.columns if df[col].unique().shape[0] == 1]
df.drop(columns=columns_to_drop, inplace=True)


# In[ ]:


df.head()


# ## Observing correlation
# 
# If two columns are highly correlated, we can safely drop them.

# In[ ]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# The only highly correlated feature we can observe is the between the start and end date (both day and month). We can drop off one of each.

# In[ ]:


df.drop(columns=['end_date_day', 'end_date_month'], inplace=True)


# ## Examining Price
# 
# Let's take a closer at price, since it seems to be numerical, but with a certain number of categories.

# In[ ]:


price_freq = df['price'].value_counts()
price_freq.head()


# In[ ]:


price_freq.tail()


# Although price is categorical, there's an important imbalance in the dataset. Prices such as \$76.30 is extremely frequent, whereas $68.97 appears only once. This is likely because the former is a standard price, and the latter is a one-time discounted price.
# 
# It is therefore wise to try to predict a numerical price rather than a making it a classification problem. We can now split the data into `X` and `y`

# In[ ]:


X_df = df.drop(columns='price')
y = df['price'].values


# ## One Hot Encoding
# 
# We will need to process the categorical data to be ready for input. The usual way to do that is to use `pd.get_dummies` or `sklearn.preprocessing.OneHotEncoder`. The former is more polyvalent, but the latter lets you output a sparse matrix instead of a regular numpy array. This is good for saving memory.

# In[ ]:


encoder = OneHotEncoder()
X = encoder.fit_transform(X_df.values)
X


# Let's take a look at the categories learned by our encoder.

# In[ ]:


for category in encoder.categories_:
    print(category[:5])


# ## Splitting into Train and Test Set

# Since we have so many data points, we will only use 10% of the data as test set.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=2019
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Linear Regression
# 
# We will start with a linear regression, perhaps the simplest algorithm you can use for predicting a numerical value. We will use the [scikit-learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = LinearRegression()\nmodel.fit(X_train, y_train)')


# In[ ]:


train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("Train Score:", train_score)
print("Test Score:", test_score)


# What is the `model.score` for a Linear Regression? According to the [sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression):
# 
# > The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.
# 
# In other words,
# 
# $$
# SS_{tot} = \sum_i (y_i - \bar{y})^2 \\
# SS_{res} = \sum_i (y_i - f_i)^2 \\
# R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
# $$
# 
# where $\bar{y}$ is the mean, $y_i$ is the true value for $i$, and $f_i$ is the predicted value for $i$, where $i$ is a data point.
# 
# *But how can we interpret our results?* According to this formula, the variance from the true value of our model is about 8x smaller than the variance of the true value from the mean. This is not bad, but let's take a look at the an actual metric that is related to variance, i.e. the MSE:

# In[ ]:


def compute_mse(model, X, y_true, name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    print(f'Mean Squared Error for {name}: {mse}')
    
compute_mse(model, X_train, y_train, 'training set')
compute_mse(model, X_test, y_test, 'test set')


# The MSE is pretty high, considering that the mean is only:

# In[ ]:


y_train.mean()


# This teaches us to not trust a single evaluation metric! Therefore, there is still some room for improvement.

# ## Aside: Defining an evaluation function

# Instead of repeating ourselves, we will build a simple function called `evaluate`, which will print the score and MSE of our models on both the training and test sets.

# In[ ]:


def build_evaluate_fn(X_train, y_train, X_test, y_test):
    def evaluate(model):
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        print("Train Score:", train_score)
        print("Test Score:", test_score)
        print()
        
        compute_mse(model, X_train, y_train, 'training set')
        compute_mse(model, X_test, y_test, 'test set')
    
    return evaluate

evaluate = build_evaluate_fn(X_train, y_train, X_test, y_test)


# # SVM
# 
# Next, we will try out a linear SVM, which is a popular algorithm [invented in the 60s by Vladimir Vapnik, and refined in the 90s by Corinna Cortes and again Vladimir Vapnik](https://en.wikipedia.org/wiki/Support-vector_machine). Although we will be using the linear models, there are multiple types of SVM that exists, which is explained in detail in the [scikit-learn docs](https://scikit-learn.org/stable/modules/svm.html). Here are some simple examples retrieved from the docs:
# 
# ![image](https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_svc_0011.png)
# 
# 
# We will be using the LinearSVR model, which is [documented here](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html).

# In[ ]:


get_ipython().run_cell_magic('time', '', 'svm = LinearSVR()\nsvm.fit(X_train, y_train);')


# In[ ]:


evaluate(svm)


# # Gradient Boosting
# 
# We will now try to use a gradient boosting machine, which is a method that combines a collection of trees (i.e. an ensemble method) to make a well-supported prediction. The wikipedia page on Gradient Boosting offers a really nice [informal introduction](https://en.wikipedia.org/wiki/Gradient_boosting#Informal_introduction) to the algorithm. We will be using the [LightGBM API](https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMRegressor), an efficient and lightweight implementation of the [original algorithm](https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting). It is described in [this paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf).
# 
# The number of trees used depends on many different factors, including dimensionality and size of dataset. It is usually a good idea to determine the optimal number of trees through Cross-Validation. We will go with 1000 trees, but feel free to try a different number.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'gbr = LGBMRegressor(n_estimators=1000)\ngbr.fit(X_train, y_train)')


# In[ ]:


evaluate(gbr)


# # Conclusion
# 
# We went through a simple workflow for preprocessing the dataset, then encoding and splitting it into training and test set. We then tested 3 different algorithms, i.e. a Linear Regression, an SVM, and Gradient Boosting. We can observe that gradient boosting, in this case, is not only faster, but significantly more accurate.

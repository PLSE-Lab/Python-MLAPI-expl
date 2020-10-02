#!/usr/bin/env python
# coding: utf-8

# # Motivation
# I recently just got a job at a big pharmaceutical company. My task is mainly to challenge the forecasts that the financial department has made for the budgets.
# The company is distributing over 100 different products, so the task is likely to be comparable to this competition.
# 
# I hope I will gain some more insight in the forecast modelling and keep improving my skill as a forecast analyst.
# 
# Any feedback is much appreciated.

# ## The beginning
# As with any other task, one is going to need some basic tools to get started.
# ### Libraries

# In[ ]:


# Data processing
import numpy as np
import pandas as pd

# Data visualization
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Eliminating unnecessary sklearn warnings
import warnings
warnings.filterwarnings('ignore')


# ## Functions
# The following function are taken from [Pedro Marcelino](https://www.kaggle.com/pmarcelino)'s kernel, [Data analysis and feature extraction with Python
# ](https://www.kaggle.com/pmarcelino/data-analysis-and-feature-extraction-with-python/notebook).

# In[ ]:


# Create table for missing data analysis
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data


# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[ ]:


# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid() 
    plt.xscale('log')
    plt.legend(loc='best') 
    plt.xlabel('Parameter') 
    plt.ylabel('Score') 
    plt.ylim(ylim)


# ## The data and the objective
# In this competition there is provided a relatively simple and clean dataset for 50 different items at 10 different stores.
# The data includes a training set with 5 years of historical data as well as a test set for 3 months.
# 
# *The objective is to predict the future 3 months of demand.*
# 
# Let's dig into the data.

# In[ ]:


df = pd.read_csv('../input/train.csv') # Load the data and save as df
df_raw = df.copy() # Save the data as raw format just in case
df.head()


# The data does indeed look very simple. Here's my definition and quick thoughts:
# - **Date:** Defined as a date. There may be a seasonal effect.
# - **Store:** The store number of where the sales were made, which goes from 1-10. Some stores may be better at selling than other stores (e.g. better local marketing, better staff etc.).
# - **Item:** The item number which goes from 1-50. Some items may have bigger sales than others (e.g. lower price, quality etc.).
# - **Sales:** The number of sales. This is our target variable.
# 
# Quick follow-up: Date may indicate seasonal effects; Some stores may be better at selling; Some items may be easier to sell.
# 
# Let's take a closer look at the data, namely the descriptives of the features.

# In[ ]:


df.describe()


# **Count:** there is obviously no missing data, as the count is equal for each column.
# 
# **Mean:** Nothing out of the ordinary here.
# 
# **Min & Max:** Nothing out of the ordinary here, as minimum sales aren't negative, nor do we have store or item below 1. Same goes for Max.
# 
# Overall the data looks simple and clean as the competition description also stated.
# 
# To confirm that there is no missing data, I can use Pedro Marcelino's function which draws a data table for the missing data.

# In[ ]:


# Draw a data table showing the missing data as percentage of total
draw_missing_data_table(df)


# There is indeed no gaps to fill.
# 
# # Minimal Viable Model
# As in any other case, it is always good to have some sort of beta version of your product, so that you can test it and benchmark it as development continues.
# 
# I start by preparing the data, then fitting the data to a multivariate regression model and finally analyze the performance of the model through learning and validation curves.
# 
# ## Preparing the data

# In[ ]:


df.dtypes


# - Date should be of date type, however this need to be parsed before used in modelling, so we drop it for now.
# - Store and item should be of categorical type.
# - Sales will not be considered because it is our target variable.

# In[ ]:


# Define date as date.
df['date'] = pd.to_datetime(df['date'])
df.drop('date', axis=1, inplace=True)


# In[ ]:


# Define store and item as categorical
df['store'] = pd.Categorical(df['store'])
df['item'] = pd.Categorical(df['item'])
df.head()


# ## Launching the model
# Let's get ready for take off!

# In[ ]:


# Transform categorical variables into dummy variables
df = pd.get_dummies(df, drop_first=True)  # To avoid dummy trap
df.head()


# In[ ]:


# Create data set to train data imputation methods
X = df[df.loc[:, df.columns != 'sales'].columns]
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)


# In[ ]:


# Debug
print('Inputs: \n', X_train.head())
print('Outputs: \n', y_train.head())
print(X.head())


# In[ ]:


# Fit linear regression
sgdreg = SGDRegressor()
sgdreg.fit(X_train, y_train)


# In[ ]:


# Model performance
scores = cross_val_score(sgdreg, X_train, y_train, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# ## Assessing Model Performance

# In[ ]:


# Plot learning curves
title = "Learning Curves (Linear Regression)"
cv = 10
plot_learning_curve(estimator=sgdreg, title=title, X=X_train, y=y_train, cv=cv, n_jobs=1);


# From the learning curve we see that the training and validation score eventually converges as the more training examples are used.
# This means that our model seems to fit well and isn't overfitted nor underfitted.
# 
# Let's check out the validation curve and see if we can optimize through parameter tuning.

# In[ ]:


# Plot validation curve
title = 'Validation Curve (Linear Regression)'
param_name = 'alpha'
param_range = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3] 
cv = 10
plot_validation_curve(estimator=sgdreg, title=title, X=X_train, y=y_train, param_name=param_name, param_range=param_range);


# ### Conclusion so far
# With the basic features our model achieves 66.5% +- 0.2%. Our model seems to be fit well, and perform best when alpha is lower than .001.

# # Let the real fun begin!
# We now have our basic model, which is modelled using the SGDRegressor. The SGDRegressor performs very well using large datasets.
# 
# For now, let's do some explorative analysis.

# In[ ]:


# Restart data set
df = df_raw.copy()
df.head()


# In[ ]:


df.dtypes


# In[ ]:


# Define date as date.
df['date'] = pd.to_datetime(df['date'])


# In[ ]:


# Define store and item as categorical
df['store'] = pd.Categorical(df['store'])
df['item'] = pd.Categorical(df['item'])
df.dtypes


# ## Items
# Some items may have a higher sell volume, due to greater demand or lower pricing. Let's see if some of the items sells better than some other.

# In[ ]:


plt.rcParams['figure.figsize']=(20,5)
sns.barplot(df['item'], df['sales'])


# Clearly some items sell better than others. We keep 'item' as a categorical feature as it is for now.
# 
# ## Stores
# Let's do the same for stores. Some stores may sell better due to geographic position, marketing, better staff etc.

# In[ ]:


plt.rcParams['figure.figsize']=(10,5)
sns.barplot(df['store'], df['sales'])


# Seems like the logic hold for this one as well. I keep the 'store' feature as it is for now.
# 
# ## Date
# The last feature to check is the date. Let's plot the date and sales and see whether there's a seasonal effect.

# In[ ]:


g = sns.FacetGrid(df, col="item", col_wrap=5)
g = g.map(plt.scatter, "date", "sales", marker="o", s=1, alpha=.5)


# It seems like every product has a peak time during mid year, so the date should have some predictive power. Let's zoom closer and see if we can define which periods demand is peaking and stagnant. We use the most recent year historical data for this.

# In[ ]:


df['month'] = df.date.dt.month
df['month'] = pd.Categorical(df['month'])
df.head()


# In[ ]:


sns.barplot(df['month'],df['sales']);


# It seems like the peak season is during April through August. Maybe I could set a categorical feature for peak time, but I will keep the monthly category as it is for now.

# In[ ]:


df.dtypes


# So far I have just worked with the features that were given in the dataset. Further feature extraction should be examined and exploited if usable.
# 
# ## Moving Averages
# Moving averages is commonly used in time series to smooth out the daily noise of the data, and can highlight short- and long term trends.
# 
# I will make use of a few different moving averages for further analysis. The types will include:
# - Simple Moving Average
# - Exponential Moving Average
# 
# As the pattern of sales seems to be pretty similiar for each year, I will shift the moving averages by a year, so the test data will have these data available.
# I choose a 3 month period (90 days), as this is the goal of prediction.

# In[ ]:


# Add SMA to dataframe
sma = df.groupby(['item', 'store'], as_index=False).apply(lambda x: x['sales'].rolling(90).mean().shift(365))
df['sma'] = sma.reset_index(level=0, drop=True)

# Add EMA to dataframe
ema = df.groupby(['item', 'store'], as_index=False).apply(lambda x: x['sales'].ewm(span=90, adjust=False).mean().shift(365))
df['ema'] = ema.reset_index(level=0, drop=True)

df.head()


# Let's have a look how the SMA and EMA follows the sale volume.

# In[ ]:


df_latest_year_one_item = df.loc[(df['date'] > '2014-12-31') & (df['item'] == 1) & (df['store'] == 10)]
plt.figure(figsize=(20,5))
plt.plot(df_latest_year_one_item[['sales','sma', 'ema']])
plt.legend(['sales','sma','ema'])
plt.show()


# As we can see from the above plot, the Simple Moving Average smooths all the noise, whereas our exponential reacts a little bit faster to trend movements.
# Note that the moving averages were shifted 365 days, but still it aligns pretty close to the actual means.
# 
# I would expect that these moving averages would be informative for our model.
# 
# One drawdown of the shifted effect, is that it leaves our first year with NaN values. I don't see much else to do, than dropping the first year.
# This shouldn't be so much of a problem, since we saw that our Minimum Viable Model were pretty good fit already after just 200.000 examples.

# In[ ]:


df.isna()['sma'].sum()


# We will be dropping 227.000 rows. I may have to sit down and think about how this can be avoided.

# In[ ]:


df.dropna(inplace=True)
df.isna()['sma'].sum()


# ## So how's the model now?
# So let's make a quick recap. The following thing has changed since we ran our Minimum Viable Model.
# - **Month:** was added as a categorical feature, going from 1-12.
# - **Moving Averages:** Simple and Exponential Moving Averages were added and shifted 365 days.
# Let's run the model and see how it performs.

# In[ ]:


# Drop date
df.drop('date', axis=1, inplace=True)


# In[ ]:


# Transform categorical variables into dummy variables
df = pd.get_dummies(df, drop_first=True)  # To avoid dummy trap


# In[ ]:


# Create data set to train data imputation methods
X = df[df.loc[:, df.columns != 'sales'].columns]
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)


# In[ ]:


# Debug
print('Inputs: \n', X_train.head())
print('Outputs: \n', y_train.head())


# In[ ]:


# Scale the features
columns_to_scale = ['sma', 'ema']
mean = X_train[columns_to_scale].mean()
std = X_train[columns_to_scale].std()
X_train[columns_to_scale] = (X_train[columns_to_scale] - mean) / std

# Fit linear regression
sgdreg = SGDRegressor(alpha=0.0001)
sgdreg.fit(X_train, y_train)


# In[ ]:


# Model performance
scores = cross_val_score(sgdreg, X_train, y_train, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# In[ ]:


# Plot learning curves
title = "Learning Curves (Linear Regression)"
cv = 10
plot_learning_curve(estimator=sgdreg, title=title, X=X_train, y=y_train, cv=cv, n_jobs=1);


# In[ ]:


# Plot validation curve
title = 'Validation Curve (Linear Regression)'
param_name = 'alpha'
param_range = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3] 
cv = 10
plot_validation_curve(estimator=sgdreg, title=title, X=X_train, y=y_train, param_name=param_name, param_range=param_range);


# # Conclusion
# - The Minimal Viable Model performs with an 66.5% +- 0.2% accuracy, whereas the final model perform with 85.3% +- 0.1% accuracy on the training data.
# - The cross validation indicates a good fit, so neither under- or overfit.
# - We have added a month category and an SMA and EMA.
# 
# Let's finish the model and submit the results.

# In[ ]:


# Restart data set
df = df_raw.copy()
df_test_raw = pd.read_csv('../input/test.csv') # Load the test data and save as df_test
df_test = df_test_raw.copy()

combine = [df, df_test]


# In[ ]:


# Define date as date.
for dataset in combine:
    dataset['date'] = pd.to_datetime(dataset['date'])


# In[ ]:


# Define store and item as categorical
for dataset in combine:
    dataset['store'] = pd.Categorical(dataset['store'])
    dataset['item'] = pd.Categorical(dataset['item'])


# In[ ]:


# Define month as categorical
for dataset in combine:
    dataset['month'] = dataset.date.dt.month
    dataset['month'] = pd.Categorical(dataset['month'])

df_test.dtypes


# In[ ]:


# Add SMA to dataframe
sma = df.groupby(['item', 'store'], as_index=False).apply(lambda x: x['sales'].rolling(90).mean())
df['sma'] = sma.reset_index(level=0, drop=True)

# Add EMA to dataframe
ema = df.groupby(['item', 'store'], as_index=False).apply(lambda x: x['sales'].ewm(span=90, adjust=False).mean())
df['ema'] = ema.reset_index(level=0, drop=True)


# In[ ]:


# Adding last 3 months of sma and ema from training to test data
sma_test = df.loc[(df['date'] < '2017-04-01') & (df['date'] >= '2017-01-01')]['sma'].reset_index(drop=True)
df_test['sma'] = sma_test
ema_test = df.loc[(df['date'] < '2017-04-01') & (df['date'] >= '2017-01-01')]['ema'].reset_index(drop=True)
df_test['ema'] = ema_test

# Shifting SMA and EMA on training data
df['sma'] = df['sma'].shift(365)
df['ema'] = df['ema'].shift(365)

df_test.head()


# In[ ]:


# Drop date and id
df.drop('date', axis=1, inplace=True)
df.dropna(inplace=True)
df_test.drop('date', axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)


# In[ ]:


# Transform categorical variables into dummy variables
df = pd.get_dummies(df, drop_first=True)  # To avoid dummy trap
df_test = pd.get_dummies(df_test, drop_first=True)  # To avoid dummy trap

# Add month 4-12 to df_test
df_test = df_test.join(pd.DataFrame(
    {
        'month_4': 0,
        'month_5': 0,
        'month_6': 0,
        'month_7': 0,
        'month_8': 0,
        'month_9': 0,
        'month_10': 0,
        'month_11': 0,
        'month_12': 0
    }, index=df_test.index
))
df_test.head()


# In[ ]:


# Prepare data for model
X_train = df[df.loc[:, df.columns != 'sales'].columns]
y_train = df['sales']
X_test = df_test

# Scale the features
columns_to_scale = ['sma', 'ema']
mean = X_train[columns_to_scale].mean()
std = X_train[columns_to_scale].std()
X_train[columns_to_scale] = (X_train[columns_to_scale] - mean) / std
X_test[columns_to_scale] = (X_test[columns_to_scale] - mean) / std


# In[ ]:


# Run SGD Regression
sgdreg = SGDRegressor(alpha=0.0001)
sgdreg.fit(X_train, y_train)

# Get prediction
prediction = sgdreg.predict(X_test)


# In[ ]:


# Add to submission
submission = pd.DataFrame({
        "id": df_test_raw['id'],
        "sales": prediction
})


# In[ ]:


# Quick look at the submission
submission.head()


# In[ ]:


# Save submission
submission.to_csv('submission.csv',index=False)


# # Last Words
# So I recently just started digging into data analysis, and this is what I've learned so far. There is probably multiply ways to improve my model, and if you as a reader got any suggestions, please let me know.
# 
# All feedback is much appreciated. Thank you for taking your time to read my analysis.

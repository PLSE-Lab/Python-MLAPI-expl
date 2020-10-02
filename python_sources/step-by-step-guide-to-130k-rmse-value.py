#!/usr/bin/env python
# coding: utf-8

# # 1. Intro
# 
# This is kernel with solution for _House Sales in King County, USA_ dataset made by Piotr Podbielski on the 2nd of April, 2019.
# 
# # 2. Solution
# Steps taken:
# 1. Load the libraries and the dataset
# 2. Analyze the dataset
# 3. Build a couple of models
# 4. Summary of the experiments
# 
# ## 2.1. Load the libraries and the dataset

# In[ ]:


# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Install required packages
get_ipython().system('pip install wheel matplotlib pandas scikit-learn xgboost seaborn hyperopt')


# In[ ]:


# Import libraries and set settings of seaborn (lib for nicer plotting)
from math import sqrt

from hyperopt import hp, tpe, STATUS_OK, Trials
from hyperopt.fmin import fmin
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns;
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
import xgboost as xgb

sns.set(rc={'figure.figsize': (11.7, 8.27)})


# In[ ]:


# Specify the filename
file_name = "../input/kc_house_data.csv"

# Read dataset to pandas dataframe (also parse date to datetime object and set column #0 as index for dataframe)
df = pd.read_csv(file_name, parse_dates=[1], date_parser=lambda x: pd.to_datetime(x, format='%Y%m%dT000000'), index_col=0)


# ## 2.2. Analyze the dataset

# In[ ]:


# Brief overview of the dataset
df.head()


# It looks like we don't have any nominal features (features which type are strings), so there is no need to convert anything to dummy variables (also known as one-hot encoding).

# In[ ]:


# How many examples and features do we have in the dataset?
df.shape


# Number of almost 22k examples looks good for the future of my model building. That amount of data is good for some basic models (for sure it wouldn't be for deep neural networks w/o regularization).

# In[ ]:


# What are the types of the features?
df.dtypes


# We will have to deal with that `datetime64` type of `date` feature, but let's leave it for later.

# In[ ]:


# Check if we need to handle with some NA values in dataframe
df.apply(lambda x: sum(x.isnull()) / len(df))


# The dataset doesn't have NA values. What a relief! :)

# In[ ]:


# Show basic statistics about our dataframe's columns (columns are ours features)
df.describe()


# The table presented above is quite raw. The plots are more assimilable way of getting knowledge about data.

# In[ ]:


# Plot histograms of the features
df.hist(bins=50, figsize=(20, 15));


# Before we categorise the features in four variable types: `continuous`, `nominal`, `dichotomous` and `ordinal`, we will look at the `date` feature, because it wasn't plotted above.

# In[ ]:


# Plot the histogram of pricing dates grouped by years and months
df['date'].groupby([df["date"].dt.year, df["date"].dt.month]).count().plot(kind="bar");


# The pricing dates were from 05.2014 to 05.2015.
# 
# Let's handle that date feature. We will create three new features based on it: year of pricing, month of pricing and concatenated year and month.

# In[ ]:


# Replace date feature with new ones
df['pricing_yr'] = df['date'].dt.year
df['pricing_month'] = df['date'].dt.month
df['pricing_yrmonth'] = df['date'].dt.year.map(str) + df['date'].dt.month.map('{:02d}'.format)
df['pricing_yrmonth'] = df['pricing_yrmonth'].astype(int)
df = df.drop(['date'], axis=1)


# We did such feature transformation, because pricing date might contain some valuable information.

# In[ ]:


# Create a dataframe for showing purposes with feature name and feature type
feature_type = ["discrete", "discrete", "continuous", "continuous", "discrete",
                "dichotomous", "ordinal", "ordinal", "ordinal", "continuous",
                "continuous", "discrete", "discrete", "discrete", "continuous", "continuous",
                "continuous", "continuous", "discrete", "ordinal", "discrete*"]
feature_array = df.drop(['price'], axis=1).columns.values
pd.DataFrame({"feature": feature_array, "type": feature_type})


# **\*** - some types of features are quite hard to define or could be easily treated as one of two types.
# 
# Variable types explanations are below.
# 
# > Most variables in a data set can be classified into one of two major types.
# >
# > *Numerical variables*
# >
# > The values of a numerical variable are numbers. They can be further classified into discrete and continuous variables.
# >
# > * A **continuous variable** is a numeric variable. Observations can take any value between a certain set of real numbers.
# > * A **discrete variable** is a numeric variable. Observations can take a value based on a count from a set of distinct whole values. A discrete variable cannot take the value of a fraction between one value and the next closest value.
# >
# > *Categorical variables*
# >
# > The values of a categorical variable are selected from a small group of categories.Categorical variables can be further categorized into ordinal and nominal variables.
# >
# > * An **ordinal variable** is a categorical variable. Observations can take a value that can be logically ordered or ranked. The categories associated with ordinal variables can be ranked higher or lower than another, but do not necessarily establish a numeric difference between each category.
# >
# > * A **nominal variable** is a categorical variable. Observations can take a value that is not able to be organized in a logical sequence.
# 
# Source: https://www.quora.com/What-are-some-types-of-features-for-machine-learning/answer/Pavan-Kumar-Kota-4
# 
# 
# There is one more left variable type:
# 
# A **dichotomous variable** is one that takes on one of only two possible values when observed or measured.
# 
# Source: http://methods.sagepub.com/reference/the-sage-encyclopedia-of-social-science-research-methods/n239.xml

# In[ ]:


# Plot the correlation of every two variables
sns.pairplot(df);


# There is a lot of small plots, but please open it in a new window and try to find some correlations.
# 
# For sure `bedrooms` and `sqft_living` features correlates well with `price`. `grade` and `sqft_above` features also looks promising.
# 
# What's alarming is this one lonely point in the second plot of the first row. It looks like outlier, so we're going to remove it.

# In[ ]:


# Remove the outlier
df = df[df["bedrooms"] < 20]


# Take a look if the correlation of bedroom and price has changed.

# In[ ]:


sns.jointplot(x="bedrooms", y="price", data=df, kind="reg");


# After removal of outlier the correlation doesn't look so awesome.

# In[ ]:


# Count the pearson correlation of features
corr_matrix = df.corr()
corr_matrix["price"].sort_values(ascending=False)


# As we can see `pricing_*` features have terrible correlation, `bedrooms` ain't so good, too. Let's leave them for now and plot the heatmap of correlations, so maybe some new things will come up.

# In[ ]:


# Source: https://seaborn.pydata.org/examples/many_pairwise_correlations.html

# Compute the correlation matrix
# Plot figsize
fig, ax = plt.subplots(figsize=(20, 20))
# Generate Color Map
colormap = sns.diverging_palette(220, 10, as_cmap=True)
# Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr_matrix, cmap=colormap, annot=True, fmt=".2f")
# Apply xticks
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns);
# Apply yticks
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
# Show plot
plt.show()


# The decision of adding `pricing_*` features wasn't good. Also, because `zipcode` and `lat` with `long` can be linked together, let's remove that features, and `pricing_*` features, too.
# 

# ## 2.3. Build a couple of models

# In[ ]:


# Remove pricing and zipcode features
df = df.drop(['pricing_yr', 'pricing_yrmonth', 'pricing_month', 'zipcode'], axis=1)


# In[ ]:


# Prepare data for model
Y = df['price'].values
X = df.drop(['price'], axis=1).values

# Split dataset on: train (for cross-validation) and test (hold-out) sets 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)


# #### Simple "train average set" model

# Let's imagine a model, which predicts for every input the average of price from train set as a result.
# 
# That simple model can be a basic benchmark for our next models. We do that because we don't have any assumptions how good our model should be. So for sure it shouldn't be worse than this simple model.
# 
# 

# In[ ]:


# Simple naive model with Cross-Validation on train set
root_mean_squared_errors = []

# When we compare models it's good to use CV, because it's going to be robust for randomness of train/val split.
kf = KFold(n_splits=10)
for train_indices, val_indices in kf.split(X_train):
    predictions = [Y_train[train_indices].mean()] * Y_train[val_indices].shape[0]
    actuals = Y_train[val_indices]
    root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))

rmse = np.mean(root_mean_squared_errors)
print("average of CV rmse score: {0:.0f}".format(rmse))


# RMSE (root mean squared error) is one of the common options of metric for the regression task.
# 
# Others are:
# * R^2
# * Mean Absolute Error
# 
# What distinguishes RMSE in regard to MAE is that RMSE will make price differences even bigger (squares them), so smaller differences will commit less than bigger ones.
# 
# More info here: https://en.wikipedia.org/wiki/Root-mean-square_deviation, https://en.wikipedia.org/wiki/Mean_absolute_error, https://en.wikipedia.org/wiki/Coefficient_of_determination
# 
# 

# #### Linear regression

# In[ ]:


# Train Linear Regression with Cross-Validation on train set
root_mean_squared_errors = []

kf = KFold(n_splits=10)
for train_indices, val_indices in kf.split(X_train):
    linreg_model = LinearRegression(n_jobs=-1).fit(X_train[train_indices], Y_train[train_indices])
    predictions = linreg_model.predict(X_train[val_indices])
    actuals = Y_train[val_indices]
    root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))

rmse = np.mean(root_mean_squared_errors)
print("average of CV rmse score: {0:.0f}".format(rmse))


# _Linear regression_ reduces RMSE by ~36% versus _simple "train average set" model_.
# I know it's quite obvious and expectable. :)
# 
# Let's now try some more sophisticated methods like trees.

# #### Random Forest

# In[ ]:


# Train RandomForest model with Cross-Validation on train set
root_mean_squared_errors = []

kf = KFold(n_splits=10)
for train_indices, val_indices in kf.split(X_train):
    rf_model = RandomForestRegressor(n_jobs=-1).fit(X_train[train_indices], Y_train[train_indices])
    predictions = rf_model.predict(X_train[val_indices])
    actuals = Y_train[val_indices]
    root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))

rmse = np.mean(root_mean_squared_errors)
print("average of CV rmse score: {0:.0f}".format(rmse))


# Out of the box _Random Forest regressor_ gives even better results. Almost ~35% better than _linear regression_. The random forest method is ensemble method, so it uses not one decision tree, but plenty of it. Let's change the number of trees (estimators) from default 10 to 100 and see if the result changes.

# #### Random Forest with `n_estimators=100`

# In[ ]:


# Train RandomForest model with Cross-Validation on train set
root_mean_squared_errors = []

kf = KFold(n_splits=10)
for train_indices, val_indices in kf.split(X_train):
    rf_model = RandomForestRegressor(n_jobs=-1, n_estimators=100).fit(X_train[train_indices], Y_train[train_indices])
    predictions = rf_model.predict(X_train[val_indices])
    actuals = Y_train[val_indices]
    root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))

rmse = np.mean(root_mean_squared_errors)
print("average of CV rmse score: {0:.0f}".format(rmse))


# It again dropped by ~6.3%. I will also try some other tree model, which XGBoost is. For now I am not familiar with how it works under the hood, but it's always frequent choice in competitions on Kaggle platform.

# #### XGBoost Regressor

# In[ ]:


# Train XGBRegressor model with Cross-Validation on train set
root_mean_squared_errors = []

kf = KFold(n_splits=10)
for train_indices, val_indices in kf.split(X_train):
    xgb_model = xgb.XGBRegressor(n_jobs=-1, n_estimators=300).fit(X_train[train_indices], Y_train[train_indices])
    predictions = xgb_model.predict(X_train[val_indices])
    actuals = Y_train[val_indices]
    root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))

rmse = np.mean(root_mean_squared_errors)
print("average of CV rmse score: {0:.0f}".format(rmse))


# _XGBoost_ also decreases RMSE a little bit.
# 
# Linear regression was linear model, trees are non-linear class of models, but they split space by hyperplanes, so let's try some model with explicitly given non-linearity functions - a MLP model.
# 
# Below there is a drawing how linear regression and trees split space.
# 
# Linear Regression | Decision Tree
# - | - 
# ![](https://littleml.files.wordpress.com/2016/06/lr_boundary_linear.png?w=244&h=244&crop=1) | ![](https://littleml.files.wordpress.com/2016/06/model_boundary_linear.png?w=244&h=244&crop=1)
# 
# ---
# Source: https://littleml.files.wordpress.com/2016/06/lr_boundary_linear.png?w=244&h=244&crop=1

# #### MLP model

# In[ ]:


# Train MLP model with Cross-Validation on train set

# Do rescaling as NN models like features from 0-1 range
scaler = MinMaxScaler(copy=True)
scaler.fit(X_train)

# Enscapsulate model training and evaluation in function
def train_mlp(X_train, Y_train, params={}, n_splits=10):
    root_mean_squared_errors_train = []
    root_mean_squared_errors = []

    kf = KFold(n_splits=n_splits)
    for train_indices, val_indices in kf.split(X_train):
        mlp_model = MLPRegressor(**params).fit(X_train[train_indices], Y_train[train_indices])
        predictions = mlp_model.predict(X_train[val_indices])
        actuals = Y_train[val_indices]
        root_mean_squared_errors_train.append(sqrt(mlp_model.loss_))
        root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))

    rmse_train = np.mean(root_mean_squared_errors_train)
    rmse_valid = np.mean(root_mean_squared_errors)
    
    return rmse_train, rmse_valid
    
X_train_scaled = scaler.transform(X_train)

rmse_train, rmse_valid = train_mlp(X_train_scaled, Y_train, n_splits=2)
print("average of CV rmse score on train set: {0:.0f}".format(rmse_train))
print("average of CV rmse score on val set: {0:.0f}".format(rmse_valid))


# As you can see MLP regressor without hypertuning works terrible. Also, we've started to printing two of the losses - on train and validation set to have some infomation about possible overfitting. Let's use auto hyper-parameters search to find proper hyperparameters.
# 
# The space of hyperparameters will be selected manually.

# In[ ]:


def objective(space):
    hidden_layers = tuple(int(space['hidden_layers']) * [int(space['hidden_neurons'])])
    
    rmse_train, rmse_valid = train_mlp(X_train_scaled, 
                                       Y_train, 
                                       params={'solver': 'adam',
                                               'hidden_layer_sizes': hidden_layers,
                                               'activation': space['activation'],
                                               'shuffle': True,
                                               'max_iter': int(space['max_iter']),
                                               'learning_rate_init': space['learning_rate_init'],
                                               'verbose': False},
                                       n_splits=2)

    return {'loss': rmse_valid, 'status': STATUS_OK}

activation_functions = ['relu', 'tanh', 'logistic']
space = {
    'activation': hp.choice('activation', activation_functions),
    'hidden_neurons': hp.quniform('hidden_neurons', 10, 50, 10),
    'hidden_layers': hp.quniform('hidden_layers', 2, 4, 1),
    'learning_rate_init': hp.loguniform('learning_rate_init', np.log(0.001), np.log(0.01)),
    'alpha': hp.loguniform('alpha', np.log(0.01), np.log(0.1)),
    'max_iter': hp.quniform('max_iter', 100, 750, 50)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=15,
            trials=trials)


# Let's plot the loss change in time.

# In[ ]:


sns.lineplot(x=np.arange(0, len(trials.losses())), y=trials.losses());


# And print the best hiperparameters set.

# In[ ]:


best


# Now run the MLP classifier again with parameters above and `n_splits=10`, so I could fairly compare it to models I've trained before.

# In[ ]:


best['activation'] = activation_functions[best['activation']]

hidden_layers = tuple(int(best['hidden_layers']) * [int(best['hidden_neurons'])])
rmse_train, rmse_valid = train_mlp(X_train_scaled, Y_train, 
                                   params={'solver': 'adam',
                                           'hidden_layer_sizes': hidden_layers,
                                           'activation': best['activation'],
                                           'shuffle': True,
                                           'max_iter': int(best['max_iter']),
                                           'learning_rate_init': best['learning_rate_init'],
                                           'verbose': False})

print("average of CV rmse score on train set: {0:.0f}".format(rmse_train))
print("average of CV rmse score on val set: {0:.0f}".format(rmse_valid))


# Basicly we can see that the MLP model overfits to training set and it would take more time to tune the hyperparameters properly. If some other methods gives reasonable results, then it's sometimes not worth investing time in the ones which don't.

# #### Final model selection
# 
# We have worse results from _MLP_ model than others, and similar results from _XGBoost_ and _Random Forest_, so it's always better to select the model with fewer hyperparameters and the one which the data scientist understands better. That's why I will pick the _Random Forest_ model as my best model and we will check how it performs on the test (hold-out) set.

# In[ ]:


# Train RandomForest model with Cross-Validation on train set
root_mean_squared_errors = []

rf_model = RandomForestRegressor(n_jobs=-1, n_estimators=300).fit(X_train, Y_train)
predictions = rf_model.predict(X_test)
actuals = Y_test
root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))

rmse = np.mean(root_mean_squared_errors)
print("average of CV rmse score on test set: {0:.0f}".format(rmse))


# On the test set, which none of the models have seen before, the result is comparable with the one calculated with Cross-Validation.

# ## 2.4. Summary of the experiments
# 
# As part of the kernel we've made a data mining part, model building part, as well as hyperparameter tuning by using `HyperOpt` module. The results seems to be good, but there are some things that could be done as further experiments:
# * feature transformation - as seen at https://playground.tensorflow.org/, making squares of features, multiplying them or taking trigonometric functions of them could help model to fit the data,
# * more complicated feature selection - it wasn't necessary in this task, because there is a couple of features, but if we somehow multiply them (in. e. using the method described above or by changing some variables from numerical to one-hot encoded), it would be beneficial to pick the ones with highest pearson correlation with `price` feature,
# * more models could be tried. Usually some datasets are fitted better by one type of models than another.

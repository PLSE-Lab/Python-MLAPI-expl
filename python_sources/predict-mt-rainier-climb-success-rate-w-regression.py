#!/usr/bin/env python
# coding: utf-8

# The aim is to predict success rate (%) in reaching the Mount Rainier peak given (1) the route and (2) the weather condition.

# # Import Libraries

# In[ ]:


# Load libraries
import sys
import scipy
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from math import sqrt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle


# In[ ]:


# get working directory
get_ipython().system(' pwd')


# # Load dataset

# In[ ]:


# load data
climbing_data = pd.read_csv('/kaggle/input/mount-rainier-weather-and-climbing-data/climbing_statistics.csv')
weather_data = pd.read_csv('/kaggle/input/mount-rainier-weather-and-climbing-data/Rainier_Weather.csv')


# ## Preliminary exploration

# In[ ]:


climbing_data.info()
weather_data.info()


# Merge the climbing and weather datasets into 1 dataset.

# # Preprocess Data (1)

# **Ways data can be preprocessed:**
# * Remove missing values
# * Remove outliers
# * Remove duplicates
# * Feature selection
# * Feature engineering
# * Feature scaling
#     * **Numerical** data = standardize or normalize
#     * **Categorical** data = one-hot encoding or dummify
# * Group data into clusters
#     * Cluster by an attribute (e.g. age, price)
#     * Cluster using k-means

# In[ ]:


# Merge 2 datasets
joined_data = pd.merge(climbing_data, weather_data, how="left", on=["Date"])
joined_data.head()


# In[ ]:


# Remove missing values
joined_data.isna().sum()
joined_data = joined_data.dropna()
joined_data.isna().sum()


# In[ ]:


# Remove rows where success percentage > 1.00
joined_data = joined_data[joined_data['Success Percentage'] <= 1]  
joined_data.describe()


# In[ ]:


# Feature selection
data = joined_data.drop(columns=["Date", "Attempted","Succeeded", "Battery Voltage AVG"])
data.info()


# In[ ]:


# dummify "Route"
dummy_Route = pd.get_dummies(data['Route'])
dummify_data = pd.concat([data, dummy_Route], axis = 1)
data = dummify_data.drop(columns=["Route"])
data.info()


# In[ ]:


# Alternative to dummify, is to code "Route"

# Change route into codes
data["Route"] = data["Route"].astype("category")
data["Route_code"] = data["Route"].cat.codes
data["Route_code"].describe() # if min is -1, then there is NA

# double check there are same no of unique Route names and Route codes
data["Route"].describe() == data["Route"].cat.codes.astype("category").describe()

# View Route code dictionary
code = data["Route"].astype('category')
code_dictionary = dict(enumerate(code.cat.categories))
print(code_dictionary)


# # Exploratory Data Analysis (EDA)

# ## Understand data with descriptive statistics

# In[ ]:


# describe
data.describe()


# In[ ]:


# head
data.head()


# In[ ]:


# shape
data.shape


# In[ ]:


# data types
data.dtypes


# In[ ]:


# attributes
attributes = data.dtypes.index
print(attributes)


# In[ ]:


# Correlation Coefficient Matrix Heatmap
correlation = data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation, xticklabels=correlation.columns.values, yticklabels=correlation.columns.values)


# In[ ]:


# Covariance Coefficient Matrix Heatmap
covariance = data.cov()
plt.figure(figsize=(15, 10))
sns.heatmap(covariance, xticklabels=covariance.columns.values, yticklabels=covariance.columns.values)


# ## Understand data with visualization

# ### Univariate plots to understand each individual attribute

# In[ ]:


# Dependent variable -- 'Success Percentage'
sns.countplot(data['Success Percentage'])


# In[ ]:


# Distribution of attribute -- "Success Percentage"
f = plt.figure(figsize=(20,4))
f.add_subplot(1,2,1)
sns.distplot(data['Success Percentage'])
f.add_subplot(1,2,2)
sns.boxplot(data['Success Percentage'])


# In[ ]:


# histogram of all attributes
data.hist(figsize=(10,6), bins = 10)
plt.show()


# In[ ]:


#box plot
plt.figure(figsize=(20,6))
sns.boxplot(data = data)


# In[ ]:


# Line plot
plt.plot(data['Temperature AVG'])
plt.title('Line plot: Temperature AVG')
plt.ylabel('Temperature AVG')
plt.show()


# ### Multivariate plots to understand relationship between attributes

# In[ ]:


# Visualize succeeded vs attempted climbs per route
data.loc[joined_data["Succeeded"]==1]

# create succeeded climbs dataset
succeded_data = joined_data[["Route","Succeeded"]].groupby("Route").sum().reset_index()
succeded_data.columns = ["Route", "Succeeded"]
#succeded_data.head()

# create attempted climbs dataset
attempts_data = joined_data[["Route","Attempted"]].groupby("Route").sum().reset_index()
attempts_data.columns=["Route", "Attempted"]
#attempts_data.head()

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

SuccessAttempt_data = [go.Bar(x=attempts_data.Route,
               y=attempts_data.Attempted, name = "Attempted climb"),
        go.Bar(x=succeded_data.Route,
               y=succeded_data.Succeeded, name = 'Successful climb'),]

layout = go.Layout(barmode='stack', title = 'Sucesssful vs Attempted climbs')

fig = go.Figure(data=SuccessAttempt_data, layout=layout)
iplot(fig)


# In[ ]:


# Visualize % succeess rate per route
success_rate_data = pd.merge(attempts_data, succeded_data, how="left", on=["Route"])
success_rate_data["Success Percentage"] = (success_rate_data.Succeeded / success_rate_data.Attempted * 100)
#success_rate_data.head(10)

import plotly.express as px
fig = px.bar(success_rate_data, x = "Route", y = "Success Percentage")
fig.show()


# In[ ]:


# scatter plot of 2 features
scatter_x = data['Success Percentage']
scatter_y = data['Temperature AVG']
plt.scatter(scatter_x, scatter_y)
plt.title('Relationship between climb success and temperature')
plt.xlabel('% successful climbs')
plt.ylabel('Average temperature')
plt.show()


# In[ ]:


# scatter plot matrix
scatter_matrix(data, figsize=(10, 10))
plt.show()


# ---

# # Modeling

# **5 Levels of ML Model Iteration:**
# 1. Fitting Parameters
# 2. Tuning Hyperparameters
# 3. Feature Engineering
# 
# ---

# **Model Types**
# 
# **- Linear models:**
# * Logistic Regression (LR)
# * Linear Discriminant Analysis (LDA)
# 
# **- Nonlinear models:**
# * K-Nearest Neighbors (KNN)
# * Classification and Regression Trees (CART)
# * Gaussian Naive Bayes (NB)
# * Support Vector Machines (SVM)
# * Ridge Regression (RR)
# 
# **- Bagging ensemble models:**
# * Random Forest (RF)
# 
# ---

# Split dataset into train/test set:

# In[ ]:


# Create x (independent, input) + y (dependent, output) variables
x = data.drop(columns=['Success Percentage'])
y = data['Success Percentage']

# Split train/validation datasets (80-20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=7)

# dimensions of train/test set
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# ## Iteration (1)

# In[ ]:


# Prepare models
models = []

# classification
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('SVM', SVC(gamma='auto')))
#models.append(('RF', RandomForestClassifier(n_estimators=100, max_features=3)))
#models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('NB', GaussianNB()))

# regression
models.append(('RFregressor', RandomForestRegressor()))
models.append(('SVR', SVR()))
models.append(('KNNregressor', KNeighborsRegressor()))
models.append(('LinearR', LinearRegression()))


# ### Optimize Models by Fitting Parameters (1)
# 
# Train models on train set to find the best parameters with cross validation & get the first performance measures on the validation set
# 
# Find different model performance metrics part of *scikit learn* here: https://scikit-learn.org/stable/modules/model_evaluation.html

# In[ ]:


# Evaluate each model's accuracy on the validation set
print('Cross Validation Score: RMSE & SD')
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_root_mean_squared_error')
	results.append(cv_results)
	names.append(name)
	print('%s: %.3f (%.3f)' % (name, -cv_results.mean(), cv_results.std()))
    
# Visualize model comparison
plt.boxplot(results, labels=names)
plt.title('Model Comparison: Cross Validation Score (RMSE)')
plt.show()


# ### Evaluate Model Performance (1)

# Performance metrics on ***train set***:

# In[ ]:


print('Train Set Performance Metrics: RMSE & MAE')
for name, model in models:
    trained_model = model.fit(x_train, y_train)
    y_train_pred = trained_model.predict(x_train)
    print('%s: %.3f (%.3f)' % (name, sqrt(mean_squared_error(y_train, y_train_pred)), (mean_absolute_error(y_train, y_train_pred))))


# Performance metrics on ***test set***:

# In[ ]:


print('Test Set Performance Metrics: RMSE & MAE')
for name, model in models:
    trained_model = model.fit(x_train, y_train)
    y_test_pred = trained_model.predict(x_test)
    print('%s: %.3f (%.3f)' % (name, sqrt(mean_squared_error(y_test, y_test_pred)), (mean_absolute_error(y_test, y_test_pred))))


# #### *EXTRA: deep error check on individual model*

# In[ ]:


# Train model
RFregressor = RandomForestRegressor().fit(x_train, y_train)


# Predict y on train set
y_train_pred = RFregressor.predict(x_train)

# Train set performance metrics
print('Train Set Performance Metrics: RMSE')
print('%.3f' % (sqrt(mean_squared_error(y_train, y_train_pred))))

# Predict y on test set
y_test_pred = RFregressor.predict(x_test)

# Test set performance metrics
print('Train Set Performance Metrics: RMSE')
print('%.3f' % (sqrt(mean_squared_error(y_test, y_test_pred))))


# In[ ]:


# Visualize predicted vs actual mountain climb 'Success Percentage' -- if perfect, then a diagonal line
plt.scatter(y_test, y_test_pred, alpha = 0.5)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.show()


# ## Iteration (2)

# ### Preprocess Data (2)
# #### Normalization (Min-Max)

# In[ ]:


# Min-Max norm all features
minmax_scaler = MinMaxScaler()
x_norm = minmax_scaler.fit_transform(x)
x = pd.DataFrame(x_norm, columns=x.columns)
x.head()


# Split *rescaled* dataset into train/test set:

# In[ ]:


# Split train/validation datasets (80-20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=7)

# dimensions of train/test set
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# ### Optimize Models by Fitting Parameters (2)

# In[ ]:


# Evaluate each model's accuracy on the validation set
print('Cross Validation Score: RMSE & SD')
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_root_mean_squared_error')
	results.append(cv_results)
	names.append(name)
	print('%s: %.3f (%.3f)' % (name, -cv_results.mean(), cv_results.std()))
    
# Visualize model comparison
plt.boxplot(results, labels=names)
plt.title('Model Comparison: Cross Validation Score (RMSE)')
plt.show()


# ### Evaluate Model Performance (2)

# Performance metrics on ***train set***:

# In[ ]:


print('Train Set Performance Metrics: RMSE & MAE')
for name, model in models:
    trained_model = model.fit(x_train, y_train)
    y_train_pred = trained_model.predict(x_train)
    print('%s: %.3f (%.3f)' % (name, sqrt(mean_squared_error(y_train, y_train_pred)), (mean_absolute_error(y_train, y_train_pred))))


# Performance metrics on ***test set***:

# In[ ]:


print('Test Set Performance Metrics: RMSE & MAE')
for name, model in models:
    trained_model = model.fit(x_train, y_train)
    y_test_pred = trained_model.predict(x_test)
    print('%s: %.3f (%.3f)' % (name, sqrt(mean_squared_error(y_test, y_test_pred)), (mean_absolute_error(y_test, y_test_pred))))


# #### *EXTRA: deep error check on individual model*

# In[ ]:


# Train model
RFregressor = RandomForestRegressor().fit(x_train, y_train)


# Predict y on train set
y_train_pred = RFregressor.predict(x_train)

# Train set performance metrics
print('Train Set Performance Metrics: RMSE')
print('%.3f' % (sqrt(mean_squared_error(y_train, y_train_pred))))

# Predict y on test set
y_test_pred = RFregressor.predict(x_test)

# Test set performance metrics
print('Train Set Performance Metrics: RMSE')
print('%.3f' % (sqrt(mean_squared_error(y_test, y_test_pred))))


# In[ ]:


# Visualize predicted vs actual mountain climb 'Success Percentage' -- if perfect, then a diagonal line
plt.scatter(y_test, y_test_pred, alpha = 0.5)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.show()


# ## Iteration (3)

# ### Optimize Models by Tuning Hyperparameters (3)

# Random Forest Regressor had the best metrics in previous iterations, therefore let's only tune the hyperparameters for this one model.

# In[ ]:


# Random Forest Regressor (RFregressor)
#Create dictionary of hyperparameters that we want to tune
RFR_params = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

# Create new RandomForestRegressor object using GridSearch
grid_RFR = GridSearchCV(RandomForestRegressor(), RFR_params, cv=3)

#Fit the model
best_model_RFR = grid_RFR.fit(x_train, y_train)

# Print the value of best hyperparameters
#print('Best n_neighbors:', best_model_RFR.best_estimator_.get_params()['n_neighbors'])
#print('Best n_neighbors:', best_model_RFR.best_estimator_.get_params()['weights'])
#print('Best n_neighbors:', best_model_RFR.best_estimator_.get_params()['metric'])
#print('Best leaf_size:', best_model_RFR.best_estimator_.get_params()['leaf_size'])
#print('Best p:', best_model_RFR.best_estimator_.get_params()['p'])
print(best_model_RFR.best_params_)


# ### Evaluate Model Performance (3)

# In[ ]:


# Predict y on train set
y_train_pred_2 = best_model_RFR.predict(x_train)

# Train set performance metrics
print('Train Set Performance Metrics: RMSE')
print('%.3f' % (sqrt(mean_squared_error(y_train, y_train_pred_2))))

# Predict y on test set
y_test_pred_2 = best_model_RFR.predict(x_test)

# Test set performance metrics
print('Train Set Performance Metrics: RMSE')
print('%.3f' % (sqrt(mean_squared_error(y_test, y_test_pred_2))))


# In[ ]:


# Visualize predicted vs actual mountain climb 'Success Percentage' -- if perfect, then a diagonal line
plt.scatter(y_test, y_test_pred_2, alpha = 0.5)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.show()


# # Save final model

# In[ ]:


# Save model to disk
FinalModel_RandomForestRegressor = 'FinalModel.sav'
pickle.dump(best_model_RFR, open(FinalModel_RandomForestRegressor, 'wb'))


# **Final Notes**
# * The best model was Random Forest Regressor. 
# * We dummified the Route, normalized the features, tuned parameters and hyperparamters.
# * The RMSE of the model's performance in predicting the % successf rate of reaching the peak of Mt Rainier was only improved from 0.439 to 0.436. The dataset is too small (1889 total rows) for room for extensive further optimization and tuning and improvements. 
# * In the future, we could try with a much larger dataset. 

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from os import chdir

# File Import
x = pd.read_csv('../input/MiningProcess_Flotation_Plant_Database.csv', decimal=",").drop_duplicates()


# I believe there might be an issue in the way this problem is set.
# 
# The frequency of the features must be aligned with the output (% Silica Concentrate). There is no point about running the algorithm on each record because we know that none of them actually predicts the average % Silica Concentrate. It is only the combination of all parameters over a period of time which gives with the averaged % Silica Concentrate.
# 
# Therefore, I am grouping and averaging all records by % Silica Concentrate. It might be considered as a shortcut but, from an industrial standpoint, I believe that makes sense.

# In[ ]:


# Values grouped by %Silica Concentrate and averaged
x_grpby = x.groupby(['% Silica Concentrate']).mean()

# Extraction of %Silica Concentrate values
y = list(x_grpby.index.values)

# Suppression of %Iron Concentrate column as it is highly correlated to %Silica Concentrate (0.8)
x_grpby.drop(labels="% Iron Concentrate", axis=1, inplace=True)

# Feature Scaling
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = pd.DataFrame(min_max_scaler.fit_transform(x_grpby), columns=x_grpby.columns)


# Dimensions reduction is always a good way to detect underlying variances within an industrial process

# In[ ]:


# Dimensions reduction with Principal Components Analysis
pca = PCA(n_components=10)
pca.fit(x_scaled)
x_pca = pca.fit_transform(x_scaled)

# Creation of the different datasets for training and testing
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.8, random_state=59)

# Optimal parameters defined on a separate script by iterations
params = {
    'n_estimators': 80,
    'max_depth': 12,
    'learning_rate': 0.1,
    'criterion': 'mse'
    }

# GBR implementation
gbr = ensemble.GradientBoostingRegressor(**params)
gbr.fit(x_train,y_train)

# GBR performance on test data
print ("R-squared (Test dataset):", gbr.score(x_test, y_test))
error = mean_squared_error(y_test, gbr.predict(x_test))
print ("MSE (Test dataset):",error)


# In[ ]:


# GBR performance 
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
ax.set(title="Gradient Boosting Regressor", xlabel="Y values (actual)", ylabel="Y values (predicted)")
ax.scatter(y_test, gbr.predict(x_test))
ax.plot([0,max(y_test)], [0,max(gbr.predict(x_test))], color='r')
fig.show()


# ![](http://)

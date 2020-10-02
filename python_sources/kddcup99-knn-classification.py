#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# dataset is taken from "https://datahub.io/machine-learning/kddcup99#resource-kddcup99_zip"

# knn clasification using k values from 1 to 40 to see which k value is good

# flattened the dataset before applying the knn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

#csv to dataframe conversion 
df = pd.read_csv("../input/kddcup99.csv")

# flattening the dataset 
df1 = df.drop(df.columns[[1, 2, 3]], axis=1)
dat1 = pd.get_dummies(df['protocol_type'])
dat2 = pd.get_dummies(df['service'])
dat3 = pd.get_dummies(df['flag'])
p = dat1.join(dat2)
df2 = p.join(dat3)
dataset = df2.join(df1)

#selecting the columns excluding lables
X = dataset.iloc[:, :-1].values

#selecting the lables column 
y = dataset.iloc[:, -1].values

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#scaling by doing normalisation
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# define the parameter values that should be searched
# for python 2, k_range = range(1, 31)
k_range = list(range(1, 11))
# print(k_range)

# create a parameter grid: map the parameter names to the values that should be searched
# simply a python dictionary
# key: parameter name
# value: list of values that should be searched for that parameter
# single key-value pair for param_grid
param_grid = dict(n_neighbors=k_range)
# print(param_grid)

# 2. run KNeighborsClassifier with k neighbours
knn = KNeighborsClassifier(n_neighbors=k_range)

# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# fit the grid with dataset
grid.fit(X_train, y_train)

# view the complete results (list of named tuples)
data = grid.cv_results_

dfdf = pd.DataFrame.from_dict(data)

print(dfdf[['param_n_neighbors','mean_test_score']])

# examine the first tuple
# we will slice the list and select its elements using dot notation and []

k_range = dfdf['param_n_neighbors']
scores = dfdf['mean_test_score']

#plot the relationship between K and accuracy
plt.figure(figsize=(12, 6))
plt.plot(k_range, scores, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Accuracy vs K Value')
plt.xlabel('Value of K for KNN')
plt.ylabel('Accuracy')

# examine the best model

# Single best score achieved across all params (k)
print('best score', grid.best_score_)

# Dictionary containing the parameters (k) used to generate that score
print('best no of neighbours', grid.best_params_)

# Actual model object fit with those best parameters
# Shows default parameters that we did not specify
print(grid.best_estimator_)

# classification report
print('classifiaction report')
print(classification_report(grid.best_estimator_.predict(X_test), y_test))


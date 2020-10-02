#!/usr/bin/env python
# coding: utf-8

# **Wine Competition**
# 
# Using only the code from [tech trainings](https://github.com/texas-a-m-data-analytics-club), you can get a 97% or higher on the wine competition.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import os


# Read in the csv file.

# In[ ]:


raw_wine_data = pd.read_csv('/kaggle/input/tdac-wine/Test_Data.csv')


# Check out the data. It's all over the place. I need to standardize it. Additionally, I don't believe the index column will contribute so I'll take that out.

# In[ ]:


raw_wine_data.head()


# Separating the type column into wine_target and removing the type and index column from the input data

# In[ ]:


wine_target = raw_wine_data['type']
raw_wine_data = raw_wine_data.drop(['index','type'], axis=1)


# Preprocessing the inputs.

# In[ ]:


vals = raw_wine_data.values
min_max_scaler = preprocessing.MinMaxScaler()
vals_scaled = min_max_scaler.fit_transform(vals)
processed_wine_data = pd.DataFrame(vals_scaled)


# Visualizing the data to ensure there is a clear distinction between the two types. Using tSNE to reduce the dimensionality of the data.

# In[ ]:


from sklearn.manifold import TSNE

decomp_wine = TSNE(n_components=2, early_exaggeration=2.0).fit_transform(processed_wine_data)


# In[ ]:


decomp_wine = pd.DataFrame(decomp_wine)


# Plotting the decomposed data

# In[ ]:


red_wine = decomp_wine[wine_target == 0]
white_wine = decomp_wine[wine_target == 1]

fig,ax=plt.subplots(1,1,figsize=(10, 10))
red_wine.plot.scatter(0,1, color='red', ax=ax, label='Red Wine')
white_wine.plot.scatter(0,1, color='blue', ax=ax, label='White Wine')


# Establishing parameters that will be tested during my grid search

# In[ ]:


parameters_logit= [{'C':[0.1,0.2,0.5],'solver':['liblinear'],'penalty':['l1','l2'],'max_iter':[1000]},
                   {'C':[0.1,0.2,0.5,1],'solver':['lbfgs'],'penalty':['l2'],'max_iter':[1000]}]


# Splitting the data into a training and test set.
# Creating a Logisitic Regression model.
# Passing the model and test parameters to the GridSearchCV function (this is a brute force way of finding the best parameters).
# Fitting the training data to the model.
# Scoring the test data.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(processed_wine_data, wine_target)
LR = LogisticRegression()
grid_search_logit=GridSearchCV(estimator=LR, param_grid=parameters_logit,scoring='accuracy',cv=10)
grid_search_logit.fit(X_train,y_train)
grid_search_logit.score(X_test, y_test)


# Openning the validation data csv file

# In[ ]:


raw_wine_val_data = pd.read_csv('/kaggle/input/tdac-wine/Val_Data.csv')


# Preprocessing the data
# 
# 
# 1.   Removing just the index column since the type column isn't present in this file
# 2.   Normalizing the data
# 3.   Storing the data in a data frame
# 
# 

# In[ ]:


del raw_wine_val_data['Index']
vals = raw_wine_val_data.values
min_max_scaler = preprocessing.MinMaxScaler()
vals_scaled = min_max_scaler.fit_transform(vals)
processed_wine_val_data = pd.DataFrame(vals_scaled)
processed_wine_val_data.head()


# Using my trained model to predict the type of wine in the validation data and storing it in a csv file.

# In[ ]:


guess = grid_search_logit.predict(processed_wine_val_data)
np.savetxt("val.csv", guess, delimiter=",")


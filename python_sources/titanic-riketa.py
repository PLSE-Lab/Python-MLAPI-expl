#!/usr/bin/env python
# coding: utf-8

# # Titanic ML competition
# 
# ## 1. Problem Definition
# 
# Using machine learning, create a model that predicts which passengers survived the Titanic shipwreck
# 
# ## 2. Data
# 
# Looking at the [dataset from Kaggle](https://www.kaggle.com/c/titanic/data), 
# This is a problem of supervised learning
# 
# There are 2 datasets:
# 1. **Train.csv** - Data on which model will be trained
# 2. **Test.csv** - Data on which prediction will be done 
# 
# ## 3. Evaluation
# 
# Use the model you trained to predict whether or not they survived the sinking of the Titanic.
# 

# ## Importing the data and preparing it for modelling

# In[ ]:


## Import data analysis tools 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[ ]:


#Read the training dataset
df = pd.read_csv("../input/titanic/train.csv") 
df.shape


# In[ ]:


#Check for the null values
df.isna().sum()


# In[ ]:


# Fill numeric rows with the median
for label, content in df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column which tells if the data was missing our not
            df[label+"_is_missing"] = pd.isnull(content)
            # Fill missing numeric values with median since it's more robust than the mean
            df[label] = content.fillna(content.median())
            
            
# Turn categorical variables into numbers
for label, content in df.items():
    # Check columns which *aren't* numeric
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to inidicate whether sample had missing value
        df[label+"_is_missing"] = pd.isnull(content)
        # We add the +1 because pandas encodes missing categories as -1
        df[label] = pd.Categorical(content).codes+1   


# In[ ]:


np.random.seed(42)

X = df.drop("Survived", axis=1)
y = df.Survived

# Split into train & test set
X_train, X_test, y_train, y_test = train_test_split(X, # independent variables 
                                                    y, # dependent variable
                                                    test_size = 0.2) # percentage of data to use for test set


# In[ ]:


# Put models in a dictionary
models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(), 
          "Random Forest": RandomForestClassifier()}

# Create function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data
    
    y_train : labels assosciated with training data
    
    """
    # Random seed for reproducible results
    np.random.seed(42)
    # Make a list to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)
model_scores




# In[ ]:


# Different RandomForestClassifier hyperparameters
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model
rs_rf.fit(X_train, y_train);


# In[ ]:


#Checkout the best parameters
rs_rf.best_params_


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Train model with the Most ideal hyperparameters\nideal_model = RandomForestClassifier(n_estimators=910,\n                                    min_samples_leaf=1,\n                                    min_samples_split=18,\n                                    max_depth = 10)\nideal_model.fit(X_train, y_train)')


# In[ ]:


# Read the test data
df_test = pd.read_csv("../input/titanic/test.csv")
df_test.head()


# In[ ]:


# Fill numeric rows with the median
for label, content in df_test.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column which tells if the data was missing our not
            df_test[label+"_is_missing"] = pd.isnull(content)
            # Fill missing numeric values with median since it's more robust than the mean
            df_test[label] = content.fillna(content.median())
            
            
# Turn categorical variables into numbers
for label, content in df_test.items():
    # Check columns which *aren't* numeric
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to inidicate whether sample had missing value
        df_test[label+"_is_missing"] = pd.isnull(content)
        # We add the +1 because pandas encodes missing categories as -1
        df_test[label] = pd.Categorical(content).codes+1 


# In[ ]:


#This column is not present in training set, so drop this column
df_test.drop("Fare_is_missing",axis=1, inplace=True)


# In[ ]:


# Make predictions on the test dataset using the best model
test_preds = ideal_model.predict(df_test)


# In[ ]:


# Create DataFrame compatible with Kaggle submission requirements
df_preds = pd.DataFrame()
df_preds["PassengerId"] = df_test["PassengerId"]
df_preds["Survived"] = test_preds
df_preds


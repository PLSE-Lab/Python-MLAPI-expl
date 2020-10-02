#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# *Neither Titanic dataset nor sklearn a new thing for any data scientist but there are some important features in scikit-learn that will make any model preprocessing and tuning easier, to be specific this notebook will cover the following concepts:*
# 
# >- ColumnTransformer
# >- Pipeline
# >- SimpleImputer
# >- StandardScalar
# >- OneHotEncoder
# >- OrdinalEncoder
# >- GridSearch

# # **Mounting Filesystem**
# 

# In[ ]:


# Input data files are available in the "../input/" directory.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # **Import Packages**
# 

# In[ ]:


# Pandas for data reading and writing
import pandas as pd
# Numpy for Numerical operations
import numpy as np
# Import ColumnTransformer
from sklearn.compose import ColumnTransformer
# Import Pipeline
from sklearn.pipeline import Pipeline
# Import SimpleImputer
from sklearn.impute import SimpleImputer
# Import StandardScaler, OneHotEncodr and OrdinalEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
# Import Random Forest for Classification
from sklearn.ensemble import RandomForestClassifier
# Import GridSearch
from sklearn.model_selection import GridSearchCV


# # **Reading Data**
# In the following cells, we will read the train and test data and check for NaNs.

# In[ ]:


# Read the train data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
# See some info
train_data.info()


# It's obvious that we had to deal with NaNs

# In[ ]:


# Load test data
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.info()


# # **Splitting Data**
# 

# In[ ]:


# Split the data into predictors and target
X_train = train_data.drop(['Survived', 'Name'], axis = 1)
X_test = test_data.drop(['Name'], axis = 1)
y_train = train_data['Survived']


# # **Continuous and Numerical features handling**
# *It's clear that we have some numerical features that have some missing values to be imputed and they have to be of the same scale also.*
# 
# *In the following cell, we will handle the numerical features separtely i.e "Age" and "Fare"*
# 

# In[ ]:


# Now, we will create a pipline for the numeric features
# Difine a list with the numeric features
numeric_features = ['Age', 'Fare']
# Define a pipeline for numer"ic features
numeric_features_pipeline = Pipeline(steps= [
    ('imputer', SimpleImputer(strategy = 'median')), # Impute with median value for missing
    ('scaler', StandardScaler())                     # Conduct a scaling step
])


# # **Categorical features handling**
# *It's clear that we have some categorical features that have some missing values to be imputed and they have to be encoded using one hot encoding.*
# 
# *In the following cell, we will handle the categorical features separtely i.e "Embarked" and "Sex"*
# 
# *Note: I choose simple imputer for the missing cells to impute with 'missing' word. My aim was to gather all missing cells in one category for further encoding.*

# In[ ]:


# Now, we will create a pipline for the categorical features
# Difine a list with the categorical features
categorical_features = ['Embarked', 'Sex']
# Define a pipeline for categorical features
categorical_features_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value = 'missing')), # Impute with the word 'missing' for missing values
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))     # Convert all categorical variables to one hot encoding
])


# # **Ordinal features handling**
# *Passenger class or 'Pclass' for short is an ordinal feature that must be handled keeping in mind that class 3 is much higher than 2 and so on.*

# In[ ]:


# Now, we will create a pipline for the ordinal features
# Define a list with the ordinal features
ordinal_features = ['Pclass']
# Define a pipline for ordinal features 
ordinal_features_pipeline = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(categories= [[1, 2, 3]]))
])


# # **Construct a comprehended preprocessor**
# *Now, we will create a preprocessor that can handle all columns in our dataset using ColumnTransformer*

# In[ ]:


# Now, we will create a transformer to handle all columns
preprocessor = ColumnTransformer(transformers= [
    # transformer with name 'num' that will apply
    # 'numeric_features_pipeline' to numeric_features
    ('num', numeric_features_pipeline, numeric_features),
    # transformer with name 'cat' that will apply 
    # 'categorical_features_pipeline' to categorical_features
    ('cat', categorical_features_pipeline, categorical_features),
    # transformer with name 'ord' that will apply 
    # 'ordinal_features_pipeline' to ordinal_features
    ('ord', ordinal_features_pipeline, ordinal_features) 
    ])


# # **Prediction Pipeline**
# *Now, we will create a full prediction pipeline that uses our preprocessor and then transfer it to our classifier of choice 'Random Forest'*.

# In[ ]:


# Now, we will create a full prediction pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                     ('classifier', RandomForestClassifier(n_estimators = 120, max_leaf_nodes = 100))])


# # **Pipeline Training**
# *Let's train our pipeline now*
# 

# In[ ]:


# Let's fit our classifier
clf.fit(X_train, y_train)


# # **Pipeline Tuning**
# *The question now, can we push it a little bit further? i.e. can we tune every single part or our Pipeline?*
# 
# *Here, I will use GridSearch to decide three things:*
# >- Simple Imputer strategy : mean or median
# >- n_estimators of Random Forest
# >- max leaf nodes of Random Forest
# 
# *Note, you can access any parameter from the outer level to the next adjacent inner one*
# 
# *For Example: to access the strategy of the Simple Imputer you can do the following*
# preprocessor__num__imputer__strategy
# 
# *Let's see this into action*
# 

# In[ ]:


param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__n_estimators': [100, 120, 150, 170, 200],
    'classifier__max_leaf_nodes' : [100, 120, 150, 170, 200]
}

grid_search = GridSearchCV(clf, param_grid, cv=10)
grid_search.fit(X_train, y_train)
print(("best random forest from grid search: %.3f"
       % grid_search.score(X_train, y_train)))
print('The best parameters of Simple Imputer and C are:')
print(grid_search.best_params_)


# # **Generate Predictions**
# *Let's generate predictions now using our grid search model and submit the results*

# In[ ]:


# Generate predictions
predictions = grid_search.predict(X_test)
# Generate results dataframe
results_df = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
# Save to csv file
results_df.to_csv('submission.csv', index = False)
print('Submission CSV has been saved!')


# In[ ]:





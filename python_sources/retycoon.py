#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# linear algebra
import numpy as np
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
pd.set_option('display.max_columns', 100)

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns
sns.set_style('darkgrid')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load real estate data from CSV
df = pd.read_csv('/kaggle/input/realestatetycoon/real_estate_data.csv')


# In[ ]:


# Exploratory Analysis
# Start with Basics
# Dataframe dimensions
df.shape


# In[ ]:


# Column datatypes
df.dtypes


# In[ ]:


# Display first 5 observations
df.head()


# In[ ]:


# Display last 5 rows of data
df.tail()


# In[ ]:


# Plot numerical distributions
# Plot histogram grid
df.hist(figsize=(14,14), xrot=-45)

# Clear the text "residue"
plt.show()


# In[ ]:


# Histogram for year_built
df.year_built.hist()
plt.show()


# In[ ]:


# Summarize numerical features
df.describe()


# In[ ]:


# Summarize categorical features
df.describe(include=['object'])


# In[ ]:


# Plot bar plot for each categorical feature
for feature in df.dtypes[df.dtypes == 'object'].index:
    sns.countplot(y=feature, data=df)
    plt.show()


# In[ ]:


# Segment tx_price by property_type and plot distributions
sns.boxplot(y='property_type', x='tx_price', data=df)


# In[ ]:


# Filter and display only df.dtypes that are 'object'
df.dtypes[df.dtypes == 'object']


# In[ ]:


# Segment by property_type and display the means within each class
df.groupby('property_type').mean()


# In[ ]:


# Segment sqft by sqft and property_type distributions
sns.boxplot(y='property_type', x='sqft', data=df)


# In[ ]:


# Segment by property_type and display the means and standard deviations within each class
df.groupby('property_type').agg(['mean', 'std'])


# In[ ]:


# Study Correlations. Calculate correlations between numeric features
correlations = df.corr()


# In[ ]:


# Change color scheme
sns.set_style("white")

# Make the figsize 10 x 8
plt.figure(figsize=(10,8))

# Plot heatmap of correlations
sns.heatmap(correlations, cmap='RdBu_r')
plt.show()

# Generate a mask for the upper triangle
mask = np.zeros_like(correlations)
mask[np.triu_indices_from(mask)] = 1

# Make the figsize 10 x 8
plt.figure(figsize=(10,8))

# Plot heatmap of annotated correlations
sns.heatmap(correlations * 100,
            cmap='RdBu_r',
            annot=True,
            fmt='.0f')

plt.show()


# In[ ]:


# Make the figsize 10 x 8
plt.figure(figsize=(10,8))

# Plot heatmap of correlations
sns.heatmap(correlations * 100,
            cmap='RdBu_r',
            annot=True,
            fmt='.0f',
            mask=mask)
plt.show()


# In[ ]:


# Correlations between two features
df[['beds', 'baths']].corr()


# In[ ]:


# Data Cleaning
# Drop duplicates
df = df.drop_duplicates()
print( df.shape )


# In[ ]:


df.head()


# In[ ]:


# Fix Structural Errors
# Display unique values of 'basement'
print( df.basement.unique() )


# In[ ]:


# Missing basement values should be 0
df.basement.fillna(0, inplace=True)


# In[ ]:


# Class distributions for 'roof'
sns.countplot(y='roof', data=df)
plt.show()


# In[ ]:


# 'composition' should be 'Composition'
df.roof.replace('composition', 'Composition', inplace=True)

# 'asphalt' should be 'Asphalt'
df.roof.replace('asphalt', 'Asphalt', inplace=True)

# 'shake-shingle' and 'asphalt,shake-shingle' should be 'Shake Shingle'
df.roof.replace(['shake-shingle', 'asphalt,shake-shingle'], 'Shake Shingle',
                inplace=True)


# In[ ]:


# Class distribution for 'roof'
sns.countplot(y='roof', data=df)
plt.show()


# In[ ]:


# Class distributions for 'exterior_walls'
sns.countplot(y='exterior_walls', data=df)
plt.show()


# In[ ]:


# 'Rock, Stone' should be 'Masonry'
df.exterior_walls.replace('Rock, Stone', 'Masonry', inplace=True)

# 'Concrete' and 'Block' should be 'Concrete Block'
df.exterior_walls.replace(['Concrete', 'Block'], 'Concrete Block', inplace=True)


# In[ ]:


# Class distributions for 'exterior_walls'
sns.countplot(y='exterior_walls', data=df)
plt.show()


# In[ ]:


# Class distributions for 'property_type'
sns.countplot(y='property_type', data=df)
plt.show()


# In[ ]:


# Filter Unwanted Outliers
# Box plot of 'tx_price' using the Seaborn library
sns.boxplot(df.tx_price)
plt.xlim(0, 1000000) # setting x-axis range to be consistent
plt.show()

# Violin plot of 'tx_price' using the Seaborn library
sns.violinplot('tx_price', data=df)
plt.xlim(0, 1000000) # setting x-axis range to be consistent
plt.show()


# In[ ]:


# Violin plot of beds
sns.violinplot(df.beds)
plt.show()

# Violin plot of sqft
sns.violinplot(df.sqft)
plt.show()

# Violin plot of lot_size
sns.violinplot(df.lot_size)
plt.show()


# In[ ]:


# Sort df.lot_size and display the top 5 samples
df.lot_size.sort_values(ascending=False).head()


# In[ ]:


df[df.lot_size == df.lot_size.max()]


# In[ ]:


# Remove lot_size outliers
df = df[df.lot_size <= 500000]

# print length of df
print( len(df) )


# In[ ]:


# Sort df.lot_size and display the top 5 samples
df.lot_size.sort_values(ascending=False).head()


# In[ ]:


# Handle Missing Data (numeric)
# Display number of missing values by feature (numeric)
df.select_dtypes(exclude=['object']).isnull().sum()


# In[ ]:


# Display number of missing values by feature (numeric)
df.select_dtypes(exclude=['object']).isnull().sum()


# In[ ]:


# Handle missing data (Categorical)
# Display number of missing values by feature (categorical)
df.select_dtypes(include=['object']).isnull().sum()


# In[ ]:


# Fill missing values in exterior_walls with 'Missing'
df['exterior_walls'].fillna('Missing', inplace=True)
df['roof'].fillna('Missing', inplace=True)


# In[ ]:


# Fill missing categorical values
for column in df.select_dtypes(include=['object']):
    df[column].fillna('Missing', inplace=True)


# In[ ]:


# Display number of missing values by feature (categorical)
df.select_dtypes(include=['object']).isnull().sum()


# In[ ]:


# Save cleaned dataframe to new file
df.to_csv('cleaned_df.csv', index=None)


# In[ ]:


# Output data files are available in the "../output/" directory.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Feature engineering
# Create Interaction Features
# Create indicator variable for properties with 2 beds and 2 baths
df['two_and_two'] = ((df.beds == 2) & (df.baths == 2)).astype(int)


# In[ ]:


df.head()


# In[ ]:


# Display percent of rows where two_and_two == 1
print( df.two_and_two.mean() )


# In[ ]:


# Create a property age feature
df['property_age'] = df.tx_year - df.year_built


# In[ ]:


# Should not be less than 0
print( df.property_age.min() )


# In[ ]:


# Number of observations with 'property_age' < 0
print( sum(df.property_age < 0) )


# In[ ]:


# Remove rows where property_age is less than 0
df = df[df.property_age >= 0]

# Print number of rows in remaining dataframe
print( len(df) )


# In[ ]:


# Number of observations with 'property_age' < 0
print( sum(df.property_age < 0) )


# In[ ]:


# Create a school score feature that num_schools * median_school
df['school_score'] = df.num_schools * df.median_school


# In[ ]:


# Display median school score
df.school_score.median()


# In[ ]:


df.school_score.describe()


# In[ ]:


# Create indicator feature for transactions between 2010 and 2013, inclusive
df['during_recession'] = ((df.tx_year >= 2010) & (df.tx_year <= 2013)).astype(int)


# In[ ]:


# Set variable a as the earlier indicator variable (combining two masks)
a = ((df.tx_year >= 2010) & (df.tx_year <= 2013)).astype(int)

# Set variable b as the new indicator variable (using "between")
b = df.tx_year.between(2010, 2013).astype(int)

# Are a and b equivalent?
print( all(a == b) )


# In[ ]:


# Create indicator feature for transactions between 2010 and 2013, inclusive
df['during_recession'] = df.tx_year.between(2010, 2013).astype(int)


# In[ ]:


# Print percent of transactions where during_recession == 1
print( df.during_recession.mean() )


# In[ ]:


# Combine Sparse Class
# Bar plot for exterior_walls
sns.countplot(y='exterior_walls', data=df)
plt.show()


# In[ ]:


# Group 'Wood Siding' and 'Wood Shingle' with 'Wood'
df.exterior_walls.replace(['Wood Siding', 'Wood Shingle'], 'Wood', inplace=True)


# In[ ]:


# List of classes to group
other_exterior_walls = ['Concrete Block', 'Stucco', 'Masonry', 'Other', 'Asbestos shingle']

# Group other classes into 'Other'
df.exterior_walls.replace(other_exterior_walls, 'Other', inplace=True)


# In[ ]:


# Bar plot for exterior_walls
sns.countplot(y='exterior_walls', data=df)
plt.show()


# In[ ]:


print( df.exterior_walls.unique() )


# In[ ]:


# Display first 5 values of 'exterior_walls'
df.exterior_walls.head()


# In[ ]:


# Bar plot for roof
sns.countplot(y='roof', data=df)
plt.show()


# In[ ]:


# Group 'Composition' and 'Wood Shake/ Shingles' into 'Composition Shingle'
df.roof.replace(['Composition', 'Wood Shake/ Shingles'],
                'Composition Shingle', inplace=True)


# In[ ]:


# List of classes to group
other_roofs = ['Other', 'Gravel/Rock', 'Roll Composition', 'Slate', 'Built-up', 'Asbestos', 'Metal']

# Group other classes into 'Other'
df.roof.replace(other_roofs, 'Other', inplace=True)


# In[ ]:


# Bar plot for roof
sns.countplot(y='roof', data=df)
plt.show()


# In[ ]:


# Add dummy variables
# Get dummy variables and display first 5 observations
pd.get_dummies( df, columns=['exterior_walls'] ).head()


# In[ ]:


# Get dummy variables and display first 5 observations
pd.get_dummies( df, columns=['roof'] ).head()


# In[ ]:


# Create new dataframe with dummy features
abt = pd.get_dummies(df, columns=['exterior_walls', 'roof', 'property_type'])


# In[ ]:


abt.head()


# In[ ]:


print( len(abt.columns) )


# In[ ]:


# Drop 'tx_year' and 'year_built' from the dataset
abt.drop(['tx_year', 'year_built'], axis=1, inplace=True)


# In[ ]:


# Save analytical base table
abt.to_csv('analytical_base_table.csv', index=None)


# In[ ]:


# Output data files are available in the "../output/" directory.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Algorithm Selection
# Import Regularized Regression algos
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# Import Tree Ensemble algos
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[ ]:


# Load ABT from Module 3
df = pd.read_csv('analytical_base_table.csv')
print(df.shape)


# In[ ]:


# Function for splitting training and test set
from sklearn.model_selection import train_test_split


# In[ ]:


# Create separate object for target variable
y = df.tx_price

# Create separate object for input features
X = df.drop('tx_price', axis=1)


# In[ ]:


# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2,
                                                   random_state=1234)


# In[ ]:


print( len(X_train), len(X_test), len(y_train), len(y_test) )


# In[ ]:


# Summary statistics of X_train
X_train.describe()


# In[ ]:


# Function for creating model pipelines
from sklearn.pipeline import make_pipeline


# In[ ]:


# For standardization
from sklearn.preprocessing import StandardScaler


# In[ ]:


# Pipeline with Standardization and Lasso Regression
make_pipeline(StandardScaler(), Lasso(random_state=123))


# In[ ]:


# Create pipelines dictionary
pipelines = {
    'lasso' : make_pipeline(StandardScaler(), Lasso(random_state=123)),
    'ridge' : make_pipeline(StandardScaler(), Ridge(random_state=123))
}


# In[ ]:


# Add a pipeline for Elastic-Net
pipelines['enet'] = make_pipeline(StandardScaler(), ElasticNet(random_state=123))


# In[ ]:


# List tuneable hyperparameters of our Lasso pipeline
pipelines['lasso'].get_params()


# In[ ]:


# Lasso hyperparameters
lasso_hyperparameters = { 
    'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10] 
}

# Ridge hyperparameters
ridge_hyperparameters = { 
    'ridge__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]  
}


# In[ ]:


# Elastic Net hyperparameters
enet_hyperparameters = { 
    'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],                        
    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]  
}


# In[ ]:


# Create hyperparameters dictionary
hyperparameters = {
    'lasso' : lasso_hyperparameters,
    'ridge' : ridge_hyperparameters,
    'enet' : enet_hyperparameters
}


# In[ ]:


# Helper for cross-validation
from sklearn.model_selection import GridSearchCV


# In[ ]:


# Create cross-validation object from Lasso pipeline and Lasso hyperparameters
model = GridSearchCV(pipelines['lasso'], hyperparameters['lasso'], cv=10, n_jobs=-1)


# In[ ]:


type(model)


# In[ ]:


# Ignore ConvergenceWarning messages
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


# In[ ]:


# Fit and tune model
model.fit(X_train, y_train)


# In[ ]:


# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    
    # Fit model on X_train, y_train
    model.fit(X_train, y_train)
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    # Print '{name} has been fitted'
    print(name, 'has been fitted.')


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


# Display fitted random forest object
fitted_models['lasso']


# In[ ]:


# Predict test set using fitted random forest
pred = fitted_models['lasso'].predict(X_test)


# In[ ]:


# Calculate and print R^2 and MAE
print( 'R^2:', r2_score(y_test, pred ))
print( 'MAE:', mean_absolute_error(y_test, pred))


# In[ ]:


# Predict test set using fitted random forest
pred = fitted_models['ridge'].predict(X_test)


# In[ ]:


# Calculate and print R^2 and MAE
print( 'R^2:', r2_score(y_test, pred ))
print( 'MAE:', mean_absolute_error(y_test, pred))


# In[ ]:


# Predict test set using fitted random forest
pred = fitted_models['enet'].predict(X_test)


# In[ ]:


# Calculate and print R^2 and MAE
print( 'R^2:', r2_score(y_test, pred ))
print( 'MAE:', mean_absolute_error(y_test, pred))


# In[ ]:


from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle


# In[ ]:


# Load ABT from Module 3
df = pd.read_csv('analytical_base_table.csv')
print(df.shape)


# In[ ]:


# Create separate object for target variable
y = df.tx_price

# Create separate object for input features
X = df.drop('tx_price', axis=1)


# In[ ]:


# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2,
                                                   random_state=1234)


# In[ ]:


# Create pipelines dictionary
pipelines = {
    'lasso' : make_pipeline(StandardScaler(), Lasso(random_state=123)),
    'ridge' : make_pipeline(StandardScaler(), Ridge(random_state=123))
}

# Add a pipeline for Elastic-Net
pipelines['enet'] = make_pipeline(StandardScaler(), ElasticNet(random_state=123))


# In[ ]:


# Lasso hyperparameters
lasso_hyperparameters = { 
    'lasso__alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 5, 10] 
}

# Ridge hyperparameters
ridge_hyperparameters = { 
    'ridge__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10]  
}

# Elastic Net hyperparameters
enet_hyperparameters = { 
    'elasticnet__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10],                        
    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]  
}


# In[ ]:


# Add a pipeline for 'rf'
pipelines['rf'] = make_pipeline(StandardScaler(),
                                RandomForestRegressor(random_state=123))

# Add a pipeline for 'gb'
pipelines['gb'] = make_pipeline(StandardScaler(),
                                GradientBoostingRegressor(random_state=123))


# In[ ]:


print( pipelines['rf'] )


# In[ ]:


print( type( pipelines['rf'] ) )


# In[ ]:


# Check that we have all 5 model families, and that they are all pipelines
for key, value in pipelines.items():
    print( key, type(value) )


# In[ ]:


# Random forest hyperparameters
rf_hyperparameters = { 
    'randomforestregressor__n_estimators' : [100, 200],
    'randomforestregressor__max_features': ['auto', 'sqrt', 0.33],
}


# In[ ]:


# Boosted tree hyperparameters
gb_hyperparameters = { 
    'gradientboostingregressor__n_estimators': [100, 200],
    'gradientboostingregressor__learning_rate' : [0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth': [1, 3, 5]
}


# In[ ]:


# Create hyperparameters dictionary
hyperparameters = {
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters,
    'lasso' : lasso_hyperparameters,
    'ridge' : ridge_hyperparameters,
    'enet' : enet_hyperparameters
}


# In[ ]:


for key in ['enet', 'gb', 'ridge', 'rf', 'lasso']:
    if key in hyperparameters:
        if type(hyperparameters[key]) is dict:
            print( key, 'was found in hyperparameters, and it is a grid.' )
        else:
            print( key, 'was found in hyperparameters, but it is not a grid.' )
    else:
        print( key, 'was not found in hyperparameters')


# In[ ]:


# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    
    # Fit model on X_train, y_train
    model.fit(X_train, y_train)
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    # Print '{name} has been fitted'
    print(name, 'has been fitted.')


# In[ ]:


# Check that we have 5 cross-validation objects
for key, value in fitted_models.items():
    print( key, type(value) )


# In[ ]:


from sklearn.exceptions import NotFittedError

for name, model in fitted_models.items():
    try:
        pred = model.predict(X_test)
        print(name, 'has been fitted.')
    except NotFittedError as e:
        print(repr(e))


# In[ ]:


for name, model in fitted_models.items():
    pred = model.predict(X_test)
    print( name )
    print( '--------' )
    print( 'R^2:', r2_score(y_test, pred ))
    print( 'MAE:', mean_absolute_error(y_test, pred))
    print()


# In[ ]:


rf_pred = fitted_models['rf'].predict(X_test)
plt.scatter(rf_pred, y_test)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()


# In[ ]:


type(fitted_models['rf'])


# In[ ]:


type(fitted_models['rf'].best_estimator_)


# In[ ]:


fitted_models['rf'].best_estimator_


# In[ ]:


import pickle

with open('final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)


#!/usr/bin/env python
# coding: utf-8

# # Business Problem

# ## Problem Description
# The dataset here is a sample of the transactions made in a retail store. The store wants to know better the customer purchase behaviour against different products. Specifically, here the problem is a regression problem where we are trying to predict the dependent variable (the amount of purchase) with the help of the information contained in the other variables.
# 
# Classification problem can also be settled in this dataset since several variables are categorical, and some other approaches could be "Predicting the age of the consumer" or even "Predict the category of goods bought". This dataset is also particularly convenient for clustering and maybe find different clusters of consumers within it.

# ## Acknowledgements
# The dataset comes from a competition hosted by Analytics Vidhya.

# ## Data Overview
# Dataset of 550 000 observations about the black Friday in a retail store, it contains different kinds of variables either numerical or categorical. It contains missing values.

# ### Target Variable
# In this approach:
# * Purchase- the purchase ammount

# ### Other Features
# * User_ID- unique id of the user
# * Product_ID- unique id of the product
# * Gender- male or female
# * Age- age category the customer belongs to
# * Occupation- Occupation of the customer
# * City_Category-city category the customer resides in
# * Stay_In_Current_City_Years- no. of years the customer has resided in the current city
# * Marital_Status- married or unmarried
# * Product_Category_1- products of category 1 
# * Product_Category_2- products of category 2
# * Product_Category_3- products of category 3
# * Purchase- Purchase amount in dollars

# ## Exploratory Data Analysis

# #### Importing the libraries

# In[ ]:


# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
# import color maps
from matplotlib.colors import ListedColormap

# Seaborn for easier visualization
import seaborn as sns

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

from math import sqrt

# Function for splitting training and test set
from sklearn.model_selection import train_test_split

# Function to perform data standardization 
from sklearn.preprocessing import StandardScaler

# Libraries to perform hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Import classes for ML Models
from sklearn.linear_model import Ridge  ## Linear Regression + L2 regularization
from sklearn.svm import SVR ## Support Vector Regressor
from sklearn.ensemble import RandomForestRegressor ## Random Forest Regressor
from sklearn.neighbors import KNeighborsRegressor ## KNN regressor
from sklearn.tree import DecisionTreeRegressor ## Decision Tree Regressor
from sklearn import linear_model ## Lasso Regressor

# Evaluation Metrics
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae

# To save the final model on disk
from sklearn.externals import joblib


# #### Loading data from csv

# In[ ]:


df = pd.read_csv('../input/BlackFriday.csv')


# Display the dimension of the dataset

# In[ ]:


df.shape


# Display the first 5 rows

# In[ ]:


df.head()


# Some features are numerical and some are categorical

# In[ ]:


df.dtypes[df.dtypes=='object']


# ## Distribution of numerical features

# In[ ]:


# Plot histogram grid
df.hist(figsize=(15,15), xrot=-45) ## Display the labels rotated by 45 degress

# Clear the text "residue"
plt.show()


# #### Observations
# From the hostagram of Product_category_1
# * There are most unmarriedd customers in the dataset.
# * The product 5 are most bought by the customers.
# * And so are product 1 and 8.
# 
# This information can be used to know about the products which have a high demand.

# In[ ]:


df.describe()


# #### Observations
# * Columns Product_Category_2 and Product_Category_3 have missing values.
# * Marital status has min and max values 0 and 1.Therefore, this might be a binary feature.

# ## Distribution of Categorical features

# In[ ]:


df.describe(include=['object'])


# #### Observations
# * There are no missing values in categorical features.
# * There are 3623 unique products.
# * Most purchases have occured from the age group of 26 to 35 among the 7 unique age groups.
# 
# This information can be used to give better suggestions on products if the age of the customers are provided.
# * Most purchases have occured from the city category 'B'.
# 
# This information can be used to give better suggestions on products if the city category is provided.

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(y='Age', data=df)


# #### Observations
# * The classes 26-35, 36-45, 18-25 are quite prevalent in the dataset.

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(y='Gender', data=df)


# #### Observations 
# * Most customers are males

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(y='City_Category', data=df)


# #### Observations
# * Most customers are from city B

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(y='Stay_In_Current_City_Years', data=df)


# ## Correlations

# * Finally, let's take a look at the relationships between numeric features and other numeric features.
# * ***Correlation*** is a value between -1 and 1 that represents how closely values for two separate features move in unison.
# * Positive correlation means that as one feature increases, the other increases; eg. a child's age and her height.
# * Negative correlation means that as one feature increases, the other decreases; eg. hours spent studying and number of parties attended.
# * Correlations near -1 or 1 indicate a strong relationship.
# * Those closer to 0 indicate a weak relationship.
# * 0 indicates no relationship.

# In[ ]:


df.corr()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr())


# **What to look for?**
# * The colorbar on the right explains the meaning of the heatmap - Dark colors indicate **strong negative correlations** and light colors indicate **strong positive correlations**.
# * Perhaps the most helpful way to interpret this correlation heatmap is to first find features that are correlated with our target variable by scanning the last column.
# * In this case, it doesn't look like many features are strongly correlated with the target variable.
# * Seems like there is negative correlation between the columns 'Purchase' and 'Product_Category_1'.

# In[ ]:


mask=np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(15,10))
with sns.axes_style("white"):
    ax = sns.heatmap(df.corr()*100, mask=mask, fmt='.0f', annot=True, lw=1, cmap=ListedColormap(['green', 'yellow', 'red','blue']))


# ## Data Cleaning

# ### Dropping the duplicates (De-duplication)

# In[ ]:


df = df.drop_duplicates()
print( df.shape )


# Looks like our data had no duplicates.

# ## Handling the missing values

# * There were missing values in the columns 'Product_Category_1' and 'Product_Category_2'.

# In[ ]:


df.Product_Category_2.unique()


# In[ ]:


df.Product_Category_2.fillna(0, inplace=True)


# In[ ]:


df.Product_Category_2.unique()


# In[ ]:


df.Product_Category_3.unique()


# In[ ]:


df.Product_Category_3.fillna(0, inplace=True)


# In[ ]:


# Display number of missing values by numeric feature
df.select_dtypes(exclude=['object']).isnull().sum()


# We don't have any numerical features with missing values

# ## Feature Engineering

# * Feature engineering is finding out new features from the existing ones.
# * This helps us isolating key information.

# #### Filtering the data

# In[ ]:


# female: 0 and male: 1
def gender(x):
    if x=='M':
        return 1
    return 0

df['Gender']=df['Gender'].map(gender)


# In[ ]:


# Defining different age groups
def agegroup(x):
    if x=='0-17':
        return 1
    elif x=='18-25':
        return 2
    elif x ==  "26-35" :
        return 3
    elif x ==  "36-45" :
        return 4
    elif x ==  "46-50" :
        return 5
    elif x ==  "51-55" :
        return 6
    elif x ==  "55+" :
        return 7
    else:
        return 0
    
df['AgeGroup']=df['Age'].map(agegroup)


# In[ ]:


df.drop(['Age'],axis=1,inplace=True)


# #### Indicator Variables

# * We know that bachelors are found mostly in the range 25 to 35.
# * Therefore we can make a feature combining the 'AgeGroup','Gender' and 'Marital_Status'

# In[ ]:


df['Bachelor']=((df.AgeGroup == 2) & (df.Marital_Status == 0) & (df.Gender == 1)).astype(int)


# In[ ]:


# Display percent of rows where Bachelor == 1
df[df['Bachelor']==1].shape[0]/df.shape[0]


# Almost 11% of the data contains bachelors.This information will be helpful to suggest products for bachelors

# #### Encoding Dummy Variables

# * Before feeding the data to the machine learning algorithm, we need to convert categorical features into numerical features.
# * Therefore we need to create dummy variables for our categorical features.

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


P = LabelEncoder()
df['Product_ID'] = P.fit_transform(df['Product_ID'])
U = LabelEncoder()
df['User_ID'] = P.fit_transform(df['User_ID'])


# In[ ]:


# Create a new dataframe with dummy variables for for our categorical features.
df = pd.get_dummies(df, columns=['City_Category', 'Stay_In_Current_City_Years'])


# In[ ]:


df.head()


# In[ ]:


df.shape


# ## Machine Learning Models

# #### Data Preparation

# In[ ]:


df.shape


# #### Test Train Split

# Since this is an enormous dataset, my machine won't be capable enough to run machine learning models.Therefore lets take a sample of 50000 data points for evaluation.

# In[ ]:


sample_df = df.sample(n=50000,random_state=100)


# In[ ]:


# Create separate object for target variable
y = sample_df.Purchase
# Create separate object for input features
X = sample_df.drop('Purchase', axis=1)


# In[ ]:


# Split X and y into train and test sets: 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# In[ ]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# #### Data Standardization
# * In data standardization we perform zero mean centering. i.e. we make the mean of all the features 0 and the standard deviation as 1.

# In[ ]:


train_mean = X_train.mean()
train_std = X_train.std()


# In[ ]:


## Standardize the train data set
X_train = (X_train - train_mean) / train_std


# In[ ]:


## Check for mean and std dev.
X_train.describe()


# In[ ]:


## Note: We use train_mean and train_std_dev to standardize test data set
X_test = (X_test - train_mean) / train_std


# In[ ]:


## Check for mean and std dev. - not exactly 0 and 1
X_test.describe()


# ### Model 1: Baseline Model

# * In this model, for every test data point, we will simply predict the average of the train labels as the output.
# * We will use this simple model to perform hypothesis testing for other complex models.

# In[ ]:


## Predict Train results
y_train_pred = np.ones(y_train.shape[0])*y_train.mean()


# In[ ]:


## Predict Test results
y_pred = np.ones(y_test.shape[0])*y_train.mean()


# In[ ]:


print("Train Results for Baseline Model:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))


# In[ ]:


print("Results for Baseline Model:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_test, y_pred)))
print("R-squared: ", r2_score(y_test, y_pred))
print("Mean Absolute Error: ", mae(y_test, y_pred))


# ### Model 2: Ridge Regression

# In[ ]:


tuned_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
model = GridSearchCV(Ridge(), tuned_params, scoring = 'neg_mean_absolute_error', cv=10, n_jobs=-1)
model.fit(X_train, y_train)


# In[ ]:


model.best_estimator_


# In[ ]:


## Predict Train results
y_train_pred = model.predict(X_train)


# In[ ]:


## Predict Test results
y_pred = model.predict(X_test)


# In[ ]:


print("Train Results for Ridge Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))


# In[ ]:


print("Test Results for Ridge Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_test, y_pred)))
print("R-squared: ", r2_score(y_test, y_pred))
print("Mean Absolute Error: ", mae(y_test, y_pred))


# #### Feature Importance

# In[ ]:


## Building the model again with the best hyperparameters
model = Ridge(alpha=0.0001)
model.fit(X_train, y_train)


# In[ ]:


indices = np.argsort(-abs(model.coef_))
print("The features in order of importance are:")
print(50*'-')
for feature in X.columns[indices]:
    print(feature)


# ### Model 3: Lasso Regression

# In[ ]:


tuned_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
model = GridSearchCV(linear_model.Lasso(), tuned_params, scoring = 'neg_mean_absolute_error', cv=10, n_jobs=-1)
model.fit(X_train, y_train)


# In[ ]:


model.best_estimator_


# In[ ]:


## Predict Train results
y_train_pred = model.predict(X_train)


# In[ ]:


## Predict Test results
y_pred=model.predict(X_test)


# In[ ]:


print("Train Results for Lasso Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))


# In[ ]:


print("Test Results for Lasso Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_test, y_pred)))
print("R-squared: ", r2_score(y_test, y_pred))
print("Mean Absolute Error: ", mae(y_test, y_pred))


# In[ ]:


## Building the model again with the best hyperparameters
model = linear_model.Lasso(alpha=0.0001)
model.fit(X_train, y_train)


# In[ ]:


indices = np.argsort(-abs(model.coef_))
print("The features in order of importance are:")
print(50*'-')
for feature in X.columns[indices]:
    print(feature)


# ### Model 4: Random Forest Regression

# In[ ]:


tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
model = RandomizedSearchCV(RandomForestRegressor(), tuned_params, n_iter=20, scoring = 'neg_mean_absolute_error', cv=5, n_jobs=-1)
model.fit(X_train, y_train)


# In[ ]:


model.best_estimator_


# In[ ]:


## Predict Train results
y_train_pred = model.predict(X_train)


# In[ ]:


## Predict Test results
y_pred = model.predict(X_test)


# In[ ]:


print("Train Results for Random Forest Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))


# In[ ]:


print("Test Results for Random Forest Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_test, y_pred)))
print("R-squared: ", r2_score(y_test, y_pred))
print("Mean Absolute Error: ", mae(y_test, y_pred))


# #### Feature Importance

# In[ ]:


## Building the model again with the best hyperparameters
model = RandomForestRegressor(n_estimators=200, min_samples_split=10, min_samples_leaf=4)
model.fit(X_train, y_train)


# In[ ]:


indices = np.argsort(-model.feature_importances_)
print("The features in order of importance are:")
print(50*'-')
for feature in X.columns[indices]:
    print(feature)


# ### Saving the Winning model to disk

# In[ ]:


joblib.dump(model, 'rfr_BlackFriday.pkl') 


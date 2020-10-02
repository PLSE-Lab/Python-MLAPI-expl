#!/usr/bin/env python
# coding: utf-8

# # Journey From rank 30000 to 3852 in Housing Prices Competition
# 
# This is my first public notebook where i want to share my journey from the first submission and scoring a rank of 30000 (might be off few hundreds) till the tenth submission where i managed to secure a rank of 3852.
# 
# First things first:
# 1. I was in search of free courses which could give me handson experience with Machine learning and thats when a friend of mine recommended the micro courses offered by Kaggle.com
# 2. These micro courses set me up in the right direction. 
# I started with **[Python](https://www.kaggle.com/learn/python)**. This is the prerequisite for most of the micro-courses.
# This gave me the right foundation and helped me with the required coding skills. 
# 3. Then i took the course on **[Pandas](https://www.kaggle.com/learn/pandas)**. This course helped me to gain the required skills for data manipulation and data analysis.
# 4. With the above two skills, i then took the **[Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)**. By the end of this course, i was very excited as it guided me to make my first submission on Kaggle learn.
# 5. The **[Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning)** course helped me with key concepts (Handling missing values, Handling categorical values, Pipelines) for improving my score and securing a higher rank.
# 
# This notebook contains the practice code which i used in order to make my updated submission. I am continuing with the other micro courses and will continue to strive in order to improve my rank and ML skills.
# 
# Hope this helps the new ML enthusiasts to continue their AI/ML journey 
# ---
# 

# As notified at the beginning, we will work with data from the [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course). 
# 
# ![Ames Housing dataset image](https://i.imgur.com/lTJVG4e.png)
# 
# 

# # **Step 1: Loading the data** 
# ## Preparing the training, validation & test datasets

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)


# # Step 2: Data Cleansing
# **Identifying the Categorical Columns and Numerical Columns**
# 1. Categorical Columns were selected based on the following conditions:
#     * (column - dtype = Object) and (cardinality(number of unique values) < 10) and (80% of the rows had data present)
# * Numerical Columns were selected based on the following conditions:
#     * (column - dtype is either 'int64' or 'float64') and (80 % of the rows had data present)

# In[ ]:


# threshold value was set. Columns with data less than the threshold value were dropped.

thres = int(0.8*len(X_train_full)) # 934

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality and less missing values
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object" and
                    X_train_full[cname].notna().sum() > thres]

# Selecting numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64'] and
                X_train_full[cname].notna().sum() > thres]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


# In[ ]:


print('Number of selected numerical columns: ',len(numerical_cols))
print('Number of selected categorical columns: ',len(categorical_cols))


# In[ ]:


# identifying the columns with missing values. These missing data in the below columns will be imputed with values

emp_cols_dict = {col:X_train[col].isnull().sum() for col in X_train.columns if X_train[col].isnull().any()}
emp_cols_dict


# # Step 3 - Data Preprocessing
# 
# 1. For preprocessing of numerical data - we will impute the data using the SimpleImputer()
# 2. For preprocessing of categorical data - we will impute the data using SimpleImputer() and then encode the data using OneHotEncoder()
# 3. Pipelines will be used to organize the sequential processing of the columnar data

# In[ ]:


# importing the necessary libraries for preprocessing, building a pipeline.

# Hyperparameter Tuning for the SimpleImputer strategy with Median yielded better results

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Preprocessing for numerical data using SimpleImputer()
numerical_transformer = SimpleImputer(strategy='median') 

# Preprocessing for categorical data using SimpleImputer() and OneHotEncoder()
categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data using a ColumnTransformer()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# # Step 4: Model definition and generating predictions

# In[ ]:



# Define model

# Using the XG boost regressor model to estimate the model parameter and make predictions 
# I experimented with the values of n_estimators ranging from 100-1000 and 
# learning_rate values from 0.1-0.09 in order to identify the best accuracy

model = XGBRegressor(n_estimators=750, learning_rate = 0.06,random_state=0)


# Bundle the preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)


# # Step 5: Generate test predictions and preparation of the submission data file
# 
# 

# In[ ]:


# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test) # Your code here


# In[ ]:


# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)


# # Key Notes
# 
# This notebook is meant for encouraging the new ML enthusiasts by showing a way to improve ranking in the Housing prices competetion.
# The notebook has room for improvement. There are several other techniques which have to be mastered by me in order to move up in the rankings. I will continue with my journey and wish all the new ML enthusiasts "happy learning".
# 

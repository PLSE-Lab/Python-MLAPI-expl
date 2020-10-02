#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This notebook explores AUTO mpg dataset. 
# 
# 
# The objective of this data science project is to predict **miles per gallon (mpg)** for a given car. 
# 
# Before seeing the data let us answer 3 questions:
# 
# 1. Is it a supervised, unsupervised, or Reinforcement Learning?
#         This is a supervised learning. We know that we have to predict miles per gallon (mpg).
# 2. Is it a classification task, a regression task, or something else?
#         We have to predict Miles per gallon. mpg is a measure of fuel economy of a car and it is a numerical value. Hence this is a regression task.
# 3. Should we use batch learning or online learning techniques?
#         As there is no continuous flow of data coming into the system, so batch learning is enough.
# 

# Let us begin with importing necessary libraries.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='svg'")


# # 1. Getting the data

# In[ ]:


df = pd.read_csv('../input/auto-mpg.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# Dataset has 398 observations and 9 feature columns.
# Independent variables are :
# 1. car name
# 2. cylinders
# 3. displacement
# 4. horsepower
# 5. weight
# 6. acceleration
# 7. model year
# 8. origin
# 
# Dependent variable is:
# 9. mpg

# Let us check the data type of the columns.

# In[ ]:


def check_data_types(data):
    data_types = data.dtypes.reset_index()
    data_types.columns = ['columnn_name','data_type']
    return data_types 


# In[ ]:


check_data_types(df)


# We have 7 numeric(3 continuous and 4 discrete) and 2 object columns. Don't you think that 'horsepower' should be a numeric value and not an object? Then, let us change the datatype of horsepower from object to numeric.

# df['horsepower'] = df['horsepower'].astype('float64')
# 
# This line of code throws as error saying "could not convert string to float: '?'". What does this mean?
# This means that this column has an entry with value '?' and this cannot be converted to float. Let us check how many rows have '?' in 'horsepower' columns.

# In[ ]:


df.loc[df.horsepower=='?']


# There are 6 rows with '?' in the column 'horsepower'. Let us get rid of these rows.

# In[ ]:


df = df.loc[df.horsepower!='?']


# Now, lets try changing the datatype of horsepower from object to float64

# In[ ]:


df['horsepower'] = df['horsepower'].astype('float64')


# In[ ]:


check_data_types(df)


# Cool!! Thats done. But are there missing values in other columns? Let us check. But before that let us save our features in different lists based on their data types.

# In[ ]:


numerical_cont = ['displacement', 'horsepower', 'weight', 'acceleration','mpg']
numerical_discrete = ['cylinders','model_year', 'origin']
categorical = ['car_name']


# Checking missing values. No missing values are present in the data.

# In[ ]:


df.isnull().any()


# # 2. Exploring the data

# Let us study the correlation between the columns.

# In[ ]:


corr_matrix = df.corr()
sns.heatmap(corr_matrix)


# From the above heatmap, we can observe that cylinders, displacement, horsepower and weight are strongly and negatively correlated with mpg. acceleration, model_year and origin are weakly and positively correlated with mpg.

# Let us plot scatter_matrix to view the relationship between the numerical predictors.

# In[ ]:


from pandas.plotting import scatter_matrix
scatter_matrix(df[numerical_cont],figsize=(12,12))
plt.show()


# From the last row of above plot, we can observe the relationship between mpg and other predictors displacement, horsepower, weight, acceleration. As observed from the correlation matrix, displacement, horsepower and weight are negatively correlated with mpg. acceleration doesn't show strong correlation with acceleration.

# Now, let us explore relationship between our dependent variable 'mpg' and numeric_discrete predictors.

# In[ ]:


sns.boxplot(x = 'cylinders', y = 'mpg', data = df, palette = "Set2")


# Mpg of cars having Cylinder 4 is the highest. 

# In[ ]:


sns.boxplot(x = 'origin', y = 'mpg', data = df, palette = "Set2")


# Origin column is basically the origin where car was manufactured. On average cars from origin 3 have higher mpg.

# In[ ]:


sns.boxplot(x = 'model year', y = 'mpg', data = df, palette = "Set2")


# median mpg of cars seem to be increasing with each year.

# # 3. Building a pipeline for further data processing

# There are 2 different pipelines for processing numerical data and categorical data:
# 1. num_pipeline has 2 transformers. Imputer for handling missing values and standard scalar for scaling the features. 
# 2. cat_pipeline has 2 transformers. Imputer for handling the missing values and OneHotEncoder for encoding categorical variables.
# 
# Note : I am going to drop 'car name' column from the analysis.

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

num_attribs = ['displacement', 'horsepower', 'weight', 'acceleration']
cat_attribs = ['cylinders','model year', 'origin']


num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore',sparse=False)),
])


# ColumnTransformer applies each transformer to the appropriate columns and concatenates the outputs along the second axis.

# In[ ]:


from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])


# Applying the pipeline to our dataset.

# In[ ]:


df_x = df.drop(columns=['mpg'])
df_y = df['mpg']
df_MPG = full_pipeline.fit_transform(df_x, df_y)


# Let us view the transformed dataset.

# In[ ]:


pd.DataFrame(df_MPG).head()


# # 4. Create Train Test Split

# Let us create train and test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_MPG, df_y, test_size = 0.2, random_state = 42)


# # 5. Fitting regression model 

# a. Lets create a baseline with Naive model. Let us assume that the estimate of mpg for test data is equal to the average of mpg values in test data.

# In[ ]:


from sklearn.metrics import mean_squared_error
mpg_mean = np.mean(y_train)
y_pred = np.full(len(X_test),mpg_mean)

print('test loss is....')
print(np.sqrt(mean_squared_error(y_pred,y_test)))


# b. Now let us fit linear regression model and see if it does any better than naive model.

# In[ ]:


from sklearn.linear_model import LinearRegression


regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print('test loss is:')
print(np.sqrt(mean_squared_error(y_pred,y_test)))

y_pred = regressor.predict(X_train)
print('train loss is:')
print(np.sqrt(mean_squared_error(y_pred,y_train)))


# Linear regression model certainly does better than our naive model. We can fit other regressors like RandomForest, XGBoost etc, but let me save that for my next task.

# This is my first data science project and I learnt a great deal executing it. Please leave a comment if you have any feedback. I look forward to learning from you. Thank you!

# In[ ]:





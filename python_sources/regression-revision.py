#!/usr/bin/env python
# coding: utf-8

# # Regression Revision with the World Happiness Report Dataset
#   **Content:**
#   1. [Introduction](#1)
#       1. [Dataset](#2)
#       1. [Plan of action](#2)
#   1. [Exploring the data](#2)  
#       1. [Example](#2)
#       1. [Your Turn](#2)
#   1. [Data Preprocessing](#2)
#   1. [Linear Regression](#3)
#       1. [Train the model and predict results](#2)
#       1. [Measure Accurracy](#2)
#       1. [Your Turn](#2)
#   1. [Polynomial Regression](#4)
#       1. [Train the model and predict results](#2)
#       1. [Measure Accurracy](#2)
#       1. [Visualisation](#2)
#   1. [Random Forest](#5)
#       1. [Your Turn](#2)

#   ## Introduction
#    We will have a look at the regression models once again. Github Repository for the relevant code can be found [here](https://github.com/KacperKubara/USAIS_Workshops/tree/master/Regression). Previous USAIS presentations on regression models can be found [here](https://drive.google.com/open?id=1k5twAgSVGUl8CGjw0qmQitPf6ij0YrIj)

#   ### Dataset
#   [World Happiness Report Dataset](https://www.kaggle.com/unsdsn/world-happiness/home) will be used to implement our Machine Learning models. I would suggest to follow the link first to understand the dataset a bit better.  

#  ### Plan of action  
#   Before we dive into training and testing the model, we will take a step back and have a closer look on the dataset. Starting from simple dataset visualisations, we are going to decide how to format the data for the Machine Learning model. Then we will try using few different Regression models and change their parameters to see what performs the best.

# ## Exploring the data   
# ### Example
# Before plugging in the data to the Machine Learning model it is good to see what the dataset contains and try to guess what features can be important for the model. We will do few simple visualisations of the dataset to understand it better.

# Importing the packages and listing the available datasets

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualisations

import os
print(os.listdir("../input")) # print all the files from the "input" directory (we provided this directory! When using locally inject path to your files) 


# As we can see from the output above, we have 3 .csv files to import:

# In[ ]:


dataset_2015 = pd.read_csv("../input/2015.csv") # Read csv file
dataset_2016 = pd.read_csv("../input/2016.csv")
dataset_2017 = pd.read_csv("../input/2017.csv")

dataset_2015 = dataset_2015.sample(frac=1).reset_index(drop=True) # randomly shuffle the rows (for the purpose of our exercise later)
dataset_2016 = dataset_2016.sample(frac=1).reset_index(drop=True)
dataset_2017 = dataset_2017.sample(frac=1).reset_index(drop=True)


# Let's get a better understanding of how the data looks like!

# In[ ]:


pd.set_option('display.max_columns', None) #  Ensures that all columns will be displayed
print(dataset_2015.head(3)) # Prints first 3 entries
print('\n\n')

print('COLUMN NAMES')
print(dataset_2015.columns) # Prints the column names
print("NUMBER OF COLUMNS: " + str(len(dataset_2015.columns))) # Prints no. columns


# 

# We can see how the variables are correlated with themselves by using the correlation matrix. Correlation matrix simply shows how each feature is correlated to another one. '1' stands for the biggest positive correlation (increase in x_1 leads to increase in x_2), '-1' stands for biggest negative correlation (increase in x_1 leads to decrease in x_2) and '0' for no correlation at all.  
#  More information about correlation matrix can be found [here](https://www.datascience.com/blog/introduction-to-correlation-learn-data-science-tutorials).

# In[ ]:


import seaborn as sns # Library for more fancy plots
corr= dataset_2015.corr() # Creates correlation matrix
sns.heatmap(corr, xticklabels=corr.columns.values,
            yticklabels=corr.columns.values) # Creats a heatmap based on correlation matrix


# Since we are going to predict the Happiness Score, we should look for the brightest or the darkest cells under the column **''Happiness Score"** . Thus, features worth trying in our future model should include **Economy (GDP per capita)**, **Family** and **Health (Life Expectancy)**. 

# ### **Your Turn**  
# You can visualise and analyse the dataset in many ways. The more time you spend on understanding the dataset, the more likely you will come up with a better choice of the ML model.  
#  ** Task for you: **
# 1.  Display the countries with the highest Happiness score in 2015 using the Bar graph (Look [here]() to get a better idea on how to do it)

# In[ ]:


# Write the code here


# ## Data Preprocessing  
#  To train the Machine Learning Model we need to provide the data  in a correct format. It is worth looking at the documentation of the specific ML model to understand how preprocess the data. In general, the steps we are going to go through prepare the data for the Regression models are :
#  1. **Choose variable which you want to predict and features to train the model on:**
#  Simply speaking 'xs' are the model features and 'y' is the output we are going to predict
#  
#  1. **Split the dataset into train and test data:**
# On the train data we can train model. Test data helps us measure how accurate the model is. There are different accuracy metrics for regression models, for example **Mean Squared Error (MSE)** or **Mean Absolute Error (MAE)**. In this tutorial we will stick to **MSE** as it is a quite popular choice for regression models. More information about the accuracy metrics can be found in Sklearn [documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).
# 
#  1. **Scale/Normalise the data**:
# We need to normalise all of the numerical features so each of them have the same significance during the model training. For example, is feature x_1 has values of magnitude 1000 and feature x_2 of magnitude 10, x_2 won't contribute much to the training of the model. Awesome explanation can be found [here](https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e).
# 

#  1. **Choose variable which you want to predict and features to train the model on:**

# In[ ]:


# Choosing variable to be predicted
y = dataset_2015['Happiness Score'] # Happiness Score chosen as a value to be predicted


# ### **Your Turn**  
# Choose the features you want to include in your model. If the features are not numerical ( E.g. column 'Country' has a string value). There should be an additional preprocessing step (LabelEncoder and OneHotEncoder in Sklearn). I've uploaded the template in the [Github repo](https://github.com/KacperKubara/USAIS_Workshops/tree/master/Other) if you want to include the categorical data in your model.   
#  ** Task for you: **  
#   **1. Choose the feature you want to train the model on. Use LabelEncoder and OneHotEncoder if you want to include the categorical data as well.**

# In[ ]:


# Write the code here


# 2. **Split the dataset into train and test data:**

# In[ ]:


# Split data into train and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


#  3. **Scale/Normalise the data**:

# In[ ]:


# Scale/Normalise the data
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

# Unscaled data - we will need it for Random Forest Regression and plotting the LinearRegression model
x_train_unscaled = x_train.copy()
x_test_unscaled  = x_test.copy()
y_train_unscaled = y_train.copy()

# Scale the data
x_train = sc_x.fit_transform(x_train)
x_test  = sc_x.transform(x_test)
y_train = y_train.values.reshape(-1, 1)
y_train = sc_y.fit_transform(y_train)

print('SCALED X_TRAIN:')
print(x_train[:5])
print('\nSCALED Y_TRAIN:')
print(y_train[:5])


# ## Linear Regression  
# Linear Regression is the simplest ML model. It is simply the best-fitting line for the dataset. Quick explanation can be found in our previous [presentation](https://www.beautiful.ai/deck/-LXYxxcxN4Hu0yzhjV6Y/AI-Society-Launch) (starts from the 10th slide): 

# ### Train the model and predict results  
#  

# In[ ]:


# Train Model
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x_train, y_train) # training the model - simple as that huh

# Predict Results
y_pred = regr.predict(x_test)
y_pred = sc_y.inverse_transform(y_pred)
print('PREDICTED VALUES (UNSCALED): ')
print(y_pred[:5])


# ### Measure Accuracy

# In[ ]:


# Measure Accuracy
from sklearn.metrics import mean_squared_error
acc = mean_squared_error(y_test, y_pred) # Mean Squared Error to measure the accuracy
print('ACCURACY(MSE): ')
print(acc)


# ### **Your Turn**  
# You can also visualise the results on 2D plot having one feature as 'X' and another one as 'Y'.
#  ** Task for you: **  
#   **1. Visualise the results with a plot (e.g. line or scatter plot)**

# In[ ]:


plt.title("Linear Regression Prediction")
plt.xlabel("Some Feature")
plt.ylabel("Happiness Score")
# Values in blue are those predicted by the model
plt.scatter(x_test_unscaled['Family'], y_pred, color = 'b')
# Values in red are orignal dataset points
plt.scatter(x_test_unscaled['Family'], y_test, color = 'r') 
# Display graph
plt.show()


# ## Polynomial Regression
#   Polynomial regression is quite similar to Linear regression. The only difference is that it will fit **polynomial function** instead of the **linear one**. In fact, you can see in code below that the code is still based on the **LinearRegression** Class

# ### Train the model and predict results  

# In[ ]:


# Import libraries for the Polynomial Regression model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Creates separate variables for each of the polynomial degree of the feature 
poly_reg = PolynomialFeatures(degree = 3)
x_poly_train = poly_reg.fit_transform(x_train)
x_poly_test  = poly_reg.fit_transform(x_test)
poly_reg.fit(x_poly_train, y_train)

# Fit the polynomial features to the LinearRegression model
lin_reg = LinearRegression()
lin_reg.fit(x_poly_test, y_test)

# Predict the result
y_pred = lin_reg.predict(x_poly_test)

# Unscale the data
y_pred = sc_y.inverse_transform(y_pred)
print('PREDICTED VALUES (UNSCALED): ')
print(y_pred[:5])


# ### Measure Accuracy

# In[ ]:


# Measure Accuracy
from sklearn.metrics import mean_squared_error
acc = mean_squared_error(y_test, y_pred)

print('ACCURACY(MSE): ')
print(acc)


# 

# ## Random Forest  
#   
# 

# Some description here

# ### **Your Turn**  
#  ** Task for you: **  
#   **1. Create a Random Forest Regression Model and fit it into the dataset. **  
#   **1. Plot the results as in the linear regression example above **
# 

# In[ ]:


# Train Model
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators = 10, random_state = 42)

# For the RandomForest we don't have to scale the data
# which is due to the different algorithm being used
# Use x_train_unscaled, y_train_unscaled and x_test_unscaled instead!

# Fit the model

# Predict Results

# Predict the values


# In[ ]:


# Measure Accuracy
from sklearn.metrics import mean_squared_error
acc = mean_squared_error(y_test, y_pred)

print('ACCURACY(MSE): ')
print(acc)


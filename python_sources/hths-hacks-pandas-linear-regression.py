#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # Pandas library for data processing and CSV file read

filepath = '../input/usa-cers-dataset/USA_cars_datasets.csv' # Directory to ge to car dataset

#print(pd.read_csv(filepath) # Print full CSV contents for debugging

price = pd.read_csv(filepath, usecols =["price"], squeeze = True) # Save price data as list
mileage = pd.read_csv(filepath, usecols =["mileage"], squeeze = True) # Save mileage data as list

print(price) # Print price to console
print(mileage) # Print mileage to console


# Parse the CSV file using pandas library
# 
# Next, try to find the variance of price and standard deviation of mileage using pandas operations.

# In[ ]:


import pandas as pd # Pandas library for data processing and CSV file read

filepath = '../input/usa-cers-dataset/USA_cars_datasets.csv' # Directory to ge to car dataset

price = pd.read_csv(filepath, usecols =["price"], squeeze = True) # Save price data as list
mileage = pd.read_csv(filepath, usecols =["mileage"], squeeze = True) # Save mileage data as list

print("Variance - " + str(pd.DataFrame(data=price).var())) # Get the variance of the car prices and print it to console
print("Standard deviation - " + str(pd.DataFrame(data=mileage).std())) # Get the standard deviation of car mileages and print it to console


# Find the variance of the price data and standard deviation of the mileage data set.
# 
# You should get 1.467998e+08 for price variance and 59705.516356 for mileage standard deviation
# 
# Next, use what you learned from the previous linear regression tutorial - regress price (x variable) against mileage (y variable)

# In[ ]:


import pandas as pd # Pandas library for data processing and CSV file read
from sklearn.model_selection import train_test_split # Sklearn library to split data set
from sklearn.linear_model import LinearRegression # Sklearn library for linear regression

filepath = '../input/usa-cers-dataset/USA_cars_datasets.csv' # Directory to ge to car dataset

price = pd.read_csv(filepath, usecols =["price"], squeeze = True) # Save price data as list
mileage = pd.read_csv(filepath, usecols =["mileage"], squeeze = True) # Save mileage data as list

x_train, x_test, y_train, y_test = train_test_split(price, mileage, test_size=0.33, random_state=42) # Create the x and y data sets

def reshape(data):
    return np.array(data).reshape(-1, 1) # Make it a numpy array and change it to be from a 1D list to 2D list

x = reshape(x_train) # Reshape x variables
y = reshape(y_train) # Reshape y variables

reg = LinearRegression() # Make the regression object
reg.fit(x, y) # Regress the data set

print("Our m: ", reg.coef_) # Print slope to console
print("Our b: ", reg.intercept_) # Print intercept to console


# Linear regress the price vs. mileage data set.
# 
# You should get -1.95964649 as the slope and 88962.12569717 as the intercept of the function.

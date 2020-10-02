#!/usr/bin/env python
# coding: utf-8

# ## Your first machine learning model 
# This tutorial describes how to write a simple machine learning model using Python and library for machine learning (panda).

# Run this cell to load the pandas library and the data files

# In[ ]:


# library to manipulate data in Python
import pandas as pd

# Path of the file to read the data set.
# we call it Xy_data containing the data of features of each house/flat `X`
# and the target variable `y` that will be used to train the model
# These files (look at the left panel, label: 'Data') are 
# in the directory: '../input/epf-3a/.'
# This directory contains :
#            - test_data.csv 
#            - test_data_reduced.csv
#            - train_data.csv 
#            - train_data_reduced.csv
# The reduced data files test_data_reduced.csv and train_data_reduced.csv consist   
# in the same data that you can found in test_data.csv and train_data.csv except for
# the non numerical columns. In this notebook, we will only work with the reduced 
# data files... figure it out ...
Xy_file_path = '../input/epf-3a/train_data_reduced.csv'
test_file_path = '../input/epf-3a/test_data_reduced.csv'

Xy_data = pd.read_csv(Xy_file_path)
test_data = pd.read_csv(test_file_path)

print("Setup Complete")


# ## Have a quick look to the data

# In[ ]:


# display the shape of the training data set shape=(nb rows, nb columns)
# and the first five rows of the set Xy : 
#    -    one row correspond to one house/flat
#    -    columns are features of each flat described in the description
# of the challenge
print("Shape of the training set: \n",Xy_data.shape)
print("First five rows of the training set: ",Xy_data.head(5))

print('-----------------------------------------------------------------------------------------')
# we do the same for the test data. 
print("Shape of the test set: ",test_data.shape)
print("First five rows of the test set: \n",test_data.head(5))

# Question: why does the test set contains one column less than the training set ?


# ## Specify the training set and the target variable
# We split the set `Xy_data` into two variables:
#    -   `X` which contains the features of each house/flat
#    -   `y` which contains the target variable which is the price of the house/flat
#    

# In[ ]:


# we select all the features of the training data except the Price
features = ['Build Year','Living Area','Number of Bedrooms','Number of Photos','Total Number of Rooms','Zip Code']
X = Xy_data[features]
# we print the first ten elements of the variable `X`
print(X.head(10))

# we select the `Price` column of the training set as the
# variable that we have to predict for the features of
# the test set.
y = Xy_data['Price']

print('-----------------------------------------------------------------------------------------')
# we print the first ten elements of the variable `y`
print(y[:10])


# ## Review of the data

# In[ ]:


# we print classical statistical elements (number of elements:count, mean, standard deviation, max, min ....) of each column of the training data X and the target variable y
print('Statistical elements for the columns of the training data: ')
print(X.describe())

print('-----------------------------------------------------------------------------------------')
print('Statistical elements for the target variable: ')
print(y.describe())


# ## Build a model and fit on the data
# 
# Then :
# 1. we choose a model (here, a linear regression model). 
# 
# We also provide, in comments, a way to load two more models: a DecisionTreeRegressor model and a RandomForestRegressor model
# 2. we fit the model that is we compute the parameters of the model with the training data X and the target variable y 
# 

# In[ ]:


'''
 How to validate the model ?
 An important point in mahcine learning is to validate the model. In fact
 if our algorithm is trained from all the data of the training set, even if the model
 has low errors on the training set, there is no guarantee that this algorithm
 will behave good on new data (data that were not used for the train the model, like the test_data).
 For this reason, a way to validate the model is to split the set X in two sets:
       - train_X containing the data used to train/fit the model
       - val_X containing the data to validate the model
 To these two sets, we associate the corresponding target variable: train_y and val_y.      
 The idea is to used train_X to fit the model and then we can predict the values of val_X with our
 model. Then we can compare these predictions with the val_y variables. 

                     |-------------------------------------|---------------|
                     |                                     |               | 
                     |                                     |               |
     X     =         |       train_X, train_y              | val_x,val_y   |
                     |                                     |               |
                     |                                     |               |
                     |-------------------------------------|---------------|
                           These data are used to            These data are
                            train the model                  used to validate
                                                             the model                    
'''   

# we load the function train_test_split to split our set X into train_X and val_X
from sklearn.model_selection import train_test_split
# we separate the data X between two sets : train_X and val_X standing for respectively training data and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# library used to load the linear regression model
from sklearn.linear_model import LinearRegression
# library used to load the DecisionTreeRegressor model
from sklearn.tree import DecisionTreeRegressor
# library used to load the RandomForestRegressor model
from sklearn.ensemble import RandomForestRegressor

# we define the model 
model = LinearRegression()
# uncomment the line below to define a DecisionTreeRegressor model
#model = DecisionTreeRegressor()
# uncomment the line below to define a RandomForestRegressor model
#model = RandomForestRegressor()

# Fit the model: we compute the parameters of the model with the training data : train_X, train_y
model.fit(train_X,train_y)


# ## Make predictions and Compute the error
# Make predictions with the model's `predict` command using `val_X` as the data. Save the results to a variable called `predictions`.

# In[ ]:


# we predict the target variable for the val_X data
predictions = model.predict(val_X)
print("First five predictions on the training data: ",predictions[:4].tolist())

# we expect that our algorithm leads to: predictions = val_y
# because val_y are the target variables of val_X
print("First five true prices:                      ", val_y.head(4).tolist())

# As you can see val_y is different from predictions. These differenteces correspond to the error of our model.
# A way to compute the error is using mean_absolute_error function
# load the function to compute the error
from sklearn.metrics import mean_absolute_error
print("The mean absolute error between val_y and predictions is: ",mean_absolute_error(val_y,predictions))


# ## Application of our model on new data and save our solution file

# In[ ]:


# we make predictions on the test data
my_test_predictions = model.predict(test_data)

print("We made ", len(my_test_predictions), "predictions")
print("First five predictions on the test data: ",my_test_predictions[:5])

# we save our solution to the file "submission.csv" that you can upload and see your rank for this competition (see instructions below)
output = pd.DataFrame({'Id': range(test_data.shape[0]),
                       'Predicted': my_test_predictions})
output.to_csv('submission.csv', index=False)


# # Test Your Work
# 
# To test your results, you'll need to join the competition (if you haven't already).  So open a new window by clicking on [this link](https://www.kaggle.com/c/epf-3a).  Then click on the **Submit Submission** button.
# 
# Next, follow the instructions below:
# 1. Begin by clicking on the blue **COMMIT** button in the top right corner of this window.  This will generate a pop-up window.  
# 2. After your code has finished running, click on the blue **Open Version** button in the top right of the pop-up window.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
# 3. Click on the **Output** tab on the left of the screen.  Then, click on the **Submit to Competition** button to submit your results to the leaderboard.
# 
# You have now successfully submitted to the competition!
# 
# 4. If you want to keep working to improve your performance, select the blue **Edit** button in the top right of the screen. Then you can change your model and repeat the process. There's a lot of room to improve your model, and you will climb up the leaderboard as you work.
# 
# # Continuing Your Progress
# There are many ways to improve your model, and **experimenting is a great way to learn at this point.**
# 
# The best way to improve your model is to add features.  Look at the list of columns and think about what might affect home prices.  Some features will cause errors because of issues like missing values or non-numeric data types. 
# 
# The **[Machine Learning course](https://www.kaggle.com/learn/intro-to-machine-learning)** micro-course will teach you basic notion in machine learning.

# In[ ]:





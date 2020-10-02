#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This will be your workspace for Kaggle's Machine Learning education track.**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.  Fork this notebook and write your code in it.
# 
# The data from the tutorial, the Melbourne data, is not available in this workspace.  You will need to translate the concepts to work with the data in this notebook, the Iowa data.
# 
# Come to the [Learn Discussion](https://www.kaggle.com/learn-forum) forum for any questions or comments. 
# 
# # Write Your Code Below
# 
# 

# The steps in this Kernel follow those outlined in DanB's Learn Machine Learning tutorial (https://www.kaggle.com/dansbecker/starting-your-ml-project).
# 
# I occasionally branch out from those steps when I want additional information about the dataframe.

# In[258]:


import pandas as pd

mb_filepath = '../input/train.csv'
print('hello world')
mb_data = pd.read_csv(mb_filepath)
print(mb_data.describe())


# The outut from the above code tells me that we have 38 variables.
# 
# My addition:  I can see that some of these seem to have more of a numerical spread than others, so I'd like to get the TYPE of variable (for each variable).  
# 
# There is a type() function, but I'd have to run it on each variable (or write a function that assessed and printed the type for each variable).  I found a function instead (select_dtypes) that allows you to specify certain kinds of variables.  So I can say "I only want to see the variables that are integers," which is what I've done below:

# In[259]:


print(mb_data.select_dtypes(include='int')) #returns variables that are integers (useful for initial mb_pred with only continuous vars below)


# The output from the above code is not as useful as it might be, because 35 out of 38 variables are integers (and some--like YrSold and MoSold) are either more categorical in nature, or not likely to have a direct impact on house prices. (Also, the output is abbreviated--it doesn't show me all the columns.)
# 
# What would be more helpful is knowing whether certain integer-type variables are in fact Ordinal (as with OverallQual).  But we may be able to infer that based on the variable name.
# 
# On to the next step--practice selecting a single column (SalePrice) and returning just the head (first couple lines) from that column:

# In[260]:


mb_price_data = mb_data.SalePrice
print(mb_price_data.head())


# Then selecting multiple columns, saving them to a new variable, and giving me descriptives on them:

# In[261]:


interest_col = ['OverallCond', 'YearBuilt']
twocols = mb_data[interest_col]
twocols.describe()


# Identifying the prediction target (y) and the predictors (x)
# 
# Predictors are based on the tutorial's recommendations

# In[262]:


y = mb_data.SalePrice

mb_pred = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = mb_data[mb_pred]

print(X) #Just to check that my predictor list is correct


# Now build the model (Define, Fit, Predict, and Evaluate):
# 
# Define type of model: decision tree regression (bc that's what the tutorial is having us practice)
# 
# Fit model: actually run the model with X and y

# In[263]:


from sklearn.tree import DecisionTreeRegressor

# Define part
mb_model = DecisionTreeRegressor()

# Fit part
mb_model.fit(X,y)


# Predict values for the first 5 houses with the set of predictor variables we specified: 

# In[264]:


print("Predicting sale price for the first 10 houses with our predictors:")
print(X.head(10)) # gives the first 10 rows so we can see the X values for those houses
print("The predictions are:")
print(mb_model.predict(X.head(10))) # shows model's predicted sale prices for the first 10 houses


# In[265]:


print("Predicting sale price for the last 10 houses with our predictors:")
print(X.tail(10)) # gives the last 10 rows so we can see the X values for those houses
print("The predictions are:")
print(mb_model.predict(X.tail(10))) # shows model's predicted sale prices for the last 10 houses


# In[266]:


print("Predicting sale price for a random sample of 10 houses with our predictors:")

import _random as rnd # package that includes a variety of functions for generating/selecting random numbers/samples/objects/etc.

rand_samp = X.sample(10) # selects a random sample of 10 rows)
print(rand_samp) # gives a random sample of 10 rows so we can see the X values for those houses
print("The predictions are:")
print(mb_model.predict(X.sample(10))) # shows model's predicted sale prices for the randome cample of 10 houses


# **Examining the few predictions we've made (a first step I've added in assessing model fit):**
# 
# The 6th prediction in the list (ID #961) strikes me as somewhat odd. 
# 
# $164,500 is the predicted price for a house with 4 bedrooms, 2 full baths, 11 rooms above ground, on a lot of over 12,000 sq ft.  
# 
# Granted, the house was built in 1977, so it is on the somewhat older side, but the fact that it is predicted to sell for less than the 5th house on the list (ID #387), built a year earlier (1976), with only 3 br and 1 full bath, a smaller lot, and no second floor, suggests that our current model may not be capturing all the variables relevant for predicting house prices. 
# 
# So I'll move onto the next step in the tutorial, which is generating summary calculations of model fit:

# In[267]:


from sklearn.metrics import mean_absolute_error

price_pred = mb_model.predict(X) #gives "in-sample" score, which is bad, according to tutorial, bc it uses same set of data to generate model and assess model fit 
mean_absolute_error(y, price_pred)


# The above calculation of model fit is the Mean Absolute Error, but it's calculated based on the same data set used to generate the model, which is NOGOOD. (Tutorial explains the why in more detail here: https://www.kaggle.com/dansbecker/model-validation)
# 
# To better check model fit, we need to use validation data--a different set of data than we used to predict the model.  So we use a function to split the training dataset into two parts, generate the model from the first part, then calculate model fit based on the second part.

# In[268]:


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0) #splits both the predictors and the outcome into two sets ("train"--training--and "val"--validation)
mb_model.fit(train_X, train_y)

# print the mean absolute error (the average difference btwn the actual value of y and the value of y predicted by our model generated using the training set of data)
val_model_pred = mb_model.predict(val_X)
print(mean_absolute_error(val_y, val_model_pred))


# Above output tells us Mean Absolute Error for model is 33,315.93.  Need a point of comparison for whether this is good or bad.  
# 
# Next step is to generate MORE models (of different types, possibly with different sets of predictors) and to compare the MAE from those models to each other.
# 
# *Notes from next module*: want to avoid **overfitting** (matching training data too closely, too many branches in tree, makes poor predictions for new data) and **underfitting** (model doesn't have enough branches, doesn't accurately predict training or test data bc not specific enough) 
# 
# Have to figure out the RIGHT NUMBER OF BRANCHES (which we do by running the model several times with different numbers of nodes--branches--and comparing the MAE of the different models)

# In[269]:


# Create a user-defined function that will generate a model 
# using a specified number of branches (max_leaf_nodes), 
# training- and test-set X values (train_pred and test_pred)
# and training- and test-set Y values (train_val and test_val) values return the MAE for a model 

def get_mae(max_leaf_nodes, train_pred, test_pred, train_val, test_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_pred, train_val) # fit a model using training set X and Y
      ## this generates coefficients, presumably 
    pred_val = model.predict(test_pred) # uses coefficients from model generated on training data to predict test set Y values given test set X values
    mae = mean_absolute_error(test_val, pred_val) # compare predicted test set Y values to ACTUAL test set Y values
    return(mae)


# Now use the for loop to actually specify a few possible number of nodes (branches) and see how models run with those numbers of nodes compare in terms of MAE :

# In[270]:


for max_leaf_nodes in [5, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("With max leaf nodes: *%d* \t\t MAE: %d" %(max_leaf_nodes, my_mae))


# I see from the output above that the lowest two MAE values are for max_leaf_nodes= 50 or 100, so I'm going to rerun the above code with a different array of possible max_leaf_nodes values to see if I can pinpoint more specifically the optimal number of nodes.

# In[271]:


for max_leaf_nodes in [5, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("With max leaf nodes: *%d* \t\t MAE: %d" %(max_leaf_nodes, my_mae))


# Again, optimal MAE is with around 50-75 nodes, so run again with decreased range:

# In[272]:


for max_leaf_nodes in [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("With max leaf nodes: *%d* \t\t MAE: %d" %(max_leaf_nodes, my_mae))


# **45** seems to be the optimal number of branches for this particular model.  So I'll re-run the model with the number of nodes set and double check that the MAE value of that model matches the value here.

# In[273]:


mb_model = DecisionTreeRegressor(max_leaf_nodes=45)
mb_model.fit(train_X, train_y)
pred_y = mb_model.predict(val_X)
print(mean_absolute_error(pred_y, val_y))


# The MAE is the same as what we got before--it should be, since it's essentially the same code, but it confirms that I can write the same thing in multiple ways (yay!).
# 
# Now, to the Random Forests module (https://www.kaggle.com/dansbecker/random-forests):

# In[275]:


from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor()
RF_model.fit(train_X, train_y)
pred_y = RF_model.predict(val_X)
print(mean_absolute_error(pred_y, val_y))


# The MAE for the Random Forest model is about 3000 (units) less than the MAe for the Decision Tree model, which is good.
# 
# Though it would be good to know more about what the standard values of MAEs are.
# 
# Now to run another model with more predictors for submission:

# In[287]:


#re-add packages
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

#re-import data
train = pd.read_csv("../input/train.csv")

#designate predictors and target
train_y = train.SalePrice
# print(train_y) ##Check
predictor_cols = ['LotArea', 'OverallQual', 'OverallCond', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
#print(predictor_cols)  ##Check
train_X = train[predictor_cols]

#make the model
simple_model = RandomForestRegressor()
simple_model.fit(train_X, train_y)

#import values from test dataset
test = pd.read_csv("../input/test.csv")

#use model to predict
test_X = test[predictor_cols] #USING SAME "predictor_cols" array here requires that test dataset have columns with EXACT same names
simple_pred = simple_model.predict(test_X)
print(simple_pred)


# Create submission file:

# In[292]:


subm = pd.DataFrame({'Id': test.Id, 'SalePrice': simple_pred})
subm.to_csv('emlini_simplepred.csv', index=False)


# In[ ]:





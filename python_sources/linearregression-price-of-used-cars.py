#!/usr/bin/env python
# coding: utf-8

# # Predicting the price of a used car

# ### Intro

# I'm trying to do the very basics with this excercise.
# My goal is to train a linear regression model with some set of variables in this data set to determine price of used car

# 1. Importing libraries
# 2. Load data
# 3. Preprocessing
# a. Exploring descriptive statistics
# b. Determing variable of interest
# c. Dealing with missing values
# 4. Exploring Probability distribution function
# 5. Removing outliers
# 6. Checking OLS assumptions
# 7. Dealing with multicollinearity
# 8. Creating dummies with categorical variables
# 9. Linear Regression model
# a. Declare x and y
# b. Scale the data
# c. Train\test split
# d. Create Regression
# e. Finding the weight and bias
# f. Testing
# 10. Conclusion
# 11. How to improve model

# ### IMPORTING LIBRARIES

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
sns.set()


# ### LOAD DATA

# In[ ]:


# Load the data from a .csv in the same folder

raw_data = pd.read_csv("/kaggle/input/1.04. Real-life example.csv")
# Exploring the first 5 row of the data
raw_data.head()


# ### PREPROCESSING

# ### Exploring the descriptive statistics of the variable

# In[ ]:


# Descriptive statistics are very useful for initial exploration of the variables
# By default, only descriptives for the numerical variables are shown
# To include the categorical ones, you should specify this with an argument(include = all)

raw_data.describe(include='all')


# ### Determining the variables of interest

# In[ ]:


# I removed the variable 'model' because it has too many unique values
data = raw_data.drop(['Model'],axis=1)

# Description of the dataframe after 'model' is removed
data.describe(include='all')


# ### Dealing with missing values

# In[ ]:


# data.isnull() # shows a df with the information whether a data point is null 
# Since True = the data point is missing, while False = the data point is not missing, we can sum them
# This will give us the total number of missing values feature-wise

## we check for missing values to improve the accuracy of our model
data.isnull().sum()


# In[ ]:


# Let's simply drop all missing values
# This is not always recommended, however, when we remove less than 5% of the data, it is okay

data_no_mv = data.dropna(axis=0)


# In[ ]:


# Let's check the descriptives without the missing values

data_no_mv.describe(include='all')


# ### Exploring probability distribution functions for each feature

# #### A great step in the data exploration is to display the probability distribution function (PDF) of a variable
# #### The PDF will show us how that variable is distributed 
# #### This makes it very easy to spot anomalies, such as outliers
# #### The PDF is often the basis on which we decide whether we want to transform a feature

# In[ ]:



sns.distplot(data_no_mv['Price'])


# ### Removing outliers

# In[ ]:


# Obviously there are some outliers present 

# Without diving too deep into the topic, we can deal with the problem easily by removing 0.5%, or 1% of the problematic samples
# Here, the outliers are situated around the higher prices (right side of the graph)
# Logic should also be applied
# This is a dataset about used cars, therefore one can imagine how $300,000 is an excessive price

# Outliers are a great issue for OLS, thus we must deal with them in some way


# In[ ]:


# Declaring a variable that will be equal to the 99th percentile of the 'Price' variable
q = data_no_mv['Price'].quantile(0.99)

# Then we can create a new df, with the condition that all prices must be below the 99 percentile of 'Price'
data_1 = data_no_mv[data_no_mv['Price']<q]

# In this way we have essentially removed the top 1% of the data about 'Price'
data_1.describe(include='all')


# In[ ]:


# Check the PDF once again, here we can see that the outliers have reduced drastically

sns.distplot(data_1['Price'])


# In[ ]:


sns.distplot(data_no_mv['Mileage'])


# In[ ]:


# We can solve other variables in a similar way

q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]


# In[ ]:


sns.distplot(data_2['Mileage'])


# In[ ]:


sns.distplot(data_no_mv['EngineV'])


# In[ ]:


# From the above we can see that| engine volume is very strange
# In such cases it makes sense to manually check what may be causing the problem
# In our case the issue comes from the fact that most missing values are indicated with 99.99 or 99
# There are also some incorrect entries like 75


# In[ ]:


# A simple Google search can indicate the natural domain of this variable
# Car engine volumes are usually (always?) below 6.5l
# This is a prime example of the fact that a domain expert (a person working in the car industry)
# may find it much easier to determine problems with the data than an outsider

data_3 = data_2[data_2['EngineV']<6.5]


# In[ ]:


sns.distplot(data_3['EngineV'])

# After plotting the graph to see PDF we can see great improvment and outliers has decreased significantly


# In[ ]:


# Finally, the situation with 'Year' is similar to 'Price' and 'Mileage'
# However, the outliers are on the low end

sns.distplot(data_no_mv['Year'])


# In[ ]:


# Declaring a variable that will be equal to the 1st percentile of the 'Year' variable
q = data_3['Year'].quantile(0.01)

# Then we can create a new df, with the condition that all year must be below the 1 percentile of 'Year'
data_4 = data_3[data_3['Year']>q]


# In[ ]:


# Check out the result now

sns.distplot(data_4['Year'])


# In[ ]:


# When we remove observations, the original indexes are preserved
# If we remove observations with indexes 2 and 3, the indexes will go as: 0,1,4,5,6
# That's very problematic as we tend to forget about it (later you will see an example of such a problem)

# Finally, once we reset the index, a new column will be created containing the old index (just in case)
# We won't be needing it, thus 'drop=True' to completely forget about it


# In[ ]:


data_cleaned = data_4.reset_index(drop=True)


# In[ ]:


# Lets have a look at our clean dataset

data_cleaned.describe(include='all')


# ### Checking OLS assumptions

# In[ ]:


# Lets do some matplotlib code and plot variables against each other on a scatter plot
# since Price is the 'y' axis of all the plots, it made sense to plot them side-by-side (so we can compare them)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')

plt.show()


# From the subplots and the PDF of price, we can easily determine that 'Price' is exponentially distributed instead of a linear relationship
# 
# A good transformation in this case is a log transformation

# In[ ]:


## Let's transform 'Price' with a log transformation
data_log = np.log(data_cleaned['Price'])

# Add the new price to our data frame
data_cleaned['log_price'] = data_log
data_cleaned


# In[ ]:


# Lets check again
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('log_price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('log_price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('log_price and Mileage')

plt.show()

# # The relationships show a clear linear relationship
# This is some good linear regression material

# Alternatively we could have transformed each of the independent variables


# In[ ]:


# Since we will be using the log price variable, we can drop the old 'Price' one
data_cleaned = data_cleaned.drop(['Price'],axis=1)


# ### Dealing With Multicollinearity

# In[ ]:


#The columns of our data frame
data_cleaned.columns.values


# In[ ]:


# sklearn does not have a built-in way to check for multicollinearity

# Here's the relevant module
from statsmodels.stats.outliers_influence import variance_inflation_factor

# we declare a variable where we put all features where we want to check for multicollinearity
# since our categorical data is not yet preprocessed, we will only take the numerical ones
variables = data_cleaned[['Mileage','Year','EngineV']]

# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns


# In[ ]:


# Explore the result
vif


# From our result we can see that 'Year' has the highest VIFs.
# If i remove 'Year' from the data frame it will cause the other varibles VIFs to reduce.
# Hence, i will remove 'Year'.

# In[ ]:


data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)


# In[ ]:


# Lets use the same method shown above to check the VIFs to see if the reduced since 'Year' has been dropped

from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns


# In[ ]:


vif
# From our result you can see that VIFs has drastically reduced for each variable


# ### Creating dummies with categorical variables

# In[ ]:


# To include the categorical data in the regression, let's create dummies
# There is a very convenient method called: 'get_dummies' which does that seemlessly
# It is extremely important that we drop one of the dummies, alternatively we will introduce multicollinearity

data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)


# In[ ]:


# Here's the result

data_with_dummies.head()


# In[ ]:


# To make our data frame more organized, we prefer to place the dependent variable in the beginning of the 
data_with_dummies.columns.values


# In[ ]:


# To make the code a bit more parametrized, let's declare a new variable that will contain the preferred order
# Conventionally, the most intuitive order is: dependent variable, indepedendent numerical variab;es and dummi

cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']


# In[ ]:


# To implement the reordering, we will create a new df, which is equal to the old one but with the new order of features
data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()


# ### Linear regression model
# 

# #### Declare x and y

# In[ ]:


# The target is the dependent variable which is the 'log_price'
targets = data_preprocessed['log_price']

# The independent variabke is everything else but the log)orice so it wise to just drop it
inputs = data_preprocessed.drop(['log_price'],axis=1)


# #### Scale the data

# In[ ]:


# Import scaling module
from sklearn.preprocessing import StandardScaler

# Create scaling object
scaler = StandardScaler()

# fit the inputs
scaler.fit(inputs)


# we scale when we want to handle disparities in units and improve performance of your model


# In[ ]:


# Scale the features and store them in a new variable 

inputs_scaled = scaler.transform(inputs)


# ### Train Test Split

# In[ ]:


# Import the split model
from sklearn.model_selection import train_test_split

# Split the variables with an 80-20 split and some random state
# To have the same split as mine, use random_state = 200
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled,targets)


# ### Create regression

# In[ ]:


# Create a regression object
reg = LinearRegression()
# Fit the regression with the scaled TRAIN inputs and targets
reg.fit(x_train,y_train)


# In[ ]:


# Let's check the outputs of the regression
# I'll store them in y_hat as this is the 'theoretical' name of the predictions
y_hat = reg.predict(x_train)


# In[ ]:


# The simplest way to compare the targets (y_train) and the predictions (y_hat) is to plot them on a scatter plot
# The closer the points to the 45-degree line, the better the prediction
plt.scatter(y_train, y_hat)
# Let's also name the axes
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
# Sometimes the plot will have different scales of the x-axis and the y-axis
# We want the x-axis and the y-axis to be the same
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[ ]:


# We can plot the PDF of the residuals and check for anomalies
sns.distplot(y_train - y_hat)

# Include a title
plt.title("Residuals PDF", size=18)

# In the best case scenario this plot should be normally distributed
# In our case we notice that there are many negative residuals (far away from the mean)
# Given the definition of the residuals (y_train - y_hat), negative values imply
# that y_hat (predictions) are much higher than y_train (the targets)
# This is food for thought to improve our model


# In[ ]:


# Find the R-squared of the model
reg.score(x_train,y_train)


# ### Finding the weight and bias

# In[ ]:


# Obtain the bias (intercept) of the regression
reg.intercept_


# In[ ]:


# Obtain the weights (coefficients) of the regression
reg.coef_


# In[ ]:


# Create a regression summary where we can compare them with one-another
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


# ### Testing

# In[ ]:


# Once we have trained our model, we can test it on a dataset that the algorithm has never seen
# Our test inputs are 'x_test', while the outputs: 'y_test'
# If the predictions are far off, we will know that our model overfitted
y_hat_test = reg.predict(x_test)


# In[ ]:


# Create a scatter plot with the test targets and the test predictions
# You can include the argument 'alpha' which will introduce opacity to the graph
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[ ]:


# lets check these predictions
# To obtain the actual prices, we take the exponential of the log_price
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()


# In[ ]:


# Include the test targets in that data frame to compare them with the predictions
df_pf['Target'] = np.exp(y_test)
df_pf

# Note that we have a lot of missing values
# There is no reason to have ANY missing values, though
# This suggests that something is wrong with the data frame


# In[ ]:


# After displaying y_test, we find what the issue is
# The old indexes are preserved (recall earlier in that code we made a note on that)
# The code was: data_cleaned = data_4.reset_index(drop=True)

# Therefore, to get a proper result, we must reset the index and drop the old indexing
y_test = y_test.reset_index(drop=True)

# Check the result
y_test.head()


# In[ ]:


# Let's overwrite the 'Target' column with the appropriate values
# Again, we need the exponential of the test log price
df_pf['Target'] = np.exp(y_test)
df_pf


# In[ ]:


# Calculate the difference between the targets and the predictions
# Note that this is actually the residual
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']


# In[ ]:


# Finally, lets see how far off we are from the result percentage-wise
# Here, we take the absolute difference in %, so we can easily order the data frame
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[ ]:


# Exploring the descriptives here gives us additional insights
df_pf.describe()


# In[ ]:


# Check these outputs manually
# To see all rows, we use the relevant pandas syntax
pd.options.display.max_rows = 999
# Moreover, to make the dataset clear, we can display the result with only 2 digits after the dot 
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Finally, we sort by difference in % and manually check the model
df_pf.sort_values(by=['Difference%'])


# ### Conclusion
# 

# Going to the bottom of the data frame we can see that they are very few predictions that are far off from the observed values.
# If you look closely at the observed column you will notice that the observed prices are extermely low.
# 

# In conclusion, our model is using mileage, EngineV, Registration, Brand and body type to predict price of a used car.
# On average it is pretty decent at predicting the price but for the last samples it isnt.
# All residuals for the outliers are negative. therefore the predictions are higher than the targets. The explanation
# maybe that we are missing an important feature which drives the price of a used car lower. factors such as model of the car that
# we removed at the beginning of the analysis or the car was damaged in some way.
# 
# i've used
# 1. data exploration
# 2. feature scaling
# 3. data visualization 
# 4. machine learning algorithm 
# 
# There is still so much to improve in our model 

# ### How to improve our model

# 1. Use a different set of variables
# 2. Remove a bigger part of the outlier observation
# 3. Use different kinds of transformations

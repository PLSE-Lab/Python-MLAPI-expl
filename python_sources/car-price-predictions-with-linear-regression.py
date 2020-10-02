#!/usr/bin/env python
# coding: utf-8

# ![cars.PNG](attachment:cars.PNG)
# Source: shutterstock

# Hello there! Welcome to my first kernel where I've designed a Linear Regression model to predict car prices. Since this is my first kernel your suggestions/feedbacks will be very much appreciated. 
# PS: Please upvote if you found it helpful :)

# # Multiple Linear Regression model
# 
# ## Car Prices Case Study
# 
# ### Problem Statement:
# A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.
# 
# They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market

# # 1. Reading and Understanding the data

# In[ ]:


#Importing necessary libraries

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
pd.options.display.max_columns = 100
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os


# In[ ]:


#Reading the data set as a dataframe

cars = pd.read_csv('../input/geely-auto/CarPriceAssignment.csv')


# In[ ]:


#Viewing the cars dataframe

cars.head()


# In[ ]:


#Dimensions of the dataframe
cars.shape


# In[ ]:


#Checking the dataframe for any null values
cars.info()


# In[ ]:


#Getting a statistical view of the numerical variables of the dataframe
cars.describe()


# In[ ]:


# Dropping car_ID variable since it has nothing to do with price

cars.drop('car_ID', axis=1, inplace=True)


# In[ ]:


cars.head(10)


# # 2. Visualising the numeric variables

# In[ ]:


plt.figure(figsize=(10,10))
sns.pairplot(cars)
plt.show()


# #### Let us check the correlations of the numeric variables with price

# In[ ]:


#Correlations of price with other numeric variables

cars[cars.columns[1:]].corr()['price'][:].round(2).sort_values(ascending=True)


# Looking at the above correlations, we can see a few numerical variables that do not quite show a considerably good linear relationship with **`price`**<br>
# These are namely:<br>
# - carheight (0.12)
# - stroke (0.08)
# - compressionratio (0.07)
# - peakrpm (-0.09)<br>
# 
# Therefore, we can go ahead and drop these variables.

# In[ ]:


cars.drop(columns = ['carheight','stroke','compressionratio','peakrpm'], axis=1, inplace=True)


# #### We will now process a few numeric and categorical variables

# In[ ]:


cars['symboling'].value_counts()


# Since `symbolising` is a categorical variable, lets convert it's numeric values to corresponding categorical values.<br>
# Here the assumption is:<br>
# - symboling value in the range -3 and 0 are considered **Low Risk**<br>
# - symboling values other than this (i.e. greater than 0) are considered **High Risk**

# In[ ]:


#Categorising symboling values -3 to 0 as Low Risk and the remaining positive values as High Risk

def categorise(x):
    if(-3 <= x <= 0):
        return "Low Risk"
    else:
        return "High Risk"
        
cars['symboling'] = cars['symboling'].apply(lambda x: categorise(x))


# In[ ]:


cars.head()


# In[ ]:





# Let us also extract the company name from the `CarName` variable and assign this new derived metric into a new categorical variable named `company`

# In[ ]:


#Extract the Company name from 'CarName' variable

cars['company'] = cars['CarName'].apply(lambda x: x.split(' ')[0])
cars['company'] = cars['company'].str.lower()


# In[ ]:


#Correcting the incorrect company names

def compName(x):
    if (x == "vw" or x == "vokswagen"):
        return "volkswagen"
    elif(x == "toyouta"):
        return "toyota"
    elif(x == "porcshce"):
        return "porsche"
    elif(x == "maxda"):
        return "mazda"
    
    else:
        return x
    
cars['company'] = cars['company'].apply(lambda x: compName(x))


# In[ ]:


#Dropping the CarName variable

cars.drop('CarName',axis=1,inplace=True)


# In[ ]:


cars.head()


# In[ ]:





# ## Visualising the categorical variables

# In[ ]:


plt.figure(figsize=(25,25))
fig_num = 0
def plot_categorical(var):       #Function to plot boxplots for all categorical variables
    plt.subplot(3,4, fig_num)
    sns.boxplot(x = var, y = 'price', data = cars)

categorical_vars = cars.dtypes[cars.dtypes==object].index
for var in categorical_vars:
    fig_num = fig_num + 1
    plot_categorical(var)

plt.show()


# From the above plots, we can see there are obvious effects of all categorical variables (except **doornumber**) on `price`. **doornumber** does not seem to have much effect on car prices. Hence, we will drop the **doornumber** variable from the dataset.

# In[ ]:


#Dropping doornumber variable from the dataset

cars.drop('doornumber', axis=1, inplace=True)


# In[ ]:


#Moving the price column to the front of the dataframe for better readability

cars = cars[['price','symboling','fueltype','aspiration','carbody','drivewheel','enginelocation','wheelbase','carlength','carwidth',
 'curbweight','enginetype','cylindernumber','enginesize','fuelsystem','boreratio','horsepower','company']]


# # 3. Data Preparation

# There are categorical variables in the data set with string values which need to be converted to numeric in order to fit a regression line.

# In[ ]:


cars.head()


# We'll now convert the binary categorical variables `symboling`, `fueltype`, `aspiration`, `enginelocation` to numeric.

# In[ ]:


def binary_map(x):
    cars[x] = cars[x].astype("category").cat.codes

binary_categorical_vars = ['symboling','fueltype','aspiration','enginelocation']
for var in binary_categorical_vars:
    binary_map(var)


# In[ ]:


cars.head()


# In[ ]:





# #### Let's check the percentage amount of each level of the categorical variables.

# In[ ]:


print("Engine Type")
print(cars['enginetype'].value_counts(normalize=True).round(2))
print("\n")
print("Drivewheel")
print(cars['drivewheel'].value_counts(normalize=True).round(2))
print("\n")
print("Carbody")
print(cars['carbody'].value_counts(normalize=True).round(2))
print("\n")
print("Fuel System")
print(cars['fuelsystem'].value_counts(normalize=True).round(2))


# Here we see that about **72%** of the **`enginetype`** variable is **ohc**. Also, others are also some form of **ohc** engines. Therefore, lets go ahead and mark the ohc type engines as 1 and others as 0. <br>

# In[ ]:


def eng_map(x):
    if("ohc" in x):
        return 1
    else:
        return 0

cars['enginetype'] = cars.enginetype.apply(lambda x: eng_map(x))


# In[ ]:


cars['enginelocation'].value_counts(normalize=True).round(2)


# As we can see, **`enginelocation`** variable almost only has 0 (front) as the value. So we can drop this variable assuming that almost all cars have engine located at the front.

# In[ ]:


cars.drop('enginelocation', axis = 1, inplace=True)


# In[ ]:


# Converting "cylindernumber" values to its corresponding number

cars['cylindernumber'].replace({"two":2,"three":3,"four":4,"five":5, "six":6,"eight":8,"twelve":12}, inplace=True)


# In[ ]:





# We will now create dummies for the variables **`drivewheel`, `carbody`, `company` and `fuelsys`**.

# In[ ]:


#Get the dummy variables for carbody and store in separate variable "carbody_dummies"

carbody_dummies = pd.get_dummies(cars['carbody'],prefix='carbody',drop_first=True)
cars = pd.concat([cars,carbody_dummies], axis=1)
cars.drop('carbody',axis=1,inplace=True)


# In[ ]:


#Get the dummy variables for drivewheel and store in separate variable "drivewheel_dummies"

drivewheel_dummies = pd.get_dummies(cars['drivewheel'],prefix='dw',drop_first=True)
cars = pd.concat([cars,drivewheel_dummies], axis=1)
cars.drop('drivewheel',axis=1,inplace=True)


# In[ ]:


#Get the dummy variables for company and store in separate variable "company_dummies"

company_dummies = pd.get_dummies(cars['company'],prefix='comp',drop_first=True)
cars = pd.concat([cars,company_dummies], axis=1)
cars.drop('company',axis=1,inplace=True)


# In[ ]:


#Get the dummy variables for fuelsystem and store in separate variable "fuelsys_dummies"

fuelsys_dummies = pd.get_dummies(cars['fuelsystem'],prefix='dw',drop_first=True)
cars = pd.concat([cars,fuelsys_dummies], axis=1)
cars.drop('fuelsystem',axis=1,inplace=True)


# In[ ]:


cars.head()


# In[ ]:


cars.shape


# In[ ]:


cars.describe()


# # 4. Splitting the data into training and testing sets
# 
# Now that we have prepared our data, we can go ahead and make the train-test split

# In[ ]:


# from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(cars, train_size = 0.7, test_size = 0.3, random_state=100)


# ### Rescaling the features

# In[ ]:


# from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[ ]:


# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['price','wheelbase','carlength','carwidth','curbweight','cylindernumber','enginesize','boreratio','horsepower']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# ### Dividing into X and Y sets for the model building

# In[ ]:


y_train = df_train.pop('price')
X_train = df_train


# # 5. Building the model

# In[ ]:


#Creating an object of LinearRegression class and using RFE to get the top 20 variables from the dataset

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm,20)
rfe = rfe.fit(X_train, y_train)


# In[ ]:


#Collecting the 20 variables selected by RFE

cols = X_train.columns[rfe.support_]
cols


# In[ ]:


#Selecting these 20 variables from X_train data set and assign to new variable X_train_rfe

X_train_rfe = X_train[cols]


# In[ ]:


#Add a constant to X_train_rfe data set using statsmodels.api library as sm

X_train_lm = sm.add_constant(X_train_rfe)


# ### Model 1

# In[ ]:


lm = sm.OLS(y_train, X_train_lm).fit()  # Running the linear model
print(lm.summary())                     # Viewing the summary of the linear model


# **Model 1:<br>
# R-squared value: 0.948<br>
# Adj. R-squared value: 0.939**

# In[ ]:


#Calculating the VIF of the variables using variance_inflation_factor library

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# `curbweight` is insignificant in presence of other variables and also has high VIF. Can be dropped.

# In[ ]:


X_train_new = X_train_rfe.drop('curbweight',axis=1)


# ### Model 2

# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train, X_train_lm).fit()
print(lm.summary())


# **Model 2:<br>
# R-squared value: 0.945<br>
# Adj. R-squared value: 0.936**

# In[ ]:


# Calculate the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# `comp_peugeot` is insignificant in presence of other variables. Can be dropped.

# In[ ]:


X_train_new = X_train_new.drop('comp_peugeot',axis=1)


# ### Model 3

# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train, X_train_lm).fit()
print(lm.summary())


# **Model 3:<br>
# R-squared value: 0.944<br>
# Adj. R-squared value: 0.936**

# In[ ]:


#Calculating the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# `comp_isuzu` is insignificant in presence of other variables. Can be dropped.

# In[ ]:


X_train_new = X_train_new.drop('comp_isuzu',axis=1)


# ### Model 4

# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train, X_train_lm).fit()
print(lm.summary())


# **Model 4:<br>
# R-squared value: 0.942<br>
# Adj. R-squared value: 0.934**

# In[ ]:


#Calculating the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# `enginesize` has high VIF. Can be dropped.

# In[ ]:


X_train_new = X_train_new.drop(['enginesize'], axis=1)


# ### Model 5

# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train, X_train_lm).fit()
print(lm.summary())


# **Model 5:<br>
# R-squared value: 0.928<br>
# Adj. R-squared value: 0.919**

# In[ ]:


#Calculating the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# `cylindernumber` is insignificant in presence of other variables and has high VIF. Can be dropped.

# In[ ]:


X_train_new = X_train_new.drop('cylindernumber', axis=1)


# ### Model 6

# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train, X_train_lm).fit()
print(lm.summary())


# **Model 6:<br>
# R-squared value: 0.928<br>
# Adj. R-squared value: 0.919**

# In[ ]:


#Calculating the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# `boreratio` is insignificant in presence of other variables and has high VIF. Can be dropped.

# In[ ]:


X_train_new = X_train_new.drop('boreratio', axis=1)


# ### Model 7

# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train, X_train_lm).fit()
print(lm.summary())


# **Model 7:<br>
# R-squared value: 0.927<br>
# Adj. R-squared value: 0.919**

# In[ ]:


#Calculating the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# `enginetype` is insignificant in presence of other variables and has high VIF. Can be dropped.

# In[ ]:


X_train_new = X_train_new.drop('enginetype', axis=1)


# ### Model 8

# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train, X_train_lm).fit()
print(lm.summary())


# **Model 8:<br>
# R-squared value: 0.923<br>
# Adj. R-squared value: 0.915**

# In[ ]:


#Calculating the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# `comp_saab` is insignificant in presence of other variables. Can be dropped.

# In[ ]:


X_train_new = X_train_new.drop('comp_saab', axis=1)


# ### Model 9

# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train, X_train_lm).fit()
print(lm.summary())


# **Model 9:<br>
# R-squared value: 0.922<br>
# Adj. R-squared value: 0.915**

# In[ ]:


#Calculating the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# `comp_subaru` is insignificant in presence of other variables. Can be dropped.

# In[ ]:


X_train_new = X_train_new.drop('comp_subaru', axis=1)


# ### Model 10

# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train, X_train_lm).fit()
print(lm.summary())


# **Model 10:<br>
# R-squared value: 0.921<br>
# Adj. R-squared value: 0.914**

# In[ ]:


#Calculating the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# `comp_audi` is insignificant in presence of other variables. Can be dropped.

# In[ ]:


X_train_new = X_train_new.drop('comp_audi', axis=1)


# ### Model 11

# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train, X_train_lm).fit()
print(lm.summary())


# **Model 11:<br>
# R-squared value: 0.919<br>
# Adj. R-squared value: 0.912**

# In[ ]:


#Calculating the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# `carwidth` has high VIF. Can be dropped.

# In[ ]:


X_train_new = X_train_new.drop('carwidth', axis=1)


# ### Model 12

# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train, X_train_lm).fit()
print(lm.summary())


# **Model 12:<br>
# R-squared value: 0.876<br>
# Adj. R-squared value: 0.867**

# In[ ]:


#Calculating the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# `carbody_sedan` is insignificant in presence of other variables. Can be dropped.

# In[ ]:


X_train_new = X_train_new.drop('carbody_sedan', axis=1)


# ### Model 13

# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train, X_train_lm).fit()
print(lm.summary())


# **Model 13:<br>
# R-squared value: 0.875<br>
# Adj. R-squared value: 0.867**

# In[ ]:


#Calculating the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# `carbody_wagon` is insignificant in presence of other variables. Can be dropped.

# In[ ]:


X_train_new = X_train_new.drop('carbody_wagon', axis=1)


# ### Model 14

# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train, X_train_lm).fit()
print(lm.summary())


# **Model 14:<br>
# R-squared value: 0.874<br>
# Adj. R-squared value: 0.868**

# In[ ]:


#Calculating the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # 6. Inferences from the final model

# From the final model (Model 14), we can conclude that the most important driving factors (variables) for predicting car prices are **horsepower, carbody, and company**.<br>
# 
# - **Horsepower** effects car prices the most with a coefficient of 0.72
# - **Carbody** is also another driving factor for car prices, mainly `hatchback` in this case. Geely Auto could focus on Hatchback designs for their cars.
# - **Company** of the car is the third factor, mainly `jaguar`, `porsche`, `bmw`, `buick`, and `volvo`. These car companies may be of interest to Geely Auto in terms of price variations, designs, configurations etc.

# ### Final training data set

# In[ ]:


X_train_new.head()


# In[ ]:


X_train_lm.head()     # Training set with the constant


# # 7. Residual analysis for the train data
# 
# So, now to check if the error terms are also normally distributed, let us plot the histogram of the error terms and see what it looks like.

# In[ ]:


y_train_pred = lm.predict(X_train_lm)
error = y_train - y_train_pred


# In[ ]:


# Plot the histogram of the error terms

fig = plt.figure()
sns.distplot(error, bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18) 


# In[ ]:


plt.figure(figsize=(5,5))
sns.regplot(y_train_pred,error)
plt.xlabel('y_train_pred')
plt.ylabel('Error')


# As seen in the above plot, there is no pattern in the error values.

# In[ ]:


# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars_test = ['price','wheelbase','carlength','carwidth','curbweight','cylindernumber','enginesize','boreratio','horsepower']

df_test[num_vars_test] = scaler.transform(df_test[num_vars_test])


# In[ ]:


df_test.head()


# ### Dividing into X and y sets for the model building

# In[ ]:


y_test = df_test.pop('price')
X_test = df_test


# Now let's use our model to make predictions.

# In[ ]:


# Creating X_test_new dataframe by only selecting variables present in the X_train training set

X_test_new = X_test[X_train_new.columns]

# Adding a constant variable
X_test_new = sm.add_constant(X_test_new)


# #### Making the predictions.

# In[ ]:


y_test_pred = lm.predict(X_test_new)


# # 8. Model Evaluation

# In[ ]:


fig = plt.figure()
plt.scatter(y_test,y_test_pred)
fig.suptitle('y_test vs y_test_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_test_pred', fontsize=16)   


# # 9. R2 score

# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_test_pred)


# # 10. Equation of best fitted line

# Therefore, the equation for our best fitted line is:
# 
# $ price = 0.72 \times horsepower - 0.051 \times carbody\_hatchback + 0.3 \times comp\_bmw + 0.42 \times comp\_buick + 0.314 \times comp\_jaguar + 0.174 \times comp\_porsche + 0.127 \times comp\_volvo $

# Thank you for viewing my kernel :)

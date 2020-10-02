#!/usr/bin/env python
# coding: utf-8

# ## Car Price Prediction (CPP)
# 
# The solution is divided into the following sections: 
# - Data understanding and exploration
# - Data cleaning
# - Data preparation
# - Model building and evaluation
# 

# ### 1. Data Understanding and Exploration
# 
# Let's first have a look at the dataset and understand the size, attribute names etc.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

# reading the dataset
cars = pd.read_csv("../input/car-data/CarPrice_Assignment.csv")


# In[ ]:


# summary of the dataset: 205 rows, 26 columns, no null values
print(cars.info())


# In[ ]:


# head
cars.head()


# #### Understanding the Data Dictionary
# 
# The data dictionary contains the meaning of various attributes; some non-obvious ones are:

# In[ ]:


# symboling: -2 (least risky) to +3 most risky
# Most cars are 0,1,2
cars['symboling'].astype('category').value_counts()


# In[ ]:


# aspiration: An (internal combustion) engine property showing 
# whether the oxygen intake is through standard (atmospheric pressure)
# or through turbocharging (pressurised oxygen intake)

cars['aspiration'].astype('category').value_counts()


# In[ ]:


# drivewheel: frontwheel, rarewheel or four-wheel drive 
cars['drivewheel'].astype('category').value_counts()


# In[ ]:


# wheelbase: distance between centre of front and rarewheels
sns.distplot(cars['wheelbase'])
plt.show()


# In[ ]:


# curbweight: weight of car without occupants or baggage
sns.distplot(cars['curbweight'])
plt.show()


# In[ ]:


# stroke: volume of the engine (the distance traveled by the 
# piston in each cycle)
sns.distplot(cars['stroke'])
plt.show()


# In[ ]:


# compression ration: ration of volume of compression chamber 
# at largest capacity to least capacity
sns.distplot(cars['compressionratio'])
plt.show()


# In[ ]:


# target variable: price of car
sns.distplot(cars['price'])
plt.show()


# #### Data Exploration
# 
# To perform linear regression, the (numeric) target variable should be linearly related to *at least one another numeric variable*. Let's see whether that's true in this case.
# 
# 
# We'll first subset the list of all (independent) numeric variables, and then make a **pairwise plot**.

# In[ ]:


# all numeric (float and int) variables in the dataset
cars_numeric = cars.select_dtypes(include=['float64', 'int'])
cars_numeric.head()


# Here, although the variable ```symboling``` is numeric (int), we'd rather treat it as categorical since it has only 6 discrete values. Also, we do not want 'car_ID'.

# In[ ]:


# dropping symboling and car_ID 
cars_numeric = cars_numeric.drop(['symboling', 'car_ID'], axis=1)
cars_numeric.head()


# Let's now make a pairwise scatter plot and observe linear relationships.

# In[ ]:


# paiwise scatter plot

plt.figure(figsize=(20, 10))
sns.pairplot(cars_numeric)
plt.show()


# This is quite hard to read, and we can rather plot correlations between variables. Also, a heatmap is pretty useful to visualise multiple correlations in one plot.

# In[ ]:


# correlation matrix
cor = cars_numeric.corr()
cor


# In[ ]:


# plotting correlations on a heatmap

# figure size
plt.figure(figsize=(16,8))

# heatmap
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()


# The heatmap shows some useful insights:
# 
# Correlation of price with independent variables:
# - Price is highly (positively) correlated with wheelbase, carlength, carwidth, curbweight, enginesize, horsepower (notice how all of these variables represent the size/weight/engine power of the car)
# 
# - Price is negatively correlated to ```citympg``` and ```highwaympg``` (-0.70 approximately). This suggest that cars having high mileage may fall in the 'economy' cars category, and are priced lower (think Maruti Alto/Swift type of cars, which are designed to be affordable by the middle class, who value mileage more than horsepower/size of car etc.)
# 
# Correlation among independent variables:
# - Many independent variables are highly correlated (look at the top-left part of matrix): wheelbase, carlength, curbweight, enginesize etc. are all measures of 'size/weight', and are positively correlated 
# 
# 
# Thus, while building the model, we'll have to pay attention to multicollinearity (especially linear models, such as linear and logistic regression, suffer more from multicollinearity).

# ## 2. Data Cleaning
# 
# Let's now conduct some data cleaning steps. 
# 
# We've seen that there are no missing values in the dataset. We've also seen that variables are in the correct format, except ```symboling```, which should rather be a categorical variable (so that dummy variable are created for the categories).
# 
# Note that it *can* be used in the model as a numeric variable also. 
# 
# 

# In[ ]:


# variable formats
cars.info()


# In[ ]:


# converting symboling to categorical
cars['symboling'] = cars['symboling'].astype('object')
cars.info()


# Netx, we need to extract the company name from the column ```CarName```. 

# In[ ]:


# CarName: first few entries
cars['CarName'][:30]


# Notice that the carname is what occurs before a space, e.g. alfa-romero, audi, chevrolet, dodge, bmx etc.
# 
# Thus, we need to simply extract the string before a space. There are multiple ways to do that.
# 
# 
# 

# In[ ]:


# Extracting carname

# Method 1: str.split() by space
carnames = cars['CarName'].apply(lambda x: x.split(" ")[0])
carnames[:30]


# In[ ]:


# Method 2: Use regular expressions
import re

# regex: any alphanumeric sequence before a space, may contain a hyphen
p = re.compile(r'\w+-?\w+')
carnames = cars['CarName'].apply(lambda x: re.findall(p, x)[0])
print(carnames)


# Let's create a new column to store the compnay name and check whether it looks okay.

# In[ ]:


# New column car_company
cars['car_company'] = cars['CarName'].apply(lambda x: re.findall(p, x)[0])


# In[ ]:


# look at all values 
cars['car_company'].astype('category').value_counts()


# Notice that **some car-company names are misspelled** - vw and vokswagen should be volkswagen, porcshce should be porsche, toyouta should be toyota, Nissan should be nissan, maxda should be mazda etc.
# 
# This is a data quality issue, let's solve it.

# In[ ]:


# replacing misspelled car_company names

# volkswagen
cars.loc[(cars['car_company'] == "vw") | 
         (cars['car_company'] == "vokswagen")
         , 'car_company'] = 'volkswagen'

# porsche
cars.loc[cars['car_company'] == "porcshce", 'car_company'] = 'porsche'

# toyota
cars.loc[cars['car_company'] == "toyouta", 'car_company'] = 'toyota'

# nissan
cars.loc[cars['car_company'] == "Nissan", 'car_company'] = 'nissan'

# mazda
cars.loc[cars['car_company'] == "maxda", 'car_company'] = 'mazda'


# In[ ]:


cars['car_company'].astype('category').value_counts()


# The ```car_company``` variable looks okay now. Let's now drop the car name variable.

# In[ ]:


# drop carname variable
cars = cars.drop('CarName', axis=1)


# In[ ]:


cars.info()


# In[ ]:


# outliers
cars.describe()


# In[ ]:


cars.info()


# ## 3. Data Preparation 
# 
# 
# #### Data Preparation
# 
# Let's now prepare the data and build the model.

# In[ ]:


# split into X and y
X = cars.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber',
       'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',
       'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',
       'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg',
       'car_company']]

y = cars['price']


# In[ ]:


# creating dummy variables for categorical variables

# subset all categorical variables
cars_categorical = X.select_dtypes(include=['object'])
cars_categorical.head()


# In[ ]:


# convert into dummies
cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)
cars_dummies.head()


# In[ ]:


# drop categorical variables 
X = X.drop(list(cars_categorical.columns), axis=1)


# In[ ]:


# concat dummy variables with X
X = pd.concat([X, cars_dummies], axis=1)


# In[ ]:


# scaling the features
from sklearn.preprocessing import scale

# storing column names in cols, since column names are (annoyingly) lost after 
# scaling (the df is converted to a numpy array)
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns


# In[ ]:


# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)


# ## 3. Model Building and Evaluation

# In[ ]:


# Building the first model with all the features

# instantiate
lm = LinearRegression()

# fit
lm.fit(X_train, y_train)


# In[ ]:


# print coefficients and intercept
print(lm.coef_)
print(lm.intercept_)


# In[ ]:


# predict 
y_pred = lm.predict(X_test)

# metrics
from sklearn.metrics import r2_score

print(r2_score(y_true=y_test, y_pred=y_pred))


# Not bad, we are getting approx. 83% r-squared with all the variables. Let's see how much we can get with lesser features.

# #### Model Building Using RFE
# 
# Let's now build a model using recursive feature elimination to select features. We'll first start off with an arbitrary number of features, and then use the ```statsmodels``` library to build models using the shortlisted features (this is also because sklearn doesn't have adjusted r-squared, statsmodels has).

# In[ ]:


# RFE with 15 features
from sklearn.feature_selection import RFE

# RFE with 15 features
lm = LinearRegression()
rfe_15 = RFE(lm, 15)

# fit with 15 features
rfe_15.fit(X_train, y_train)

# Printing the boolean results
print(rfe_15.support_)           
print(rfe_15.ranking_)  


# In[ ]:


# making predictions using rfe model
y_pred = rfe_15.predict(X_test)

# r-squared
print(r2_score(y_test, y_pred))


# In[ ]:


# RFE with 6 features
from sklearn.feature_selection import RFE

# RFE with 6 features
lm = LinearRegression()
rfe_6 = RFE(lm, 6)

# fit with 6 features
rfe_6.fit(X_train, y_train)

# predict
y_pred = rfe_6.predict(X_test)

# r-squared
print(r2_score(y_test, y_pred))


# Note that RFE with 6 features is giving about 88% r-squared, compared to 89% with 15 features. 
# Should we then choose more features for slightly better performance?
# 
# A better metric to look at is adjusted r-squared, which penalises a model for having more features, and thus weighs both the goodness of fit and model complexity. Let's use statsmodels library for this.
# 

# #### Model Building and Evaluation 

# In[ ]:


# import statsmodels
import statsmodels.api as sm  

# subset the features selected by rfe_15
col_15 = X_train.columns[rfe_15.support_]

# subsetting training data for 15 selected columns
X_train_rfe_15 = X_train[col_15]

# add a constant to the model
X_train_rfe_15 = sm.add_constant(X_train_rfe_15)
X_train_rfe_15.head()


# In[ ]:


# fitting the model with 15 variables
lm_15 = sm.OLS(y_train, X_train_rfe_15).fit()   
print(lm_15.summary())


# Note that the model with 15 variables gives about 93.9% r-squared, though that is on training data. The adjusted r-squared is 93.3.

# In[ ]:


# making predictions using rfe_15 sm model
X_test_rfe_15 = X_test[col_15]


# # Adding a constant variable 
X_test_rfe_15 = sm.add_constant(X_test_rfe_15, has_constant='add')
X_test_rfe_15.info()


# # Making predictions
y_pred = lm_15.predict(X_test_rfe_15)


# In[ ]:


# r-squared
r2_score(y_test, y_pred)


# Thus, the test r-squared of model with 15 features is about 89.4%, while training is about 93%. Let's compare the same for the model with 6 features.

# In[ ]:


# subset the features selected by rfe_6
col_6 = X_train.columns[rfe_6.support_]

# subsetting training data for 6 selected columns
X_train_rfe_6 = X_train[col_6]

# add a constant to the model
X_train_rfe_6 = sm.add_constant(X_train_rfe_6)


# fitting the model with 6 variables
lm_6 = sm.OLS(y_train, X_train_rfe_6).fit()   
print(lm_6.summary())


# making predictions using rfe_6 sm model
X_test_rfe_6 = X_test[col_6]


# Adding a constant  
X_test_rfe_6 = sm.add_constant(X_test_rfe_6, has_constant='add')
X_test_rfe_6.info()


# # Making predictions
y_pred = lm_6.predict(X_test_rfe_6)


# In[ ]:


# r2_score for 6 variables
r2_score(y_test, y_pred)


# Thus, for the model with 6 variables, the r-squared on training and test data is about 89% and 88.5% respectively. The adjusted r-squared is about 88.6%.

# ### Choosing the optimal number of features
# 
# Now, we have seen that the adjusted r-squared varies from about 93.3 to 88 as we go from 15 to 6 features, one way to choose the optimal number of features is to make a plot between n_features and adjusted r-squared, and then choose the value of n_features.

# In[ ]:


n_features_list = list(range(4, 20))
adjusted_r2 = []
r2 = []
test_r2 = []

for n_features in range(4, 20):

    # RFE with n features
    lm = LinearRegression()

    # specify number of features
    rfe_n = RFE(lm, n_features)

    # fit with n features
    rfe_n.fit(X_train, y_train)

    # subset the features selected by rfe_6
    col_n = X_train.columns[rfe_n.support_]

    # subsetting training data for 6 selected columns
    X_train_rfe_n = X_train[col_n]

    # add a constant to the model
    X_train_rfe_n = sm.add_constant(X_train_rfe_n)


    # fitting the model with 6 variables
    lm_n = sm.OLS(y_train, X_train_rfe_n).fit()
    adjusted_r2.append(lm_n.rsquared_adj)
    r2.append(lm_n.rsquared)
    
    
    # making predictions using rfe_15 sm model
    X_test_rfe_n = X_test[col_n]


    # # Adding a constant variable 
    X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')



    # # Making predictions
    y_pred = lm_n.predict(X_test_rfe_n)
    
    test_r2.append(r2_score(y_test, y_pred))


# In[ ]:


# plotting adjusted_r2 against n_features
plt.figure(figsize=(10, 8))
plt.plot(n_features_list, adjusted_r2, label="adjusted_r2")
plt.plot(n_features_list, r2, label="train_r2")
plt.plot(n_features_list, test_r2, label="test_r2")
plt.legend(loc='upper left')
plt.show()


# Based on the plot, we can choose the number of features considering the r2_score we are looking for. Note that there are a few caveats in this approach, and there are more sopisticated techniques to choose the optimal number of features:
# 
# - Cross-validation: In this case, we have considered only one train-test split of the dataset; the values of r-squared and adjusted r-squared will vary with train-test split. Thus, cross-validation is a more commonly used technique (you divide the data into multiple train-test splits into 'folds', and then compute average metrics such as r-squared across the 'folds'
# 
# - The values of r-squared and adjusted r-squared are computed based on the training set, though we must *always look at metrics computed on the test set*. For e.g. in this case, the test r2 actually goes down with increasing n - this phenomenon is called 'overfitting', where the performance on training set is good because the model has in some way 'memorised' the dataset, and thus the performance on test set is worse.
# 
# Thus, we can choose anything between 4 and 12 features, since beyond 12, the test r2 goes down; and at lesser than 4, the r2_score is too less.
# 
# In fact, the test_r2 score doesn't increase much anyway from n=6 to n=12. It is thus wiser to choose a simpler model, and so let's choose n=6.
# 

# ### Final Model
# 
# Let's now build the final model with 6 features.

# In[ ]:


# RFE with n features
lm = LinearRegression()

n_features = 6

# specify number of features
rfe_n = RFE(lm, n_features)

# fit with n features
rfe_n.fit(X_train, y_train)

# subset the features selected by rfe_6
col_n = X_train.columns[rfe_n.support_]

# subsetting training data for 6 selected columns
X_train_rfe_n = X_train[col_n]

# add a constant to the model
X_train_rfe_n = sm.add_constant(X_train_rfe_n)


# fitting the model with 6 variables
lm_n = sm.OLS(y_train, X_train_rfe_n).fit()
adjusted_r2.append(lm_n.rsquared_adj)
r2.append(lm_n.rsquared)


# making predictions using rfe_15 sm model
X_test_rfe_n = X_test[col_n]


# # Adding a constant variable 
X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')



# # Making predictions
y_pred = lm_n.predict(X_test_rfe_n)

test_r2.append(r2_score(y_test, y_pred))


# In[ ]:


# summary
lm_n.summary()


# In[ ]:


# results 
r2_score(y_test, y_pred)


# ### Final Model Evaluation
# 
# Let's now evaluate the model in terms of its assumptions. We should test that:
# - The error terms are normally distributed with mean approximately 0
# - There is little correlation between the predictors
# - Homoscedasticity, i.e. the 'spread' or 'variance' of the error term (y_true-y_pred) is constant

# In[ ]:


# Error terms
c = [i for i in range(len(y_pred))]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
plt.show()


# In[ ]:


# Plotting the error terms to understand the distribution.
fig = plt.figure()
sns.distplot((y_test-y_pred),bins=50)
fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 
plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label
plt.ylabel('Index', fontsize=16)                          # Y-label
plt.show()


# In[ ]:


# mean
np.mean(y_test-y_pred)


# Now it may look like that the mean is not 0, though compared to the scale of 'price', -380 is not such a big number (see distribution below).

# In[ ]:


sns.distplot(cars['price'],bins=50)
plt.show()


# In[ ]:


# multicollinearity
predictors = ['carwidth', 'curbweight', 'enginesize', 
             'enginelocation_rear', 'car_company_bmw', 'car_company_porsche']

cors = X.loc[:, list(predictors)].corr()
sns.heatmap(cors, annot=True)
plt.show()


# Though this is the most simple model we've built till now, the final predictors still seem to have high correlations. One can go ahead and remove some of these features, though that will affect the adjusted-r2 score significantly (you should try doing that). 
# 
# 
# Thus, for now, the final model consists of the 6 variables mentioned above.

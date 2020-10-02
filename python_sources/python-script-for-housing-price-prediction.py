# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# -*- coding: utf-8 -*-
"""

There are Five parts to these codes
1 - Study of Categorical data
2 - 

@author: VivekChutke
"""
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import seaborn as sns


from scipy import stats
from sklearn import linear_model # Scikit learn library that implements generalized linear models
from sklearn import neighbors # provides functionality for unsupervised and supervised neighbors-based learning methods
from sklearn.metrics import mean_squared_error # Mean squared error regression loss
from sklearn import preprocessing # provides functions and classes to change raw feature vectors
from math import log

#df = pd.read_csv('/Users/malleshb/Documents/PPT/ClassMaterial/Unit2Regression/HousePrice_Regression/kc_house_data.csv')
df = pd.read_csv('../input/kc_house_data.csv')
df= df.drop(['date','id'],axis=1) # Note: axis=1 denotes that we are referring to a column, not a row

# Part 1 -----  Study of Categorical data -- 
# -----   Base Features  - All the Features except Following Data
# ---- Categorical Features - cat_cols = ['floors', 'view', 'condition', 'grade', 'sqft_basement', 'yr_renovated']
# ------ Use Base featues + One Cat Feature, Perform Linear regression -- Total 6 models
# Q1 : Plot RMSE and R for each model on the Same plot and observe effect of each Feature
train_data, test_data = train_test_split(df, train_size = 0.8)

df['waterfront'] = df['waterfront'].astype('category',ordered=True)
df['view'] = df['view'].astype('category',ordered=True)
df['condition'] = df['condition'].astype('category',ordered=True)
df['grade'] = df['grade'].astype('category',ordered=False)
df['zipcode'] = df['zipcode'].astype(str)

# A joint plot is used to visualize the bivariate distribution
sns.jointplot(x="sqft_living", y="price", data=df, kind = 'reg', size = 10)
sns.jointplot(x="sqft_basement", y="price", data=df, kind = 'reg', size = 10)
sns.jointplot(x="yr_renovated", y="price", data=df, kind = 'reg', size = 10)
plt.show()

# Make the two variables as categorical data
df['basement_present'] = df['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)
df['basement_present'] = df['basement_present'].astype('category', ordered = False)

df['renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
df['renovated'] = df['renovated'].astype('category', ordered = False)

# Read these four as categorical variables
cat_cols = ['floors', 'view', 'condition', 'grade']

for cc in cat_cols:
    dummies = pd.get_dummies(df[cc], drop_first=False)
    dummies = dummies.add_prefix("{}#".format(cc))
    df.drop(cc, axis=1, inplace=True)
    df = df.join(dummies)


#------ part2 --- Making use of a function for iterative model evaluation
#------ Part 1 above can be performed easily using this function
# Q2 : Perform the same analysis by performing iterations on the following Function
# A function that take one input of the dataset 
# and return the RMSE, R, and Bias-parameter (intercept)

train_data, test_data = train_test_split(df, train_size = 0.8)

def simple_linear_model(train, test, input_feature):
    lm = linear_model.LinearRegression() # Create a linear regression object
    lm.fit(train.as_matrix(columns = [input_feature]), train.as_matrix(columns = ['price'])) # Train the model
    RMSE = mean_squared_error(test.as_matrix(columns = ['price']), 
                              lm.predict(test.as_matrix(columns = [input_feature])))**0.5 # Calculate the RMSE on test data
    return RMSE, lm.intercept_[0], lm.coef_[0][0]


RMSE, w0, w1 = simple_linear_model(train_data, test_data, 'sqft_living')
print(RMSE, w0, w1)


#-------- part 3 Using large number of variables in modeling
# ------  Use Features = Base Features + ALL Categorical features + dummies_Zipcode
# ------   Q3 : Use three sets of dummies_zipcodes [2, 5, 70] in numbers
# -------  And plot RMSE for each case, for both J_train  and J_test 

lm = linear_model.LinearRegression() # Create a linear regression object
dummies_zipcodes = pd.get_dummies(df['zipcode'], drop_first=True)
dummies_zipcodes.reset_index(inplace=True)
dummies_zipcodes = dummies_zipcodes.add_prefix("{}#".format('zipcode'))
df.drop('zipcode', axis=1, inplace=True)
# total 70 zipcodes exist


other_features = ['sqft_living']
dummies_zipcodes_partial = dummies_zipcodes[['zipcode#98004','zipcode#98102','zipcode#98109','zipcode#98112','zipcode#98039','zipcode#98040']]
#dummies_zipcodes_partial = dummies_zipcodes[['zipcode#98004']]

df_short = df.select_dtypes(include=['float64', 'int64'])
df_new = df[other_features].join(dummies_zipcodes_partial)
y = df['price']


# Model evaluation using train_test split
X_train, X_test, y_train, y_test = train_test_split(df_new, y, test_size=0.3, random_state=0)
lm.fit(X_train, y_train)

print(metrics.mean_squared_error(y_test, lm.predict(X_test)))
print('Variance score: %.2f' % lm.score(X_test, y_test))



# ---------- Part4  : Get RMSE for test, train of different size of data-sets
# Assignment : Randomize the data. Then plot the learning curves  for J_train  and J_test
#              Train vs. Test error for 10 data-set splits  from 0% to 90%

for i in range(1,10):
    X_train, X_test, y_train, y_test = train_test_split(df_new, y, test_size=0.1*i, random_state=0)
    lm.fit(X_train, y_train)
    print("Size of Train data : ", 0.1*i)
    print(metrics.mean_squared_error(y_train, lm.predict(X_train)))
    print(metrics.mean_squared_error(y_test, lm.predict(X_test)))
    print('Variance score: %.2f' % lm.score(X_test, y_test))
    
    
# --------- Part 5 :  Add polynomial features for sqft_living to boost the accuracy
# ------  Use Features = Base Features + ALL Categorical features + Exotic_features
# ------- By Plotting Train and Test error, with one Exotic_feature at a time,
# -----   Compare the J_test and J_train to check for bias

dummies_zipcodes_partial = dummies_zipcodes[['zipcode#98004','zipcode#98102','zipcode#98109','zipcode#98112','zipcode#98039','zipcode#98040']]
df_new = df_short[other_features].join(dummies_zipcodes_partial)

#Exotic1A
df_new['sqft_living_squared'] = df_new['sqft_living'].apply(lambda x: x**2) 
#Exotic1B
df_new['sqft_living_cubed'] = df_new['sqft_living'].apply(lambda x: x**3) 
#Exotic1C  
df_new['sqft_living_quad'] = df_new['sqft_living'].apply(lambda x: x**4) 


X_train, X_test, y_train, y_test = train_test_split(df_new, y, test_size=0.3, random_state=0)
lm.fit(X_train, y_train)

print(metrics.mean_squared_error(y_test, lm.predict(X_test)))
print('Variance score: %.2f' % lm.score(X_test, y_test))
print(metrics.mean_squared_error(y_train, lm.predict(X_train)))
print('Variance score: %.2f' % lm.score(X_train, y_train))


#  - ---    Lets add polynomial terms of few other features also
del df_new
other_features = ['sqft_living', 'basement_present', 'renovated']

dummies_zipcodes_partial = dummies_zipcodes[['zipcode#98004','zipcode#98102','zipcode#98109','zipcode#98112','zipcode#98039','zipcode#98040']]
df_new = df[other_features].join(dummies_zipcodes_partial)

# #Exotic2 : sqft_living cubed
df_new['sqft_living_cubed'] = df['sqft_living'].apply(lambda x: x**3) 

# #Exotic3 : bedrooms_squared: this feature will mostly affect houses with many bedrooms.
df_new['bedrooms_squared'] = df['bedrooms'].apply(lambda x: x**2) 

# #Exotic4 : bedrooms times bathrooms gives what's called an "interaction" feature. It is large when both of them are large.
df_new['bed_bath_rooms'] = df['bedrooms']*df['bathrooms']

# #Exotic5 : bringing large values closer together and spreading out small values.
df_new['log_sqft_living'] = df['sqft_living'].apply(lambda x: log(x))


# ----   Part6 :   Cross validation methods

scores = cross_val_score(lm, df_new, y, cv=5)
print(scores)

# Assign 3 :  Make 10 blocks, average of coefficients (PLot price vs. sqft-living)

# ---- Part7 : Regularisation using L2 norm
# ----  Perform Regularisation for Features = ['floors', 'view', 'condition', 'grade'] and,
# ----  Polynomial features of Sqft_living
#  --- Make a plot of J_train and J_test with and without regularisation, for different values of lambda (alpha)
del df_new

cat_cols = ['floors', 'view', 'condition', 'grade']    
df_short = df.select_dtypes(include=['float64', 'int64'])
df_new = pd.concat([df, dummies],axis=1)
df_new['sqft_living_squared'] = df_new['sqft_living'].apply(lambda x: x**2) 
df_new['sqft_living_cubed'] = df_new['sqft_living'].apply(lambda x: x**3)     
df_new['sqft_living_quad'] = df_new['sqft_living'].apply(lambda x: x**4)
 

from sklearn.linear_model import Ridge
regr = linear_model.Ridge(alpha=10000000000000.0)
#clf = Ridge(alpha=1.0)
regr.fit(X_train, y_train)
print(metrics.mean_squared_error(y_test, regr.predict(X_test)))
print('Variance score: %.2f' % regr.score(X_test, y_test))


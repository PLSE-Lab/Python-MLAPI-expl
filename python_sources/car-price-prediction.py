#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Image
import os
get_ipython().system('ls ../input/')


# 

# # Problem Statement
# 
# A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts. 
# 
#  
# 
# They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:
# 
# Which variables are significant in predicting the price of a car
# How well those variables describe the price of a car
# 
# Based on various market surveys, the consulting firm has gathered a large dataset of different types of cars across the Americal market. 

# ![](http://)

# # Step 1: Reading and Understanding the Data

# In[ ]:


import numpy as np
import pandas as pd
# Supress Warnings

import warnings
warnings.filterwarnings('ignore') 


# In[ ]:


car = pd.read_csv("../input/header/CarPrice_Assignment.csv")


# In[ ]:


# Check the head of the dataset
car.head()


# Inspecting the various important aspects of the given dataframe

# In[ ]:


car.shape


# In[ ]:


car.info()


# In[ ]:


car.describe()


# # Data Cleaning and Preparation

# In[ ]:


# we  need to consider only company name as the independent variable for model building. 


# In[ ]:


#Splitting company name from CarName column
CompanyName = car['CarName'].apply(lambda x : x.split(' ')[0])
car.insert(3, "CompanyName", CompanyName)
car.drop("CarName", axis = 1, inplace = True)
car.head()


# In[ ]:



car.CompanyName.unique()


# In[ ]:


# Correcting Invalid values(Spelling Errors)

car.CompanyName = car.CompanyName.str.lower()

def replace_name(a,b):
    car.CompanyName.replace(a, b, inplace = True)
    

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

car.CompanyName.unique()


# In[ ]:



#Checking for duplicates
car.loc[car.duplicated()]


# In[ ]:


pd.set_option('display.max_columns',500)


# In[ ]:


car.head()


# In[ ]:


car['CompanyName'].value_counts()


# ## we can see that Toyota is the top choice

# # Visualising the Data

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# Visualising the numeric variables

# In[ ]:


#Visualising Price vs variuos parameters

def scatter (x,fig):
    plt.subplot(5,2,fig)
    plt.scatter(car[x],car['price'])
    plt.title(x+' vs Price')
    plt.ylabel('Price')
    plt.xlabel(x)


plt.figure(figsize=(10,20))

scatter('carlength', 1)
scatter('carwidth', 2)
scatter('carheight', 3)
scatter('curbweight', 4)

plt.tight_layout()
    


# In[ ]:


#Deduction::

#carheight doesn't show any significant correlation with price.

#width, length and weight seems to have a poitive correlation with price as can be easily inferred by above.


# In[ ]:


np.corrcoef(car['carlength'], car['carwidth'])[0,1]


# In[ ]:


#visualising Price vs few more important variables

def pairplot(x,y,z):
    sns.pairplot(car, x_vars = [x,y,z], y_vars = ['price'], size = 4, aspect = 1, kind='scatter')
    plt.show()
    
pairplot('compressionratio', 'horsepower', 'peakrpm')
pairplot('enginesize', 'boreratio', 'stroke')
pairplot('wheelbase', 'citympg', 'highwaympg')


# In[ ]:


#Deductions:

#citympg, highwaympg have a negative correlation with price as can be easily inferred from above 
#however boreratio, enginesize, horsepower, wheelbase - seem to have a  positive correlation with price..


# In[ ]:


# visualising which numeric variables has good correlation with price  i.e which affects the price mostly.
sns.pairplot(car)
plt.show()


# In[ ]:


##width, length and weight seems to have a poitive correlation with price
#citympg, highwaympg have a negative correlation with price as can be easily inferred from above 
#however boreratio, enginesize, horsepower, wheelbase - seem to have a  positive correlation with price
#carheight doesn't show any significant correlation with price.


# In[ ]:


#Binning all the Car Companies based on average price of each Company.

car['price'] = car['price'].astype('int')
temp = car.copy()
car1 = temp.groupby(['CompanyName'])['price'].mean()
temp = temp.merge(car1.reset_index(), how='left',on='CompanyName')
bins = [0,10000,20000,40000]
car_bin=['Budget','Medium','Luxurious']
car['Range'] = pd.cut(temp['price_y'],bins,right=False,labels=car_bin)
car.head()


# #Visualising Categorical Data

# In[ ]:


# CompanyName
# Symboling
#fueltype
#enginetype
#carbody
#doornumber
#enginelocation
#fuelsystem
#cylindernumber
#aspiration
#drivewheel


# In[ ]:


plt.figure(figsize=(25, 6))

plt.subplot(1,3,1)
plt1 = car.fueltype.value_counts().plot('bar')
plt.title('Fuel Type Histogram')
plt1.set(xlabel = 'Fuel Type', ylabel='Fuel type Freq')


plt.subplot(1,3,2)
plt1 = car.carbody.value_counts().plot('bar')
plt.title('Car Type Histogram')
plt1.set(xlabel = 'Car Type', ylabel='car type Freq')

plt.subplot(1,3,3)
plt1 = car.CompanyName.value_counts().plot('bar')
plt.title('Company Histogram')
plt1.set(xlabel = 'Car company', ylabel='company freq')

# Deductions:

#sedan is most preffered type.
# Gas Fueled cars are much more than diesel fueled.
# again same inference that "Toyota" is most favored car company.
# In[ ]:


plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('Symboling versus Price')
sns.boxplot(x=car.symboling, y=car.price, palette=("cubehelix"))

plt.subplot(1,2,2)
plt.title('Symboling Histogram')
sns.countplot(car.symboling, palette=("cubehelix"))

plt.show()


# In[ ]:


# Inferences::

#symboling 0 &1 has maximum frequency
# symboling -1,-2 has high price and thats justified since -1,-2 are good
# also 3 has same price range as -2
# There is a decrease at 1


# In[ ]:


plt.figure(figsize=(25, 6))
df = pd.DataFrame(car.groupby(['fueltype'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Fuel Type versus Average Price')
plt.show()


# In[ ]:


#Deduction:
# Diesel fueled has higher price than Gas fueled


# In[ ]:


plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'drivewheel', y = 'price', data = car)
plt.subplot(2,3,2)
sns.boxplot(x = 'aspiration', y = 'price', data = car)
plt.subplot(2,3,3)
sns.boxplot(x = 'cylindernumber', y = 'price', data = car)
plt.subplot(2,3,4)
sns.boxplot(x = 'enginetype', y = 'price', data = car)
plt.subplot(2,3,5)
sns.boxplot(x = 'symboling', y = 'price', data = car)
plt.subplot(2,3,6)
sns.boxplot(x = 'fueltype', y = 'price', data = car)
plt.show()


# In[ ]:


# Deductions::
#rwd has higher price than fwd and 4wd
#aspiration turbo has higher price than std
# 8-cylinder has highest price and 4-cylinder lowest
# -1 & -2 synboling has higher prices that means auto is pretty safe and thats justified.
# Engine type Ohcv has higher range then comes dohc and then i 


# In[ ]:


plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'carbody', y = 'price', data = car)
plt.subplot(2,3,2)
sns.boxplot(x = 'doornumber', y = 'price', data = car)
plt.subplot(2,3,3)
sns.boxplot(x = 'enginelocation', y = 'price', data = car)
plt.subplot(2,3,4)
sns.boxplot(x = 'fuelsystem', y = 'price', data = car)


# In[ ]:


# Deductions::
#hardtop has highest price range followed by convertible
# 4-door has slightly has price than 2-doors
# rear engine location has pretty higher price range as compared to front engine location
# mpfi and idi fuel system has higher price range than others


# In[ ]:


plt.figure(figsize=(25, 6))

df = pd.DataFrame(car.groupby(['drivewheel','fuelsystem','Range'])['price'].mean().unstack(fill_value=0))
df.plot.bar()
plt.title('Car Range versus Average Price')
plt.show()


# #High price ranged cars has rwd drivewheel with idi or mpfi fuelsystem.

# In[ ]:


# see  General spread of car-price 

plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Car-Price Distribution Plot')
sns.distplot(car.price)

plt.subplot(1,2,2)
plt.title('Car-Price Spread')
sns.boxplot(y=car.price)

plt.show()


# In[ ]:


print(car.price.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1]))

#useful observations from above plots

# we have data points far more away spread from mean as can be seen in the box plot.
# 85% of the prices are below 18,500, whereas the remaining 15% are between 18,500 and 45,400.) 
# Much difference between Mean and Median 
# In[ ]:





# In[ ]:





# ##important and significant vaiables after visualisation##

# In[ ]:


#Range,wheel base,Fuel type,Boreratio,Aspiration,Cylinder Number,Drivewheel,
#Curbweight,Car width, Car length, Engine Size, carbody, Horse Power,Engine type 


# In[ ]:


cars_regression = car[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 'carlength','carwidth', 'Range']]
cars_regression.head()


# In[ ]:


sns.pairplot(cars_regression)
plt.show()


# # Data Preparation

# In[ ]:




# Defining the map function for Dummy variables

def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df

# Applying the function to the car_Regression

cars_regression = dummies('Range',cars_regression)
cars_regression= dummies('carbody',cars_regression)
cars_regression = dummies('aspiration',cars_regression)
cars_regression = dummies('enginetype',cars_regression)
cars_regression = dummies('drivewheel',cars_regression)
cars_regression = dummies('cylindernumber',cars_regression)
cars_regression = dummies('fueltype',cars_regression)


# In[ ]:


cars_regression.head()


# In[ ]:


cars_regression.shape


# # Splitting the Data into Training and Testing Sets

# In[ ]:


from sklearn.model_selection import train_test_split

#We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)
df_train, df_test = train_test_split(cars_regression, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[ ]:


df_train.shape


# In[ ]:


df_test.shape


# # Feature scaling

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()
num_vars = ['wheelbase', 'enginesize', 'curbweight', 'horsepower', 'boreratio','carwidth','carlength','price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# In[ ]:


#Plotting Heatmap to see the Correlation
plt.figure(figsize = (30, 25))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# # Highly correlated variables to price are - highend , enginesize, horsepower,carwidth and curbweight.

# In[ ]:


# dividing dataframe into x and y variables:

y_train = df_train.pop('price')


# In[ ]:


x_train = df_train


# In[ ]:


y_train.shape


# In[ ]:


x_train.shape


# # Bulding the Model

# In[ ]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()
lm.fit(x_train,y_train)
rfe = RFE(lm, 10)
rfe = rfe.fit(x_train, y_train)


# In[ ]:


list(zip(x_train.columns,rfe.support_,rfe.ranking_))


# In[ ]:


x_train.columns[rfe.support_]


# In[ ]:


x_train_rfe = x_train[x_train.columns[rfe.support_]]
x_train_rfe.head()


# In[ ]:


## Invoking Functions to Build Model and  to Check VIF

def build_model(X,y):
    X = sm.add_constant(X) #to add constant
    lm = sm.OLS(y,X).fit() # fitting the model
    print(lm.summary()) # model summary
    return X
    
def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)


# In[ ]:


x_train_new = build_model(x_train_rfe,y_train)


# In[ ]:


#Calculating the Variance Inflation Factor
checkVIF(x_train_new)


# In[ ]:


# dropping curbweight to to high VIF value (it has high multicolinearity)
x_train_new = x_train_new.drop(["curbweight"], axis = 1)


# Model 1

# In[ ]:


x_train_new = build_model(x_train_new,y_train)


# In[ ]:


checkVIF(x_train_new)


# we can see VIF has been decreased for most of the variables automatically
# 

# In[ ]:


#dropping engine size due to its high p-value
x_train_new = x_train_new.drop(["enginesize"], axis = 1)


# Model 2

# In[ ]:


x_train_new = build_model(x_train_new,y_train)


# In[ ]:


checkVIF(x_train_new)


# In[ ]:


# dropping sedan becasue of its high VIF value
x_train_new = x_train_new.drop(["sedan"], axis = 1)


# Model 3

# In[ ]:


x_train_new = build_model(x_train_new,y_train)


# In[ ]:


checkVIF(x_train_new)


# In[ ]:


# dropping wagon due to high p-value as it becomes insignificant then.


# In[ ]:


x_train_new = x_train_new.drop(["wagon"], axis = 1)


# Model 4

# In[ ]:


x_train_new = build_model(x_train_new,y_train)


# In[ ]:


checkVIF(x_train_new)


# In[ ]:


# dropping dohcv and check stats


# In[ ]:


x_train_new = x_train_new.drop(["dohcv"], axis = 1)


# Model 4

# In[ ]:


x_train_new = build_model(x_train_new,y_train)


# In[ ]:


#removing three because of high p-value

x_train_new = x_train_new.drop(["three"], axis = 1)


# Model 5

# In[ ]:


x_train_new = build_model(x_train_new,y_train)


# In[ ]:


checkVIF(x_train_new)


# In[ ]:





# # Residual Analysis

# In[ ]:



lm = sm.OLS(y_train,x_train_new).fit()
y_train_price = lm.predict(x_train_new)


# In[ ]:


# Plotting the histogram of the error terms

fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  
plt.xlabel('Errors', fontsize = 18)


# # As we can see Error terms are approximately normally distributed so our assumptions are satisfied.

# In[ ]:





# # Prediction and Evaluation

# #Applying the scaling on the test sets

# In[ ]:


#Scaling the test set
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','carlength','carwidth','price']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])


# In[ ]:


df_test.describe()


# In[ ]:


#Dividing into X and y


# In[ ]:


y_test = df_test.pop('price')


# In[ ]:


x_test = df_test


# In[ ]:


# # Now let's use our model to make predictions.

x_train_new = x_train_new.drop('const',axis=1)


# In[ ]:


# Creating X_test_new dataframe by dropping variables from X_test
x_test_new = x_test[x_train_new.columns]


# In[ ]:


# Adding a constant variable 
x_test_new = sm.add_constant(x_test_new)


# # Making Predictions

# In[ ]:


y_pred = lm.predict(x_test_new)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(y_test, y_pred)


# # we can see there is almost 4 % difference between R-Square of Test and Train Set which means our Model is performing well.

# # Model Evaluation

# In[ ]:


#Let's now plot the graph for actual versus predicted values.



# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_predict', fontsize=20)              # Plotting heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)   


# In[ ]:


print(lm.summary())


# # Useful Deductions

# p-values - p-values for all the coefficients are less than the significance level of 0.05. - that means all the predictors 
# are statistically significant.
#  
# R-sqaured and Adjusted R-squared (extent of fit) - 0.899 and 0.896 - 90% variance explained.
#  
# F-stats and Prob(F-stats) (overall model fit) - 308 and 1.04e-67(approx. 0.0) - Model fir is significant and explained 90% variance is noy just by chance.
# 

# # Equation of best fit line::

# We can see that the equation of our best fitted line is:
# 
# price = .395 x carwidth + .279 x Luxurious + .44 x horsepower -.0414 x hatchback - .0824

# In[ ]:





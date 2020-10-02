#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement

# A Chinese automobile company aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts. 
# 
# They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:
# 
# 1. Which variables are significant in predicting the price of a car
# 
# 2. How well those variables describe the price of a car
# 
# Based on various market surveys, the consulting firm has gathered a large dataset of different types of cars across the Americal market.
# 

# ## Business Goal

# You are required to model the price of cars with the available independent variables.
# 
# It will be used by the management to understand how exactly the prices vary with the independent variables.
# They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. 
# 
# Further, the model will be a good way for management to understand the pricing dynamics of a new market. 

# Note : There is a variable named CarName which is comprised of two parts - the first word is the name of 'car company' and the second is the 'car model'. For example, chevrolet impala has 'chevrolet' as the car company name and 'impala' as the car model name.

# ## Step 1: Reading and Understanding the Data

# In[ ]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[ ]:


# Read the csv
carprice = pd.read_csv("../input/car-price-dataset/CarPrice_Assignment.csv")


# In[ ]:


# Read the head
carprice.head()


# In[ ]:


# 205 rows and 26 columns
carprice.shape


# In[ ]:


carprice.info()


# In[ ]:


carprice.describe()


# In[ ]:


# Checking Null values%
round(100*(carprice.isnull().sum()/len(carprice.index)),2)
# There are no NULL values


# ## Step 2: Data Cleaning

# In[ ]:


# Drop the car_ID column as to does not hold any significance for developing the model
carprice.drop(['car_ID'], axis = 1, inplace = True)


# In[ ]:


# Convert CarName to lower string
carprice['CarName'] = carprice['CarName'].str.lower()

# Create a new column called company from the first word in CarName Values
carprice['company'] = carprice['CarName'].str.split(' ').str[0]

print(carprice['company'].value_counts())


# In[ ]:


# Perform corrections in the company names
carprice['company'].replace(to_replace="vokswagen", value = 'volkswagen', inplace=True)
carprice['company'].replace(to_replace="vw", value = 'volkswagen', inplace=True)
carprice['company'].replace(to_replace="toyouta", value = 'toyota', inplace=True)
carprice['company'].replace(to_replace="porcshce", value = 'porsche', inplace=True)
carprice['company'].replace(to_replace="maxda", value = 'mazda', inplace=True)

print(carprice['company'].value_counts())


# In[ ]:


# Drop the CarName column now as we have created a new column of company which will be used in analysis and modeling
carprice.drop(['CarName'], axis = 1, inplace = True)


# In[ ]:


print(carprice['enginelocation'].value_counts(),"\n")


# In[ ]:


# Almost all the values of enginelocation are front; hence dropping that column
carprice.drop(['enginelocation'], axis = 1, inplace = True)


# ## Step 3: Visualizing the Data

# ### Plot of Target Column -  Price

# In[ ]:


plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Car Price Distribution Plot')
sns.distplot(carprice.price)

plt.subplot(1,2,2)
plt.title('Car Price Spread')
sns.boxplot(y=carprice.price)

plt.show()
carprice.price.describe()


# Summary:
# 
# The plot is right-skewed, meaning that the most prices in the dataset are low(Below 15,000). The data points are far spread out from the mean, which indicates a high variance in the target columns that is car prices.(75% of the prices are below 16,503, whereas the remaining 25% are between 16,502 and 45,400)

# ### Analysis of Car Company vs Car Price

# In[ ]:


print(carprice['company'].value_counts(),"\n")
print(carprice.groupby("company").price.mean().sort_values(ascending=False))

plt.figure(figsize=(16, 8))

plt.subplot(2,1,1)
ax1 = sns.countplot(y="company", data = carprice)
ax1.set(ylabel='Car Company', xlabel='Count of Cars')

plt.subplot(2,1,2)
ax2 = sns.barplot(y="company", x = "price" , data = carprice)
ax2.set(ylabel='Car Company', xlabel='Average Car Price')

plt.show()


# Summary:
# 
# Toyoto seems to be the prefered company.
# 
# Jaguar, Buick and Porsche are the top3 companies in terms of average price.

# ### Analysis of Symboling vs Car Price

# In[ ]:


print(carprice['symboling'].value_counts(),"\n")
print(carprice.groupby("symboling").price.mean())

plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
ax1 = sns.countplot(x="symboling", data = carprice)
ax1.set(xlabel='Insurance Risk Rating', ylabel='Count of Cars')

plt.subplot(1,2,2)
ax2 = sns.barplot(x="symboling", y = "price" , data = carprice)
ax2.set(xlabel='Insurance Risk Rating', ylabel='Average Car Price')

plt.show()


# Summary:
# 
# Most of the cars seem to get 0 or 1 insurance rating.
# 
# Very few cars have got -2 rating.

# ### Analysis of Fuel Type vs Car Price

# In[ ]:


print(carprice['fueltype'].value_counts(),"\n")
print(carprice.groupby("fueltype").price.mean())

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
ax1 = sns.countplot(x="fueltype", data = carprice)
ax1.set(xlabel='Type of Fuel', ylabel='Count of Cars')
plt.subplot(1,2,2)
ax2 = sns.barplot(x="fueltype", y = "price" , data = carprice)
ax2.set(xlabel='Type of Fuel', ylabel='Average Car Price')
plt.show()


# Summary:
# 
# Average Car Price is higher for diesel cars as compared to non diesel cars.
# 
# Also note that number of observations of diesel cars is quite less as compared to non diesel cars.

# ### Analysis of Aspiration vs Car Price

# In[ ]:


print(carprice['aspiration'].value_counts(),"\n")
print(carprice.groupby("aspiration").price.mean())
plt.figure(figsize=(10, 5))
ax = sns.barplot(x="aspiration", y = "price" , data = carprice)
ax.set(xlabel='Aspiration', ylabel='Average Car Price')
plt.show()


# Summary:
# 
# Turbo cars have higher average price than standard cars.

# ### Analysis of Number of Doors vs Car Price

# In[ ]:


print(carprice['doornumber'].value_counts(),"\n")
print(carprice.groupby("doornumber").price.mean())

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
ax1 = sns.countplot(x="doornumber", data = carprice)
ax1.set(xlabel='Number of Doors', ylabel='Count of Cars')
plt.subplot(1,2,2)
ax2 = sns.barplot(x="doornumber", y = "price" , data = carprice)
ax2.set(xlabel='Number of Doors', ylabel='Average Car Price')
plt.show()


# Summary:
# 
# There is almost no variance of average price by the number of doors category.
# 
# Also note that there are decent number of observations for both 2 and 4 number of doors.

# ### Analysis of Car Body vs Car Price

# In[ ]:


print(carprice['carbody'].value_counts(),"\n")
print(carprice.groupby("carbody").price.describe())

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
ax1 = sns.countplot(x="carbody", data = carprice)
ax1.set(xlabel='Car Body', ylabel='Count of Cars')
plt.subplot(1,2,2)
ax2 = sns.barplot(x="carbody", y = "price" , data = carprice)
ax2.set(xlabel='Car Body', ylabel='Average Car Price')
plt.show()


# Summary:
# 
# Sedan and hatchback are the top 2 most common cars.
# 
# Hard top and convertible cars are more expensive than other type of cars.
# 
# Based upon the analyis, also we replaced the hard top and convertible values to a single value hardtop_or_convertible.

# In[ ]:


# Based upon data visualization of car body, replace hardtop and convertible values to a single value
carprice['carbody'].replace(to_replace="hardtop", value = 'hardtop_or_convertible', inplace=True)
carprice['carbody'].replace(to_replace="convertible", value = 'hardtop_or_convertible', inplace=True)
print(carprice['carbody'].value_counts(),"\n")


# ### Analysis of Number of Cylinders vs Car Price

# In[ ]:


print(carprice['cylindernumber'].value_counts(),"\n")
print(carprice.groupby("cylindernumber").price.mean().sort_values(ascending=False))

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
ax1 = sns.countplot(x="cylindernumber", data = carprice)
ax1.set(xlabel='Number of cylinders', ylabel='Count of Cars')
plt.subplot(1,2,2)
ax2 = sns.barplot(x="cylindernumber", y = "price" , data = carprice)
ax2.set(xlabel='Number of cylinders', ylabel='Average Car Price')
plt.show()


# Summary:
# 
# Most of cars seem to have 4 cylinders.
# 
# Average car price is highest for 8 and 12 cylinders cars.

# ### Analysis of Engine Type vs Car Price

# In[ ]:


print(carprice['enginetype'].value_counts(),"\n")
print(carprice.groupby("enginetype").price.mean().sort_values(ascending=False))

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
ax1 = sns.countplot(x="enginetype", data = carprice)
ax1.set(xlabel='Engine Type', ylabel='Count of Cars')
plt.subplot(1,2,2)
ax2 = sns.barplot(x="enginetype", y = "price" , data = carprice)
ax2.set(xlabel='Engine Type', ylabel='Average Car Price')
plt.show()


# Summary:
# 
# Most cars are of ohc engine type.
# 
# dohcv car type is the most expensive but there is jut once observation.

# ### Analysis of Fuel System vs Car Price

# In[ ]:


print(carprice['fuelsystem'].value_counts(),"\n")
print(carprice.groupby("fuelsystem").price.mean().sort_values(ascending=False))

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
ax1 = sns.countplot(x="fuelsystem", data = carprice)
ax1.set(xlabel='Fuel System', ylabel='Count of Cars')
plt.subplot(1,2,2)
ax2 = sns.barplot(x="fuelsystem", y = "price" , data = carprice)
ax2.set(xlabel='Fuel System', ylabel='Average Car Price')
plt.show()


# Summary:
# 
# mpfi and 2bbl are most common type of fuel systems. 
# 
# mpfi and idi having the highest price range. 

# ### Analysis of Numeric Columns vs Car Price
# 

# In[ ]:


# Derive a new column that is Fuel economy from citympg and highwaympg
carprice['fueleconomy'] = (0.55 * carprice['citympg']) + (0.45 * carprice['highwaympg'])

# Drop both citympg and highwaympg
carprice.drop(['citympg','highwaympg'], axis = 1, inplace = True)


# In[ ]:


def pp(x,y,z):    
    sns.pairplot(carprice, x_vars=[x,y,z], y_vars='price',size=4, aspect=1, kind='scatter')
    plt.show()

pp('carwidth', 'carlength', 'curbweight')
pp('carheight','enginesize', 'boreratio' )
pp('stroke','compressionratio', 'horsepower')
pp('peakrpm','wheelbase', 'fueleconomy')


# Summary:
# 
# carwidth, carlength, curbweight, boreratio seems to have a positive correlation with price.
# 
# carheight doesn't show any significant trend with price.
# 
# enginesize and horsepower - seem to have a significant positive correlation with price.
# 
# fueleconomy - seem to have a significant negative correlation with price.

# ### Check the correlation coefficients to see which variables are highly correlated

# In[ ]:


plt.figure(figsize = (14, 8))
sns.heatmap(carprice.corr(), annot = True, cmap="YlGnBu")
plt.show()


# Summary:
# 
# Price is positively correlated with curbweight, car length, car width, engine size and horse power.
# 
# Price is negatively correlated with fuel economy (-0.70).
# 
# car length and car width are themselves correlated (0.84).
# 
# Also curb weight is correlated with car length and car width.

# ## Step 4: Data Preparation For Modeling

# ### Derived Variables

# In[ ]:


# Create a new column called company category having values Budget, Mid_Range and Luxury based upon 
# company average price of their cars.
# If company average price < 10000 then Budget
# Else If company average price >= 10000 and < 20000 then Mid_Range
# Else If company average price > 20000 then Luxury
carprice["company_average_price"] = round(carprice.groupby('company')["price"].transform('mean'))

carprice['company_category'] = carprice["company_average_price"].apply(lambda x : "budget" if x < 10000
                                                                       else ("mid_range" if 10000 <= x < 20000
                                                                       else "luxury"))
plt.figure(figsize=(12, 6))
sns.boxplot(x = 'company_category', y = 'price', data = carprice)
plt.show()

print(carprice.groupby("company_category").company.count())


# In[ ]:


# Drop company and company_average_price after deriving company_category which will be used for modeling
carprice.drop(['company','company_average_price'], axis = 1, inplace = True)


# In[ ]:


# Based upon data visualization of drivewheel, derive a single numeric column of drivewheel_rwd where if 1 then it implies 
# rwd else 4wd or fwd
carprice["drivewheel_rwd"] = np.where(carprice["drivewheel"].str.contains("rwd"), 1, 0)

# Drop drivewheel column
carprice.drop(['drivewheel'], axis = 1, inplace = True)


# In[ ]:


# Based upon data visualization, derive a single numeric column of cylindernumber_four where if 1 then it implies 
# 4 cylinders car else not
carprice["cylindernumber_four"] = np.where(carprice["cylindernumber"].str.contains("four"), 1, 0)

# Drop cylindernumber column
carprice.drop(['cylindernumber'], axis = 1, inplace = True)


# In[ ]:


# Based upon data visualization, derive a single numeric column of enginetype_ohc where if 1 then it implies ohc
# cylinder type else not
carprice["enginetype_ohc"] = np.where(carprice["enginetype"].str.contains("ohc"), 1, 0)

# Drop enginetype column
carprice.drop(['enginetype'], axis = 1, inplace = True)


# In[ ]:


print(carprice.shape)
# Right now we have 23 columns
carprice.head(10)


# ### Dummy Variables

# In[ ]:


# Create dummy variables for the remaining categorical variables
carprice_dummy = carprice.loc[:, ['company_category','doornumber','fueltype','aspiration','carbody','fuelsystem']]
carprice_dummy.head()
dummy = pd.get_dummies(carprice_dummy, drop_first = True)
print(dummy.shape)
dummy.head(10)


# In[ ]:


# Concatenate carprice and dummy data frames
carprice = pd.concat([carprice, dummy], axis = 1)

# Drop the original categorical columns once we have the corresponding derived numerical columns
carprice.drop(['company_category','doornumber','fueltype','aspiration','carbody','fuelsystem'], axis = 1, inplace = True)


# In[ ]:


carprice.head(10)


# In[ ]:


# Now we have 32 columns and all are numeric which can be used for modeling
carprice.shape


# In[ ]:


carprice.info()


# ## Step 5: Splitting the Data into Training and Testing Sets
# 
# First basic step for regression is performing a train-test split.

# In[ ]:


# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)

df_train, df_test = train_test_split(carprice, train_size = 0.7, test_size = 0.3, random_state = 100)


# ### Rescaling the features

# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


# Apply scaler() to all the columns except the dummy variables and target variable
num_vars = ['wheelbase', 'carlength','carwidth','carheight','curbweight', 'enginesize', 'boreratio', 
            'stroke','compressionratio','horsepower','peakrpm','fueleconomy']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

df_train.head()


# ### Dividing into X and Y sets for the model building

# In[ ]:


# Set y_train to the target column
y_train = df_train.pop('price')
# Set X_train to the independent variables
X_train = df_train


# ## Step 6: Building a linear model

# ### RFE

# In[ ]:


# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[ ]:


# Running RFE with the output number of the variable equal to 12
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 12) # running RFE
rfe = rfe.fit(X_train, y_train)


# In[ ]:


(list(zip(X_train.columns,rfe.support_,rfe.ranking_)))


# In[ ]:


col = X_train.columns[rfe.support_]
col


# In[ ]:


# Create function definitions to build model and check VIF
def build_model(X,y):
    X = sm.add_constant(X) #Adding the constant
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


# ### Building model using statsmodel, for the detailed statistics

# In[ ]:


# Creating X_train_rfe dataframe with RFE selected variables
X_train_rfe = X_train[col]

#Build model and check VIF
X_train_lm = build_model(X_train_rfe,y_train)
checkVIF(X_train_lm)


# ### Rebuilding the model after dropping fuelsystem_idi (High P value and High VIF)

# In[ ]:


X_train_lm = X_train_lm.drop(["fuelsystem_idi"], axis = 1)
print(X_train_lm.columns)

#Build model and check VIF
X_train_lm = build_model(X_train_lm,y_train)
checkVIF(X_train_lm)


# ### Rebuilding the model after dropping carlength (High P value and High VIF)

# In[ ]:


X_train_lm = X_train_lm.drop(["carlength"], axis = 1)
print(X_train_lm.columns)

# Build model and check VIF
X_train_lm = build_model(X_train_lm,y_train)
checkVIF(X_train_lm)


# ### Rebuilding the model after dropping fueleconomy (High P value and High VIF)

# In[ ]:


X_train_lm = X_train_lm.drop(["fueleconomy"], axis = 1)
print(X_train_lm.columns)

# Build model and check VIF
X_train_lm = build_model(X_train_lm,y_train)
checkVIF(X_train_lm)


# In[ ]:


X_train_lm = X_train_lm.drop(["fueltype_gas"], axis = 1)
print(X_train_lm.columns)
X_train_lm = build_model(X_train_lm,y_train)
checkVIF(X_train_lm)


# ### Rebuilding the model after dropping curbweight (High VIF, all P values lower than 0.05)

# In[ ]:


X_train_lm = X_train_lm.drop(["curbweight"], axis = 1)
print(X_train_lm.columns)
X_train_lm = build_model(X_train_lm,y_train)
checkVIF(X_train_lm)


# ### Rebuilding the model after dropping peakrpm (High P value)

# In[ ]:


X_train_lm = X_train_lm.drop(["peakrpm"], axis = 1)
print(X_train_lm.columns)
X_train_lm = build_model(X_train_lm,y_train)
checkVIF(X_train_lm)


# ### Rebuilding the model after dropping carbody_sedan (High VIF, all P values lower than 0.05)

# In[ ]:


X_train_lm = X_train_lm.drop(["carbody_sedan"], axis = 1)
print(X_train_lm.columns)
X_train_lm = build_model(X_train_lm,y_train)
checkVIF(X_train_lm)


# ### Rebuilding the model after dropping carbody_wagon (High P value)

# In[ ]:


X_train_lm = X_train_lm.drop(["carbody_wagon"], axis = 1)
print(X_train_lm.columns)
X_train_lm = build_model(X_train_lm,y_train)
checkVIF(X_train_lm)


# ### Now we have a final model with all variables of low P value and low VIF

# ## Step 7: Residual Analysis of the train data
# 
# So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# In[ ]:


lm = sm.OLS(y_train,X_train_lm).fit()

y_train_pred = lm.predict(X_train_lm)

residual = y_train_pred - y_train


# In[ ]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot(residual, bins = 20)
fig.suptitle('Train Data Distribution of Error Terms')                  # Plot heading 
plt.xlabel('Errors')       


# In[ ]:


# Plot the scatter plot of the error terms
fig = plt.figure()
sns.scatterplot(y_train, residual)
fig.suptitle('Train Data Scatter Plot of Error Terms')
plt.ylabel('Errors') 


# ## Step 8: Making Predictions Using the Final Model
# 
# Now that we have fitted the model and checked the normality of error terms, it's time to go ahead and make predictions using the final model.

# In[ ]:


num_vars = ['wheelbase', 'carlength','carwidth','carheight','curbweight', 'enginesize', 'boreratio', 
            'stroke','compressionratio','horsepower','peakrpm','fueleconomy']

df_test[num_vars] = scaler.transform(df_test[num_vars])


# ### Dividing X_test and y_test

# In[ ]:


y_test = df_test.pop('price')
X_test = df_test


# In[ ]:


# Now let's use our model to make predictions.
X_train_lm = X_train_lm.drop(['const'], axis=1)

# Creating X_test_new dataframe by dropping variables from X_test using the final X_train_lm.columns
X_test_new = X_test[X_train_lm.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)


# In[ ]:


# Making predictions
y_test_pred = lm.predict(X_test_new)
print(y_test_pred)


# ## Step 9: Model Evaluation
# 
# Let's now plot the graph for actual versus predicted values.

# ### r2 score of train data

# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_train, y_train_pred)


# ### r2 score of test data

# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_test_pred)


# ### RMSE of train and test data

# In[ ]:


#Returns the mean squared error; we'll take a square root
print(np.sqrt(mean_squared_error(y_train, y_train_pred)))
np.sqrt(mean_squared_error(y_test, y_test_pred))


# ### Plot y_train vs y_train_pred and y_test vs y_test_pred

# In[ ]:


# Plotting y_test and y_test_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_train,y_train_pred)
fig.suptitle('y_train vs y_train_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_train', fontsize=18)                          # X-label
plt.ylabel('y_train_pred', fontsize=16)                     # Y-label

fig = plt.figure()
plt.scatter(y_test,y_test_pred)
fig.suptitle('y_test vs y_test_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_test_pred', fontsize=16)                     # Y-label


# In[ ]:


# Final Model Summary
lm.summary()


# ### We can see that the equation of our best fitted line is:
# 
# price = 0.3957 carwidth + 0.4402 horsepower + 0.2794 company_category_luxury - 0.0414 carbody_hatchback - 0.0824
# 
# F-statistic is also high that is 308 -  overall model fit is significant.

# ### Plot the histogram of the test data error terms

# In[ ]:


residual = y_test - y_test_pred
fig = plt.figure()
sns.distplot(residual, bins = 20)
fig.suptitle('Test Data Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)


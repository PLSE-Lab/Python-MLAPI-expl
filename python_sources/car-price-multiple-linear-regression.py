#!/usr/bin/env python
# coding: utf-8

# **Problem Statement**
# 
# A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.
# 
# They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market.
# 
# 
# **Company wants to know**
# 
# 
# 1.   Which variables are significant in predicting the price of a car
# 2.   How well those variables describe the price of a car
# 
# Based on various market surveys, the consulting firm has gathered a large dataset of different types of cars across the American market.
# 
# **Business Goal**
# 
# You are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market. 
# 

# In[ ]:


# Import Packages
import pandas as pd              # Data Analysis package
import numpy as np               
import matplotlib.pyplot as plt  # Data Virtualization package
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns            # Data Virtualization package
import warnings                  # Supress Warnings
warnings.filterwarnings('ignore')


# **Step 1 : Reading and Understanding the dataset**

# In[ ]:


# Create dataframe by reading the car price dataset
car_details = pd.read_csv('../input/CarPrice_Assignment.csv')


# In[ ]:


# Read the first 5 observations from the dataframe
car_details.head()


# In[ ]:


# Print the shape of the car_details dataframe
car_details.shape


# We have 205 observations in car_details dataframe. Let's describe it.

# In[ ]:


# Describe car_details dataframe
car_details.describe()


# In[ ]:


# Get the detailed information
car_details.info()


# **Step 2 : Data Cleaning and Preparation**

# In[ ]:


# Split the company name from CarName variable
companyname = car_details['CarName'].apply(lambda name : name.split(' ')[0])


# In[ ]:


# Dropping the CarName variable as it is not needed
car_details.drop(columns = {'CarName'}, axis = 1, inplace = True)


# In[ ]:


# Adding the companyname as a new variable
car_details.insert(loc = 3, column = 'companyname', value = companyname)


# In[ ]:


# Get the list of first 5 observations
car_details.head()


# In[ ]:


# Check the unique values in companyname variable
car_details['companyname'].unique()


# From the above result, we have spelling mistakes in the companyname variable. The following list shows the words which has spelling mistakes. Words in bold is the correct spelling.
# 
# 
# 1.   **volkswagen** = vokswagen, vw
# 2.   **mazda** = maxda
# 3.   **porsche** = porcshce
# 4.   **toyota** = toyouta
# 
# 
# 

# In[ ]:


# Convert the data into lowercase
car_details['companyname'] = car_details['companyname'].str.lower()


# In[ ]:


# Define a function to rename the spelling mistakes
def renameCompanyName(error_data, correct_data):
  car_details['companyname'].replace(error_data, correct_data, inplace = True)


# In[ ]:


# Call renameCompanyName function
renameCompanyName('vw','volkswagen')
renameCompanyName('vokswagen','volkswagen')
renameCompanyName('maxda','mazda')
renameCompanyName('porcshce','porsche')
renameCompanyName('toyouta','toyota')


# In[ ]:


# Check the unique values in companyname variable
car_details['companyname'].unique()


# In[ ]:


# Checking for duplicate values in car_details dataframe
car_details.loc[car_details.duplicated()]


# As a result, there is no duplicated values.

# **Step 3 : Data Virtualization**

# In[ ]:


# Let's understand the price of the car
plt.figure(figsize=(15,6)) # Set width and height for the plots

plt.subplot(1,2,1) # Set the rows, columns and their indexing position
sns.distplot(a = car_details.price)

plt.subplot(1,2,2) # Set the rows, columns and their indexing position
sns.boxplot(y = car_details.price)


# From the distribution plot, we can see that there is right skewed and most of the car prices are below 20000.
# 
# In the box plot, there are outliers which shows some of the car prices are significantly higher.

# In[ ]:


# Let's see the mean, median and other percentile for the car prices
car_details.price.describe(percentiles = [0.25, 0.5, 0.75, 0.85, 0.95, 1])


# 
# 
# *   There is a significance difference between mean and median of the car prices.
# *   There is a high variance in the car prices whereas 85 % of the car prices falls within 18500 and the remaining 15% ranges between 18500 and 45400.
# 
# 

# **Virtualize the categorical variables**

# In[ ]:


# Let's virtualize the car companies, car types and fuel types
plt.figure(figsize = (20,6))

plt.subplot(1,3,1)
plt1 = car_details.companyname.value_counts().plot('bar')
plt.title('Companies')
plt1.set(xlabel = 'Car Company', ylabel='Frequency of Car Company')

plt.subplot(1,3,2)
plt1 = car_details.carbody.value_counts().plot('bar')
plt.title('Car Type')
plt1.set(xlabel = 'Car Type', ylabel='Frequency of Car type')

plt.subplot(1,3,3)
plt1 = car_details.fueltype.value_counts().plot('bar')
plt.title('Fuel Type')
plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of fuel type')


# **toyota** is the most favored car company.
# 
# **sedan** is the car type being used by most of the car companies.
# 
# **gas** is used mostly than diesel
# 

# In[ ]:


# Let's virutalize the engine types
plt1 = car_details.enginetype.value_counts().plot('bar')
plt.title('Engine Type')
plt1.set(xlabel = 'Engine Type', ylabel='Frequency of Engine Type')


# **dhc** is the most favored engine type for the car companies

# Let's compare the average prices for car companies and engine type.

# In[ ]:


plt.figure(figsize=(20,6))

plt.subplot(1,3,1)
plt1 = car_details.groupby('companyname')['price'].mean().sort_values(ascending = False).plot('bar')
plt1.set(xlabel = 'Car Company', ylabel = 'Average Price')

plt.subplot(1,3,2)
plt1 = car_details.groupby('enginetype')['price'].mean().sort_values(ascending = False).plot('bar')
plt1.set(xlabel = 'Engine Type', ylabel = 'Average Price')

plt.subplot(1,3,3)
plt1 = car_details.groupby('fueltype')['price'].mean().sort_values(ascending = False).plot('bar')
plt1.set(xlabel = 'Fuel Type', ylabel = 'Average Price')


# **jaguar, buick, porsche, bmw, volvo** are the top 5 car companies having higher average price.
# 
# **dohcv** is the most favored engine type.
# 
# **diesel** has the highest average prices than **gas**. Most of the car companies preferred **gas**
# as their fuel type.

# In[ ]:


plt.figure(figsize=(18,5))

plt.subplot(1,2,1)
plt1 = car_details.enginelocation.value_counts().sort_values(ascending = False).plot('bar')
plt1.set(xlabel = 'Engine Location', ylabel = 'Frequency of Engine Location')

plt.subplot(1,2,2)
plt1 = car_details.groupby('enginelocation')['price'].mean().sort_values(ascending = False).plot('bar')
plt1.set(xlabel = 'Engine Location', ylabel = 'Average Price')


# **front** is the most favored engine location used by most of the car companies.
# **rear** has the highest average price then **front**.

# **Step 4 : Extracting new features**

# In[ ]:


# Calculating the fuel economy by using highwaympg and citympg
car_details['fueleconomy'] = (0.45 * car_details['highwaympg']) + (0.55 * car_details['citympg'])


# In[ ]:


# Calculating the stroke ratio by using boreratio and stroke
car_details['strokeratio'] = car_details['boreratio'] / car_details['stroke']


# In[ ]:


# Categorizing the car companies based on average car price
car_details['price'] = car_details['price'].astype('int')
temp1 = car_details.copy()
temp2 = temp1.groupby('companyname')['price'].mean()
temp1 = temp1.merge(temp2.reset_index(), how = 'left', on = 'companyname')
bins = [0, 10000, 20000, 40000]
cars_bins = ['Low', 'Medium', 'High']
car_details['carsrange'] = pd.cut(temp1['price_y'], bins, right = False, labels = cars_bins)


# In[ ]:


plt.figure(figsize = (15,6))
plt.title('Fuel Economy vs Price')
sns.scatterplot(x = car_details['fueleconomy'], y = car_details['price'])
plt.xlabel('Fuel Economy')
plt.ylabel('Price')


# There is a negative correlation between fuel economy and price and it is significant

# In[ ]:


plt.figure(figsize=(15,6))
sns.heatmap(car_details.corr(), annot = True, cmap='YlGnBu')


# price has high correlation for the list of features
# 
# 
# *   carlength = 0.68
# *   carwidth = 0.76
# *   curbweight = 0.84
# *   enginesize = 0.87
# *   horsepower = 0.81
# 
# 

# **Step 5 : Dummy Variables**

# In[ ]:


categorical_variables = ['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation',
                       'enginetype', 'cylindernumber', 'fuelsystem', 'carsrange']

def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df= pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df
  
for variable in categorical_variables:
   car_details = dummies(variable, car_details)


# In[ ]:


car_details.shape


# In[ ]:


car_details.head()


# In[ ]:


#Removing car_ID and companyname as it is not required for model building
car_details.drop(columns =['car_ID','companyname'], inplace = True)


# In[ ]:


car_details.shape


# **Step 6 : Test Train Data Split and Feature Scaling**

# In[ ]:


# Importing train_test_split to train the data for model building
from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(car_details, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[ ]:


# Use MinMaxScaler to apply scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_vars = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
            'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'fueleconomy', 'strokeratio', 'price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[ ]:


df_train.describe()


# In[ ]:


df_train.head()


# In[ ]:


#Dividing data into X and y variables
y_train = df_train.pop('price')
X_train = df_train


# **Step 7 : Model Building**

# In[ ]:


#RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


lm = LinearRegression()
lm.fit(X_train,y_train)
rfe = RFE(lm, 10)
rfe = rfe.fit(X_train, y_train)


# In[ ]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[ ]:


X_train.columns[rfe.support_]


# Building model using statsmodel, for the detailed statistics

# In[ ]:


X_train_rfe = X_train[X_train.columns[rfe.support_]]
X_train_rfe.head()


# In[ ]:


def buildModel(X,y):
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


# **MODEL I**

# In[ ]:


X_train_new = buildModel(X_train_rfe,y_train)


# **hardtop** has high p-value and dropping it.

# In[ ]:


X_train_new = X_train_new.drop(['hardtop'], axis = 1)


# **MODEL II**

# In[ ]:


X_train_new = buildModel(X_train_new,y_train)


# All the features has p-value less than 0.05. Let's check VIF.

# In[ ]:


vif_df = X_train_new.drop(['const'], axis = 1)


# In[ ]:


checkVIF(vif_df)


# **curbweight** has high VIF and dropping it.

# In[ ]:


X_train_new = X_train_new.drop(['curbweight'], axis = 1)


# **MODEL III**

# In[ ]:


X_train_new = buildModel(X_train_new,y_train)


# **wagon** has p-value and dropping this feature.

# In[ ]:


X_train_new = X_train_new.drop(['wagon'], axis = 1)


# **MODEL IV**

# In[ ]:


X_train_new = buildModel(X_train_new,y_train)


# All the features has good p-value and let's look for VIF.

# In[ ]:


vif_df = X_train_new.drop(['const'], axis = 1)


# In[ ]:


checkVIF(vif_df)


# **horsepower** has high VIF and dropping it.

# In[ ]:


X_train_new = X_train_new.drop(['horsepower'], axis = 1)


# **MODEL V**

# In[ ]:


X_train_new = buildModel(X_train_new,y_train)


# Dropping **hatchback**. Hence it has high p-value.

# In[ ]:


X_train_new = X_train_new.drop(['hatchback'], axis = 1)


# **MODEL VI**

# In[ ]:


X_train_new = buildModel(X_train_new,y_train)


# Dropping **three** and it has high p-value.

# In[ ]:


X_train_new = X_train_new.drop(['three'], axis = 1)


# **MODEL VII**

# In[ ]:


X_train_new = buildModel(X_train_new,y_train)


# **dohcv** has high p-value and hence dropping it.

# In[ ]:


X_train_new = X_train_new.drop(['dohcv'], axis = 1)


# **MODEL VIII**

# In[ ]:


X_train_new = buildModel(X_train_new,y_train)


# All the features has good p-value and check for VIF.

# In[ ]:


vif_df = X_train_new.drop(['const'], axis = 1)


# In[ ]:


checkVIF(vif_df)


# Features **High, carwidth** and **rear** has good p-value and good VIF.
# 
# High represents the car companies which has high average car price.
# 
# carwidth represents the width of the car.
# 
# rear represents the engine location.

# **Residual Analysis of Model**

# In[ ]:


lm = sm.OLS(y_train,X_train_new).fit()
y_train_price = lm.predict(X_train_new)


# In[ ]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  
plt.xlabel('Errors', fontsize = 18) 


# Error terms seem to be normally distributed, so the assumption on the linear modeling seems to be fulfilled.

# **Step 9 : Prediction and Evaluation**

# In[ ]:


# Scaling the test data
num_vars = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
            'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'fueleconomy', 'strokeratio', 'price']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])


# In[ ]:


#Dividing into X and y
y_test = df_test.pop('price')
X_test = df_test


# In[ ]:


# Now let's use our model to make predictions.
X_train_new = X_train_new.drop('const',axis=1)
# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)


# In[ ]:


# Making predictions
y_pred = lm.predict(X_test_new)


# Evaluation of test data

# In[ ]:


from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)


# In[ ]:


print(lm.summary())


# In[ ]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              
plt.xlabel('y_test', fontsize=18)                          
plt.ylabel('y_pred', fontsize=16)  


# For test data and train data, R-squrared values are 0.83 and 0.86 respectively.
# 
# Adjusted R-squared value for train data is 0.86.
# 
# p-values for all the features has less than 0.05 and it is statistically significant.
# 
# Hence, 86% of variance explained. Prob (F-statistic) has 9.90e-60 (approx. 0.0) - Model fit is significant.

# In[ ]:





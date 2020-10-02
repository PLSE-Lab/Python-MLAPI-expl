#!/usr/bin/env python
# coding: utf-8

# # Prediction of car prices
# 
# ### Problem Statement
# A Chinese automobile company __Geely Auto__ aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts. 
# 
# They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. 
# 
# __The company wants to know:__ <br>
# Which variables are significant in predicting the price of a car.<br>
# How well those variables describe the price of a car.<br>
# 
# Based on various market surveys, the consulting firm has gathered a large dataset of different types of cars across the Americal market. <br>
# 
#  
# 
# ### Business Goal 
# 
# You are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. 
# Further, the model will be a good way for management to understand the pricing dynamics of a new market. 
# 

# ## Reading and Understanding Data

# In[ ]:


# supress warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Importing all required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import xticks


# In[ ]:


df = pd.DataFrame(pd.read_csv("../input/CarPrice_Assignment.csv"))


# In[ ]:


df.head()


# ### Data Inspection

# In[ ]:


df.shape
# Data has 26 columns and 205 rows.


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.columns


# ### Data Cleaning

# In[ ]:


#checking duplicates
sum(df.duplicated(subset = 'car_ID')) == 0
# No duplicate values


# In[ ]:


# Checking Null values
df.isnull().sum()*100/df.shape[0]
# There are no NULL values in the dataset, hence it is clean.


# ## __Exploratory Data Analysis ( EDA )__

# ## Univariate Analysis

# ### Price : Target Variable

# In[ ]:


df.price.describe()


# In[ ]:


sns.distplot(df['price'])


# In[ ]:


# Inference
# Mean and median of price are significantly different.
# Large standard deviation indicates that there is considerable variance in the prices of the automobiles.
# Price values are right-skewed, most cars are priced at the lower end (9000) of the price range.


# ### Car ID

# In[ ]:


# car_ID : Unique ID for each observation


# ### Symboling

# In[ ]:


# symboling : Its assigned insurance risk rating
#             A value of +3 indicates that the auto is risky,
#             -3 that it is probably pretty safe.(Categorical)


# In[ ]:


# Let's see the count of automobile in each category and percent share of each category.


# In[ ]:


plt1 = sns.countplot(df['symboling'])
plt1.set(xlabel = 'Symbol', ylabel= 'Count of Cars')
plt.show()
plt.tight_layout()


# In[ ]:


df_sym = pd.DataFrame(df['symboling'].value_counts())
df_sym.plot.pie(subplots=True,labels = df_sym.index.values, autopct='%1.1f%%', figsize = (15,7.5))
# Unsquish the pie.
plt.gca().set_aspect('equal')
plt.show()
plt.tight_layout()


# In[ ]:


# Let's see average price of cars in each symbol category.


# In[ ]:


plt1 = df[['symboling','price']].groupby("symboling").mean().plot(kind='bar',legend = False,)
plt1.set_xlabel("Symbol")
plt1.set_ylabel("Avg Price (Dollars)")
xticks(rotation = 0)
plt.show()


# In[ ]:


# Inference
# More than 50% of cars are with symbol 0 or 1.
# Average price of car is lower for 0,1 & 2 symbol category.


# ### Car Name

# In[ ]:


df.CarName.values[0:10]


# In[ ]:


# It is observed that Car Name consists of two parts 'car company' + ' ' + 'Car Model'
# Let's split out car company to a new column.


# In[ ]:


df['brand'] = df.CarName.str.split(' ').str.get(0).str.upper()


# In[ ]:


len(set(df.brand.values))


# In[ ]:


# Let's see companies and their no of models.


# In[ ]:


fig, ax = plt.subplots(figsize = (15,5))
plt1 = sns.countplot(df['brand'], order=pd.value_counts(df['brand']).index,)
plt1.set(xlabel = 'Brand', ylabel= 'Count of Cars')
xticks(rotation = 90)
plt.show()
plt.tight_layout()


# In[ ]:


# It's noticed that in brand names,
# VOLKSWAGON has three different values as VOLKSWAGEN, VOKSWAGEN and VW
# MAZDA is also spelled as MAXDA
# PORSCHE as PORSCHE and PORCSCHE.
# Let's fix these data issues.
df['brand'] = df['brand'].replace(['VW', 'VOKSWAGEN'], 'VOLKSWAGEN')
df['brand'] = df['brand'].replace(['MAXDA'], 'MAZDA')
df['brand'] = df['brand'].replace(['PORCSHCE'], 'PORSCHE')
df['brand'] = df['brand'].replace(['TOYOUTA'], 'TOYOTA')


# In[ ]:


fig, ax = plt.subplots(figsize = (15,5))
plt1 = sns.countplot(df['brand'], order=pd.value_counts(df['brand']).index,)
plt1.set(xlabel = 'Brand', ylabel= 'Count of Cars')
xticks(rotation = 90)
plt.show()
plt.tight_layout()


# In[ ]:


df.brand.describe()


# In[ ]:


# Inference
# Toyota, a Japanese company has the most no of models.


# In[ ]:


# Let's see average car price of each company.


# In[ ]:



df_comp_avg_price = df[['brand','price']].groupby("brand", as_index = False).mean().rename(columns={'price':'brand_avg_price'})
plt1 = df_comp_avg_price.plot(x = 'brand', kind='bar',legend = False, sort_columns = True, figsize = (15,3))
plt1.set_xlabel("Brand")
plt1.set_ylabel("Avg Price (Dollars)")
xticks(rotation = 90)
plt.show()


# In[ ]:


#df_comp_avg_price


# In[ ]:


df = df.merge(df_comp_avg_price, on = 'brand')


# In[ ]:


df['brand_category'] = df['brand_avg_price'].apply(lambda x : "Budget" if x < 10000 
                                                     else ("Mid_Range" if 10000 <= x < 20000
                                                           else "Luxury"))


# In[ ]:


# Inference:
# Toyota has considerably high no of models in the market.
# Brands can be categorised as Luxury, Mid Ranged, Budget based on their average price.
# Some of the Luxury brans are


# ### Fuel Type

# In[ ]:


# Let's see how price varies with  Fuel Type


# In[ ]:


df_fuel_avg_price = df[['fueltype','price']].groupby("fueltype", as_index = False).mean().rename(columns={'price':'fuel_avg_price'})
plt1 = df_fuel_avg_price.plot(x = 'fueltype', kind='bar',legend = False, sort_columns = True)
plt1.set_xlabel("Fuel Type")
plt1.set_ylabel("Avg Price (Dollars)")
xticks(rotation = 0)
plt.show()


# In[ ]:


# Inference
# Diesel cars are priced more than gas cars.


# ### Aspiration

# In[ ]:


df_aspir_avg_price = df[['aspiration','price']].groupby("aspiration", as_index = False).mean().rename(columns={'price':'aspir_avg_price'})
plt1 = df_aspir_avg_price.plot(x = 'aspiration', kind='bar',legend = False, sort_columns = True)
plt1.set_xlabel("Aspiration")
plt1.set_ylabel("Avg Price (Dollars)")
xticks(rotation = 0)
plt.show()


# In[ ]:


# Inference
# Cars with turbo aspiration engine are priced more than standard ones.


# ### Door Numbers

# In[ ]:


df_door_avg_price = df[['doornumber','price']].groupby("doornumber", as_index = False).mean().rename(columns={'price':'door_avg_price'})
plt1 = df_door_avg_price.plot(x = 'doornumber', kind='bar',legend = False, sort_columns = True)
plt1.set_xlabel("No of Doors")
plt1.set_ylabel("Avg Price (Dollars)")
xticks(rotation = 0)
plt.show()


# In[ ]:


# Inference
# Number of doors doesn't seem to have much effect on price.


# ### Car Body

# In[ ]:


df_body_avg_price = df[['carbody','price']].groupby("carbody", as_index = False).mean().rename(columns={'price':'carbody_avg_price'})
plt1 = df_body_avg_price.plot(x = 'carbody', kind='bar',legend = False, sort_columns = True)
plt1.set_xlabel("Car Body")
plt1.set_ylabel("Avg Price (Dollars)")
xticks(rotation = 0)
plt.show() 


# In[ ]:


# Inference 
# Hardtop and convertible are the most expensive whereas hatchbacks are the cheapest.


# ### Drivewheel

# In[ ]:


df_drivewheel_avg_price = df[['drivewheel','price']].groupby("drivewheel", as_index = False).mean().rename(columns={'price':'drivewheel_avg_price'})
plt1 = df_drivewheel_avg_price.plot(x = 'drivewheel', kind='bar', sort_columns = True,legend = False,)
plt1.set_xlabel("Drive Wheel Type")
plt1.set_ylabel("Avg Price (Dollars)")
xticks(rotation = 0)
plt.show()


# In[ ]:


# Inference
# Cars with Rear wheel drive have a higher price value.


# ### Wheel base

# In[ ]:


plt1 = sns.scatterplot(x = 'wheelbase', y = 'price', data = df)
plt1.set_xlabel('Wheelbase (Inches)')
plt1.set_ylabel('Price of Car (Dollars)')
plt.show()


# In[ ]:


# Most cars has a wheel base around 95 inches.
# Price has a slight positive correlation with wheelbase.


# ## Car Dimensions

# In[ ]:


# Let's see how price varies with Car's length, width,height and weight.


# In[ ]:


fig, axs = plt.subplots(2,2,figsize=(15,10))
plt1 = sns.scatterplot(x = 'carlength', y = 'price', data = df, ax = axs[0,0])
plt1.set_xlabel('Length of Car (Inches)')
plt1.set_ylabel('Price of Car (Dollars)')
plt2 = sns.scatterplot(x = 'carwidth', y = 'price', data = df, ax = axs[0,1])
plt2.set_xlabel('Width of Car (Inches)')
plt2.set_ylabel('Price of Car (Dollars)')
plt3 = sns.scatterplot(x = 'carheight', y = 'price', data = df, ax = axs[1,0])
plt3.set_xlabel('Height of Car (Inches)')
plt3.set_ylabel('Price of Car (Dollars)')
plt3 = sns.scatterplot(x = 'curbweight', y = 'price', data = df, ax = axs[1,1])
plt3.set_xlabel('Weight of Car (Pounds)')
plt3.set_ylabel('Price of Car (Dollars)')
plt.tight_layout()


# In[ ]:


# Inference
# Length width and weight of the car is positively related with the price.
# There is not much of a correlation with Height of the car with price.


# ## Engine Specifications

# ### Engine Type, Cylinder, Fuel System

# In[ ]:


fig, axs = plt.subplots(1,3,figsize=(20,5))
#
df_engine_avg_price = df[['enginetype','price']].groupby("enginetype", as_index = False).mean().rename(columns={'price':'engine_avg_price'})
plt1 = df_engine_avg_price.plot(x = 'enginetype', kind='bar', sort_columns = True, legend = False, ax = axs[0])
plt1.set_xlabel("Engine Type")
plt1.set_ylabel("Avg Price (Dollars)")
xticks(rotation = 0)
#
df_cylindernumber_avg_price = df[['cylindernumber','price']].groupby("cylindernumber", as_index = False).mean().rename(columns={'price':'cylindernumber_avg_price'})
plt1 = df_cylindernumber_avg_price.plot(x = 'cylindernumber', kind='bar', sort_columns = True,legend = False, ax = axs[1])
plt1.set_xlabel("Cylinder Number")
plt1.set_ylabel("Avg Price (Dollars)")
xticks(rotation = 0)
#
df_fuelsystem_avg_price = df[['fuelsystem','price']].groupby("fuelsystem", as_index = False).mean().rename(columns={'price':'fuelsystem_avg_price'})
plt1 = df_fuelsystem_avg_price.plot(x = 'fuelsystem', kind='bar', sort_columns = True,legend = False, ax = axs[2])
plt1.set_xlabel("Fuel System")
plt1.set_ylabel("Avg Price (Dollars)")
xticks(rotation = 0)
plt.show()


# In[ ]:


# Inference
# DOHCV and OHCV engine types are priced high.
# Eight and twelve cylinder cars have higher price.
# IDI and MPFI fuel system have higher price.


# ### Engine Size, Bore Ratio, Stroke, Horsepower & Compression Ratio

# In[ ]:


fig, axs = plt.subplots(3,2,figsize=(20,20))
#
plt1 = sns.scatterplot(x = 'enginesize', y = 'price', data = df, ax = axs[0,0])
plt1.set_xlabel('Size of Engine (Cubic Inches)')
plt1.set_ylabel('Price of Car (Dollars)')
#
plt2 = sns.scatterplot(x = 'boreratio', y = 'price', data = df, ax = axs[0,1])
plt2.set_xlabel('Bore Ratio')
plt2.set_ylabel('Price of Car (Dollars)')
#
plt3 = sns.scatterplot(x = 'stroke', y = 'price', data = df, ax = axs[1,0])
plt3.set_xlabel('Stroke')
plt3.set_ylabel('Price of Car (Dollars)')
#
plt4 = sns.scatterplot(x = 'compressionratio', y = 'price', data = df, ax = axs[1,1])
plt4.set_xlabel('Compression Ratio')
plt4.set_ylabel('Price of Car (Dollars)')
#
plt5 = sns.scatterplot(x = 'horsepower', y = 'price', data = df, ax = axs[2,0])
plt5.set_xlabel('Horsepower')
plt5.set_ylabel('Price of Car (Dollars)')
#
plt5 = sns.scatterplot(x = 'peakrpm', y = 'price', data = df, ax = axs[2,1])
plt5.set_xlabel('Peak RPM')
plt5.set_ylabel('Price of Car (Dollars)')
plt.tight_layout()
plt.show()


# In[ ]:


# Inference
# Size of Engine, bore ratio, and Horsepower has positive correlation with price.


# ## City Mileage & Highway Mileage

# In[ ]:


# A single variable mileage can be calculated taking the weighted average of 55% city and 45% highways.


# In[ ]:


df['mileage'] = df['citympg']*0.55 + df['highwaympg']*0.45


# In[ ]:


# Let's see how price varies with mileage.


# In[ ]:


plt1 = sns.scatterplot(x = 'mileage', y = 'price', data = df)
plt1.set_xlabel('Mileage')
plt1.set_ylabel('Price of Car (Dollars)')
plt.show()


# In[ ]:


# Inference 
# Mileage has a negative correlation with price.


# ## Bivariate Analysis

# ### Brand Category - Mileage

# In[ ]:


# It is expected that luxury brands don't care about mileage. Let's find out how price varies with brand category and mileage.


# In[ ]:


plt1 = sns.scatterplot(x = 'mileage', y = 'price', hue = 'brand_category', data = df)
plt1.set_xlabel('Mileage')
plt1.set_ylabel('Price of Car (Dollars)')
plt.show()


# ### Brand Category - Horsepower

# In[ ]:


plt1 = sns.scatterplot(x = 'horsepower', y = 'price', hue = 'brand_category', data = df)
plt1.set_xlabel('Horsepower')
plt1.set_ylabel('Price of Car (Dollars)')
plt.show()


# ### Mileage - Fuel Type

# In[ ]:


plt1 = sns.scatterplot(x = 'mileage', y = 'price', hue = 'fueltype', data = df)
plt1.set_xlabel('Mileage')
plt1.set_ylabel('Price of Car (Dollars)')
plt.show()


# ### Horsepower - Fuel Type

# In[ ]:


plt1 = sns.scatterplot(x = 'horsepower', y = 'price', hue = 'fueltype', data = df)
plt1.set_xlabel('Horsepower')
plt1.set_ylabel('Price of Car (Dollars)')
plt.show()


# ## Summary Univariate and Bivriate Analysis:
# __From the above Univariate and bivariate analysis we can filter out variables which does not affect price much.__ <br>
# __The most important driver variable for prediction of price are:-__
# 1. Brand Category
# 2. Fuel Type
# 3. Aspiration
# 4. Car Body
# 5. Drive Wheel
# 6. Wheelbase
# 7. Car Length
# 8. Car Width
# 9. Curb weight
# 10. Engine Type
# 11. Cylinder Number
# 12. Engine Size
# 13. Bore Ratio
# 14. Horsepower
# 15. Mileage
# 

# # Linear Regression Model

# In[ ]:


auto = df[['fueltype', 'aspiration', 'carbody', 'drivewheel', 'wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginetype',
       'cylindernumber', 'enginesize',  'boreratio', 'horsepower', 'price', 'brand_category', 'mileage']]


# In[ ]:


auto.head()


# ## Visualising the Data

# ### Visualising Numeric Variables
# 
# Let's make a pairplot of all the numeric variables

# In[ ]:


plt.figure(figsize=(15, 15))
sns.pairplot(auto)
plt.show()


# ### Visualising Categorical Variables
# 
# Let's make a boxplot for categorical variables.

# In[ ]:


plt.figure(figsize=(10, 20))
plt.subplot(4,2,1)
sns.boxplot(x = 'fueltype', y = 'price', data = auto)
plt.subplot(4,2,2)
sns.boxplot(x = 'aspiration', y = 'price', data = auto)
plt.subplot(4,2,3)
sns.boxplot(x = 'carbody', y = 'price', data = auto)
plt.subplot(4,2,4)
sns.boxplot(x = 'drivewheel', y = 'price', data = auto)
plt.subplot(4,2,5)
sns.boxplot(x = 'enginetype', y = 'price', data = auto)
plt.subplot(4,2,6)
sns.boxplot(x = 'brand_category', y = 'price', data = auto)
plt.subplot(4,2,7)
sns.boxplot(x = 'cylindernumber', y = 'price', data = auto)
plt.tight_layout()
plt.show()


# ## Data Preparation

# ### Dummy Variables

# In[ ]:


# Categorical Variables are converted into Neumerical Variables with the help of Dummy Variable 


# In[ ]:


cyl_no = pd.get_dummies(auto['cylindernumber'], drop_first = True)


# In[ ]:


auto = pd.concat([auto, cyl_no], axis = 1)


# In[ ]:


brand_cat = pd.get_dummies(auto['brand_category'], drop_first = True)


# In[ ]:


auto = pd.concat([auto, brand_cat], axis = 1)


# In[ ]:


eng_typ = pd.get_dummies(auto['enginetype'], drop_first = True)


# In[ ]:


auto = pd.concat([auto, eng_typ], axis = 1)


# In[ ]:


drwh = pd.get_dummies(auto['drivewheel'], drop_first = True)


# In[ ]:


auto = pd.concat([auto, drwh], axis = 1)


# In[ ]:


carb = pd.get_dummies(auto['carbody'], drop_first = True)


# In[ ]:


auto = pd.concat([auto, carb], axis = 1)


# In[ ]:


asp = pd.get_dummies(auto['aspiration'], drop_first = True)


# In[ ]:


auto = pd.concat([auto, asp], axis = 1)


# In[ ]:


fuelt = pd.get_dummies(auto['fueltype'], drop_first = True)


# In[ ]:


auto = pd.concat([auto, fuelt], axis = 1)


# In[ ]:


auto.drop(['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginetype', 'cylindernumber','brand_category'], axis = 1, inplace = True)


# ## Model Building

# ## Splitting the Data into Training and Testing sets

# In[ ]:


from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(auto, train_size = 0.7, test_size = 0.3, random_state = 100)


# ### Rescaling the Features

# In[ ]:


# We will use min-max scaling


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


# Apply scaler() to all the columns except the 'dummy' variables
num_vars = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize','boreratio', 'horsepower', 'price','mileage']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# In[ ]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# ### Dividing into X and Y sets for the model building

# In[ ]:


y_train = df_train.pop('price')
X_train = df_train


# ### RFE
# Recursive feature elimination

# In[ ]:


# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[ ]:


# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 10)             # running RFE
rfe = rfe.fit(X_train, y_train)


# In[ ]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[ ]:


col = X_train.columns[rfe.support_]
col


# ### Building model using statsmodel, for the detailed statistics

# In[ ]:


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]


# In[ ]:


# Adding a constant variable 
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)


# In[ ]:


lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model


# In[ ]:


#Let's see the summary of our linear model
print(lm.summary())


# In[ ]:


# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Dropping curbweight as p-value is high.
X_train_new1 = X_train_rfe.drop(["twelve"], axis = 1)


# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new1)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#Let's see the summary of our linear model
print(lm.summary())


# In[ ]:


# This leaves mileage insignificant.


# In[ ]:


# Dropping hardtop as p value is high.
X_train_new2 = X_train_new1.drop(["mileage"], axis = 1)


# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new2)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#Let's see the summary of our linear model
print(lm.summary())


# In[ ]:


# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new2
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Dropping mileage as p-value is high.
X_train_new3 = X_train_new2.drop(["curbweight"], axis = 1)


# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new3)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#Let's see the summary of our linear model
print(lm.summary())


# In[ ]:


# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new3
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Dropping sedan as VIF value is high.
X_train_new4 = X_train_new3.drop(["sedan"], axis = 1)


# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new4)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#Let's see the summary of our linear model
print(lm.summary())


# In[ ]:


# Dropping wagon as p value is high.
X_train_new5 = X_train_new4.drop(["wagon"], axis = 1)


# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new5)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#Let's see the summary of our linear model
print(lm.summary())


# In[ ]:


# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new5
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Dropping dohcv to see if any change in model.
X_train_new6 = X_train_new5.drop(["dohcv"], axis = 1)


# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new6)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#Let's see the summary of our linear model
print(lm.summary())


# ## Residual Analysis of the train data
# So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# In[ ]:


y_train_price = lm.predict(X_train_lm)


# In[ ]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label


# ## Making Predictions
# Applying the scaling on the test sets

# In[ ]:


num_vars = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize','boreratio', 'horsepower', 'price','mileage']

df_test[num_vars] = scaler.transform(df_test[num_vars])


# In[ ]:


y_test = df_test.pop('price')
X_test = df_test


# In[ ]:


# Now let's use our model to make predictions.

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[['carwidth', 'horsepower', 'Luxury', 'hatchback']]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)


# In[ ]:


# Making predictions
y_pred = lm.predict(X_test_new)


# ## Model Evaluation

# In[ ]:


from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)


# In[ ]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label


# #### We can see that the equation of our best fitted line is:
# 
# __price = 0.3957 carwidth + 0.4402 horsepower + 0.2794 luxury -0.0414 hatchback -0.0824__     

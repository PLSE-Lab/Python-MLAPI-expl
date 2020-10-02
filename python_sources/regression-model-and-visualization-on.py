#!/usr/bin/env python
# coding: utf-8

#  # REGRESSION MODEL - PREDICTING PRICE OF PRE OWNED CARS
#  ### @author: Samyak

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### Setting graph size

# In[ ]:


sns.set(rc = {"figure.figsize": (10, 8)})


# ### Reading Data and getting info about data

# In[ ]:


data_price = pd.read_csv("../input/pre-owned-cars/cars_sampled.csv")
cars = data_price.copy()
cars.info()
cars.head()


# In[ ]:


cars.describe()


# In[ ]:


# To set float values upto 3 decimal places
pd.set_option("display.float_format", lambda x: "%.3f" % x)
cars.describe()


# ### Dropping unwanted columns

# In[ ]:


cols = ["name", "dateCrawled", "dateCreated", "postalCode", "lastSeen"]
cars = cars.drop(columns = cols, axis =1)


# ### Removing duplicates from the data

# In[ ]:


cars.drop_duplicates(keep="first", inplace=True)


# In[ ]:


cars.isnull().sum()


# ## Varialbe Analysis
# ### 1.Year Of Registration

# In[ ]:


yearwise = cars["yearOfRegistration"].value_counts().sort_index()
print(cars["yearOfRegistration"].describe())
print("yearOfRegistration Greater than 2018:", sum(cars["yearOfRegistration"] > 2018))
print("yearOfRegistration Lesser than 2018:",  sum(cars["yearOfRegistration"] < 1950))


# In[ ]:


sns.regplot(x="yearOfRegistration", y="price", scatter=True, fit_reg=False, data=cars)


# As we can see that the data is compact at one point i.e. not clearly scattered due many off range values.

# #### Removing Null values

# In[ ]:


cars = cars.dropna(axis = 0)
cars.isnull().sum()


# ### 2. Varialbe price

# In[ ]:


price_count = cars["price"].value_counts().sort_index()
cars["price"].describe()
print("Price Greater than 150000:",sum(cars["price"] > 150000))
print("Price Lesser than 100:",sum(cars["price"] < 100))
sns.distplot(cars["price"])


# Here also we can see that the data is compact at one point i.e. not clearly scattered due many off range values.

# ### 3.Varialbe PowerPS

# In[ ]:


power_count = cars["powerPS"].value_counts().sort_index()
cars["powerPS"].describe()
print("powerPS greater than: ",sum(cars["powerPS"] > 500))
print("powerPS smaller than: ",sum(cars["powerPS"] < 10))


# In[ ]:


fig, (axis1, axis2) = plt.subplots(1,2,figsize=(14,8))

sns.boxplot(cars["powerPS"], ax=axis1)
sns.regplot(x="powerPS", y="price", scatter=True, fit_reg=False, data=cars, ax=axis2)


# Same case with this also,
# we can see that the data is compact at one point i.e. not clearly scattered due many or large number of off range values.

# ## Ranging the data to make it more usefull

# In[ ]:


cars = cars[
    (cars.yearOfRegistration >= 1950)
    & (cars.yearOfRegistration <= 2018)
    & (cars.price <= 150000)
    & (cars.price >= 100)
    & (cars.powerPS <= 500)
    & (cars.powerPS >= 10)
]


# In[ ]:


cars["monthOfRegistration"] /= 12


# #### Adding Age

# In[ ]:


cars["Age"] = (2018-cars["yearOfRegistration"])+cars["monthOfRegistration"]
cars["Age"] = round(cars["Age"], 2)
cars["Age"].describe()


# In[ ]:


#Since age is deployed therefor removing
cols1 = ["yearOfRegistration", "monthOfRegistration"]
cars = cars.drop(columns = cols1, axis = 1)
cars1 = cars.copy()
cars1.head()


# ### Vissualizing Parameters after narrowing the range form dataframe
# ### 4. Variable Age
# 

# In[ ]:


# Age
sns.distplot(cars["Age"])


# ### 5. Variable Price (Output variable)

# In[ ]:


# Price
fig, (axis1, axis2) = plt.subplots(1,2,figsize=(14,8))

sns.distplot(cars["price"], ax=axis1)
sns.boxplot(y=cars["price"], ax=axis2, palette="Set2")


# In[ ]:


fig, (axis1, axis2) = plt.subplots(1,2,figsize=(14,8))

sns.boxplot(y=cars["Age"], ax=axis1)
axis1.set_title("BoxPlot: Price vs Age")

sns.regplot(x="Age", y="price", scatter=True, fit_reg=False, data=cars1, ax=axis2)
axis2.set_title("ScatterPlot: Price vs Age")


# In[ ]:


# PowerPS
fig, (axis1, axis2) = plt.subplots(1,2,figsize=(15,9))

sns.distplot(cars["powerPS"], ax=axis1)
sns.boxplot(y=cars["powerPS"], ax=axis2, palette="Set1")


# In[ ]:


sns.regplot(x="powerPS", y="price", scatter=True, fit_reg=False, data=cars1, scatter_kws={"color": "purple"})


# ## Comparing and Analyzing each and every varaible with price
# ### And removing Insignificant columns

# In[ ]:


#Seller
print(cars["seller"].value_counts())
print(pd.crosstab(cars["seller"], columns="count", normalize=True))

fig, (axis1, axis2) = plt.subplots(1,2,figsize=(14,8))

sns.countplot(x="seller", data=cars1, ax=axis1, palette="Set2")
sns.boxplot(x="seller", y="price", data=cars1, ax=axis2)
#Fewer cars have commercial which is innsignificant
#does not affect price as seen in boxplot
cars1 = cars1.drop(columns=["seller"], axis=1)


# In[ ]:


# Offertype
print(cars["offerType"].value_counts())
print(pd.crosstab(cars["offerType"], columns="count", normalize=True))

fig, (axis1, axis2) = plt.subplots(1,2,figsize=(14,8))

sns.countplot(x="offerType", data=cars1, ax=axis1, palette="Set2")
sns.boxplot(x="offerType", y="price", data=cars1, ax=axis2, palette="Set1")
#does not affect price as seen in boxplot
cars1 = cars1.drop(columns=["offerType"], axis=1)


# In[ ]:


# ABtest
print(cars["abtest"].value_counts())
print(pd.crosstab(cars["abtest"], columns="count", normalize=True))

fig, (axis1, axis2) = plt.subplots(1,2,figsize=(14,8))

sns.countplot(x="abtest", data=cars1, ax=axis1)
sns.boxplot(x="abtest", y="price", data=cars1, ax=axis2)
axis2.set_title("Preice vs ABtest")

#does not affect price as seen in boxplot
cars1 = cars1.drop(columns=["abtest"], axis=1)


# In[ ]:


# VehicleType
print(cars["vehicleType"].value_counts())
print(pd.crosstab(cars["vehicleType"], columns="count", normalize=True))

fig, (axis1, axis2) = plt.subplots(2,1,figsize=(12,16))

sns.countplot(x="vehicleType", data=cars1, ax=axis1)
sns.boxplot(x="vehicleType", y="price", data=cars1, ax=axis2)
#affecting the price


# In[ ]:


#gearbox
print(cars["gearbox"].value_counts())
print(pd.crosstab(cars["gearbox"], columns="count", normalize=True))

fig, (axis1, axis2) = plt.subplots(1,2,figsize=(14,8))

sns.countplot(x="gearbox", data=cars1, ax=axis1)
sns.boxplot(x="gearbox", y="price", data=cars1, ax=axis2)
axis2.set_title("Price vs gearbox")#affecting the price


# In[ ]:


#Model of car
print(cars["model"].value_counts())
print(pd.crosstab(cars["model"], columns="count", normalize=True))
#affecting the price


# In[ ]:


# kilometer
print(cars["kilometer"].value_counts())
print(pd.crosstab(cars["kilometer"], columns="count", normalize=True))

fig, (axis1, axis2) = plt.subplots(2,1,figsize=(12,16))

sns.countplot(x="kilometer", data=cars1, ax=axis1)
sns.boxplot(x="kilometer", y="price", data=cars1, ax=axis2)
#affecting the price


# In[ ]:


#fuelType
print(cars["fuelType"].value_counts())
print(pd.crosstab(cars["fuelType"], columns="count", normalize=True))

fig, (axis1, axis2) = plt.subplots(1,2,figsize=(14,8))

sns.countplot(x="fuelType", data=cars1, ax=axis1)
sns.boxplot(x="fuelType", y="price", data=cars1, ax=axis2)
axis2.set_title("Price vs fuelType")
#affecting the price


# In[ ]:


# Brand
fig, (axis1, axis2) = plt.subplots(2,1,figsize=(12,18))

sns.countplot(y="brand", data=cars1, ax=axis1)
sns.boxplot(x="price", y="brand", data=cars1, ax=axis2)
axis2.set_title("Brand vs Price")
#affecting the price


#  ## CORRELATION

# In[ ]:


cars_select = cars.select_dtypes(exclude=[object])
corelation = cars_select.corr()
round(corelation, 3)
cars_select.corr().loc[:, "price"].abs().sort_values(ascending=False)[1:]
# powerPS have some decent affect on the price i.e 58%


# In[ ]:


sns.heatmap(corelation, vmin = -1, vmax = 1, annot = True, cmap="Blues")
plt.show()


# In[ ]:


cars1.describe()


# In[ ]:


cars2 = cars1.copy()
#converting categorical variable in 0/1 format or dummy format
cars2 = pd.get_dummies(cars1, drop_first=True)
cars2.head()


# ## Building ML Model

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# ### Seprating the values for Linear Regression Model

# In[ ]:


x1 = cars2.drop(["price"], axis = "columns", inplace = False )
y1 = cars2["price"]


# ### Plotting  the variable price

# In[ ]:


prices = pd.DataFrame({"1. Before": y1, "2. After":np.log(y1)})
prices.hist()


# #### Transforming file as a loarithmic value

# In[ ]:


y1 = np.log(y1)
y1


# In[ ]:


#Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.3, random_state=0)


# In[ ]:


#Findin mean value on test data
test_mean = np.mean(y_test)
print(test_mean)
test_mean = np.repeat(test_mean, len(y_test))
print(test_mean)


# Root mean squared value error (RMSE)

# In[ ]:


rmse = np.sqrt(mean_squared_error(y_test, test_mean))
print(rmse)


# ### Initialinzing the Model

# In[ ]:


linear_reg = LinearRegression(fit_intercept = True)


# In[ ]:


model_fit = linear_reg.fit(x_train, y_train)


# In[ ]:


cars_prediction = linear_reg.predict(x_test)
cars_prediction


# In[ ]:


len(x_test) == len(y_test)


# SE and RMSE for predictive values

# In[ ]:


mse1 = mean_squared_error(y_test, cars_prediction)
rmse1 = np.sqrt(mse1)
print(mse1)
print(rmse1)


#  R SQUARED VALUE

# In[ ]:


r2_test = model_fit.score(x_test, y_test)
r2_train = model_fit.score(x_train, y_train)


# In[ ]:


print(r2_test, r2_train)


# Regression Diaagnostic - Residual plot analysis

# In[ ]:


reiduals = y_test - cars_prediction
sns.regplot(x=cars_prediction, y=reiduals, fit_reg=False, scatter=True)
reiduals.describe()


# ## RANDOM FOREST MODEL

# In[ ]:


rf = RandomForestRegressor(n_estimators=100, max_features="auto",
                           max_depth=100, min_samples_split=10,
                           min_samples_leaf=4, random_state=1)


# In[ ]:


model_rf = rf.fit(x_train, y_train)


# In[ ]:


rf_prediction = rf.predict(x_test)


# SE and RMSE for predictive values

# In[ ]:


mse1 = mean_squared_error(y_test, rf_prediction)
rmse1 = np.sqrt(mse1)
print(mse1)
print(rmse1)


#  R SQUARED VALUE

# In[ ]:


r2_test = model_rf.score(x_test, y_test)
r2_train = model_rf.score(x_train, y_train)


# In[ ]:


print(r2_test, r2_train)


# END

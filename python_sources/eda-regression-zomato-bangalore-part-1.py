#!/usr/bin/env python
# coding: utf-8

# This kernel explores the Zomato Bangalore dataset uploaded by Himanshu Poddar and attempts to predict restaurant ratings with regression. This is Part One of my three-part analysis, and uses only the numeric features and categorical features with <100 levels.
# 
# The kernel consists of:
# 
# * Data cleaning (identifying and dropping duplicates, reformatting features)
# * Exploratory Data Analysis and observations
# * Data visualizations
# * Preprocessing and prediction with regression models
# * Model evaluation (MSE, MAPE, R^2) 
# * Results summary

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from statsmodels.regression.linear_model import OLS
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

zomato = pd.read_csv("../input/zomato.csv", na_values = ["-", ""])
# Making a copy of the data to work on
data = zomato.copy()


# In[ ]:


data.shape
# The dataset has 51717 rows and 17 columns


# In[ ]:


data.info()
# Each row represents a restaurant and each column is a feature of the restaurant


# In[ ]:


data.head(3)


# In[ ]:


data.tail(3)


# ### Observations
# - The dataset contains missing values
# - Location information is captured more accurately by "location" than by "listed_in(city)"
# - There is some information overlap between rest_type and listed_in(type) 
# - There's something strange about the menu_item column - let's take a closer look

# In[ ]:


data["menu_item"].value_counts()[:1]


# 39617 entries in the column are empty lists [] 

# In[ ]:


data.isnull().sum()


# It appears both menu_item and dish_liked have over 50% of their data missing.

# In[ ]:


data.address[1]


# Zip codes are not included in the addresses. So they may not be useful for analysis but can be used to identify duplicate data.

# In[ ]:


# Renaming and removing commas in the cost column 
data = data.rename({"approx_cost(for two people)": "cost"}, axis=1)
data["cost"] = data["cost"].replace(",", "", regex = True)


# In[ ]:


# Converting numeric columns to their appropriate dtypes
data[["votes", "cost"]] = data[["votes", "cost"]].apply(pd.to_numeric)


# In[ ]:


# Examining restaurant types in the column "listed_in(type)"
data["listed_in(type)"].value_counts()


# In[ ]:


# Examining the top 20 restaurant types in the column "rest_type"
data["rest_type"].value_counts()[:10]


# There is an information overlap between these two features and we can see that rest_type is more informative. Additionally, the uploader has mentioned there is duplication of data because many restaurants are categorised under multiple types in listed_in(type). We will handle this duplication before proceeding further.

# In[ ]:


# Group and aggregate duplicate restaurants that are listed under multiple types in listed_in(type)
grouped = data.groupby(["name", "address"]).agg({"listed_in(type)" : list})
newdata = pd.merge(grouped, data, on = (["name", "address"]))


# In[ ]:


# Examine the duplicates
newdata.head(3)
# The duplicates can be seen in column "listed_in(type)_x"


# In[ ]:


# Drop rows which have duplicate information in "name", "address" and "listed_in(type)_x"
newdata["listed_in(type)_x"] = newdata["listed_in(type)_x"].astype(str) # converting unhashable list to a hashable type
newdata.drop_duplicates(subset = ["name", "address", "listed_in(type)_x"], inplace = True)


# In[ ]:


newdata.shape


# The reduced dataset has 12499 restaurants - **a substantial reduction from 51717 !**

# In[ ]:


newdata.describe(include = "all")


# ## Observations
# - There are 8792 unique restaurant names, of which **Cafe Coffee Day** has the highest occurrence (54)
# - There are 93 unique locations in Bangalore of which **Whitefield** has the highest number of restaurants (885). Note that this is different from the original dataset's "top" location, BTM, which shows the importance of removing duplicates
# - The most common restaurant type is "Quick Bites" (5024 occurrences)
# - The most common listed type is Delivery (8715) 
# - Biryani is the most popular dish, but we can't be sure about this as dish_liked is missing over half its data
# - There are 2609 unique levels in the cuisines column, this is because restaurants are categorised under many different combinations of cuisines
# - Average cost for two at Bangalore restaurants is Rs 487 and there is very **high variance** (standard deviation Rs 390)
# - Average number of votes per restaurant is 180 and here too there is **high variance**
# - Majority of restaurants allow online ordering but don't allow online table booking
# - NEW is the most common entry in the rating column - this represents unrated new restaurants. We will look at the ratings more closely later
# - Like menu_item, reviews_list also contains many empty lists (2511)
# 

# In[ ]:


# Converting the restaurant names to rownames 
newdata.index = newdata["name"]


# In[ ]:


# Identifying the top 10 cuisines in Bangalore?
pd.DataFrame(newdata.groupby(["cuisines"])["cuisines"].agg(['count']).sort_values("count", ascending = False)).head(10)


# Despite being a southern city, Bangalore has **more North Indian restaurants than South Indian**. Bangaloreans also really seem to love their biryani as no other dish has an entire cuisine category to itself.

# In[ ]:


# Dropping unnecessary columns
newdata.drop(["name", "url", "phone", "listed_in(city)", "listed_in(type)_x", "address", "dish_liked",  "listed_in(type)_y", "menu_item", "cuisines", "reviews_list"], axis = 1, inplace = True)


# In[ ]:


newdata.head(3)


# These are the features we'll use to build our regression model.

# In[ ]:


# Converting restaurant ratings to a numeric variable
newdata["rating"] = newdata["rate"].str[:3] # Extracting the first three characters of each string in "rate"
newdata.drop("rate", axis = 1, inplace = True)


# Instead of representing a rating as 3.5/5, we are now representing it as just 3.5.
# 
# Next we will remove the "NEW" level from ratings as it is not predictable.

# In[ ]:


# Recreating dataset without NEW restaurants
newdata = newdata[newdata.rating != "NEW"] 


# In[ ]:


newdata.isnull().sum()


# We will drop rows that have missing values in the target variable. The remaining missing values in other features will be imputed later.

# In[ ]:


newdata = newdata.dropna(subset = ["rating"])


# Now we can convert ratings to a numeric column.

# In[ ]:


newdata["rating"] = pd.to_numeric(newdata["rating"])


# ## Data visualizations

# In[ ]:


# Plotting the distribution of restaurant ratings
plt.figure(figsize = (10, 5))
plt.hist(newdata.rating, bins = 20, color = "r")
plt.show()


# ### Observations
# - **3.7 is the most common rating**, i.e. most Bangaloreans have above-average dining experiences when they go out. 
# - There are very few ratings between 2 to 2.5 and 4.5 to 5, and hardly any under 2.

# In[ ]:


# Plotting the distribution of locations
plt.figure(figsize = (30, 20))
ax = sns.barplot(data = newdata, x = newdata.location.value_counts().index, y = newdata.location.value_counts())
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right") # to make the labels more readable
plt.show()


# In[ ]:


# Printing restaurant value counts for the top 25 locations
newdata["location"].value_counts()[:25]


# ### Observations
# 
# - After Whitefield, the maximum number of restaurants are in BTM, HSR, Marathahalli and Electronic City
# - Koramangala has been split blockwise or it would be at the top with the others
# 
# 
# Let's see which locations have the **highest rated** restaurants.

# In[ ]:


# Top 5 locations with the highest ratings
(pd.DataFrame(newdata.groupby("location")["rating"].mean())).sort_values("rating", ascending = False).head(5)


# Which locations are the **most expensive** to dine in?

# In[ ]:


# Top 5 most expensive locations (cost = cost for two)
(pd.DataFrame(newdata.groupby("location")["cost"].mean())).sort_values("cost", ascending = False).head(5)


# ### Observations
# - The top two locations with high ratings are also the two most expensive locations (Sankey Road and Lavelle Road)
# - In general we can see that restaurants around the MG Road area are more expensive
# 

# In[ ]:


# Identifying the high rated fancy restaurants on Sankey Road
newdata[(newdata["location"] == "Sankey Road") & (newdata["rating"] >= 4 )]


# Almost all of them are located in one 5 star hotel!
# 
# What about Lavelle Road?

# In[ ]:


newdata[(newdata["location"] == "Lavelle Road") & (newdata["rating"] >= 4 )][:10]


# In[ ]:


# Visualizing the relationship between rating and cost
plt.figure(figsize = (10, 5))
plt.scatter(newdata.rating, newdata.cost)
plt.show()


# ### Observations
# Interestingly, restaurants rated between 4.5 and 5.0 are **cheaper** than those rated between 4.0 and 4.5.
# 
# We have explored and cleaned the dataset and can now apply preprocessing steps that are necessary for model-building.
# 

# ## Data preprocessing

# In[ ]:


# Separating the predictors and target
predictors = newdata.drop("rating", axis = 1)
target = newdata["rating"]


# In[ ]:


# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(predictors, target, random_state = 0, test_size = 0.3)


# In[ ]:


# Preprocessing the predictors
num_cols = ["votes", "cost"]
cat_cols = ["location", "rest_type", "online_order", "book_table"]

num_imputer = SimpleImputer(strategy = "median") 
# Imputing numeric columns with the median (not mean because of the high variance)
num_imputed = num_imputer.fit_transform(X_train[num_cols])
scaler = StandardScaler()
# Scaling the numeric columns to have a mean of 0 and standard deviation of 1
num_preprocessed = pd.DataFrame(scaler.fit_transform(num_imputed), columns = num_cols)

cat_imputer = SimpleImputer(strategy = "most_frequent")
# Imputing categorical columns with the mode
cat_imputed = pd.DataFrame(cat_imputer.fit_transform(X_train[cat_cols]), columns = cat_cols)
# Dummifying the categorical columns
cat_preprocessed = pd.DataFrame(pd.get_dummies(cat_imputed, prefix = cat_cols, drop_first = True))


# In[ ]:


# Joining the numeric and categorical columns and checking their shape
predictors = pd.concat([num_preprocessed, cat_preprocessed], axis=1)


# After building one regression model I had found that one feature ("rest_type_Quick Bites") had a high VIF of 12, indicating multicollinearity. We will drop this feature from our predictors.

# In[ ]:


# Dropping the feature with a high VIF 
predictors.drop("rest_type_Quick Bites", axis = 1, inplace = True)
predictors.shape


# In[ ]:


Y = list(y_train)


# ## Model Building

# In[ ]:


# Building an Ordinary Least Squares regression model
import statsmodels.api as sm
X = sm.add_constant(predictors)
ols = sm.OLS(Y, X).fit()


# In[ ]:


# Predicting on the train data
pred_train = np.around(ols.predict(X), 1)
pred_train[:5] # checking the first 5 predictions


# In[ ]:


# Preprocessing the test data and predicting on it
test_num_imputed = num_imputer.transform(X_test[num_cols])
test_num_preprocessed = pd.DataFrame(scaler.transform(test_num_imputed), columns = num_cols)

test_cat_imputed = pd.DataFrame(cat_imputer.transform(X_test[cat_cols]), columns = cat_cols)
test_cat_preprocessed = pd.DataFrame(pd.get_dummies(test_cat_imputed, prefix = cat_cols))

test_predictors = pd.concat([test_num_preprocessed, test_cat_preprocessed], axis=1)
test_predictors.drop("rest_type_Quick Bites", axis = 1, inplace = True)

# Accounting for missing columns in the test set caused by dummification
missing_cols = set(predictors) - set(test_predictors)
# Adding missing columns to test set with default value equal to 0
for c in missing_cols:
    test_predictors[c] = 0
# Ensuring the order of column in the test set is in the same order than in train set
test_predictors = test_predictors[predictors.columns]

test_X = sm.add_constant(test_predictors)
test_Y = list(y_train)

# Prediction
pred_test = np.around(ols.predict(test_X), 1)
pred_test[:5] # first five rating predictions


# ### Evaluation

# In[ ]:


mean_squared_error(y_train, pred_train)


# In[ ]:


mean_squared_error(y_test, pred_test)


# In[ ]:


# Finding the Mean Absolute Percentage Error
def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape(y_train, pred_train)


# In[ ]:


mape(y_test, pred_test)


# In[ ]:


# Printing the model summary
ols.summary()


# ### Observation
# 
# Adjusted R-squared is very low.
# 
# Let's try Decision Tree-based regression with boosting.

# In[ ]:


# Regression with XGBoost
xgb = XGBRegressor(n_estimators = 100, max_depth = 8, gamma = 0.5, colsample_bytree = 0.8, random_state = 0)
xgb.fit(predictors, y_train)

pred_train = xgb.predict(predictors)
pred_test = xgb.predict(test_predictors)


# In[ ]:


mean_squared_error(y_train, pred_train)


# In[ ]:


mean_squared_error(y_test, pred_test)


# In[ ]:


mape(y_train, pred_train)


# In[ ]:


mape(y_test, pred_test)


# ### Results summary
# 
# OLS linear regression predicted with approximately 8% Mean Absolute Percentage Error on the train and test sets, after checking for multicollinearity and dropping the feature with a high VIF (rest_type_Quick Bites). Adjusted R-squared was low (0.30), indicating that the model does not explain the variance in restaurant ratings, i.e. it is underfitting. 
# 
# We then tried regression with XGBoost and experimented with hyperparameters. Here the train MAPE was 6.5% and test MAPE was 7%. 
# 
# Other types of models and feature transformation may improve performance. In Part Two of my analysis we will transform the ratings into a multi-class categorical feature and see if classification models are able to do better.

# In[ ]:





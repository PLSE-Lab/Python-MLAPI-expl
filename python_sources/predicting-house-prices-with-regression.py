#!/usr/bin/env python
# coding: utf-8

# # Regression Guide to Predict House Price
# Hi! If you read my previous [kernel](https://www.kaggle.com/samsonqian/titanic-guide-with-sklearn-and-eda) of predicting survivors of the Titanic, you saw a thorough guide of EDA, visualization, and classification. This kernel is going to be about another kind of Supervised Machine Learning, regression, where we predict numeric values (house price) instead of categories (survived/didn't survive). This guide will go more in depth of data preprocessing and modelling, because the data set we will work with is much larger and complex than the Titanic data. Let's get started! 
# 
# *Please upvote if this kernel helps you! Feel free to fork this notebook to play with the code yourself.* If you may have any questions about the code, or any step of the process, please comment and I will clear up any confusion.

# My next kernel will be about Deep Learning and Neural Networks, so please follow me and stay tuned for that!

# ## Classification vs. Regression
# The problem we are dealing with in this kernel is predicting house prices from features of the house (ie. how many rooms it has). Because we are trying to predict a continuous value instead of a binary value (ie. Titanic survivors), this is a regression problem. For a guide of classification, please visit [here](https://www.kaggle.com/samsonqian/titanic-guide-with-sklearn-and-eda).

# # Contents
# 1. [Importing Packages](#p1)
# 2. [Loading and Inspecting Data](#p2)
# 3. [Imputing Null Values](#p3)
# 4. [Feature Engineering](#p4)
# 5. [Creating, Training, Evaluating, Validating, and Testing ML Models](#p5)
# 6. [Submission](#p6)

# <a id="p1"></a>
# # 1.  Importing Packages
# We use the same modules as we would use for any problem working with data. We have numpy and pandas to work with numbers and data, and we have seaborn and matplotlib to visualize data. We would also like to filter out unnecessary warnings.

# In[ ]:


import numpy as np 
import pandas as pd 

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# <a id="p2"></a>
# # 2. Loading and Inspecting Data
# With various Pandas functions, we load our training and test data set as well as inspect it to get an idea of the data we're working with. Wow! That is a large data set; just take a look at its shape. We're going to have to understand our data before modelling. 

# In[ ]:


training = pd.read_csv("../input/train.csv")
testing = pd.read_csv("../input/test.csv")


# In[ ]:


training.head()


# In[ ]:


training.describe()


# In[ ]:


training.shape


# > That is a very large data set! We are going to have to do a lot of work to clean it up

# In[ ]:


training.keys()


# Since there are so many columns to work with, let's inspect the correlations to get a better idea of which columns correlate the most with the Sale Price of the house. If there are features that don't do a good job predicting the Sale Price, we can just eliminate them and not use them in our model.

# In[ ]:


correlations = training.corr()
correlations = correlations["SalePrice"].sort_values(ascending=False)
features = correlations.index[1:6]
correlations


# <a id="p3"></a>
# # 3. Imputing Null Values
# With data this large, it is not surprising that there are a lot of missing values in the cells. In order to effectively train our model we build, we must first deal with the missing values. There are missing values for both numerical and categorical data. We will see how to deal with both.
# 
# For numerical imputing, we would typically fill the missing values with a measure like median, mean, or mode. For categorical imputing, I chose to fill the missing values with the most common term that appeared from the entire column. There are other ways to do the imputing though, and I ecnourage you to test out your own creative ways!

# ## Places Where NaN Means Something
# If you look at the data description file provided, you will see that for some categories, NaN actually means something. This means that if a value is NaN, the house might not have that certain attribute, which will affect the price of the house. Therefore, it is better to not drop, but fill in the null cell with a value called "None" which serves as its own category.

# In[ ]:


training_null = pd.isnull(training).sum()
testing_null = pd.isnull(testing).sum()

null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])


# In[ ]:


null_many = null[null.sum(axis=1) > 200]  #a lot of missing values
null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]  #not as much missing values


# In[ ]:


null_many


# In[ ]:


#you can find these features on the description data file provided

null_has_meaning = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]


# In[ ]:


for i in null_has_meaning:
    training[i].fillna("None", inplace=True)
    testing[i].fillna("None", inplace=True)


# ## Imputing "Real" NaN Values
# These are the real NaN values that we have to deal with accordingly because they were not recorded.

# In[ ]:


from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")


# In[ ]:


training_null = pd.isnull(training).sum()
testing_null = pd.isnull(testing).sum()

null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])


# In[ ]:


null_many = null[null.sum(axis=1) > 200]  #a lot of missing values
null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]  #few missing values


# In[ ]:


null_many


# LotFrontage has too many Null values and it is a numerical value so it may be better to just drop it.

# In[ ]:


training.drop("LotFrontage", axis=1, inplace=True)
testing.drop("LotFrontage", axis=1, inplace=True)


# In[ ]:


null_few


# GarageYrBlt, MasVnrArea, and MasVnrType all have a fairly decent amount of missing values. MasVnrType is categorical so we can replace the missing values with "None", as we did before. We can fill the others with median.

# In[ ]:


training["GarageYrBlt"].fillna(training["GarageYrBlt"].median(), inplace=True)
testing["GarageYrBlt"].fillna(testing["GarageYrBlt"].median(), inplace=True)
training["MasVnrArea"].fillna(training["MasVnrArea"].median(), inplace=True)
testing["MasVnrArea"].fillna(testing["MasVnrArea"].median(), inplace=True)
training["MasVnrType"].fillna("None", inplace=True)
testing["MasVnrType"].fillna("None", inplace=True)


# Now, the features with a lot of missing values have been taken care of! Let's move on to the features with fewer missing values.

# In[ ]:


types_train = training.dtypes #type of each feature in data: int, float, object
num_train = types_train[(types_train == int) | (types_train == float)] #numerical values are either type int or float
cat_train = types_train[types_train == object] #categorical values are type object

#we do the same for the test set
types_test = testing.dtypes
num_test = types_test[(types_test == int) | (types_test == float)]
cat_test = types_test[types_test == object]


# **Numerical Imputing**
# 
# We'll impute with median since the distributions are probably very skewed.

# In[ ]:


#we should convert num_train and num_test to a list to make it easier to work with
numerical_values_train = list(num_train.index)
numerical_values_test = list(num_test.index)


# In[ ]:


print(numerical_values_train)


# >These are all the numerical features in our data.

# In[ ]:


fill_num = []

for i in numerical_values_train:
    if i in list(null_few.index):
        fill_num.append(i)


# In[ ]:


print(fill_num)


# >These are the numerical features in the data that have missing values in them. We will impute these features with a for-loop below. 

# In[ ]:


for i in fill_num:
    training[i].fillna(training[i].median(), inplace=True)
    testing[i].fillna(testing[i].median(), inplace=True)


# **Categorical Imputing**
# 
# Since these are categorical values, we can't impute with median or mean. We can, however, use mode. We'll impute with the most common term that appears in the entire list.

# In[ ]:


categorical_values_train = list(cat_train.index)
categorical_values_test = list(cat_test.index)


# In[ ]:


print(categorical_values_train)


# >These are all the categorical features in our data

# In[ ]:


fill_cat = []

for i in categorical_values_train:
    if i in list(null_few.index):
        fill_cat.append(i)


# In[ ]:


print(fill_cat)


# >These are the categorical features in the data that have missing values in them. We'll impute with the most common term below. 

# In[ ]:


def most_common_term(lst):
    lst = list(lst)
    return max(set(lst), key=lst.count)
#most_common_term finds the most common term in a series

most_common = ["Electrical", "Exterior1st", "Exterior2nd", "Functional", "KitchenQual", "MSZoning", "SaleType", "Utilities", "MasVnrType"]

counter = 0
for i in fill_cat:
    most_common[counter] = most_common_term(training[i])
    counter += 1


# In[ ]:


most_common_dictionary = {fill_cat[0]: [most_common[0]], fill_cat[1]: [most_common[1]], fill_cat[2]: [most_common[2]], fill_cat[3]: [most_common[3]],
                          fill_cat[4]: [most_common[4]], fill_cat[5]: [most_common[5]], fill_cat[6]: [most_common[6]], fill_cat[7]: [most_common[7]],
                          fill_cat[8]: [most_common[8]]}
most_common_dictionary


# >This shows the most common term for each of the categorical features that we're working with. We'll replace the null values with these.

# In[ ]:


counter = 0
for i in fill_cat:  
    training[i].fillna(most_common[counter], inplace=True)
    testing[i].fillna(most_common[counter], inplace=True)
    counter += 1


# Good! That should take care of the last couple of missing values. Let's check our work by looking at how many null values remain. If we are successful, the code below should print an empty table.

# In[ ]:


training_null = pd.isnull(training).sum()
testing_null = pd.isnull(testing).sum()

null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])
null[null.sum(axis=1) > 0]


# Yup! An empty table.

# <a id="p4"></a>
# # 4. Feature Engineering
# Ok, now that we have dealt with all the missing values, it looks like it's time for some feature engineering, the second part of our data preprocessing. We need to create feature vectors in order to get the data ready to be fed into our model as training data. This requires us to convert the categorical values into representative numbers.

# First, let's take a look at our target.

# In[ ]:


sns.distplot(training["SalePrice"])


# In[ ]:


sns.distplot(np.log(training["SalePrice"]))


# It appears that the target, SalePrice, is very skewed and a transformation like a logarithm would make it more normally distributed. Machine Learning models tend to work much better with normally distributed targets, rather than greatly skewed targets. By transforming the prices, we can boost model performance.

# In[ ]:


training["TransformedPrice"] = np.log(training["SalePrice"])


# Now, let's take a look at all the categorical features in the data that need to be transformed.

# In[ ]:


categorical_values_train = list(cat_train.index)
categorical_values_test = list(cat_test.index)


# In[ ]:


print(categorical_values_train)


# In[ ]:


for i in categorical_values_train:
    feature_set = set(training[i])
    for j in feature_set:
        feature_list = list(feature_set)
        training.loc[training[i] == j, i] = feature_list.index(j)

for i in categorical_values_test:
    feature_set2 = set(testing[i])
    for j in feature_set2:
        feature_list2 = list(feature_set2)
        testing.loc[testing[i] == j, i] = feature_list2.index(j)


# In[ ]:


training.head()


# In[ ]:


testing.head()


# Great! It seems like we have changed all the categorical strings into a representative number. We are ready to build our models!

# <a id="p5"></a>
# # 5. Creating, Training, Evaluating, Validating, and Testing ML Models
# Now that we've preprocessed and explored our data, we have a much better understanding of the type of data that we're dealing with. Now, we can began to build and test different models for regression to predict the Sale Price of each house. We will import these models, train them, and evaluate them. In classification, we used accuracy as a evaluation metric; in regression, we will use the R^2 score as well as the RMSE to evaluate our model performance. We will also use cross validation to optimize our model hyperparameters.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score, KFold


# **Defining Training/Test Sets**
# 
# We drop the Id and SalePrice columns for the training set since those are not involved in predicting the Sale Price of a house. The SalePrice column will become our training target. Remember how we transformed SalePrice to make the distribution more normal? Well we can apply that here and make TransformedPrice the target instead of SalePrice. This will improve model performance and yield a much smaller RMSE because of the scale.

# In[ ]:


X_train = training.drop(["Id", "SalePrice", "TransformedPrice"], axis=1).values
y_train = training["TransformedPrice"].values
X_test = testing.drop("Id", axis=1).values


# **Splitting into Validation**
# 
# It is always good to split our training data again into validation sets. This will help us evaluate our model performance as well as avoid overfitting our model.

# In[ ]:


from sklearn.model_selection import train_test_split #to create validation data set

X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0) #X_valid and y_valid are the validation sets


# **Linear Regression Model**

# In[ ]:


linreg = LinearRegression()
parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}
grid_linreg = GridSearchCV(linreg, parameters_lin, verbose=1 , scoring = "r2")
grid_linreg.fit(X_training, y_training)

print("Best LinReg Model: " + str(grid_linreg.best_estimator_))
print("Best Score: " + str(grid_linreg.best_score_))


# In[ ]:


linreg = grid_linreg.best_estimator_
linreg.fit(X_training, y_training)
lin_pred = linreg.predict(X_valid)
r2_lin = r2_score(y_valid, lin_pred)
rmse_lin = np.sqrt(mean_squared_error(y_valid, lin_pred))
print("R^2 Score: " + str(r2_lin))
print("RMSE Score: " + str(rmse_lin))


# In[ ]:


scores_lin = cross_val_score(linreg, X_training, y_training, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lin)))


# **Lasso Model**

# In[ ]:


lasso = Lasso()
parameters_lasso = {"fit_intercept" : [True, False], "normalize" : [True, False], "precompute" : [True, False], "copy_X" : [True, False]}
grid_lasso = GridSearchCV(lasso, parameters_lasso, verbose=1, scoring="r2")
grid_lasso.fit(X_training, y_training)

print("Best Lasso Model: " + str(grid_lasso.best_estimator_))
print("Best Score: " + str(grid_lasso.best_score_))


# In[ ]:


lasso = grid_lasso.best_estimator_
lasso.fit(X_training, y_training)
lasso_pred = lasso.predict(X_valid)
r2_lasso = r2_score(y_valid, lasso_pred)
rmse_lasso = np.sqrt(mean_squared_error(y_valid, lasso_pred))
print("R^2 Score: " + str(r2_lasso))
print("RMSE Score: " + str(rmse_lasso))


# In[ ]:


scores_lasso = cross_val_score(lasso, X_training, y_training, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lasso)))


# **Ridge Model**

# In[ ]:


ridge = Ridge()
parameters_ridge = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False], "solver" : ["auto"]}
grid_ridge = GridSearchCV(ridge, parameters_ridge, verbose=1, scoring="r2")
grid_ridge.fit(X_training, y_training)

print("Best Ridge Model: " + str(grid_ridge.best_estimator_))
print("Best Score: " + str(grid_ridge.best_score_))


# In[ ]:


ridge = grid_ridge.best_estimator_
ridge.fit(X_training, y_training)
ridge_pred = ridge.predict(X_valid)
r2_ridge = r2_score(y_valid, ridge_pred)
rmse_ridge = np.sqrt(mean_squared_error(y_valid, ridge_pred))
print("R^2 Score: " + str(r2_ridge))
print("RMSE Score: " + str(rmse_ridge))


# In[ ]:


scores_ridge = cross_val_score(ridge, X_training, y_training, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_ridge)))


# **Decision Tree Regressor Model**

# In[ ]:


dtr = DecisionTreeRegressor()
parameters_dtr = {"criterion" : ["mse", "friedman_mse", "mae"], "splitter" : ["best", "random"], "min_samples_split" : [2, 3, 5, 10], 
                  "max_features" : ["auto", "log2"]}
grid_dtr = GridSearchCV(dtr, parameters_dtr, verbose=1, scoring="r2")
grid_dtr.fit(X_training, y_training)

print("Best DecisionTreeRegressor Model: " + str(grid_dtr.best_estimator_))
print("Best Score: " + str(grid_dtr.best_score_))


# In[ ]:


dtr = grid_dtr.best_estimator_
dtr.fit(X_training, y_training)
dtr_pred = dtr.predict(X_valid)
r2_dtr = r2_score(y_valid, dtr_pred)
rmse_dtr = np.sqrt(mean_squared_error(y_valid, dtr_pred))
print("R^2 Score: " + str(r2_dtr))
print("RMSE Score: " + str(rmse_dtr))


# In[ ]:


scores_dtr = cross_val_score(dtr, X_training, y_training, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_dtr)))


# **Random Forest Regressor**

# In[ ]:


rf = RandomForestRegressor()
paremeters_rf = {"n_estimators" : [5, 10, 15, 20], "criterion" : ["mse" , "mae"], "min_samples_split" : [2, 3, 5, 10], 
                 "max_features" : ["auto", "log2"]}
grid_rf = GridSearchCV(rf, paremeters_rf, verbose=1, scoring="r2")
grid_rf.fit(X_training, y_training)

print("Best RandomForestRegressor Model: " + str(grid_rf.best_estimator_))
print("Best Score: " + str(grid_rf.best_score_))


# In[ ]:


rf = grid_rf.best_estimator_
rf.fit(X_training, y_training)
rf_pred = rf.predict(X_valid)
r2_rf = r2_score(y_valid, rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_valid, rf_pred))
print("R^2 Score: " + str(r2_rf))
print("RMSE Score: " + str(rmse_rf))


# In[ ]:


scores_rf = cross_val_score(rf, X_training, y_training, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_rf)))


# ## Evaluation Our Models
# Now that we've built and trained a couple of different regression models, let's compare all of them and see which of them is the best one we should use to predict on the test set. Let's create a visualize a table to compare their different evaluation metrics.

# In[ ]:


model_performances = pd.DataFrame({
    "Model" : ["Linear Regression", "Ridge", "Lasso", "Decision Tree Regressor", "Random Forest Regressor"],
    "Best Score" : [grid_linreg.best_score_,  grid_ridge.best_score_, grid_lasso.best_score_, grid_dtr.best_score_, grid_rf.best_score_],
    "R Squared" : [str(r2_lin)[0:5], str(r2_ridge)[0:5], str(r2_lasso)[0:5], str(r2_dtr)[0:5], str(r2_rf)[0:5]],
    "RMSE" : [str(rmse_lin)[0:8], str(rmse_ridge)[0:8], str(rmse_lasso)[0:8], str(rmse_dtr)[0:8], str(rmse_rf)[0:8]]
})
model_performances.round(4)

print("Sorted by Best Score:")
model_performances.sort_values(by="Best Score", ascending=False)


# In[ ]:


print("Sorted by R Squared:")
model_performances.sort_values(by="R Squared", ascending=False)


# In[ ]:


print("Sorted by RMSE:")
model_performances.sort_values(by="RMSE", ascending=True)


# The RMSEs are small because of the log transformation we performed. So even a 0.1 RMSE may be significant in this case. 

# I decided to choose Random Forest Regressor to use on the test set because I believe it will perform the best based on the statistics printed above. It was a high R^2 value and a lower RMSE. Feel free to try another model and let me know if you get even better results!

# In[ ]:


rf.fit(X_train, y_train)


# <a id="p6"></a>
# # 6. Submission
# Let's use our optimized model to predict on the Test Set! We will create a dataframe with the predictions and the IDs to submit.

# Remember how we transformed the Sale Price by taking a log of all the prices? Well, now we need to change that back to the original scale. We can do this with numpy's exp function, which will reverse the log. It is the same as raising *e* to the power of the argument (prediction). (e^pred)

# In[ ]:


submission_predictions = np.exp(rf.predict(X_test))


# In[ ]:


submission = pd.DataFrame({
        "Id": testing["Id"],
        "SalePrice": submission_predictions
    })

submission.to_csv("prices.csv", index=False)
print(submission.shape)


# If you made it all the way here, thank you and congratulations on learning Regression! Now you should know both types of Supervised Machine Learning. To view my kernel on Classification, click [here](https://www.kaggle.com/samsonqian/titanic-guide-with-sklearn-and-eda) Please upvote and share if this kernel helped you! Also, please feel free to fork this kernel and play around with the code and models. There is always room for improvement in preprocessing and building models. But most importantly, remember that the best way to learn is to perform these projects hands on. Look forward to my future kernels!

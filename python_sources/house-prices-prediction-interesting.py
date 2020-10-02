#!/usr/bin/env python
# coding: utf-8

# *Hi, Welcome to my First ever documented Kernel, hope you like it.*
# 
# ---
# # Here are the steps I followed,
# 1.   **Importing Packages and Datasets.**
# 2.   **Checking for Null Values.**
# 3.   **Dividing the Dataset into Numeric and Categorical**
# 3.   **Handling Null Values.**
# 4.   **Normalizing Numerical Variables.**
# 5.   **Transforming Categorical Variables.**
# 6.   **Merging the Datasets**
# 7.   **Applying Basic Algorithms**
# 8.   **Applying Advanced Algorithms** - [NOT DONE YET]
# 9.   **Checking for Accuracy**
# 10. **Creating Submission Output Files**
# 
# ---
# Note - We divide the whole process of Coding into **2 - Stages**,
# 1. "**Train Data Handling**" stage.
# 2. "**Test Data Handling**" stage.
# ---
# 
# ---
# # 1 - Importing importent packages and Datasets

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Here is how our data looks.. 

# In[ ]:


train.head()


# Here are the names of the variables we have in **train** dataset

# In[ ]:


#index,variable name
np.array(list(zip(train.Id,train.columns)))


# # 2 - Checking for NULL values 
# 
# What are the variables that have NULL in **train** dataset?

# In[ ]:


np.array(list(zip(train.Id,train.columns[train.isnull().any()].tolist())))


# Variables that contain Null values in **test** data?

# In[ ]:


np.array(list(zip(train.Id,test.columns[test.isnull().any()].tolist())))


# # 3 - Dividing the data into 2 datasets
# to make the handling of NULL values easy, we divide the **train** and **test** into two seperate datasets which have Numeric data in one dataset and categorical in another dataset
# 
# 1.**train**
# 
#     1 - trainNum
#     
#     2 - trainCat
# 2.**test**
#     
#     1 - testNum
#     
#     2 - testCat

# In[ ]:


#train data
trainNum = train.select_dtypes(include=[np.number])
trainCat = train.select_dtypes(include=[object])
#test data
testNum = test.select_dtypes(include=[np.number])
testCat = test.select_dtypes(include=[object])


# What variables in **trainNum** have NULL values in it?

# In[ ]:


trainNum.columns[trainNum.isnull().any()].tolist()


# What variables in **trainCat** have NULL values in it?

# In[ ]:


np.array(list(zip(train.Id,trainCat.columns[trainCat.isnull().any()].tolist())))


# ---
# # "**Train Data Handling**" stage
# ---
# # 4.1 - Handling NULL values in Numerical Variables
# * We go with one of the best way to handle the NULL values in the data, which is to calculate the **mean()** of each variable and **fill** them wherever **NULL** is **True** aka where NULL values are present.

# In[ ]:


trainNum["LotFrontage"].fillna(trainNum["LotFrontage"].mean(), inplace = True)
trainNum["MasVnrArea"].fillna(trainNum["MasVnrArea"].mean(), inplace = True)


# # Simply add **mean()** to null values in a Numeric Dataset?
# Considering the fact that we can fill every numeric variable with mean.. sometimes it defays the purpose of that variable, **how?**
# 
# we simply add mean in null values instead of thinking about that variable in **BUSINESS PROSPECTVIVE** in which case..
# 
# our current 3rd numeric variable.. which is **"GarageYrBlt"** is nothing but * **"**year in which **Garage is built"** *
# 
# think what will happen if we simply add **mean** to that variable in all null values..
# 
# ***Think otherwise!***  i will leave it to you!
# 
# * but what we are doing here is counting all the **unique values** and filling the NA with **most repeated value**

# In[ ]:


trainNum["GarageYrBlt"].fillna(trainNum["GarageYrBlt"].value_counts().idxmax(), inplace = True)


# # 4.2 - Handling NULL values in Categorical variables
# 
# here what we check first for is howmany values in a Categorical variable are Null..?
# 
# in such case.. we are going to delete some variables that have **null values** more then **30%** in them.
# 
# so, we created another dataset as **trainCat1** with all those variables deleted

# In[ ]:


trainCat1 = trainCat.drop(["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"], axis  = 1)


# # Fill NULL in Categorical dataset with the **MODE** of their variable
# in other words what we are doing is, just as we did above for "GarageYrBlt" variable, we are counting for all the unique variables and we are imputing the NULL values with the **unique value** that is **repeated the most**

# In[ ]:


trainCat1["MasVnrType"].fillna(trainCat1["MasVnrType"].value_counts().idxmax(), inplace = True)
trainCat1["BsmtCond"].fillna(trainCat1["BsmtCond"].value_counts().idxmax(), inplace = True)
trainCat1["BsmtExposure"].fillna(trainCat1["BsmtExposure"].value_counts().idxmax(), inplace = True)
trainCat1["BsmtFinType1"].fillna(trainCat1["BsmtFinType1"].value_counts().idxmax(), inplace = True)
trainCat1["BsmtFinType2"].fillna(trainCat1["BsmtFinType2"].value_counts().idxmax(), inplace = True)
trainCat1["BsmtQual"].fillna(trainCat1["BsmtQual"].value_counts().idxmax(), inplace = True)
trainCat1["Electrical"].fillna(trainCat1["Electrical"].value_counts().idxmax(), inplace = True)
trainCat1["GarageCond"].fillna(trainCat1["GarageCond"].value_counts().idxmax(), inplace = True)
trainCat1["GarageFinish"].fillna(trainCat1["GarageFinish"].value_counts().idxmax(), inplace = True)
trainCat1["GarageQual"].fillna(trainCat1["GarageQual"].value_counts().idxmax(), inplace = True)
trainCat1["GarageType"].fillna(trainCat1["GarageType"].value_counts().idxmax(), inplace = True)


# # 5 - Normalizing **Numeric variables**
# we use **LableEncoder** from **PreProcessing** sub - package, which is located in **SKLearn** package to normalize the Numeric variables

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[ ]:


#normalizing numeric data
trainNum["MSSubClass"] = le.fit_transform(trainNum["MSSubClass"].values)
trainNum["OverallQual"] = le.fit_transform(trainNum["OverallQual"].values)
trainNum["OverallCond"] = le.fit_transform(trainNum["OverallCond"].values)
trainNum["YearBuilt"] = le.fit_transform(trainNum["YearBuilt"].values)
trainNum["YearRemodAdd"] = le.fit_transform(trainNum["YearRemodAdd"].values)
trainNum["GarageYrBlt"] = le.fit_transform(trainNum["GarageYrBlt"].values)
trainNum["YrSold"] = le.fit_transform(trainNum["YrSold"].values)


# # 6 - Transform **Non-Numeric** to **Numeric**!
# 
# while used the **LableEncoder** to Normalize the Numeric data, we can also use it to **Transform** the **Non-Numeric data** to **Numeric data**.
# 
# >lets say, we have a demo variable as **Age**
# 
# >Then the Age will be having two type of values as **Male** and **Female**
# 
# >now what this Transformer does is, it transforms 
# 
# >**Male** = **0** and **Female** = **1** and **fills** that variable with **0's** and **1's**
# 
# In a similar fashion we are going to transform variables in our **trainCat** dataset in a "One GO" and create an another dataset as "**trainCatTransformed**"

# In[ ]:


#trainCat data transformed..
trainCatTransformed = trainCat1.apply(le.fit_transform)


# # 7 - Merging the **trainNum** and **trainCatTransformed** into a single dataset
# now that we have handled both **trainNum** and **trainCat** 
# 
# we merge both dataset to make a single dataset which will act as a **Cleaned**, **Normalized** and **Transformed**  final train dataset, we save it as **trainFinal**

# In[ ]:


trainFinal = pd.concat([trainNum, trainCatTransformed], axis = 1)


# And here is the first 5 rows of our **trainFinal** data

# In[ ]:


trainFinal.head()


# # "**Test Data Handling**" stage

# What are the variables that have NULL values in **test** dataset?

# In[ ]:


np.array(list(zip(train.Id,test.columns[test.isnull().any()].tolist())))


# What are the variables that have NULL in **testNum**?

# In[ ]:


np.array(list(zip(train.Id,testNum.columns[testNum.isnull().any()].tolist())))


# variables that have Null in **testCat**

# In[ ]:


np.array(list(zip(train.Id,testCat.columns[testCat.isnull().any()].tolist())))


# in the same approach.. lets handle NULL values in **testNum** with their **mean** values

# In[ ]:


testNum["BsmtFinSF1"].fillna(testNum["BsmtFinSF1"].mean(), inplace = True)
testNum["BsmtFinSF2"].fillna(testNum["BsmtFinSF2"].mean(), inplace = True)
testNum["BsmtUnfSF"].fillna(testNum["BsmtUnfSF"].mean(), inplace = True)
testNum["TotalBsmtSF"].fillna(testNum["TotalBsmtSF"].mean(), inplace = True)
testNum["BsmtFullBath"].fillna(testNum["BsmtFullBath"].mean(), inplace = True)
testNum["BsmtHalfBath"].fillna(testNum["BsmtHalfBath"].mean(), inplace = True)
testNum["GarageCars"].fillna(testNum["GarageCars"].mean(), inplace = True)
testNum["GarageArea"].fillna(testNum["GarageArea"].mean(), inplace = True)
testNum["LotFrontage"].fillna(testNum["LotFrontage"].mean(), inplace = True)
testNum["MasVnrArea"].fillna(testNum["MasVnrArea"].mean(), inplace = True)
#you remember the reason for below one right?
testNum["GarageYrBlt"].fillna(testNum["GarageYrBlt"].value_counts().idxmax(), inplace = True)


# time for **testCat** to get fixed..

# In[ ]:


# we are droping variables with percent of missing values more than 30%
testCat1 = testCat.drop(["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"], axis  = 1)


# In[ ]:


#filling the NULL values with the MODE of their Variables
testCat1["MSZoning"].fillna(testCat1["MSZoning"].value_counts().idxmax(), inplace = True)
testCat1["BsmtCond"].fillna(testCat1["BsmtCond"].value_counts().idxmax(), inplace = True)
testCat1["BsmtExposure"].fillna(testCat1["BsmtExposure"].value_counts().idxmax(), inplace = True)
testCat1["BsmtFinType1"].fillna(testCat1["BsmtFinType1"].value_counts().idxmax(), inplace = True)
testCat1["BsmtFinType2"].fillna(testCat1["BsmtFinType2"].value_counts().idxmax(), inplace = True)
testCat1["BsmtQual"].fillna(testCat1["BsmtQual"].value_counts().idxmax(), inplace = True)
testCat1["Exterior1st"].fillna(testCat1["Exterior1st"].value_counts().idxmax(), inplace = True)
testCat1["GarageCond"].fillna(testCat1["GarageCond"].value_counts().idxmax(), inplace = True)
testCat1["GarageFinish"].fillna(testCat1["GarageFinish"].value_counts().idxmax(), inplace = True)
testCat1["GarageQual"].fillna(testCat1["GarageQual"].value_counts().idxmax(), inplace = True)
testCat1["GarageType"].fillna(testCat1["GarageType"].value_counts().idxmax(), inplace = True)
testCat1["Utilities"].fillna(testCat1["Utilities"].value_counts().idxmax(), inplace = True)
testCat1["Exterior2nd"].fillna(testCat1["Exterior2nd"].value_counts().idxmax(), inplace = True)
testCat1["MasVnrType"].fillna(testCat1["MasVnrType"].value_counts().idxmax(), inplace = True)
testCat1["KitchenQual"].fillna(testCat1["KitchenQual"].value_counts().idxmax(), inplace = True)
testCat1["Functional"].fillna(testCat1["Functional"].value_counts().idxmax(), inplace = True)
testCat1["SaleType"].fillna(testCat1["SaleType"].value_counts().idxmax(), inplace = True)


# Time to Normalize **testNum** dataset!

# In[ ]:


#normalizing numeric data
testNum["MSSubClass"] = le.fit_transform(testNum["MSSubClass"].astype(str))
testNum["OverallQual"] = le.fit_transform(testNum["OverallQual"].astype(str))
testNum["OverallCond"] = le.fit_transform(testNum["OverallCond"].astype(str))
testNum["YearBuilt"] = le.fit_transform(testNum["YearBuilt"].astype(str))
testNum["YearRemodAdd"] = le.fit_transform(testNum["YearRemodAdd"].astype(str))
testNum["GarageYrBlt"] = le.fit_transform(testNum["GarageYrBlt"].astype(str))
testNum["YrSold"] = le.fit_transform(testNum["YrSold"].astype(str))


# In[ ]:


#Transforming categorical data
testCatTransformed = testCat1.apply(le.fit_transform)


# now that we have handled both **testNum** and **testCatTransformed**
# 
# Time to marge them into a final dataset

# In[ ]:


testFinal = pd.concat([testNum, testCatTransformed], axis = 1)


# here are the first 5 rows of our **testFinal** dataset

# In[ ]:


testFinal.head()


# # 8 - Applying Basic Algorithms
# Lets apply some Basic Regression Models like,
# 
# 1 - Linear Regression
# 
# 2 - Lasso Regression
# 
# 3 - Ridge Regression
# 
# we use **linear_model** from **SKLearn** to import all Regression models

# In[ ]:


from sklearn import linear_model


# we train our Model using the **Indipendent training variables** as "**X**" dataset while "**Id**" variable is just an indexing variable that is not related to our Objective
# 
# and, **Dependent training variables** as "**y**" dataset

# In[ ]:


X = trainFinal.drop(["Id","SalePrice"],axis = 1)
y = trainFinal["SalePrice"]


# # 9 - Checking for Accuracy
# 
# **Linear Regression**

# In[ ]:


LR = linear_model.LinearRegression()
LR.fit(X,y)
#Liner Regression Score
LR.score(X,y)


# **Lasso Regression**

# In[ ]:


Lasso = linear_model.Lasso(alpha=0.1)
Lasso.fit(X,y)
#Lasso Regression Score
Lasso.score(X,y)


# **Ridge Regression**

# In[ ]:


Ridge = linear_model.Ridge(0.01)
Ridge.fit(X,y)
#Ridge Regression Score
Ridge.score(X,y)


# # 10 - Applying Advanced Algorithms
# ---
# 
# ---
# 
# # ==PENDING==
# ---
# 
# ---

# # 11.1 - Creating Submission Output Files
# 
# Here, we are simply creating a dataframe with **Id** and **SalePrice** variables as in the Criteria for submission.

# In[ ]:


#Linear Regression Output
submissionLR = pd.DataFrame({
        "Id":test.Id,
        "SalePrice": LR.predict(testFinal.drop("Id",axis=1))
    })

#Lasso Regression Output
submissionLasso = pd.DataFrame({
        "Id":test.Id,
        "SalePrice": Lasso.predict(testFinal.drop("Id",axis=1))
    })

#Ridge Regression Output
submissionRidge = pd.DataFrame({
        "Id":test.Id,
        "SalePrice": Ridge.predict(testFinal.drop("Id",axis=1))
    })


# # 11.2 - Saving outputs to **.csv** files

# In[ ]:


submissionLR.to_csv('salesPrice_LR.csv', index=False)
submissionLasso.to_csv('salesPrice_Lasso.csv', index=False)
submissionRidge.to_csv('salesPrice_Ridge.csv', index=False)


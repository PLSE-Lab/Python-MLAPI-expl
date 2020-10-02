#!/usr/bin/env python
# coding: utf-8

# # Women in Kaggle Workshop for Beginners
# Philadelphia,April 26, 2018
# 
# In this notebook, we will go through a kaggle competition together. This tutorial assumes that you have some basic knowlege of Kaggle and its competitions, statistical models, and Python. By the end of this notebook, you should be able to submit your first prediction to the house price competition. 
# 
# Please folk this notebook so that you can edit the codes to do the exercises. 
# 
# ## What we will do today: 
# 
# 1. Data Loading
# 2. Simple Data Exploration
# 3. Handling Missing Values
# 4. Encoding Categorical Variables
# 5. Modeling & Evaluation
# 6. Create Submission File
# 
# ## Target today:
# 
# Submit your first prediction file to the kaggle competition. 
# 
# ## What we will cover later:
# 
# 1. Indepth Explotary Data Analysis (EDA) and Visualization
# 2. Feature Selection and Importance
# 3. Model Tuning
# 
# ## References
# 
# This notebook is written for a workshop organized by the Women in Kaggle Philly Meetup group.
# 
# This notebook is built upon ideas from:
# https://www.kaggle.com/learn/machine-learning
# https://www.kaggle.com/meikegw/filling-up-missing-values
# 
# 

# ## 1. Data Loading
# 
# The first step of any Kaggle competition is to start a kernel and load data. 
# 
# The kaggle competition files are usually stored online and are accesible through an address like "../input/train.csv" and "../input/test.csv". Note that the file names might change, but you can confirm at the competition page under the "Input" tab. 
# 
# For a simple competition like this one, there will be at least two files, one for training your models (train.csv), and one for testing your predictions and caculating the LeaderBoard score (test.csv). 

# In[ ]:


# Import libraries/packages that we need for the analysis
# Pandas is a python library widely used for manipulating datasets. You'll probably need it for any Kaggle script.
import pandas as pd

# read the training and testing data and store it in a dataframe called "train"
train = pd.read_csv('../input/train.csv') 


# In[ ]:


# Exercise: write a line to read the testing dataset and store it in a dataframe called "test"


# ## 2. Simple Data Exploration
# 
# There are some simple commands for taking a glimpse of the datasets, for example, "head" for showing the first few lines, "columns" for showing column (feature) names, and "describe" for showing basic summary statistics.  You append the command to a dataset with an dot in between. 
# 
# **2.1 Check the first few lines**

# In[ ]:


# show the first few lines of the training datasets
train.head()


# In[ ]:


# Exercise: write your own code to show the first few lines of the testing datasets. 



# Compare the dimensions of the training and testing datasets. What's the difference and why?
# 

# **2.2 Check column names (features)**

# In[ ]:


# Show column names for the training dataset. For a detailed description of the variables, refer to the data tab
train.columns


# In[ ]:


# Exercies: print column names for the testing dataset



# **2.3 Check data type**

# In[ ]:


# List data type for each feature
train.dtypes

# A summary table of data types
train.dtypes.value_counts()

# Show all categorical features
train.columns[train.dtypes=="object"]


# In[ ]:


# Exercise: list data type for each feature in the testing data


# Exercise: show a summary table of data types for the testing data


# Exercise: show all int64 features in the testing data


# **2.3 Basic summary statistics**

# In[ ]:


# Show summary statistics for the training and testing dataset
# Note that it only shows numerical variables
train.describe()

#Show summary statistics for one single column/variables
train.LotArea.describe()

#Show summary statistics for multiple columns
columns = ['OverallQual', 'YearBuilt', 'SalePrice']
train[columns].describe()

#describe() can show summary statistics for categorical variable, but not mixed types of variables
cat=['PavedDrive','Heating']
train[cat].describe()

#frequency table for a categorical variable
train.Heating.value_counts()


# In[ ]:


#Exercise: choose three variables from the training dataset and show the summary statistics


#Exercise: Show the frequency table (value counts) for "PavedDrive" in the training datasets


# 
# 

# ## 3. Dealing with missing values
# 
# Many machine learning models do not deal with missing values and categorical variables automatically, so we need to transform them before getting into the models. Some people refer to these basic steps as "data cleaning". We'll first look at missing values. 
# 
# There are multiple ways of dealing with missing values, but first of all, we need to detect missing values in our data. Below we'll look at the training data first. 
# 
# ###  Detecting missing values
# 
# 

# In[ ]:


# Detecting missing values in the training dataset
train_missing = train.isnull().sum()
train_missing[train_missing>0].sort_values(ascending=False)


# Apparently, there are missing values in many features, which will become a problem when building models. 
# 
# Of course, we can simply drop the columns or rows with missing values: 

# In[ ]:


# Drop rows with missing data. Note that we are assinging the result to a new dataframe because we are not really changing the original dataframe
train_drop_rows=train.dropna(axis=1)

# Drop columns with missing data
train_drop_columns=train.dropna(axis=0)


# However, that is not desirable. If we drop rows, we'll lose cases; if we drop columns, we'll lose features. Most importantly, this may introduce bias into our analysis since data might be missing for a reason. 
# 
# Let's look at those features with missing data closely and figure out the best way
# 
# (in this process, you can also use describe() and value_counts() to facilitate your observation. I omit these steps here. )
# 
# * **PoolQC**: Pool quality. If this is missing, it's most likely that the house doesn't have a pool. We can check against "PoolArea" to see if this is true. 
# * **MiscFeature**: Miscellaneous feature not covered in other categories. Missing values probably mean that there are no special features. Since this is a categorical variable, let's treat these houses with missing values as another category and impute "None". 
# * **Alley**: Type of alley access. Probably no alley access if missing. Impute "None". 
# * **Fence**: Fence quality. Probably no fence if missing. Impute "None". 
# * **FireplaceQu**: Fireplace quality. Probably no fireplace ifmissing. Impute "None". 
# * **LotFrontage**: Linear feet of street connected to property. This shouldn't be zero since every house should be connected to the street. We'll take a simple path and impute the mean. 
# * **Garage variables**: Note that the number of missing cases is 81 for all garage variables. A reasonable guess is that these houses do not have a garage. However, there are other garage variables in the dataset that have no missing values. Why? Let's check and decide later. 
# * **Basement variables**: Same as the garage variables. Check against other basement variables and decide later. 
# * **MasVnrType / MasVnrArea**: Masonry veneer type and area. Both have 8 cases missing, probably the same cases. If we decide to set these 8 cases as no masonry venner, we should impute 0 for the area. 
# * **Electrical**: Just one missing value and it won't affect much. Let's impute the most common value, which is "SBrkr". 
# 

# ** Based on our observation, we should proceed through the following steps (we'll deal with categorical variables first and then numerical ones)**
# 
# 1. Impute "None" for missing values in MiscFeature, Alley, Fence, FireplaceQu
# 2. Check MasVnrType against MasVnrArea. If missing in the same rows, we'll impute MasVnrType as "None" and MasVnrArea as 0. 
# 3. Impute "SBrkr" for Electrical
# 4. Check PoolQC against PoolArea to see if missing values in PoolQC mean no pool.
# 5. Check garage variables 
# 6. Check basement variables
# 7. Impute mean for LotFrontage
# 
# Let's proceed accordingly.
# 
# ** 1)  Impute "None" for missing values in MiscFeature, Alley, Fence, FireplaceQu, MasVnrType**

# In[ ]:


## 1. Impute "None" for missing values in MiscFeature, Alley, Fence, FireplaceQu
missing=["MiscFeature", "Alley", "Fence", "FireplaceQu"]
train[missing]=train[missing].fillna("None")


# **2)  Check MasVnrType against MasVnrArea.**

# In[ ]:


MasVnrType_Missing= train['MasVnrType'].isnull()==True
train[['MasVnrType','MasVnrArea']][MasVnrType_Missing]

## You can also write in one line: 
## train[['MasVnrType, 'MasVnrArea']][train['MasVnrType'].isnull()==True]


# So the 8 missing cases are the same for these two variables. The most reasonable guess is that they don't have MasVnr. 

# In[ ]:


# Check current categories in MasVnrType to make sure
train.MasVnrType.value_counts()

# "None" is the most common vategory. Let's impute missing values in MasVnrType as "None"
train.MasVnrType = train.MasVnrType.fillna("None")


# In[ ]:


# Exercise: impute missing values in MasVnrArea as 0


# **3)  Impute "SBrkr" for Electrical**

# In[ ]:


# Exercies: impute "SBrkr" for missing values in Electrical


# **4) Check PoolQC against PoolArea to see if missing values in PoolQC mean no pool.**

# In[ ]:


# Check if those with missing PoolQC data has a 0 in the PoolArea variable
train['PoolArea'][train['PoolQC'].isnull()==True].describe()


# In[ ]:


# The answer is YES. So missing values in PoolQC should be imputed as "None"
# Exercise: impute PoolQC missing values with "None"


# **5)  Check garage variables **
# 

# In[ ]:


# Let's look at the column names again for all garage variables
train.columns

# Copy them here
garage_cols=['GarageType','GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual','GarageCond']

# Choose those houses with missing values in GarageType, and check on all garage variables
train[garage_cols][train['GarageType'].isnull()==True]


# In[ ]:


# It seems obvious that those houses with missing values in garage variables do not have a garage. 
# Impute "None" for all of them
train[garage_cols] = train[garage_cols].fillna("None")


# **6)  Check basement variables**
# 
# It's the same to check basement variables. Try it by yourself.

# In[ ]:


# Exercise: check basement variables and impute accordingly


# **7)  Impute mean for LotFrontage**
# 
# The last step is to impute mean for LotFrontage. We can easily calculate the mean and replace the missing value with it as we did before using fillna(). However, we want to fill the missing values in the training and testing datasets with the same mean. We'll try something different for this purpose.
# 
# There is a powerful "Imputer" from the sklearn library that can efficiently do this. 

# In[ ]:


# Import the imputer
from sklearn.preprocessing import Imputer

# Define the parameters for the imputer; these are actually the default and you can leave them blank
my_imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

#Imputer does not work with one-dimensional data. Let's put in another numerical variable just to simplify the process
two_cols=["LotFrontage", "GarageArea"]
train[two_cols] = my_imputer.fit_transform(train[two_cols])

# Note that we use "fit_transform", which means we fit the imputer to our training data first to get the imputed value, 
# which is the mean, and then impute this value. To impute the same mean in the testing dataset, we use:
test[two_cols] = my_imputer.transform(test[two_cols])


# **Now let's check missing values again. Remember how to do that?**
# 

# In[ ]:


# Exercise: Check missing values again on the training set


# **Great! We have no missing values in the training set now**
# 
# Now, remember that for most of the variables we only did imputations in the training data.  Let's check how things are going for the testing data.

# In[ ]:


# Exercise: Check missing values on the test set


# There are more variables with missing values in the test dataset. 
# 
# We'll first impute those variables that are also missing in the training dataset using the same rules:
# 

# In[ ]:


# Impute following the same rule as we did for the training data
missing=["MiscFeature", "Alley", "Fence", "FireplaceQu", "MasVnrType", "PoolQC"] 
test[missing]=test[missing].fillna("None")
test.MasVnrArea=test.MasVnrArea.fillna(0)

# Note that GarageCars is a numerical variable, fill in 0 first before filling in None for the other variables
test.GarageCars=test.GarageCars.fillna(0)
test[garage_cols]=test[garage_cols].fillna("None")

# Note that there are numerical basement variables. Impute 0 first before filling in None for categorical variables
basement_cols_num=["BsmtFullBath","BsmtHalfBath","BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF"]
test[basement_cols_num] = test[basement_cols_num].fillna(0)
test[basement_cols]=test[basement_cols].fillna("None")


# In[ ]:


# Look at missing data in the test dataset again
test_missing = test.isnull().sum()
test_missing[test_missing>0].sort_values(ascending=False)


# In[ ]:


# For the remaining variables, since there are very few missing and they are all categorical, 
# we'll impute the most frequent category for all of them
remain = ["MSZoning", "Functional", "Utilities", "SaleType", "KitchenQual", "Exterior2nd", "Exterior1st"]
for feature in remain:
    test[feature] = test[feature].fillna(test[feature].value_counts().index[0])


# In[ ]:


# Exercise: Look at missing data in the test dataset again



# Hooray! Now we have no missing data anywhere. 
# 
# Note that a common practice in actual kaggle competition is to merge the training and testing dataset before dealing with missing data and performing any feature engineering in order to avoid duplicate codes. There are some concerns about [data leakage ](https://www.kaggle.com/dansbecker/data-leakage), but it depends on your data structure, the nature of your missing data, and your imputation strategy. In this house price dataset, when using our simple imputation strategies,  it should be fine. 
# 
# Next we'll look at categorical variables.
# 
# ## 4. Encoding Categorical Variables
# 
# Many machine learning models do not deal with categorical variables and we need to transform them into numerical ones. There are many ways to do this, but the most popular and easy way is to split categorical variables into several dummy dichotomous variables, and assigning 1 and 0 to them. This is called "One-Hot-Encoding". 
# 
# There is an one liner for applying one-hot-encoding to all categorical variables in the dataset using "pd.get_dummies": 
# 

# In[ ]:


# First check datatypes of the training dataset:
train.columns[train.dtypes=="object"]
test.columns[test.dtypes=="object"]

# One-liner for one-hot-encoding
one_hot_encoded_train = pd.get_dummies(train)
one_hot_encoded_test = pd.get_dummies(test)

# To make sure traing and testing datasets are encoded in the same way: 
final_train, final_test = one_hot_encoded_train.align(one_hot_encoded_test,
                                                                    join='left', 
                                                                    axis=1)

# Now check on categorical variables in the encoded training data:
final_train.columns[final_train.dtypes=="object"]


# In[ ]:


# Exercise: check on categorical variables in the encoded testing data



# Now we are ready to build some models.
# 
# ## 5. Modeling & Evaluation
# 
# Our target is to predict the sale price of a house given all the features.  For this, we'll train the model using our training data, and applying the trained model to the test data.
# 
# One common problem is overfitting: if we use all of the training data, it's possible that our model would fit the training dataset perfectly, but may not work so well when applying to a new dataset. To better evaluate our model, one popular strategy is to split the training data into two parts: training and validation. This means that we'll use only part of the data to train our models, and then use the validation data to evaluate and pick the best model. 
# 
# Today we'll look at three models: Linear Regression, Random Forest, and X[GBoost](https://www.kaggle.com/dansbecker/learning-to-use-xgboost). We're not going into the statistical details. 
# 
# ** First, let's split our training dataset into two parts: training and validation.**
# 
# ### 5.1 Split the dataset

# In[ ]:


# import the libraries we need 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Assign the target and predicor variables. It's conventional to call the target y and predictors X. 
# Our target variable is SalePrice, and we'll use all other variables as predictors. 
y = final_train.SalePrice
X = final_train.drop(['SalePrice',"Id"], axis=1)

# Devide the training data into training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(X,  #predictor variables
                                                  y,  #target variable
                                                  train_size=0.8, #percentage of the training data
                                                  test_size=0.2,  #precentage of the validation data
                                                  random_state=42) #this is a random seed. Use the same number and you'll get the same result.


# **Now we're ready to train our models. **
# 
# ### 5.2 Linear Regression 
# 
# We'll firstly train a simple linear regression model and get an evaluation score.  There are many ways to evaluate a model, and different competitions use different evaluation methods. Here we use a very simple one: the mean absolute error (MAE)
# 
# MAE means taking the absolute distance between each prediction and the target (this is the "error"), and then average all errors. The smaller the MAE, the better the model. **

# In[ ]:


# Define our model
model_tree = LinearRegression()

# Use the predictors and target in the training data to train the model
model_tree.fit(X_train, y_train)

# Use the predictors in the validation data to generate predictions
prediction_val = model_tree.predict(X_val)

# Compare the actual y in the validation data with our predictions and generate MAE score
mean_absolute_error(y_val, prediction_val)


# ### 5.3 Random Forest Model
# 
# Next  we'll train a Random Forest Model. 
# Simply replace LinearRegression() with RandomForestRegressor(), and give your model a different name, like model_random_forest
# 

# In[ ]:


## Exercise: write your own code for a Random Forest Model


# You'll notice that everytime your run the random forest model, the resulting MAE scores are different.
# This is because the model is random. You can use a seed to ensure the same result, eg. RandomForestRegressor(random_state=5)
# Anyway your result should be significantly better than the simple linear regression model. 


# ### 5.4 XGBoost Model
# 

# In[ ]:


## Exercise: replace RandomForestRegressor with XGBRegressor, rename your model, and get the MAE score



# ** To evaluate the three models, we can write a simple function to get the MAE score without duplicate codes:**

# In[ ]:


def model_score(model, X_train, X_val, y_train, y_val):
    model = model
    model.fit(X_train, y_train)
    prediction_val = model.predict(X_val)
    return mean_absolute_error(y_val, prediction_val)


# In[ ]:


models = [LinearRegression(), RandomForestRegressor(random_state=5), XGBRegressor()]
for model in models:
    mae = model_score(model, X_train, X_val, y_train, y_val)
    print(mae)


# ** Apparently, the XGB model is significantly better. We can get better results from model tuning, but today we're happy enough with this.  we'll submit our prediction with XGBoost.**

# In[ ]:


# Train the XGB model if you haven't done so in the exercise
model_XGB = XGBRegressor()
model_XGB.fit(X_train, y_train)

# In our example, the purpose of validation is to avoid overfitting during evaluation. 
# Now we have picked the best model, it's OK to fit it on the full dataset for training. 
# More data is usually better. Let's re-train the model with full training dataset. 
X_retrain = final_train.drop(["SalePrice", "Id"], axis=1)
y_retrain = final_train.SalePrice
model_XGB_retrain = XGBRegressor()
model_XGB_retrain.fit(X_retrain, y_retrain)


# ## 6. Create Submission File 

# In[ ]:


# Get predictors in the test file
X_test = final_test.drop(['SalePrice', 'Id'], axis=1)

# Use your trained model to predict the SalePrice in the test dataset
# We'll test both the one trained with 80% data and the one retrained with full training data
prediction_test = model_XGB.predict(X_test)
prediction_retrain = model_XGB_retrain.predict(X_test)

# Create a dataframe with two columns: ID from the test file, and prediction from the model
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': prediction_test})

# Write this dataframe into a CSV file. You could use any filename. 
submission.to_csv('my_submission.csv', index=False)


# In[ ]:


# Exercise: create a dataframe for the prediction from the retrained model


# Exercise: write the dataframe from the retrained model into a csv file. Name it submission_retrain.csv. 


# Now go back to this kernel page, you'll find a new tab named "Output", and the submission file is right there.
# Click on "Submit to Competition" and your first submission to Kaggle competition is done! 
# 

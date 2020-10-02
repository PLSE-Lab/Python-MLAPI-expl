#!/usr/bin/env python
# coding: utf-8

# # Walkthrough of Preprocessing || EDA || Modelling Results
# 
# ### Author : [Joshua Yeo][1]
# 
# ### Updated : 1 May 2020
# 
# Greetings! Thank you for taking the time to view this notebook. If you use any parts of this notebook, it would be greatly appreciated if you would give credit by linking back to this notebook! (: 
# 
# [1]: https://github.com/Joshuayeo95
# 
# ------

# ## Notebook Overview
# 
# The goal of this notebook is to provide a __comprehensive walkthrough__ of the different stages of solving a machine learning regression problem. 
# 
# Throughout this notebook, i detail my thought process as i work through the problem and provide insights wherever i can. Please note that this is a pretty lengthy notebook, and i have created a table of contents so it is easier to revisit the notebook.
# 
# I hope that this notebook serves as a reference for newer Kagglers on how to approach a Regression Problem. For experienced Kagglers that come across this notebook, i welcome any feedback/criticism of my methodologies and ways to improve. 
# 
# *If you enjoyed reading this notebook, an upvote would be appreciated! Thank you and have a great day! (:*

# ## Table of Contents
# 
# #### [1. Understanding the Problem](#problem)
# 
# 
# #### [2. Initial Data Preprocessing](#preprocessing)
# * [Getting a Feel of our Data](#initial_analysis)
# * [Handling Missing Data](#missing_data)
# * [Correcting Data Types](#data_types)
#     
#     
# #### [3. Exploratory Data Analysis and Further Processing](#eda)
# * [Target Variable Distribution](#target)
# * [Numerical Variables - Univariate Analysis](#num_var_univariate)
# * [Numerical Variables - Bivariate Analysis](#num_var_bivariate)
# * [Categorical Variables - Univariate Analysis](#cat_var_univariate)
# * [Categorical Variables - Bivariate Analysis](#cat_var_bivariate)
# 
# 
# #### [4. Modelling](#modelling)
# * [Preparing our Data](#preparing_data)
# * [Linear Models - Linear Regression](#linear)
# * [Linear Models - Ridge Regression](#ridge)
# * [Linear Models - Lasso Regression](#lasso)
# * [Linear Regression after Feature Selection](#linear_small)
# * [Ensemble - Random Forest Regression](#rforest)
# * Ensemble - Gradient Boosting Regression (work in progress)
# * Potential Feature Engineering (work in progress)
# 
# #### 5. Evaluating Model Performances (work in progress)

# # 1. Understanding the Problem <a id='problem'></a>

# The goal of this competition is to __predict__ housing prices. Given that our target varible (prices) are continuous in nature, we are trying to solve a __regression problem__.
# 
# Given that we are focused on obtaining the __most accurate predictions__, we just need to select the model that performs the best, given an evaluation metric. 
# 
# For this competition, the evaluation metric chosen is the __Root Mean Squared Error (RMSE)__, and we will be trying to find the model that gives us the __lowest RMSE__.
# 
# The models that we will be considering in this notebook include:
# 
# * __Linear Models__ : Linear Regression / Ridge Regression / Lasso Regression
# 
# * __Ensemble Models__ : Random Forest / Gradient Boosting 
# 
# Within linear models, we can compare the effects of different types of __Regularisation__ on the performance. As for the ensemble methods, we can compare performances between __Bootstrap Aggregation__ (Random Forest) and __Boosting__. Given that the dataset is not very large, we did not choose to implement deep learning models.
# 
# Note : The correct evaluation metric to use varies from problem to problem. This [article][1] compares the differences between the two most common metrics for regression problems, the __Mean Absolute Error (MAE)__ and the __RMSE__.
# 
# [1]: https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d

# # 2. Preprocessing the Data <a id='preprocessing'></a>
# 
# After we have defined the problem statement and know how to evaluate our model performance, we need to preprocess the data before the modelling stage.
# 
# There are often mistakes or issues with our data that we need to address before we can actually fit our model using them. 
# 
# Common things to look out for in our data:
# * Typos
# * Outliers
# * Missing Values
# * Incorrect Data Types
# * Whitespace in Column Headers

# ### Importing Libraries (expand to view)

# In[ ]:


# Standard Tools
from scipy import stats
import pandas as pd
import numpy as np
import pickle
import os

# Visualisation Tools
import matplotlib.pyplot as plt 
import seaborn as sns

# Modelling Tools
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import statsmodels as sm

# Misc
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
plt.style.use('fivethirtyeight')

# Setting the random state
np.random.seed(8888)
SEED = 8888

# # File names 
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# ### 2.1 Loading the Data and Initial Analysis <a id='initial_analysis'></a>
# 
# Firstly, we import our data and get an overall feel of what it contains.
# 
# We usually want to take a quick scan at the first few rows of our data and look our for any potential mistakes mentioned previously.
# 
# Furthermore, i highly recommend looking through any __README__ or __DESCRIPTION__ files that come with the dataset. Often, they provide information on how categorical variables were encoded and may also provide explaination for encoding observations as missing.

# In[ ]:


# Loading our data
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

print(f'The dataset has {df.shape[0]} rows and {df.shape[1]} columns.')


# In[ ]:


# Getting a feel for how our data looks like
df.head()


# In[ ]:


# Dropping column Id as we will use the dataframe indexing
df = df.drop('Id', axis=1)


# Next, we can scan the data types of our variables and check back with our data and see if they are correct. 
# 
# Determining whether they are correct or not may require some domain knowledge but most of the time can be identified with a bit of common sense. (: 
# 
# i.e Prices are numerical but they are classified as 'object', which is definitely a mistake.
# 
# Note: Expand the cell below to view the data types.

# In[ ]:


# Quick scan for missing values and wrong datatypes
df.info()


# From our initial analysis, we observe that we have variables with missing data and some with potentially wrong datatypes (ordinal variables). We will address these issues below.

# We can also check for skewness or potential outliers in the numerical variables by using the .describe() method, which provides summary statistics for our numerical variables.

# In[ ]:


df.describe()


# ### 2.2 Handling Missing Data <a id='missing_data'></a>
# 
# __Missing data__ is common in most data sets. While there are algorithms that can handle missing data (i.e. xgboost), not all of them can.
# 
# As we will be comparing different models, it is good practice to handle these missing values.
# 
# There are 2 options when dealing with missing data:
# 1. __Imputation__
# 2. __Dropping Column / Observations__
# 
# For imputation, the method varies depending on the type of variable:
# * __Numerical__ - Imputation with Mean / Median / K-Nearest Neighbours
# * __Categorical__ - Imputation with Mode / K-Nearest Neighbours / Create New Category
# 
# Note: For __*time series*__ data, we can impute numerical variables using __Linear Interpolation / Back-filling / Forward-filling__.
# 
# With regards to dropping variables, i do not know of any 'rules' but the __rules of thumb__ that i usually employ are:
# * Drop variables with more than 80-90% missing data
# * If fraction of missing values is negligible/small compared to number of observations, drop rows that are missing data.
# 
# If anyone has any suggestions/advice on when to drop data, please do let me know! (:

# In[ ]:


# Helper function to help check for missing data
def variable_missing_percentage(df, save_results=False):
    '''
    Function that shows variables that have missing values and the percentage of total observations that are missing.
    
    Arguments:
        df : Pandas DataFrame
        save_results : bool, default is False
            Set as True to save the Series with the missing percentages.
    
    Returns:
        percentage_missing : Pandas Series
            Series with variables and their respective missing percentages.
    '''
    percentage_missing = df.isnull().mean().sort_values(ascending=False) * 100
    percentage_missing = percentage_missing.loc[percentage_missing > 0].round(2)
    missing_variables = len(percentage_missing)
    
    if len(percentage_missing) > 0:
        print(f'There are a total of {missing_variables} variables with missing values. Percentage of total missing:')
        print()
        print(percentage_missing)
    
    else:
        print('The dataframe has no missing values in any column.')
    
    if save_results:
        return percentage_missing


# In[ ]:


variable_missing_percentage(df)


# After looking into the description.txt file, we find that the cause of most of the missing variables are due to the property not having the associated feature. Therefore, we will be dropping the variables that have high percentage of missing data ( > 80%) as they would not generalise well to new data.

# In[ ]:


def drop_missing_variables(df, threshold, verbose=True):
    '''Function that removes variables that have missing percentages above a threshold.
    
    Arguments:
        df : Pandas DataFrame
        threshold : float
            Threshold missing percentage value in decimals.
        verbose : bool, default is True
            Prints the variables that were removed.
            
    Returns:
        df : Pandas DataFrame with variables removed
    '''
    shape_prior = df.shape
    vars_to_remove = df.columns[df.isnull().mean() > threshold].to_list()
    df = df.drop(vars_to_remove, axis=1)
    shape_post = df.shape
    
    print(f'The original DataFrame had {shape_prior[1]} variables.')
    print(f'The returned DataFrame has {shape_post[1]} variables.')
    
    if verbose:
        print()
        print('The following variables were removed:')
        print(vars_to_remove)
        
    return df


# In[ ]:


df = drop_missing_variables(df, 0.8)


# In[ ]:


# Dropping features that are related to the ones we just removed
df = df.drop(['MiscVal', 'PoolArea'], axis=1)


# For the following categorical variables, we will be replacing the missing values as a new category that indicates the property does not have that feature.

# In[ ]:


df.FireplaceQu = df.FireplaceQu.fillna('NoFirePlace')

basement_variables = ['BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
df[basement_variables] = df[basement_variables].fillna('NoBasement')

garage_variables = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
df[garage_variables] = df[garage_variables].fillna('NoGarage')


# In[ ]:


# Checking which variables are still missing
variable_missing_percentage(df)


# We hypothesize that the Garage Year Built variable is highly correlated to the variable Year House Built.

# In[ ]:


df.YearBuilt.corr(df.GarageYrBlt).round(2)


# As expected, the two variables are highly correlated and we will be dropping the variable 'GarageYrBlt' as most of its information is captured in the variable 'YearBuilt'.

# In[ ]:


df = df.drop('GarageYrBlt', axis=1)


# For the variables that have low missing percentages (<1%), we will just drop the missing observations.

# In[ ]:


df = df.dropna(how='any', subset=['MasVnrType', 'MasVnrArea', 'Electrical'])


# For the variable 'LotFrontage', we chose to use a K-Nearest Neighbours imputation of the missing value. The rationale is that houses come in different sizes which would definitely influence the Lot Frontage of the property. Therefore, we believe that similar sized houses would have similar Lot Frontage, therefore, being a better imputation than just taking the median.

# In[ ]:


# Installing missingpy package, remember to turn internet on in the settings! 
get_ipython().system('pip install missingpy')

from missingpy import KNNImputer
knn_imputer = KNNImputer(n_neighbors=5, weights='distance', metric='masked_euclidean')

df.LotFrontage = knn_imputer.fit_transform(np.array(df.LotFrontage).reshape(-1,1))


# In[ ]:


# Making sure we have tackled all missing variables
variable_missing_percentage(df)


# Now that we have sucessfully dealt with the missing values, we need to check that they have correct data types.

# ### 2.3 Correcting Data Types <a id='data_types'></a>
# 
# In our initial analysis, we noted some variables being stored as numerical when they should be categorical, as well as some categorical variables that can be stored as ordinal variables due to the inate hierarchy in the variable. Therefore, this section seeks to address these problems.

# In[ ]:


# Changing numeric variables to categorical
df = df.replace({
    'MSSubClass' : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 50 : "SC50", 60 : "SC60",
                    70 : "SC70", 75 : "SC75", 80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                    150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
    'MoSold' : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
})


# In[ ]:


# Converting categorical variables to an interval scale as they are ordinal in nature.
df = df.replace({
    'ExterQual' : {'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'ExterCond' : {'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'BsmtQual' : {'NoBasement' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'BsmtCond' : {'NoBasement' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'BsmtExposure' : {'NoBasement' : 0, 'No' : 1, 'Mn' : 2, 'Av' : 3, 'Gd' : 4},
    'BsmtFinType1' : {'NoBasement' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6},
    'BsmtFinType2' : {'NoBasement' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6},
    'HeatingQC' : {'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'KitchenQual' : {'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'FireplaceQu' : {'NoFirePlace' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'GarageFinish' : {'NoGarage' : 0, 'Unf' : 1, 'RFn' : 2, 'Fin' : 3},
    'GarageQual' : {'NoGarage' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'GarageCond' : {'NoGarage' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
})

# Creating a list of our ordinal variables
ordinal_vars = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2','HeatingQC', 'KitchenQual',
    'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond'
]


# In[ ]:


# Checking our datatypes once again
df.info()


# In[ ]:


# Changing features to their correct data types
df.BsmtCond = df.BsmtCond.astype('int64')
df.BsmtFinType2 = df.BsmtFinType2.astype('int64')
df.FireplaceQu = df.FireplaceQu.astype('int64')


# Now that we are done with preprocessing our data, we can move on to exploratory data analysis of our data!

# # 3. Exploratory Data Analysis <a id='eda'></a>
# 
# In this section, we will be taking a closer look at our data and using visualisations to better understand our data. As the type of analysis differs for numerical and categorical data, we will analyse them separately in their own sections.
# 
# For numerical variables, __univariate analysis__ will consist of checking the __variable's distribution__ (focusing on skewness) and performing __transformations__ if necessary. We will then analyse how each variable relates to our target variable, SalesPrice. 
# 
# For __categorical variables__, we will look into the __variable's distribution across different categories__, before seeing the effect that each category has on our target variable. 
# 
# *Note: I have previously defined some helper functions that i use across various projects and have collapsed the code for cleaner presentation. Feel free to expand them for more clarity on what they are doing.* 

# In[ ]:


def change_variables_to_categorical(df, vars_to_change=[]):
    '''Function that changes all non-numeric variables to categorical datatype.
    
    Arguments:
        df : Pandas DataFrame
        vars_to_change : list, default is an empty list
            If a non-empty list is passed, only the variables in the list are converted to 
            categorical datatype.
    
    Returns:
        df : Pandas DataFrame with categorical datatypes converted.
    '''
    categorical_variables = df.select_dtypes(exclude='number').columns.to_list()
    
    if len(vars_to_change) > 0:
        categorical_variables = vars_to_change
    
    for var in categorical_variables:
        df[var] = df[var].astype('category')
        
    return df


# In[ ]:


def numerical_categorical_split(df):
    '''Function that creates a list for numerical and categorical variables respectively.
    '''
    numerical_var_list = df.select_dtypes(include='number').columns.to_list()
    categorical_var_list = df.select_dtypes(exclude='number').columns.to_list()
    
    return numerical_var_list, categorical_var_list


# In[ ]:


# Changing datatypes from 'objects' to 'category' --> More memory efficient
df = change_variables_to_categorical(df)


# In[ ]:


# Creating lists of numerical and categorical features
numerical_vars, categorical_vars = numerical_categorical_split(df)

# Splitting 2 dataframes, one for numeric variables and another for categorical
numerical_df = df[numerical_vars]
categorical_df = df[categorical_vars]


# In[ ]:


# Checking if both have same number of observations
print(numerical_df.shape, categorical_df.shape)


# ### 3.1 Target Variable : SalePrice <a id='target'></a>
# 
# First, we look at the distribution of the target variable and check for any negative values.

# In[ ]:


any(numerical_df.SalePrice <= 0)


# Next, let's check the distribution and skewness of the target.

# In[ ]:


print(f'Skewness of SalePrice : {round(stats.skew(df.SalePrice),2)}')

fig = plt.figure(figsize=(7,4))
ax = sns.distplot(numerical_df.SalePrice, fit=stats.norm)
ax.set_title('Distribution of SalePrice', size=18, y=1.05)
plt.show();


# It appears that we have a bit of positive skew, we will try taking the logarithmic transformation to see if it alleviates this issue.

# In[ ]:


numerical_df['LogSalePrice'] = np.log(numerical_df.SalePrice)

print(f'Skewness of LogSalePrice : {round(stats.skew(numerical_df.LogSalePrice),2)}')

fig = plt.figure(figsize=(7,4))
ax = sns.distplot(numerical_df.LogSalePrice, fit=stats.norm)
ax.set_title('Distribution of LogSalePrice', size=18, y=1.05)
plt.show();


# The distribution of LogSalePrice is much more normally distributed, therefore, we will be using the log-transform as our new target variable. 
# 
# For our bivariate analysis in later sections, we will be comparing the independent variables against the LogSalePrice.
# 
# Note: For submission later, we need to take the __exponent of our predicted results__ to get the predicted SalePrice.
# 
# Note: [Effect of transforming the targets in regression model.][1]
# 
# [1]: https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py

# In[ ]:


numerical_df = numerical_df.drop('SalePrice', axis=1)


# ### 3.2.1 Numerical Variables - Univariate Analysis <a id='num_var_univariate'></a>
# 
# Typically if the number of variables are small, i like to plot the individual distributions of the variables to get a better understanding of our data.
# 
# However, in reality, the number of features within datasets are often too numerous and it takes too much time to look at each variable individually.
# 
# Therefore, in these situations, it is more efficient to analyse the variables that are likely to cause problems in our modelling later on. One of the main problems is the variables having large positive or negative skew.
# 
# Usually, variables are considered skewed if they have skew of magnitude > 0.5, and highly skewed when the magnitude > 1. 
# 
# Hence, in this section, we will first identify which variables are very skewed and use visualisations to guide us on what approaches to handle these issues.
# 
# Reference : [Rule of thumb for identifying skewed variables.][1]
# 
# [1]: https://stats.stackexchange.com/questions/245835/range-of-values-of-skewness-and-kurtosis-for-normal-distribution

# In[ ]:


def check_variable_skew(df, threshold=1, verbose=True):
    '''Function that checks each variable in the dataframe for their skewness.
    
    Arguments:
        df : Pandas DataFrame
        threshold : int, default = 1
            The threshold that we allow for skewness within the variable.
        verbose : bool, default = True
            Prints out highly skewed variables and their values.
        
    Returns:
        highly_skewed_vars_list : list
    '''
    skewness = df.apply(lambda x : np.abs(stats.skew(x)))
    skewed_vars = skewness.loc[skewness >= threshold].sort_values(ascending=False).round(2)
    
    if len(skewed_vars) == 0:
        print('There are no variables that are highly skewed.')
        return []
    
    skewed_vars_list = skewed_vars.index.to_list()
    
    print(f'The following {len(skewed_vars_list)} variables are highly skewed:')
    print()
    for var in skewed_vars_list:
        print(var, '\t', skewed_vars.loc[var])
      
    return skewed_vars_list


# In[ ]:


def skewness_subplots(df, skewed_vars_list, n_cols=4, fig_size=(18,12)):
    '''Function that plots the distribution of each variable within a grid.
    
    Arguments:
        df : Pandas DataFrame
        skewed_vars_list : list
            List of variables to plot histograms for.
        n_cols : int, default = 4
            Number of columns for the grid
    '''
    num_vars = len(skewed_vars_list)
    n_rows = int(np.ceil(num_vars / n_cols))
    df_skewed_vars = df[skewed_vars_list]
    
    fig = plt.figure(figsize=fig_size)
    plt.suptitle('Distributions for Highly Skewed Variables', y=1.03, size=18)

    for i, col in enumerate(skewed_vars_list):
        skew = np.round(stats.skew(df[col]), 2)
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        sns.distplot(df[col], ax=ax, kde=False, bins=50)
        ax.set_title(f'Skew : {skew}', size=16)
    
    plt.tight_layout()    
    plt.show();


# In[ ]:


# Creating a list of variables that are highly skewed
highly_skewed_vars = check_variable_skew(numerical_df)


# Since we have some variables that are highly skewed, we would like to visualise them to better understand how to handle them. The figure below is a grid of the distributions of these variables.

# In[ ]:


skewness_subplots(numerical_df, highly_skewed_vars, fig_size=(18,18))


# From the figure, we observe that there are some variables such as '3SsnPorch' and 'LowQualFinSF' that have very __narrow distributions__ (most of its values are concentrated with very few/one values). 
# 
# An option is to binary encode these variables into a categorical variable of having the most common value versus other values. However, as these variables have values that are mostly concentrated in one value, there is not much variation and hence not much information within the variable. Therefore, we will be dropping such variables with more than 80% of their values being concentrated in a single value. 
# 
# As some of the variables are postively skewed with no negative values, therefore, we will apply a logarithmic transformation and check if it helps alleviate the skewness.
# 
# For the other oridinal variables that are skewed we will just leave them as is. 

# In[ ]:


def most_frequent_value_proportion(df, threshold=0.8, verbose=True):
    '''Function that returns series with variables and their most frequent values respectively.
    
    Arguments:
        df : Pandas DataFrame
        threshold : float
            Threshold for the maximum allowed proportion of a single value/class. 
            
    Returns:
        most_frequent_series : Pandas Series
            Variables as index and values as proportions for their most common value.
    '''
    most_frequent_pct = []
    for col in df.columns:
        most_frequent = df[col].value_counts(normalize=True).sort_values(ascending=False).iloc[0]
        most_frequent_pct.append(np.round(most_frequent,2))
    
    most_frequent_series = pd.Series(most_frequent_pct, index=df.columns)
    most_frequent_series = most_frequent_series.loc[most_frequent_series >= threshold]
    most_frequent_series = most_frequent_series.sort_values(ascending=False)
    
    if verbose:
        print(f'The following {len(most_frequent_series)} variables have a high concentration (>{threshold*100}%) of their values in one value only.')
        print()
        print(most_frequent_series)
    
    return most_frequent_series


# In[ ]:


narrow_dist_vars = most_frequent_value_proportion(numerical_df, threshold=0.8)


# In[ ]:


# Dropping narrowly distributed variables
numerical_df = numerical_df.drop(narrow_dist_vars.index.to_list(), axis=1)


# In[ ]:


# List of positvely skewed variables
pos_skewed_vars = list(set(highly_skewed_vars) - set(narrow_dist_vars.index.to_list()) - set(ordinal_vars))


# In[ ]:


def make_log_variables(df, variables_list, drop=False):
    '''Function to make new columns of the logarithmic transformation of a list of variables.
    Arguments:
        df : Pandas DataFrame
        variables_list : list
            List of variables to log-transform.
        drop : bool, default = False
            Pass as true to drop the original variables.
    Returns:
        df : Pandas DataFrame with new variables.
        log_var_list : list
            List of the log-transformed variable names.
    '''
    # Checking for negative values for each variable
    any_neg_value = np.sum((df[variables_list] < 0).all(axis=0))
    if any_neg_value:
        raise ValueError('There are one or more columns with negative values and cannot be log-transformed.')
    
    log_var_list = []
    
    for var in variables_list:
        log_var_name = 'Log' + var
        df[log_var_name] = np.log1p(df[var])
        log_var_list.append(log_var_name)
    
    if drop:
        df = df.drop(variables_list, axis=1)
    
    return df, log_var_list


# In[ ]:


# Creating log-transformations for our highly skewed variables and saving the new variables in a list
numerical_df, log_var_list = make_log_variables(numerical_df, pos_skewed_vars, drop=False)


# Now that we have created new features by log-transforming the positively skewed variable, let's see if there are any improvements. 

# In[ ]:


print(f'Prior to log-transformation, there were {len(pos_skewed_vars)} variables that were highly positively skewed.')


# In[ ]:


highly_skewed_vars = check_variable_skew(numerical_df[log_var_list])


# It looks like the log transformation has corrected the skews for all but one variable. Therefore, we will __keep the log transformed variables for the successful transformation and drop the original variables__.
# 
# For the variable TotalBsmtSF that is still positively skewed, we will take a closer look at both its original and transfromed distributions.

# In[ ]:


# Dropping original variables
pos_skewed_vars.remove('TotalBsmtSF')
numerical_df = numerical_df.drop(pos_skewed_vars, axis=1)


# In[ ]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax1.set_title(f'Skew : {stats.skew(numerical_df.TotalBsmtSF):.2f}', y=1.03)
sns.distplot(numerical_df.TotalBsmtSF, ax=ax1, bins=50)

ax2.set_title(f'Skew : {stats.skew(numerical_df.LogTotalBsmtSF):.2f}', y=1.03)
sns.distplot(numerical_df.LogTotalBsmtSF, ax=ax2, bins=50)

fig.tight_layout()
plt.show();


# From the figure above, we observe that the actually the original distribution is less skewed than the log-transformed variable. The skew present in the original distribution is likely to be caused by outliers.
# 
# Using a scatteplot against the target variable, we can check for outliers.

# In[ ]:


fig = sns.scatterplot(numerical_df.TotalBsmtSF, numerical_df.LogSalePrice)


# The observation that is greater than 6000 is clearly an outlier, therefore we will drop the observation and check back on our variable's skew after.

# In[ ]:


# Removing the observation from both our numerical and caregorical data frames
outliers = numerical_df.loc[numerical_df.TotalBsmtSF > 5000].index.to_list()


# In[ ]:


# Dropping the outlier from both numerical and categorical data frames
numerical_df = numerical_df.drop(outliers, axis=0)
categorical_df = categorical_df.drop(outliers, axis=0)
print(numerical_df.shape, categorical_df.shape)


# In[ ]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax1.set_title(f'Skew : {stats.skew(numerical_df.TotalBsmtSF):.2f}', y=1.03)
sns.distplot(numerical_df.TotalBsmtSF, ax=ax1, bins=50)

ax2.set_title(f'Skew : {stats.skew(numerical_df.LogTotalBsmtSF):.2f}', y=1.03)
sns.distplot(numerical_df.LogTotalBsmtSF, ax=ax2, bins=50)

fig.tight_layout()
plt.show();


# After removing the outlier, the original variable is much more normally distributed, therefore we will remove the log-transformed variable from out dataset. 

# In[ ]:


numerical_df = numerical_df.drop('LogTotalBsmtSF', axis=1)


# Now that we have finished our univariate analysis and cleaning of our numerical data, we can proceed to bivariate analysis! 

# ### 3.2.2 Numeric Variables - Bivariate Analysis <a id='num_var_bivariate'></a>
# 
# For bivariate analysis, we are interested in the following:
# 1. Relationship between each independent variable and target (LogSalePrice)
# 2. Relationship between the independent variables - Checking for multicollinearity
# 
# We can obtain a quantitative understanding of the relationship by calculating the pearson's coefficient between the variables. 
# 
# We can visualise the relationships using scatterplots and a correlation heatmap.
# 
# Note : For a clearer understanding of why detecting multicollinearity is important, i found this [article][1] very helpful.
# 
# *Future Work : Calculate correlations between binary - continuous variables using [Point Biserial Correlation][2].*
# 
# *Future Work : Calculate correlations between categorical - continuous variables using Spearman's Rho.*
# 
# 
# [1]: https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/
# [2]: https://www.statisticssolutions.com/point-biserial-correlation/

# In[ ]:


# Rearranging our dataframe for easier interpretation of heatmap
log_sale_price = numerical_df.LogSalePrice
numerical_df = numerical_df.drop('LogSalePrice', axis=1)
numerical_df['LogSalePrice'] = log_sale_price


# In[ ]:


sns.set(style="white")

# Compute the correlation matrix
corr = numerical_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 16))
plt.title('Correlation Heatmap', size=20)

cmap = sns.diverging_palette(220, 10, as_cmap=True)
heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, fmt='.2f', vmin=-1, vmax=1.0, center=0, square=True,
                      linewidths=.5, cbar_kws={"shrink": .7}, annot=True, annot_kws={"size": 8})

bottom, top = ax.get_ylim()
heatmap.set_ylim(bottom + 0.5, top - 0.5)


# First, let's look at the __last row__ in the heatmap, which shows the __correlation between each individual variable and the target__. We observe that most variables are postively correlated to our target, with the overall quality ('OverallQual') and logarithmic transformation of the ground living area ('LogGrLivArea') being particularly strong predictors. This makes sense as larger houses are expected to cost more.
# 
# As the variables 'OverallCond' and 'YrSold' have almost no relationship with the target, we will drop these variables. 
# 
# However, there may potentially be __multicollinearity__ present in our data, with some pairs of independent variables having high correlation with each other. In terms of multicollinearity between independent variables, there is no hard cutoff to remove variables. But as a general rule of thumb, attention should be placed on variable pairs that have correlations around 0.8 or higher. Highly correlated variable pairs:
# 1. LogBsmtFinSF1 and BsmtFinType1 
# 2. 1stFlrSF and TotalBsmtSF
# 3. TotRmsAbrGrd and LogGrLivArea
# 4. FirePlaceQu and FirePlaces
# 5. GarageCars and GarageArea
# 
# We will drop GarageCars as it is clearly correlated with the size of the garage (GarageArea).
# 
# For the rest of the correlated variable pairs, we will just keep them in mind for now, and proceed with modelling with the variables included. Later on, we can use regularisation which helps deal with multicollinearity. 

# In[ ]:


numerical_df = numerical_df.drop(['OverallCond', 'YrSold', 'GarageCars'], axis=1)


# In[ ]:


corr_matrix_unstacked = corr.unstack().sort_values(ascending=False).drop_duplicates()
correlated_pairs = corr_matrix_unstacked.loc[corr_matrix_unstacked >= 0.75].index.to_list()
correlated_pairs


# The correlation heatmap has provided us a rough idea of the relationships between the variables. Now, we will plot the scatter plots of each variable against the target variable.

# In[ ]:


def scatter_subplots(df, target, hue=None, n_cols=4, fig_size=(12,12)):
    '''Function that plots the scatterplots of each variable against the target variable within a grid.
    
    Arguments:
        df : Pandas DataFrame with target variable included
        target : str
            Target feature name
        hue : str, default = None
            Column in the data frame that should be used for colour encoding
        n_cols : int, default = 4
            Number of columns for the grid
    '''
    independent_vars_list = list(df.columns)
    independent_vars_list.remove(target)
    num_vars = len(independent_vars_list)
    n_rows = int(np.ceil(num_vars / n_cols))
    
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=fig_size)
    plt.suptitle(f'Scatterplots of Independent Variables against {target}', y=1.02, size=18)

    for i, col in enumerate(independent_vars_list):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        sns.scatterplot(x=col, y=target, hue=hue, data=df, ax=ax)
    
    plt.tight_layout()
    plt.show();


# In[ ]:


scatter_subplots(numerical_df, 'LogSalePrice', fig_size=(16,24))


# From the scatterplots, it doesn't seem like we have any obvious outliers. Now that we have a good understanding of our numerical variables, we can move on to looking at our categorical variables.

# ### 3.3.1 Categorical Variables Analysis <a id='cat_var_univariate'></a>
# 
# Similar to numerical variables univariate analysis, we are interested in the distribution of categorical variables across their individual classes. We also do not want categorical variables that are majority concentrated within one class. 

# In[ ]:


# Adding the target variable to the categorical dataframe
categorical_df['LogSalePrice'] = numerical_df.LogSalePrice
print(categorical_df.shape, numerical_df.shape)


# In[ ]:


def annotate_plot(ax, dec_places=1, annot_size=14):
    '''Function that annotates plots with their value labels.
    Arguments:
        ax : Plot Axis.
        dec_places : int
            Number of decimal places for annotations.
        annot_size : int
            Font size of annotations.
    '''
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), '.{}f'.format(dec_places)),
            (p.get_x() + p.get_width() / 2., p.get_height(),),
            ha='center', va='center',
            xytext=(0,10), textcoords='offset points', size=annot_size
        )


# In[ ]:


def var_categories_countplots(df, n_cols=3, orientation='v', x_rotation=45, y_rotation=0, palette='pastel', fig_size=(18,12)):
    '''Function that plots the class distribution for categorical variables.
    
    Arguments:
        df : Pandas DataFrame
        n_cols : int, default = 3
            Number of columns for the subplot grid.
        orientation : str, default = 'v'
            Plot orientation, with 'v' for vertical and 'h' for horizontal.
        x_rotation : int, default = 45
            Rotation of the x-axis labels.
        palette : str, default = 'pastel'
            Seaborn color palette for plotting.
    '''
    categorical_vars = df.select_dtypes(exclude='number').columns.to_list()
    num_vars = len(categorical_vars)
    n_rows = int(np.ceil(num_vars / n_cols))
    
    fig = plt.figure(figsize=fig_size)
    plt.suptitle('Class Distributions for Categorical Variables', y=1.01, size=24)
    
    for i, col in enumerate(categorical_vars):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        sns.countplot(x=df[col], ax=ax, orient=orientation, palette=palette)
        ax.set_ylabel('Frequency')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation)
        
        annotate_plot(ax, dec_places=0, annot_size=12) # Annotating plot with count labels
    
    plt.tight_layout()
    plt.show();


# In[ ]:


# Finding variables that have more than 80% of their values in one category
highly_imbalanced_vars = most_frequent_value_proportion(categorical_df, threshold=0.8, verbose=False)
highly_imbalanced_vars_list = highly_imbalanced_vars.index.to_list()
print(f'The following {len(highly_imbalanced_vars)} variables have more than 80% of their data concentrated in only one class:')
print()
print(highly_imbalanced_vars)

# Note: Function was defined earlier in univariate analysis of numerical variables


# Let's visualise these variables and get a better understanding of their frequency distributions. Problematic variables would be those that have many categories but are sparsely distributed. For binary variables, the imbalanced class issue might not be as bad, as long as the imbalance is not too great. 

# In[ ]:


var_categories_countplots(categorical_df[highly_imbalanced_vars_list], fig_size=(16,16))


# From the diagram above, we observe that the class imbalance within these variables are quite extreme, therefore, we will be dropping these variables from the analysis. This would help alleviate the problem of 'The Curse of Dimensionality' as we would have less dummy variables when we encode our data for modelling later on.

# In[ ]:


categorical_df = categorical_df.drop(highly_imbalanced_vars_list, axis=1)


# Let's check the count distributions for the remaining categorical variables to check if there any more problematic variables.

# In[ ]:


var_categories_countplots(categorical_df, fig_size=(20,16))


# Looking at the variable distributions, we see that we still have some variables have classes that have very few observations. Therefore, we may want to lump similar classes together or under one category as 'Others'.
# 
# For variables that are not so obvious how to encode or if we are unsure how to encode them, we will just leave them as is.

# In[ ]:


categorical_df.MSZoning = categorical_df.MSZoning.apply(lambda x :
                                                        x if x == 'RL'
                                                        else x if x == 'RM'
                                                        else 'Others')

categorical_df.LotShape = categorical_df.LotShape.apply(lambda x :
                                                        x if x =='Reg'
                                                        else 'Irregular')

categorical_df.LotConfig = categorical_df.LotConfig.apply(lambda x :
                                                          x if x == 'Inside'
                                                          else x if x == 'CulDSac'
                                                          else x if x == 'Corner'
                                                          else 'FR')

categorical_df.RoofStyle = categorical_df.RoofStyle.apply(lambda x :
                                                          x if x =='Gable'
                                                          else x if x == 'Hip'
                                                          else 'Others')

categorical_df.MasVnrType = categorical_df.MasVnrType.apply(lambda x :
                                                            x if x == 'None'
                                                            else x if x == 'Stone'
                                                            else 'Brk')

categorical_df.Foundation = categorical_df.Foundation.apply(lambda x :
                                                            x if x =='BrkTil'
                                                            else x if x == 'CBlock'
                                                            else x if x == 'PConc'
                                                            else 'Others')

categorical_df.GarageType = categorical_df.GarageType.apply(lambda x : 
                                                            x if x == 'Attchd'
                                                            else x if x == 'BuiltIn'
                                                            else x if x == 'Detchd'
                                                            else x if x == 'NoGarage'
                                                            else 'Others')


# ### 3.3.2 Categorical Variables - Bivariate Analysis
# 
# Now that we have finished processing the categorical variables individually, we can explore the relationship between their individual levels and the target variable. 

# In[ ]:


def var_categories_boxplots(df, target, hue=None, n_cols=3, orientation='v', x_rotation=45, y_rotation=0, palette='pastel', fig_size=(18,12)):
    '''Function that plots the class distribution for categorical variables against target variable.
    
    Arguments:
        df : Pandas DataFrame
        target : str
            Target variable name.
        hue : str, default = None
            Column in the data frame that should be used for colour encoding.
        n_cols : int, default = 3
            Number of columns for the subplot grid.
        orientation : str, default = 'v'
            Plot orientation, with 'v' for vertical and 'h' for horizontal.
        x_rotation : int, default = 45
            Rotation of the x-axis labels.
        palette : str, default = 'pastel'
            Seaborn color palette for plotting.
    '''
    categorical_vars = df.select_dtypes(exclude='number').columns.to_list()
    num_vars = len(categorical_vars)
    n_rows = int(np.ceil(num_vars / n_cols))
    
    fig = plt.figure(figsize=fig_size)
    plt.suptitle('Categorical Variables vs Target', y=1.01, size=24)
    
    for i, col in enumerate(categorical_vars):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        sns.boxplot(x=df[col], y=df[target], ax=ax, hue=hue, orient=orientation, palette=palette)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation)
    
    plt.tight_layout()
    plt.show();


# In[ ]:


var_categories_boxplots(categorical_df, 'LogSalePrice', n_cols=3, fig_size=(20,30))


# Phew! We finally finished our preprocessing and exploratory data analysis for both our numerical and categorical data!
# 
# We can finally move on to the fun part, modelling our data!

# # 4. Modelling <a id='modelling'></a>
# 
# In this section we will be trying different machine learning algorithms and comparing their performances. For most of these algorithms, there will be some hyper-parameter tuning involved as well.
# 
# Models that we will be comparing:
# 1. Linear Models - Linear / Lasso / Ridge Regression
# 2. Ensemble Bagging Model - Random Forest
# 3. Ensemble Boosting Model - Gradient Boosting Regression

# ### 4.1 Preparing Our Data <a id='preparing_data'></a>
# 
# Initially, we split our data to analyse the different data types, therefore we need to concat the separate data frames back together.
# 
# Furthermore, most algorithms in Scikit-Learn do not accept categorical variables, therefore we need to one hot encode our variables as well.

# In[ ]:


df = pd.concat([categorical_df.drop('LogSalePrice', axis=1), numerical_df], axis=1).reset_index(drop=True)

# One Hot Encoding
df = pd.get_dummies(df)

y = df.LogSalePrice
X = df.drop('LogSalePrice', axis=1)
print(X.shape, y.shape)
X.head()


# Now, we need to split our data into training (80%) and testing (20%) data.
# 
# For hyper-parameter tunining, we will be using a 5-fold cross validation on the training dataset.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


# In[ ]:


kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)


# As per the submission requirements, we will be using the Root Mean Squared Error as our evaluation metric.

# In[ ]:


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Creating a sklearn scorer instance
rmse_scorer = {'RMSE' : make_scorer(rmse, greater_is_better=False, needs_proba=False, needs_threshold=False)}


# In[ ]:


# Kwargs for cross_validate method
cv_kwargs = {
    'scoring' : rmse_scorer,
    'cv' : kfold,
    'n_jobs' : -1,
    'return_train_score' : True,
    'verbose' : False,
    'return_estimator' : True
}


# For Cross Validation, we will be using the [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) method from scikit-learn.
# 
# I prefer using this method over [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) as it provides more information and i can check if the model is overfitting by comparing the training and validation scores.

# ### 4.2 Linear Regression Model - Baseline <a id='linear'></a>
# 
# The linear regression model is a good baseline model for comparison. There are no hyper parameters to tune and there is no need to transform the data before fitting the model.

# In[ ]:


def save_cv_results(cv_results, scoring_name='score', verbose=True):
    '''Function to save the training and testing results from cross validation into a dataframe.
    
    Arguments:
        cv_results : dict
            Dictionary of results from scikit-learn's cross_validate method.
        scoring_name : str, default = 'score'
            Name of scorer used in the cross_validate method. If no custom scorer was passed, default should be 'score'.
            In the cv_results dictionary, there should be keys 'train_score' and 'test_score'
            If custom scorer was passed as the scoring method, the cv_results dictionary should have 'train_scoring_name'.
        verbose : bool, default = True
            Prints the mean training and testing scores and fitting times. 
            
    Returns:
        results_df : Pandas DataFrame with training and test scores.
    
    '''
    train_key = 'train_' + scoring_name
    test_key = 'test_' + scoring_name
    
    # Sklearn scorer flips the sign to negative so we need to flip it back
    train_scores = [-result for result in cv_results[train_key]]
    test_scores = [-result for result in cv_results[test_key]]
    
    indices = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    
    results_df = pd.DataFrame({'TrainScores' : train_scores, 'TestScores' : test_scores}, index=indices)
    
    if verbose:
        avg_train_score = np.mean(train_scores)
        avg_test_score = np.mean(test_scores)
        avg_training_time = np.mean(cv_results['fit_time'])
        avg_predict_time = np.mean(cv_results['score_time'])
        
        title = 'Cross Validation Results Summary'
        print(title)
        print('=' * len(title))
        print(f'Avg Training {scoring_name}', '\t', '{:.6f}'.format(avg_train_score))
        print(f'Avg Testing {scoring_name}', '\t', '{:.6f}'.format(avg_test_score))
        print()
        print('Avg Fitting Time', '\t', '{:.4f}s'.format(avg_training_time))
        print('Avg Scoring Time', '\t', '{:.4f}s'.format(avg_predict_time))
    
    return results_df


# In[ ]:


def training_vs_testing_plot(results, fig_size=(5,5), title_fs=18, legend_fs=12):
    '''Function that plots the training and testing scores obtained from cross validation.'''
    
    fig = plt.figure(figsize=fig_size)
    plt.style.use('fivethirtyeight')
    plt.title('Cross Validation : Training and Testing Scores', y=1.03, x=0.6, size=title_fs)
    plt.plot(results.TrainScores, color='b', label='Training')
    plt.plot(results.TestScores, color='r', label='Testing')
    plt.legend(loc='center left', bbox_to_anchor=(1.02,0.5), ncol=1, fontsize=legend_fs)
    plt.show();


# In[ ]:


def get_best_estimator(cv_results, scoring_name='score'):
    ''' Function that returns the best estimator found during cross valiation.
    Arguments:
        cv_results : dict
            Results from Sklearn's cross_validate method.
        scoring_name : str, default = 'score'
            Custom scoring name if a custom scorer was passed during cross validation.
            Default 'score' should be used when using sci-kit learn's  scoring metrics. 
    
    Returns:
        best_estimator : Sklearn estimator object
            Best estimator found during cross validation.
    '''
    test_key = 'test_' + scoring_name
    
    # Sklearn flips the sign during scoring so we need to flip it back
    scores = [-result for result in cv_results[test_key]]
    max_score_index = scores.index(max(scores))
    best_estimator = cv_results['estimator'][max_score_index]
    
    return best_estimator


# Note: As we pass a custom scoring *__loss__* function as the scorer into __scikit-learn's cross_validate method__, the scores returned will be __negative__. In the functions above (hidden), we have made the necessary adjustments to return positive results.

# In[ ]:


linreg = LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)

linreg_cv_results = cross_validate(linreg, X_train, y_train, **cv_kwargs)
linreg_cv_results


# In[ ]:


# Saving cross validation results to a dataframe
linreg_cv_scores = save_cv_results(linreg_cv_results, scoring_name='RMSE')

# Plotting training vs testing RMSE scores
training_vs_testing_plot(linreg_cv_scores)


# From the diagram, we observe that the model trained on the 5th fold had the lowest training error, but had the highest testing error. By visualising the training and test scores, we are able to tell which models are overfitting.
# 
# The best model from our cross validation is the one trained on the second fold. Despite having the highest training error, it had the lowest testing error, which indicates that it is better able to generalise and predict unseen data.
# 
# Therefore, we will save the model in fold 2 and evaluate it once again on the holdout testing data (20% from the initial split) for comparison with other models.

# In[ ]:


# Saving best estimator from cross validation
best_linreg = get_best_estimator(linreg_cv_results, scoring_name='RMSE')


# In[ ]:


def holdout_set_evaluation(model, X_train, y_train, X_test, y_test, model_name, scoring_name):
    '''Function that evaluates the performance on the holdout dataset.
    
    Arguments:
        model : sklearn estimator object
        model_name : str
            String to be passed as the index for the dataframe.
        scoring_name : str
            Evluation metric used as column header.
    
    Returns:
        rmse_score : Pandas DataFrame
            Column is the scoring_name, index is the model_name and value is the model performance on the holdout dataset.
    '''    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse_score = rmse(y_test, y_pred) # calls rmse function
    rmse_score = pd.DataFrame({scoring_name : [rmse_score]}, index=[model_name])
    
    return rmse_score.round(4)


# In[ ]:


# Saving our result to a dataframe
linreg_result = holdout_set_evaluation(best_linreg, X_train, y_train, X_test, y_test, model_name='Linear', scoring_name='RMSE')
linreg_result


# Now that we have our baseline result, we can use this as a benchmark to evaluate the performance of other algorithms.

# In[ ]:


# Creating a new dataframe to store model results
model_results = linreg_result.copy(deep=True)


# Previously, we identified the possibilility of multicollinearity in our model due to having highly correlated pairs of variables. 
# 
# We can check for multicollinearity by looking at the [Variance Inflation Values (VIFs)][1] of the variable coefficients from our Linear model. As a rule of thumb, VIFs greater than 5 are signs of potential multicollinearity.  
# 
# [1]: https://online.stat.psu.edu/stat462/node/180/

# In[ ]:


print('Highly Correlated Pairs of Variables:')
print(correlated_pairs)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

vif_X = add_constant(X)
vifs = pd.Series([
    variance_inflation_factor(vif_X.values, i) for i in range(vif_X.shape[1])],
    index=vif_X.columns
)

# Dropping inf values due to dummy variables
vifs = vifs.loc[vifs != np.inf].sort_values(ascending=False)
vifs.loc[vifs > 5]


# From the results above, we observe that there is definitely Multicollinearity in our variables, and more than the number of variables we identified previously.
# 
# Given that there is Multicollinearity present, does that mean our model is unusable?
# 
# Actually, [not really][1]! The primary concern of multicollinearity is that the estimates of the regression model become unstable and the standard errors of the coefficients become inflated.
# 
# This problem only really affects us when we are trying to __use regression for causal inference__ on the effects of our independent variables on our target. However in terms of __predictive performance__, multicollinearity does not adversely affect the results. 
# 
# Furthermore, the next two models (Ridge and Lasso) that we will be working with have regularisation paramters that help reduce the negative effect of multicollinearity.
# 
# For Reference:
# * [Discussion on Multicollinearity in Machine Learning.][2]
# * [Times where Multicollinearity is not that big of an issue.][3]
# 
# [1]: https://www.lexjansen.com/wuss/2018/131_Final_Paper_PDF.pdf
# [2]: https://stats.stackexchange.com/questions/168622/why-is-multicollinearity-not-checked-in-modern-statistics-machine-learning
# [3]: https://statisticalhorizons.com/multicollinearity

# ### 4.3 Ridge Regression - L2 Regularization <a id='ridge'></a>
# 
# Ridge Regression is a regularisation method to reduce the variance of the model in exchange for a tolerable increase in the bias of our model.  
# 
# The penalty term (lambda / alpha) regularizes the coefficients such that if the coefficients take large values the optimization function is penalized. So, ridge regression __shrinks the coefficients__ and it helps to reduce the model complexity and multi-collinearity.
# 
# For more information on how Ridge Regression works, i found [this article][1] very helpful.
# 
# References : [Rules of thumb for applying Ridge Regression][2]
# 
# [1]: https://towardsdatascience.com/ridge-regression-for-better-usage-2f19b3a202db
# [2]: https://stats.stackexchange.com/questions/169664/what-are-the-assumptions-of-ridge-regression-and-how-to-test-them

# In[ ]:


# Range of alphas to iterate over
alphas_vector = np.arange(1,200)


# In[ ]:


def alpha_tuning_results(alphas_vector, X_train, y_train, cv_kwargs, model_name='Ridge', scoring_name='score', x_axis_log_scale=False, fig_size=(5,5)):
    ''' Function to obtain the average training and testing RMSE across different alpha values.
    
    Arguments:
        alphas_vector : array
            array of alpha values to iterate through and fit the model
        cv_kwargs : dict
            kwargs for the cross_validate method
        model_name : str, default = 'Ridge'
            Type of model to fit training data. Any other str input will fit a Lasso model.
        scoring_name : str, default = 'score'
            Scoring used in sklearn's cross_validate method. Use default value if no custom scorer was used.
            Else, enter the name of the scorer used when making scorer.
        x_axis_log_scale : bool, default = False
            Set the X-axis to log scale. Useful when tuning Lasso Alpha.
        
    Returns:
        results_df : Pandas DataFrame with average training and testing RMSE per alpha. 
            
    '''
    results_df = pd.DataFrame(columns=['Avg_Train_RMSE', 'Avg_Test_RMSE'])

    for alpha in alphas_vector:
        if model_name == 'Ridge':
            model = Pipeline(steps=[
                ('Standardise',  StandardScaler()),
                ('Ridge', Ridge(alpha=alpha, fit_intercept=True, random_state=SEED))
            ])
        else:
            model = Pipeline(steps=[
                ('Standardise', StandardScaler()),
                ('Lasso', Lasso(alpha=alpha, fit_intercept=True, random_state=SEED))
            ])
        
        cv_results = cross_validate(model, X_train, y_train, **cv_kwargs)
        train_key = 'train_' + scoring_name
        test_key = 'test_' + scoring_name
        # Sklearn scorer flips the sign to negative so we need to flip it back
        train_scores = [-result for result in cv_results[train_key]]
        test_scores = [-result for result in cv_results[test_key]]
        avg_train_rmse = np.mean(train_scores)
        avg_test_rmse = np.mean(test_scores)
        
        results_df.loc[alpha] = [avg_train_rmse, avg_test_rmse]
    
    # Visualising the results
    fig = plt.figure(figsize=fig_size)
    plt.style.use('fivethirtyeight')
    plt.title(f'Training and Testing {scoring_name} for Different Alpha Values', y=1.03, x=0.6, size=16)
    plt.ylabel(f'{scoring_name}', size=14)
    plt.xlabel('Alpha', size=14)
    
    if x_axis_log_scale:
        plt.xscale('log')
    
    plt.plot(results_df.Avg_Train_RMSE, color='b', label='Training')
    plt.plot(results_df.Avg_Test_RMSE, color='r', label='Testing')
    plt.legend(loc='center left', bbox_to_anchor=(1.02,0.5), ncol=1, prop={'size': 14})
    plt.plot();
    
    return results_df


# In[ ]:


# Function to perform cross validation for each alpha value and plot the average RMSE obtained for each alpha value
ridge_results_df = alpha_tuning_results(np.arange(1,200), X_train, y_train, cv_kwargs, model_name='Ridge', scoring_name='RMSE')


# In[ ]:


# Getting the alpha which has the lowest testing RMSE
optimal_ridge_alpha = ridge_results_df.Avg_Test_RMSE.idxmin()
print(f'The optimal alpha from cross validation : {optimal_ridge_alpha}')


# From the figure above, as alpha increases, the regularisation strength increases, which shrinks the coefficients to a larger extent. This bias of shrinking the coefficients towards 0 causes the model to [underfit][1] to the training data, which can be observed by the increasing training error. 
# 
# However, this allows our model to generalise better to unseen data, as seen from the decrease in testing error as alpha increases. Although there will come a point where the marginal increase in bias out weighs the marginal decrease in variance, and performance starts to deteriorate due to the model underfitting to unseen data as well.
# 
# From the cross validation, we found that the optimal alpha is 128, and we can proceed to testing the model on the holdout dataset.
# 
# [1]: https://stats.stackexchange.com/questions/351990/why-dont-we-want-to-choose-a-big-lambda-in-ridge-regression

# In[ ]:


# Creating model with optimal alpha
ridge = Pipeline(steps=[
    ('Standardise',  StandardScaler()),
    ('Ridge', Ridge(alpha=optimal_ridge_alpha, fit_intercept=True, random_state=SEED))
])

# Cross validation results
ridge_cv_results = cross_validate(ridge, X_train, y_train, **cv_kwargs)

# Saving scores from cv results
ridge_cv_scores = save_cv_results(ridge_cv_results, scoring_name='RMSE')

# Plotting training vs testing RMSE scores
training_vs_testing_plot(ridge_cv_scores)


# In[ ]:


# Saving best estimator from cross validation
best_ridge = get_best_estimator(ridge_cv_results, scoring_name='RMSE')

# Evaluating resutls 
ridge_result = holdout_set_evaluation(best_ridge, X_train, y_train, X_test, y_test, model_name='Ridge', scoring_name='RMSE')


# In[ ]:


model_results = model_results.append(ridge_result)
model_results


# Nice! The Ridge Regression outperformed the baseline Linear Regression model!
# 
# This shows the benefits of regularisation and how it helps the model generalise to newer predictions.

# __*Alternatively*__, we could have used a cross validated __GridSearch__ to find the optimal alpha as well, and the steps are shown below. 
# 
# Ultimately, we will end up with the same alpha but its just a personal preference that i like using cross_validate and making my own functions.

# In[ ]:


# Using a GridSearch to find the optimal alpha for the ridge regression 
ridge_pipe = Pipeline(steps=[
    ('Standardise', StandardScaler()),
    ('Ridge', Ridge(fit_intercept=True, random_state=SEED))
])

ridge_params = {'Ridge__alpha' : np.arange(1,200)}

ridge_gscv = GridSearchCV(ridge_pipe, ridge_params, scoring=rmse_scorer['RMSE'], n_jobs=-1, cv=kfold, return_train_score=True)
ridge_gscv.fit(X_train, y_train)


# In[ ]:


# Best alpha 
ridge_gscv.best_params_


# ## 4.4 Lasso Regression - L1 Regularisation <a id='lasso'></a>
# 
# Lasso Regression is another form of regularisation that we can use to help our model generalise better to unseen data. Similar to Ridge Regression, it incorporates a penalty term (lambda / alpha) to the cost function. 
# 
# While Ridge Regression regularises the model by shrinking all coefficients towards 0 but never reaching 0, L1 regularisation by Lasso is able to shrink some non-informative variables' coefficients all the way to 0. 
# 
# This property of Lasso Regression is useful as it helps us with feature selection. By setting their coefficients to 0, the Lasso model is essentially removing them from the model.
# 
# This [article][1] does a good comparison between Ridge and Lasso regression.
# 
# [1]: https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b

# For Lasso Regression, what i usually like to do is see how the alpha changes in orders of 10.
# 
# Similar to Ridge Regression, the regularisation strength decreases as we approach 0, and in the case where alpha = 0, it is the same as Linear Regression.

# In[ ]:


# Range of alphas to iterate over
alphas_vector = np.logspace(-6,0,7)

# Function to perform cross validation for each alpha value and plot the average RMSE obtained for each alpha value
lasso_results_df = alpha_tuning_results(alphas_vector, X_train, y_train, cv_kwargs, model_name='Lasso', scoring_name='RMSE', x_axis_log_scale=True)


# In[ ]:


lasso_results_df.Avg_Test_RMSE.idxmin()


# We can see that the optimal alpha for Lasso Regression is likely to be somewhere between 0.0001 and 0.01.
# 
# Therefore, let's iterate through these values and plot their respective RMSEs.

# In[ ]:


alphas_vector = np.linspace(0.0001, 0.01, 100)

# Function to perform cross validation for each alpha value and plot the average RMSE obtained for each alpha value
lasso_results_df = alpha_tuning_results(alphas_vector, X_train, y_train, cv_kwargs, model_name='Lasso', scoring_name='RMSE', x_axis_log_scale=True)


# In[ ]:


# Getting the alpha which has the lowest testing RMSE
optimal_lasso_alpha = lasso_results_df.Avg_Test_RMSE.idxmin()
optimal_lasso_alpha


# Now that we have obtained the optimum alpha, we can fit a new Lasso model and see how it performs during cross validation.

# In[ ]:


lasso_pipe = Pipeline(steps=[
    ('Standardise', StandardScaler()),
    ('Lasso', Lasso(alpha=optimal_lasso_alpha, fit_intercept=True, random_state=SEED))
])

lasso_cv_results = cross_validate(lasso_pipe, X_train, y_train, **cv_kwargs)

# Saving cross validation results 
lasso_cv_scores = save_cv_results(lasso_cv_results, scoring_name='RMSE')

# Plotting training vs testing RMSE scores
training_vs_testing_plot(lasso_cv_scores)


# In[ ]:


# Saving best model from cross validation
best_lasso = get_best_estimator(lasso_cv_results, scoring_name='RMSE')

# Evaluating Lasso performance
lasso_result = holdout_set_evaluation(best_lasso, X_train, y_train, X_test, y_test, model_name='Lasso', scoring_name='RMSE')


# In[ ]:


model_results = model_results.append(lasso_result)
model_results


# Comparing the results of the linear models, Lasso performed the best.
# 
# The poorer performance of the Linear and Ridge models could be due to multicollinearity. Ridge is able to outperform the Linear model as it is able to reduce the impacts of multicollinearity, by shrinking the coefficients of correlated pairs. 
# 
# In the case of Lasso, it handles multicollinearity by dropping the less important variable from the correlated pair of variables. 

# As mentioned previously, Lasso helps to do feature selection by setting variable coefficients to 0. Let's see which variables were removed from the Lasso Regression.

# In[ ]:


best_lasso.fit(X_train, y_train)

# We need to access the 'named_steps' to get our Lasso estimator as we are using a Pipeline
lasso_model = best_lasso.named_steps['Lasso']
lasso_coefs = pd.Series(lasso_model.coef_, index=X.columns)
zero_coefs = lasso_coefs.loc[lasso_coefs == 0].index.to_list()

print(f'Original training data frame had {X.shape[1]} variables.')
print(f'Lasso selection removed {len(zero_coefs)} variables.')
print(f'Number of variables with non-zero coefficients : {X.shape[1] - len(zero_coefs)}')


# We observe that a large number of our variables were deemed as insignificant by the Lasso selection.
# 
# We will save the remaining variables as a new dataset and check if the lower dimensions help to alleviate the curse of dimensionality and improve our results for the earlier models. 

# ## 4.5 Linear Regression with Smaller Dataset <a id='linear_small'></a>

# In[ ]:


# Creating new training and testing datasets after removing variables
small_df = df.drop(zero_coefs, axis=1)

Xsmall = small_df.drop('LogSalePrice', axis=1)
Xsmall_train, Xsmall_test, y_train, y_test = train_test_split(Xsmall, y, test_size=0.2, random_state=SEED)
print(Xsmall_train.shape, Xsmall_test.shape)


# In[ ]:


linreg_small = LinearRegression(fit_intercept=True, n_jobs=-1)

# Cross validation 
linreg_small_cv_results = cross_validate(linreg_small, Xsmall_train, y_train, **cv_kwargs)

# Saving cross validation results
linreg_small_cv_scores = save_cv_results(linreg_small_cv_results, scoring_name='RMSE')

# Plotting training vs testing RMSE scores
training_vs_testing_plot(linreg_small_cv_scores)


# In[ ]:


# Saving best estimator from cross validation
best_linreg_small = get_best_estimator(linreg_small_cv_results, scoring_name='RMSE')

# Saving our result to a dataframe so it is easier to append the performances of other models
linreg_small_result = holdout_set_evaluation(best_linreg_small, Xsmall_train, y_train, Xsmall_test, y_test, model_name='Linear_SmallDf', scoring_name='RMSE')

# Evaluating results
model_results = model_results.append(linreg_small_result)


# In[ ]:


model_results


# From the table above, we can see that by removing the features with 0 coefficients from the Lasso regression, we were able to improve the model performance of the Linear model! 
# 
# This shows that Lasso Regression can be useful for feature selection prior to the modelling process. 
# 
# We will proceed to ensemble modelling using the smaller dataset.

# ## 4.6 Random Forest Regression <a id='rforest'></a>
# 
# Having seen the performance of the linear models, let's compare their performance against other non-parametric ensemble methods.
# 
# The first ensemble model we will try is the Random Forest, which uses Bootstrap Aggregating (bagging). Unlike linear models, Random Forests have many hyper parameters to tune and we will compare the performance a baseline versus a fully tuned model.
# 
# If you are new to Random Forest Regression or decision trees in general, i would highly recommend checking out the YouTube channel, [StatQuest][1].
# 
# Note : I took inspiration for some of the functions from this [kernel][2].
# 
# [1]: https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&list=PLblh5JKOoLUIcdlgu78MnlATeyx4cEVeR&index=11&t=0s
# [2]: https://www.kaggle.com/hadend/tuning-random-forest-parameters

# In[ ]:


# Baseline Untuned Random Forest Model
baseline_rforest = RandomForestRegressor(random_state=SEED)
baseline_rf_cv_results = cross_validate(baseline_rforest, Xsmall_train, y_train, **cv_kwargs)

# Saving cross validation results
baseline_rf_cv_scores = save_cv_results(baseline_rf_cv_results, scoring_name='RMSE')

# Plotting training vs testing RMSE scores
training_vs_testing_plot(baseline_rf_cv_scores)


# In[ ]:


# Saving best estimator from cross validation
best_baseline_rf = get_best_estimator(baseline_rf_cv_results, scoring_name='RMSE')

# Saving our result to a dataframe so it is easier to append the performances of other models
baseline_rf_result = holdout_set_evaluation(best_baseline_rf, Xsmall_train, y_train, Xsmall_test, y_test, model_name='BaselineRF', scoring_name='RMSE')


# In[ ]:


model_results = model_results.append(baseline_rf_result)
model_results


# The untuned Random Forest Regression has worse performance than the linear models. Let's see how much the model performance increases as we tuned the hyper parameters.
# 
# So how do we go about tuning the random forest? One method is to create a range of values for each hyper parameter and shove it all into a GridSearchCV, and it will output the best combination of hyper parameters. However, this is computationally expensive and may not be feasible for larger datasets.
# 
# Personally, i like to tune each parameter separately and visualise how it affects the model performance. After i have a good idea how each hyper parameter affects the model performance, i will fit a smaller and more targeted parameter grid into the GridSearch.
# 
# The hyper parameters that i mainly care about are the regularisation parameters:
# 1. min_samples_split - number of samples to have in a node to split
# 2. min_samples_leaf - number of samples to have in each leaf node
# 3. max_leaf_nodes - maximum number of leaf nodes
# 4. max_depth - maximum depth of the tree can grow
# 5. max_features - percentage of total features included to train each tree
# 
# Note: For further reading, you can check out these [Bayesian Optimisation Methods][1] for hyper parameter tuning as well.
# 
# [1]: https://roamanalytics.com/2016/09/15/optimizing-the-hyperparameter-of-which-hyperparameter-optimizer-to-use/

# In[ ]:


def rforest_tuning_scores(model, X_train, y_train, parameter, param_range, scorer, cv, flip_scores=True):
    
    gridsearch = GridSearchCV(model, param_grid={parameter : param_range}, scoring=scorer,
                              cv=cv, n_jobs=-1, return_train_score=True, verbose=False)
    
    gridsearch.fit(X_train, y_train)
    cv_results = gridsearch.cv_results_
    
    if flip_scores:
        train_scores = [-result for result in cv_results['mean_train_score']]
        test_scores = [-result for result in cv_results['mean_test_score']]
    else:
        train_scores = [cv_results['mean_train_score']]
        test_scores = [cv_results['mean_test_score']]
        
    results_df = pd.DataFrame({'TrainScores' : train_scores, 'TestScores' : test_scores}, index=param_range)
    
    return results_df


# The parameter grid below is a dictionary of the hyper paramters and an array of values to iterate through. I focus my attention on these variables as they help to regularise the model. 
# 
# * max_features - Helps to decorrelate the trees as a subset of features are selected when building each individual tree. 
# * The others control how deep each tree is allowed to grow and reduces overfitting.
# 
# Note : The parameters in max_features are the options availble in scikit-learn, with 'auto' being all features. These values for max_features are those that have been  empirically proven to give the best results.

# In[ ]:


param_grid = {
    'max_features' : ['auto', 'sqrt', 'log2'],
    'max_depth' : [1, 2, 5, 10, 20, 30, 50, None],
    'min_samples_split' : np.arange(2, 30, step=2),
    'min_samples_leaf' : np.arange(1, 20, step=1)
}


# In[ ]:


n_cols = 2
n_vars = len(param_grid)
n_rows = int(np.ceil(n_vars / n_cols))
index = 0

fig = plt.figure(figsize=(12,6))
plt.suptitle('Training and Test Scores for Different Hyper Parameters', y=1.03, size=20)

for parameter, param_range in dict.items(param_grid):
    results_df = rforest_tuning_scores(baseline_rforest, Xsmall_train, y_train, parameter=parameter, param_range=param_range,
                                       scorer=rmse_scorer['RMSE'], cv=kfold, flip_scores=True)

    ax = fig.add_subplot(n_rows, n_cols, index+1)
    plt.plot(results_df.TrainScores, color='b', label='Training')
    plt.plot(results_df.TestScores, color='r', label='Testing')
    
    plt.xlabel(parameter, size=14)
    plt.legend(loc='center left', bbox_to_anchor=(1.02,0.5), ncol=1, prop={'size': 14})
    
    index += 1

plt.tight_layout()
plt.show();


# From the figure above, we observe that as the max_leaf_nodes exceeds 60, as there is little improvement in testing data despite improving training performance. 
# 
# In similar fashion, a depth greater than 10 does not seem to improve the model's performance.
# 
# As for min_samples_split and min_samples_leaf, it appears that smaller values offer better performance. 
# 
# As for the max features to include, their performances are similar but taking the square roots seems to have marginally better performance.
# 
# This helps us shrink the grid search space as we are able to narrow our search to smaller range of values per parameter.

# In[ ]:


# New parameter tuning grid
rforest_tuning_grid = {
    'max_depth' : np.arange(8,13 , step=1),
    'max_features' : ['auto', 'sqrt', 'log2'],
    'min_samples_split' : np.arange(2, 6, step=1),
    'min_samples_leaf' : np.arange(1, 6, step=1),
}


# In[ ]:


# Dictionary where we will store the optimal kwargs for the tuned random forest model
rforest_kwargs = {
    'n_estimators' : 100,
    'criterion' : 'mse',
    'max_features' : 'sqrt',
    'bootstrap' : True,
    'n_jobs' : -1,
    'random_state' : SEED, 
}


# In[ ]:


np.random.seed(8888)
gs_rforest = GridSearchCV(estimator=RandomForestRegressor(**rforest_kwargs), param_grid=rforest_tuning_grid,
                          scoring=rmse_scorer['RMSE'], n_jobs=-1, cv=kfold, refit=True)

gs_rforest.fit(Xsmall_train, y_train)


# In[ ]:


# Saving the best estimator from the gird search
tuned_rforest = gs_rforest.best_estimator_

# Updating our random forest kwargs with the optimal hyper parameter values
rforest_kwargs.update(gs_rforest.best_params_)
rforest_kwargs


# Now that we have tuned the hyper parameters of the Random Forest Regression model, we can evaluate its performance against the initial untuned version as well as the linear models.

# In[ ]:


# Evaluating tuned random forest results
tuned_rf_result = holdout_set_evaluation(tuned_rforest, Xsmall_train, y_train, Xsmall_test, y_test, model_name='TunedRF', scoring_name='RMSE')


# In[ ]:


model_results = model_results.append(tuned_rf_result)
model_results


# From the table above, we see that the tuned Random Forest Model has outperformed the untuned version, albeit only slightly. This is because the tuned parameters were very similar to the default settings of the untuned model. 
# 
# However, we see that even after tuning, the Random Forest Regression does not outperform even the simple Linear Regression model. This shows tthat when modelling a datset, there is *__'No Free Lunch'__*! 
# 
# More advanced/complex models will not outperform simple models all the time!

# # Upcoming Sections
# 
# If you are reading this, thanks for making it this far! I hope this notebook has been helpful! (:
# 
# I plan to update this notebook over the next few days covering the sections:
# * Ridge Regression post feature engineering
# * Gradient Boosting 
# * Model Comparisons and Conclusion
# 
# Looking forward to any feedback / comments on ways to improve! If you have any questions, i'll try to answer to the best of abilities. (:
# 
# Thanks and have a great day!

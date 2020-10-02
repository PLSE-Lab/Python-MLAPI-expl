#!/usr/bin/env python
# coding: utf-8

# **THIS KERNEL IS A WORK IN PROGRESS**
# 
# # Exploratory analysis and feature engineering
# In this kernel I will put together code snippets and analysis taken from some other kernels, improve them when possible and also provide a bit of new insight. I will be using code coming from these kernels:
# * [Serigne - Stacked Regressions to predict House Prices](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook)
# * [Pedro Marcelino - Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
# 
# I will focus my explanations on whatever I am doing different than the other kernels, but I will also mention what I'm doing equally and why it is being done.
# 
# The kernel will be divided in the following sections:
# 1. Introduction: introducing the problem, dataset and variables.
# 2. EDA: understanding the dataset.
# 3. Feature Engineering: improve the statistical quality of the variables, in order to train better models.
# 4. Model training: Train different models and compare their performance.
# 5. Conclusions.
# 
# # Introduction
# 
# The task consists on predicting the price of houses in the USA based on a series of characteristics, either concerning the edification (e.g. YearBuilt, HouseStyle), the zone/neighborhood (e.g. Street, Neighborhood) or spatial measurements of the house and terrain (e.g. LotArea, GrLivArea). I will assume that you are familiar with the variable names and descriptions, which can be found right on the competition's homepage.
# 
# From a machine learning perspective it is a regression problem and the performance of a model can be measured for example through the root mean square error. In general it is a bit more difficult to determine if a model is "good" for regression problems, as the RMSE is relative to the dependent variable's scale. We will explore whether it makes sense to measure the relative error also.
# 
# Before starting let's import the necessary Python libraries and load the datasets:

# In[55]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # also for visualization
from scipy import stats # general statistical functions

import warnings
warnings.filterwarnings('ignore') # ignore warnings from the different libraries


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

import os
print(os.listdir("../input")) # check directory contents


# In[56]:


# Import and put the train and test datasets in pandas dataframes

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # EDA
# 
# Lets start by droping the "Id" variable and doing a distribution plot for the "SalePrice" variable.
# 

# In[57]:


# Drop the 'Id' colum since it's unnecessary for prediction process
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# In[58]:


# Distribution plot for SalePrice
from scipy.stats import norm
sns.distplot(train['SalePrice'] , fit=norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')


# We can see that the variable roughly resembles a normal distribution, but it's a bit skewed. Lets take a note about that and try to improve it in the next section.
# 
# Next I will analyze the dependence of the independent variables with "SalePrice". I will take a different approach and plot them all at once with a single function (one plot each). The idea is to visually inspect the empirical distributions and identify useless variables and outliers.

# In[59]:


# Determine which kind of variables are present

train.dtypes.unique()


# In[60]:


# Plotting function

def explore_variables(target_name, dt):
    for col in dt.drop(target_name, 1).columns:
        if train.dtypes[train.columns.get_loc(col)] == 'O': # categorical variable
            f, ax = plt.subplots()
            fig = sns.boxplot(x=col, y=target_name, data=dt)
            ax = sns.swarmplot(x=col, y=target_name, data=dt, color=".25", alpha=0.2)
            fig.axis(ymin=0, ymax=800000)
        else: # numerical variable
            fig, ax = plt.subplots()
            ax.scatter(x=dt[col], y=dt[target_name])
            plt.ylabel(target_name, fontsize=13)
            plt.xlabel(col, fontsize=13)
            plt.show()


# In[61]:


explore_variables('SalePrice', train)


# Wow, that was an overwhelming amount of information. Notice that some variables were read as numerical, eventhough they are categorical in nature. However, the scatterplots obtained from these variables were fairly useful, so we will leave them like they are.
# 
# I have noticed the following:
# 
# #### Outliers in numerical variables
# These variables have some points that are sitting alone and don't follow the general trend. We will probably drop these **observations** (if they are few in total):
# * LotFrontage
# * LotArea
# * BsmtFinSF1
# * TotalBsmtSF
# * 1stFlrSF
# * GrLivArea
# 
# #### Useless variables
# The vast majority of the observations of these variables belong to the same class or value. They simply don't have enough observations in different classes to be able to generalize. Better drop them:
# * PoolArea
# * PoolQC
# * Street
# * Condition2
# * RoofMatl
# * Heating
# * MiscFeature
# * MiscVal
# * Utilities
# 
# #### Outliers in categorical variables
# There are a lot of "outliers" (actually, extreme values would be more correct) in each of the boxplots. However, take into account that we are only plotting two variables each time, so that the extreme values could be explained by others variables. Additionally, they are important in number, so we will include them in the modeling.
# 
# Lets see how small gets the dataset after dropping the extreme observations:

# In[62]:


print("Original size: {}".format(train.shape))

# Drop extreme observations
conditions = [train['LotFrontage'] > 250,
             train['LotArea'] > 100000,
             train['BsmtFinSF1'] > 4000,
             train['TotalBsmtSF'] > 5000,
             train['1stFlrSF'] > 4000,
             np.logical_and(train['GrLivArea'] > 4000, train['SalePrice'] < 300000)]

print("Outliers: {}".format(sum(np.logical_or.reduce(conditions))))


# # Preprocessing
# 
# We can then proceed to drop the outliers and useless variables.

# In[63]:


# drop outliers
train = train[np.logical_or.reduce(conditions)==False]


# In[64]:


# drop useless variables
train.drop(labels=['PoolArea', 
                   'PoolQC', 
                   'Street', 
                   'Condition2', 
                   'RoofMatl', 
                   'Heating', 
                   'MiscFeature', 
                   'MiscVal', 
                   'Utilities'], axis=1, inplace=True);


# Next we will analyze how many missing values do we have in the dataset, like in the kernel from Marcelino,

# In[65]:


#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# Then we have the following options for each of the variables:
# 1. Drop the observations (can lead to a large data loss).
# 2. Drop the variable.
# 3. Impute missing values.
# 
# We will look at them case by case (some of them as groups). First 'Alley':

# In[66]:


train['Alley'].value_counts()


# Most of the observations are NAs, i.e. no alley access. It is safe to drop the variable.

# In[67]:


train.drop('Alley', axis=1, inplace=True)


# Lets continue with 'Fence':

# In[68]:


train['Fence'].value_counts()


# In this case NAs actually mean no fence. We could introduce a new value 'NoFence' for the NAs and all other values could be grouped together as 'Fence', obtaining a binary variable. Lets do that.

# In[69]:


# replace NaNs

values = {'Fence': 'NoFence'}
train.fillna(value=values, inplace=True)


# In[70]:


# combine other values into one

train.loc[train['Fence'] != 'NoFence', 'Fence'] = 'Fence'


# Lets continue with 'FireplaceQu':

# In[71]:


train['FireplaceQu'].value_counts()


# In this case NA means no fireplace. Lets fill the NAs with 'NoFireplace', we will worry about the encoding in the next section. We will also drop the 'Fireplaces' variable, because the quality variable contains the interesting information (whether a house has a fireplace and of which quality)

# In[72]:


# replace NaNs

values = {'FireplaceQu': 'NoFireplace'}
train.fillna(value=values, inplace=True)


# In[73]:


# drop 'Fireplaces'

train.drop('Fireplaces', axis=1, inplace=True)


# Lets continue with 'LotFrontage'. In this case we have a numerical variable with missing values, and a significant amount at that. We could drop the observations, but that would lead to a significant loss of data. We will then impute the NAs with either the mean or the median:

# In[74]:


sns.distplot(train['LotFrontage'].dropna())

print("Mean: {}. Median: {}".format(np.mean(train['LotFrontage']), np.median(train['LotFrontage'].dropna())))


# Mean and median are very near to one another. Lets just use the median:

# In[75]:


# replace NaNs

values = {'LotFrontage': np.median(train['LotFrontage'].dropna())}
train.fillna(value=values, inplace=True)


# Lets continue with the Garage* variables. In all of them NA means no garage, we can then impute the missing values. 
# 
# 'GarageYrBlt' seems kind of irrelevant, because it should not be different in most cases to that of the building, therefore we will drop it.

# In[76]:


# drop 'GarageYrBlt'

train.drop('GarageYrBlt', axis=1, inplace=True)


# In[77]:


# replace NaNs

values = {var:'NoGarage' for var in ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']}
train.fillna(value=values, inplace=True)


# Lets continue with the Bsmt* variables. In all of them NA means no basement, we can then impute the missing values. I am impressed about most buildings having a basement!

# In[78]:


# replace NaNs

values = {var:'NoBsmt' for var in ['BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']}
train.fillna(value=values, inplace=True)


# Lets continue with MasVnr*. In these case the NAs are real missing data, but we can impute them as type 'None'/0, which is the most frequent value.

# In[79]:


# replace NaNs

values = {'MasVnrType': 'None', 'MasVnrArea':0}
train.fillna(value=values, inplace=True)


# We are left with 'Electrical'. There is one real NA that we will impute according to the empirical distribution of the variable. Lets take a look:

# In[80]:


train['Electrical'].value_counts()


# Here the two main options are 'standard circuit breaker' and 'fuse box'. Lets impute the missing value and make the variable binary.

# In[81]:


# replace NaNs

values = {'Electrical': 'SBrkr'}
train.fillna(value=values, inplace=True)


# In[82]:


# combine other values into one

train.loc[train['Electrical'] != 'SBrkr', 'Electrical'] = 'Fusebox'


# Lets check if we still have missing values:

# In[83]:


#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# # Feature Engineering
# 
# Now we will proceed to further transform variables that have a not very useful empirical distribution, trying to make them useful for statistical learning. In the process we will create new variables. I will be using the EDA plots and empirical distributions.
# 
# In other kernels I have seen that they analize the normality in the distribution of all variables, including SalePrice. However, I have to say that not all models care about this, as in the case of SVM or tree-based models. It is only important if the objective is to do inference, and even then only the residuals should be analyzed. However, in the case of the dependent variable it can be appropiate, if there is a underlying reason. In our case it is logical to assume that the different observations for a house have an porcentual effect on the base price of the house. For example, having a central heating will increase the price of a house normally costing 100k to 110k, while it will increase a house costing 50k to 55k (i.e. 10% increase). That behaviour can be modeled easier with the log of the variable.
# 
# In previous versions of this kernel there was QQ-Plot and empirical distribution analysis of the price and log(price), but as said earlier that doesn't make sense. We will try transformed and untransformed variants in the modeling part (next kernel).

# Now we will transform some of the nominal variables so that they have less categories. It will however not be exhaustive, there will be variables with inconvenient distributions left. Hopefully the models won't get confused. 
# 
# Lets just plot and modify the variables:

# In[84]:


explore_variables('SalePrice', train.loc[:, ['SalePrice', 'LotShape'] ])


# In[85]:


# Turn irregular lot shapes into one class

train.loc[train['LotShape'] != 'Reg', 'LotShape'] = 'Irregular'


# In[86]:


explore_variables('SalePrice', train.loc[:, ['SalePrice', 'LandContour'] ])


# In[87]:


# Turn non leveled land contours into one class

train.loc[train['LandContour'] != 'Lvl', 'LandContour'] = 'Unleveled'


# In[88]:


explore_variables('SalePrice', train.loc[:, ['SalePrice', 'LotConfig'] ])


# In[89]:


# Turn FR* observations into one class

train.loc[np.logical_or(train['LotConfig'] == 'FR2', train['LotConfig'] == 'FR3'), 'LotConfig'] = 'FR'


# In[90]:


explore_variables('SalePrice', train.loc[:, ['SalePrice', 'LandSlope'] ])


# In[91]:


# Turn Severe slopes into Moderate (there were too few)

train.loc[train['LandSlope'] == 'Sev', 'LandSlope'] = 'Mod'


# In[92]:


explore_variables('SalePrice', train.loc[:, ['SalePrice', 'MasVnrType'] ])


# In[93]:


# Turn brick veneer types into one class (for example BrkCmn)

train.loc[train['MasVnrType'] == 'BrkFace', 'MasVnrType'] = 'BrkCmn'


# In[94]:


explore_variables('SalePrice', train.loc[:, ['SalePrice', 'ExterCond'] ])


# In[95]:


# Put poor and excellent external conditions into other conditions

train.loc[train['ExterCond'] == 'Po', 'ExterCond'] = 'Fa'
train.loc[train['ExterCond'] == 'Ex', 'ExterCond'] = 'Gd'


# In[96]:


# incorporate log(1+x) transformation !NOT ANYMORE!

# train["SalePrice"] = np.log1p(train["SalePrice"])


# ## Data Cleaning and Formatting
# 
# Now we will correct some data format problems.

# In[97]:


# Transform nominal variables that were read as numeric back into nominal

#MSSubClass=The building class
train['MSSubClass'] = train['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
train['OverallCond'] = train['OverallCond'].astype(str)


#Month sold transformed into categorical feature.
train['MoSold'] = train['MoSold'].astype(str)


# In[98]:


# transform year variables and drop not useful ones

train['SoldYrAgo'] = 2019 - train['YrSold']
train.drop('YrSold', axis=1, inplace=True)

train['BuiltYrAgo'] = 2019 - train['YearBuilt']
train.drop('YearBuilt', axis=1, inplace=True)

train.drop(labels=['YearRemodAdd', 'GarageFinish'], axis=1, inplace=True)


# Now we will see if we can drop or put together features that are strongly correlated with one another.

# In[99]:


#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True);


# In[100]:


# Lets see which correlations are very large in absolute value

corrmat = train.corr()
corrmat[abs(corrmat) > 0.7] = 1
corrmat[abs(corrmat) <= 0.7] = 0

plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True);


# Strong correlations with SalePrice are actually good for a future prediction. Then we obtain the following pairs of correlated variables:
# * GarageCars - GarageArea: We can drop GarageArea, as it is proportional to the number of cars.
# * TotalBsmtSF - 1stFlrSF: Here it seems that when there is a basement, its area correlates to the ground floor area, not a big surprise. I think that the presence or ausence of the basement is more important than its area, therefore we can drop TotalBsmtSF.
# * TotalRmsAbvGrd - GrLivArea: The correlation is quite simple, more rooms translate generally to a greater living area. It may be convenient to create a new variable AreaPerRoom that may provide some aditional information.
# 
# Lets do this changes.

# In[101]:


# drop GarageArea
train.drop('GarageArea', axis=1, inplace=True)


# In[102]:


# drop TotalBsmtSF
train.drop('TotalBsmtSF', axis=1, inplace=True)


# In[103]:


# create AreaPerRoom
train['AreaPerRoom'] = train['GrLivArea'] / train['TotRmsAbvGrd']


# Now we will continue to encode the nominal variables. In some kernels I have seen the use of the LabelEncoder for ordinal variables. In my opinion this is a big risk, because there is no real underlying scale (e.g. we cannot say "Fair" * 2 = "Good"). I will stick to dummy coding and let the models figure out the ordinality.

# In[104]:


# dummy coding

train = pd.get_dummies(train)


# In[105]:


train.shape


# Finally lets save the modified dataset, we will use it in the next kernel for modelling.

# In[106]:


train.to_csv('preprocessed_training.csv', index=False)


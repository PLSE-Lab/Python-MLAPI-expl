#!/usr/bin/env python
# coding: utf-8

# # Kaggle Competition - House Prices: Advanced Regression Techniques

# - *Author: Annie Pi*
# - *Last Updated: April 11, 2018*

# This notebook attempts to minimize RMSE using only feature engineering and regression methods. It uses the House Prices dataset from this Kaggle competition: <https://www.kaggle.com/c/house-prices-advanced-regression-techniques>.
# 
# By primarily focusing on N/A imputation (section 4), I was able to achieve a final score of 0.11582 (top 7% of the competition).
# 
# The sections in the notebook are as follows:
# 1. Data Reading and Preparation
# 2. Inspecting Sale Price
# 3. Removing Sale Price Outliers
# 4. Inspecting Unusual and Missing Values
# 5. Factorize Features
# 6. Correcting Skewness
# 7. Creating Additional Variables
# 8. Dummifying Variables
# 9. Resplitting
# 10. Checking for Significant Outliers
# 11. Modeling and Prediction

# ## 1. Data Reading and Preparation 

# To begin with, I start by loading libraries that will be used throughout the notebook and then read and inspect my dataset.

# In[1]:


# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy.stats import skew
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option("max_columns", None)


# In[2]:


# Load data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


# Check for duplicates and sizes of the training and testing set
print("Training")
print(train.shape)
print(len(train.Id.unique()))
print("\n")
print("Testing")
print(test.shape)
print(len(test.Id.unique()))


# There are no duplicates, so later I will drop the ID column. 

# ## 2. Inspecting Sale Price

# Before doing anything else, I want to inspect the target variable, Sale Price.

# In[9]:


# Plot histogram of SalePrice and log(SalePrice)
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
sns.distplot(train.SalePrice)
plt.subplot(1,2,2)
sns.distplot(np.log(train.SalePrice))


# As the two histograms show, Sale Price is skewed to the left, but by applying a log transformation to it, we can make it more normally distributed, which we need for a linear model. So I go ahead and take the log plus one.

# In[10]:


train['SalePrice'] = np.log1p(train['SalePrice'])


# ## 3. Removing Sale Price Outliers

# Since I am limited to regression models, which tend to be heavily unfluenced by outliers. I look first at what variables are strongly correlated with Sale Price.

# In[11]:


# Create Saleprice correlation matrix
k = 10 #number of variables for heatmap
corrmat = train.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# I plot the two most highly correlated variables, OverallQual and GrLiveArea, against Sale Price. 

# In[12]:


# Plot grlivarea vs. saleprice and overallqual vs. saleprice
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(train.OverallQual, train.SalePrice, marker = "s")
plt.title("OverallQual vs. SalePrice")
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")

plt.subplot(1,2,2)
plt.scatter(train.GrLivArea, train.SalePrice, marker = "s")
plt.title("GrLivArea vs. SalePrice")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")


# In general, it seems that OverallQual follows a steady pattern with SalePrice. However, GrLvArea appears to have two points that don't follow the same pattern where GrLvArea is greater than 4000, yet SalePrice is around the same price as properties with half the area. So, I remove these two points.

# In[13]:


# Drop two outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<12.5)].index)


# ## 4. Inspecting Unusual and Missing Values

# So that I don't have to repeat the same steps for the train and test data, I first combine the two data sets into one dataset and save the IDs and shapes that I can separate them again later.

# In[14]:


# Save ID values
train_ID = train['Id']
test_ID = test['Id']

# Save shapes
ntrain = train.shape[0]
ntest = test.shape[0]

# Store Sale Price as target value
y_target = train.SalePrice.values

# Combine train and test into new dataset called Combined and drop SalePrice
combined = pd.concat((train, test)).reset_index(drop=True)
combined.drop(['SalePrice'], axis=1, inplace=True)
print("combined size is : {}".format(combined.shape))


# I inspect the numerical variables to see if there are any unusual values.

# In[16]:


# Select only numerical variables
numerical_features = combined.select_dtypes(exclude = ["object"])

# Inspect summary statistics for each numerical variable
numerical_features.describe()


# Most of the values look fairly typical, but I see the max for GarageYrBlt is 2207, which seems like a typo since this is almost 200 years into the future.

# In[810]:


combined[combined["GarageYrBlt"] == 2207]


# By taking a closer look at this value, I see that the house was remodeled in 2007 and 2207 seems likely a typo of this year, so I update the value to 2007.

# In[811]:


combined.loc[combined["GarageYrBlt"] == 2207,'GarageYrBlt'] = 2007 


# I also need to address any NA values before I can start working on my model, as missing values will lower its predictive power.

# In[812]:


# NA discovery
naCols = combined.isnull().sum()[combined.isnull().sum()>0]
print(naCols)
print("\n")
print("There are " + str(len(naCols)) + " columns with missing values.")


# Before I start filling in missing values, I see that there are many columns with missing values that form natural groups -- for example, there are a number of variables all related to Garage. I would expect if there is no Garage that *all* garage variables would be empty or "N/A". To verify if this is true, I create a function to pass in a list of variables and count how many missing values there are by row.
# 
# If the number of missing values is greater than 0, but less than the number of variables I pass in, I flag the row for further inspection.

# In[813]:


# Create function to detect rows with missing values greater than 0, but less than number of variables being passed
def countMissing(var):
    print(var)
    for index, row in combined.iterrows():
        missing = 0
        for i in var:
            if (type(row[i]) == float) or (type(row[i]) == int):
                if (row[i] == 0):
                    missing += 1 
                else:
                    if (np.isnan(row[i])) or (row[i] == "None"):
                        missing += 1
            else:
                if (row[i] == "None"):
                    missing += 1
        if (missing > 0) and (missing < len(var)):
            print("Index: " + str(index) + " Number Missing: " + str(missing))
    print("\n")


# In[814]:


# Create lists of related variables
poolVariables = ["PoolQC", "PoolArea"]
basementVariables = ["BsmtQual", "BsmtCond", "BsmtExposure", "TotalBsmtSF"]
garageVariables = ["GarageType", "GarageYrBlt", "GarageArea"]
veneerVariables = ["MasVnrArea", "MasVnrType"]
roofVariables = ["RoofStyle", "RoofMatl"]
kitchenVariables = ["KitchenAbvGr", "KitchenQual"]
fireplaceVariables = ["Fireplaces", "FireplaceQu"]

# Call function on all lists of related variables
for i in (poolVariables, basementVariables, garageVariables, veneerVariables, roofVariables, kitchenVariables, fireplaceVariables):
    countMissing(i)


# I see there are many rows being flagged and I start by examining the issues with pool variables.

# In[815]:


# Inspect rows flagged for missing Pool variables
combined.loc[[2418,2501,2597]]


# I see that there are three rows with a PoolArea and no PoolQc. As having a specific number for PoolArea seems unlikely to be something that happened by accident, I assume a pool exists, and try to figure out how to impute PoolQC by looking at existing PoolQC values.

# In[819]:


# Inspect rows with PoolArea and PoolQC filled in
combined[(combined["PoolArea"].notnull()) & (combined["PoolArea"] != 0)]


# Looking at the OverallQual and the PoolQC, there seems to be some correlation. For example, OverallQual of 8 and above has an Ex (Excellent) PoolQC. And the 6 or 7 ratings alternate between Fa (Fair) and Gd (Good). There are too few values to establish a definite relationship, but it makes sense that OverallQual of the house might say something about the quality of the pool as well, so I use this value to impute the three missing values for PoolQC. 

# In[820]:


# Impute missing values for PoolQC
combined.loc[2418, 'PoolQC'] = 'Fa' 
combined.loc[2501, 'PoolQC'] = 'Gd' 
combined.loc[2597, 'PoolQC'] = 'Fa'


# Next, looking at the basement variables, I see there are several rows that are missing BsmtExposure, BsmtCond, or BsmtQual, even though other basement columns are filled out.

# In[821]:


# Inspect rows flagged for missing Basement variables
combined.loc[[947,1485,2038,2183,2215,2216,2346,2522]]


# In[822]:


# Inspect values for BsmtCond, BsmtExposure, and BsmtQual
print(combined['BsmtCond'].value_counts())
print("\n")
print(combined['BsmtQual'].value_counts())
print("\n")
print(combined['BsmtExposure'].value_counts())


# I look at the value counts for these three variables and see that a majority of basements are TA (Typical) condition and have no exposure, so I can fill in BsmtCond and BsmtQual using the mode. However, BsmtQual is split fairly evenly between TA (Typical) and Gd (Good) values, so for the two missing values I look again at OverallQual and see that it is 4 for these two rows, so I go with TA (Typical). 

# In[823]:


# Impute missing Basement values
combined.loc[947, 'BsmtExposure'] = 'No' 
combined.loc[1485, 'BsmtExposure'] = 'No' 
combined.loc[2346, 'BsmtExposure'] = 'No'
combined.loc[2038, 'BsmtCond'] = 'TA'
combined.loc[2183, 'BsmtCond'] = 'TA'
combined.loc[2346, 'BsmtCond'] = 'TA'
combined.loc[2215, 'BsmtQual'] = 'TA'
combined.loc[2216, 'BsmtQual'] = 'TA'


# In[824]:


# Inspect rows flagged for missing Veneer variables
combined.loc[[623,687,772,1229,1240,1298,1332,1667,2317,2450,2608]]


# There are a couple of different cases with Veneer.
# 
# 1. Rows with MasVnrArea but no MasVnrType - as with Pool, I assume MasVnrArea was not entered by mistake and MasvnrType should not be "None", so I impute "BrkFace" for MasVnrType (the second most common value after None).
# 
# 2. Rows with no MasVnrArea but a MasVnrType - I impute the mean MasVnrArea for the MasVnrType.
# 
# 3. Rows with 1 for MasVnrArea and None for MasVnrType. It seems unlikely that the square area in feet is 1, so I assume there is no veneer and change MasVnrArea to 0.

# In[825]:


# Show summary statistics for MasVnrArea by MasVnrType
combined.groupby(['MasVnrType'])['MasVnrArea'].describe()


# In[826]:


# Fill in rows with MasVnrArea but no MasVnrType
for i in (623,2608,1298,1332,1667):
    combined.loc[i, "MasVnrType"] = "BrkFace"

# Fill in rows with no MasVnrArea but a MasVnrType
combined.loc[687, 'MasVnrArea'] = 261.67
combined.loc[1240, 'MasVnrArea'] = 247
combined.loc[2317, 'MasVnrArea'] = 261.67

#Fill in rows with 1 for MasVnr area and None for MAsVnrType
for i in (772,1249,2450):
    combined.loc[i, "MasVnrArea"] = 0


# In[827]:


# Inspect rows flagged for missing Garage variables
combined.loc[[2124,2574]]


# Now looking at the flagged rows for garage, the first row appears to have an actual garage, as GarageArea is filled out as well as GarageCars. Therefore, I impute the values based on the YearBuilt or mode values. However, the second row has no other values filled out for Garage, so I set the GarageType to None. 

# In[828]:


# Impute missing Garage values
combined.loc[2124, 'GarageYrBlt'] = combined.loc[2124, 'YearBuilt']
combined.loc[2124, 'GarageCond'] = combined['GarageCond'].mode()[0]
combined.loc[2124, 'GarageFinish'] = combined['GarageFinish'].mode()[0]
combined.loc[2124, 'GarageQual'] = combined['GarageQual'].mode()[0]

combined.loc[2574, 'GarageType'] = 'None' 


# In[829]:


# Inspect rows flagged for missing Kitchen values
combined.loc[[953,1553,2585,2857]]


# There are rows with TA quality but 0 kitchens - I assume these should actually be 1 kitchen, as this is by far the most common number of kitchens. For KitchenQual, it's again split between TA and Gd, but I fill in TA as OverallQual is a 5. 

# In[830]:


# Inspect value counts for KitchenAbvGr and KitchenQual
print(combined["KitchenAbvGr"].value_counts())
print("\n")
print(combined["KitchenQual"].value_counts())


# In[831]:


# Impute missing KitchenAbvGr values
for i in (953,2585,2857):
    combined.loc[i, "KitchenAbvGr"] = 1

# Impute missing KitchenQual values
combined.loc[1553, "KitchenQual"] = "TA"


# Now that we know have corrected many erroneous missing values, we can use the Data Description from Kaggle to automatically fill in some of the remaining missing values.  

# In[832]:


# Impute missing values based on Data Description

# Alley: Data description says NA means 'no alley access'
combined["Alley"].fillna("None", inplace=True)

# Bsmt : Data description says NA for basement features is "no basement"
combined["BsmtQual"].fillna("None", inplace=True)
combined["BsmtCond"].fillna("None", inplace=True)
combined["BsmtExposure"].fillna("None", inplace=True)
combined["BsmtFinType1"].fillna("None", inplace=True)
combined["BsmtFinType2"].fillna("None", inplace=True)

# Fence : Data description says NA means "no fence"
combined["Fence"].fillna("None", inplace=True)

# FireplaceQu : Data description says NA means "no fireplace"
combined["FireplaceQu"].fillna("None", inplace=True)

# Garage : Data description says NA for garage features is "no garage"
combined["GarageType"].fillna("None", inplace=True)
combined["GarageFinish"].fillna("None", inplace=True)
combined["GarageQual"].fillna("None", inplace=True)
combined["GarageCond"].fillna("None", inplace=True)

# MiscFeature : Data description says NA means "no misc feature"
combined["MiscFeature"].fillna("None", inplace=True)

# PoolQC : Data description says NA means "no pool"
combined["PoolQC"].fillna("None", inplace=True)

# Print remaining variables with missing values
naCols = combined.isnull().sum()[combined.isnull().sum()>0]
print(naCols)
print("\n")
print("There are " + str(len(naCols)) + " columns with missing values.")


# For the remaining columns, we have to make assumptions about the missing values. I check BsmtFullBath and see that BsmtCond is "None" so there is no basement. In the same row, I also see that TotalBsmtSF, BsmtSinF1, and BsmtSinF2 show up as NA, so I replace these with 0.
# 
# I assume the same thing is happening with the NA values of GarageArea and GarageCars, so I replace these with 0 as well.

# In[833]:


# Check rows where BsmtFullBath is null
combined[combined.BsmtFullBath.isnull()]


# In[834]:


# Impute missing values for Basement and Garage variables
combined["BsmtFullBath"].fillna(0, inplace=True)
combined["BsmtHalfBath"].fillna(0, inplace=True)
combined["BsmtFinSF1"].fillna(0, inplace=True)
combined["BsmtFinSF2"].fillna(0, inplace=True)
combined["BsmtUnfSF"].fillna(0, inplace=True)
combined["TotalBsmtSF"].fillna(0, inplace=True)
combined["GarageArea"].fillna(0, inplace=True)
combined["GarageCars"].fillna(0, inplace=True)

# Print remaining variables with missing values
naCols = combined.isnull().sum()[combined.isnull().sum()>0]
print(naCols)
print("\n")
print("There are " + str(len(naCols)) + " columns with missing values.")


# To fill in MSZoning, I look at a related field: MSSubClass. It appears that for different MSSubClasses, different MSZoning values appear more frequently. Therefore, I use the most frequent value by MSSubClass to impute values for MSZoning.

# In[835]:


# Check most common MSZoning values for each MSSubClass
combined.groupby(['MSSubClass'])['MSZoning'].describe()


# In[836]:


# Inspect missing MSZoning values
combined[combined["MSZoning"].isnull()]


# In[837]:


# Impute missing MSZoning values based on mode by MSSubClass
combined.loc[1913,"MSZoning"] = "RM"
combined.loc[2214,"MSZoning"] = "RL"
combined.loc[2248,"MSZoning"] = "RM"
combined.loc[2902,"MSZoning"] = "RL"


# For the columns that only have 1 or 2 missing values, I check the distributions of values, and it seems most of the columns have one value that appears much more frequently than the others. For example in "Electrical," "SBrkr" appears 91.5% of the time. 

# In[838]:


# Check value counts for variables with only 1 or 2 missing variables
for col in ('Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'SaleType', 'Utilities', 'KitchenQual'):
    print(combined[col].value_counts())
    print("\n")


# I also notice looking at the distributions that Utilities has 1 value for "NoSeWa" and the rest are "AllPub." Since all the values are basically the same and only the training or test set will have "NoSeWa," I decide to drop this from the dataset.
# 
# For the rest of the columns, I fill in missing values with the mode.`

# In[839]:


# Drop utilities
combined = combined.drop(['Utilities'], axis=1)

# Fill in mode for other variables
for col in ('Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'SaleType', 'KitchenQual'):
    combined[col].fillna(combined[col].mode()[0], inplace=True)
    
# Print remaining variables with missing values
naCols = combined.isnull().sum()[combined.isnull().sum()>0]
print(naCols)
print("\n")
print("There are " + str(len(naCols)) + " columns with missing values.")


# Looking at the results where GarageYrBlt is NA, I see that the other Garage columns have been filled with "None" so I replace GarageYrBlt with 0. 

# In[840]:


combined[combined["GarageYrBlt"].isnull()]


# In[841]:


# Impute missing values for GarageYrBuilt
combined["GarageYrBlt"].fillna(0, inplace=True)


# Finally, I look at LotFrontage, which is defined as the linear feet of street connected to the property.

# In[842]:


# Plot distribution of LotFrontage
sns.distplot(combined['LotFrontage'].dropna());


# Looking at the distribution of values, there is a wide range of values from roughly 25 to 150, with a few values even above 300. So I don't want to impute a single value for all the missing ones. I try instead to split up the dataset into groups by neighborhood. 

# In[843]:


# Show summary statistics for LotFrontage by Neighborhood
combined.groupby("Neighborhood")['LotFrontage'].describe()


# By neighborhood, there appear to be narrower ranges of lot frontage, so I impute based on the median within each neighborhood.

# In[844]:


# Impute median LotFrontage by neighborhood for missing values
combined["LotFrontage"] = combined.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# For MasVnrArea and MasVnrType, I already checked for discrepancies between missing value sand corrected these. So I assume for the rest of the missing values, there is no veneer and MasVnrArea is 0 and MasVnrType is None.

# In[845]:


# Impute missing values for MasVnrArea and MasVnrType
combined["MasVnrArea"].fillna(0, inplace=True)
combined["MasVnrType"].fillna("None", inplace=True)


# In[846]:


# Print remaining variables iwth missing columns
naCols = combined.isnull().sum()[combined.isnull().sum()>0]
print(naCols)
print("\n")
print("There are " + str(len(naCols)) + " columns with missing values.")


# Now there are no more missing values!

# ## 5. Factorize Features

# Some numerical features are actually categorical. As this affects how the features are treated in a regression model, I recode two numerical features, MSSubClass and MoSold, as strings. 

# In[847]:


# Replace values for MSSubclass and MoSold
combined = combined.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })


# ## 6. Correcting Skewness

# Since Sale Price was skewed, it is very possible that other variables are skewed as well. I check for variables where skewness is above 0.75.

# In[848]:


skew = combined.skew(numeric_only=True) > 0.75

print(skew[skew==True])
print("\n")
print("There are " + str(len(skew[skew==True])) + " skewed variables.")


# In[849]:


# Apply log transformation to skew values in training
from scipy.special import boxcox1p

for index_val, series_val in skew.iteritems():
    if series_val == True:
        combined[index_val] = boxcox1p(combined[index_val], 0.15)
        
skew = combined.skew(numeric_only=True) > 0.75

# Print remaining skew variables
print(skew[skew==True])
print("\n")
print("There are " + str(len(skew[skew==True])) + " skewed variables.")


# There are still 9 skewed variables, even after applying a log transformation, so I inspect these in a histogram.

# In[850]:


plt.figure(figsize=(20,10))

# Plot histogram for remaining skew variables
for col in ('3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath', 'EnclosedPorch', 'KitchenAbvGr', 'LowQualFinSF', 'MiscVal', 'PoolArea', 'ScreenPorch'):
    plt.figure()
    sns.distplot(combined[col])


# Most of the histograms show skewness because of values at 0, which is hard to correct with a transformation. I may consider dropping some of these variables later, but for now I leave them alone.

# ## 7. Creating Additional Variables

# It may be interesting to create new variables for the model that can impact SalePrice. I create one that sees how many years have passed between a house being remodelled and sold, as it seems logical that more recently remodelled houses will sell for higher.

# In[851]:


combined["YearsSinceRemodelled"] = combined["YrSold"] - combined["YearRemodAdd"]


# For another data quality check, I see if there are any negative values for this variable.

# In[852]:


# Check if YearsSinceRemodelled is less than 0
combined[combined["YearsSinceRemodelled"] < 0]


# There are two houses that say they were remodelled after they were sold (and even one that says it was built after it was sold). I assume these are mistakes and change the YrSold to the YearRemodAdd year.

# In[853]:


# Update YrSold for two rows
combined.loc[2293, 'YrSold'] = 2008
combined.loc[2547, 'YrSold'] = 2009


# There are many square footage variables in the dataset, yet no overall square footage, so I create a totalSF variable, combining the basement SF with the 1st and 2nd floor SF. 

# In[854]:


# Create new variable: total square footage
combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']


# ## 8. Dummifying Variables

# Since I cannot have strings for my linear model, I have to convert all my categorical variables into dummies.

# In[855]:


# Concatenate categorical and numerical features
print("Previous number of variables: " + str(len(combined.columns)))
combined = pd.get_dummies(combined)
print("New number of variables: " + str(len(combined.columns)))


# I have created over 200 new variables by dummifying categorical variables and some of these may be insignificant. So I create a function to test for which variables are 99.9% zero values and drop these from my dataset to help reduce overfitting.

# In[856]:


# Create function to check for variables with near zero variance (99.9% zero values)
def countZeroes(var):
    nearZeroVariables = []
    for i in var:
        zeroValues = 0
        for index, row in combined.iterrows():
            if row[i] == 0:
                zeroValues += 1
        if zeroValues > 0.999 * len(combined):
            nearZeroVariables.append(i)
            print("Variable " + str(i) + ": " + str(zeroValues))
    combined.drop(nearZeroVariables, axis=1, inplace=True)


# In[859]:


# Run function on all columns and drop ones with 99.9% zero values
colnames = list(combined)
countZeroes(colnames)


# From running this entire notebook and checking my lasso coefficients, I see that some variables are being used in my model with a high coefficient and are also nearly all zero values, so I drop these too to prevent overfitting.

# In[860]:


# Drop additional variables to prevent overfitting
combined.drop(["MSZoning_C (all)", "Condition2_PosN", "MSSubClass_SC160", "Street_Grvl", "Street_Pave"], axis=1, inplace=True)


# In[861]:


# Display how many columns are remaining
len(combined.columns)


# ## 9. Re-splitting

# Now that I have completed my feature engineering, I can split my combined dataset back into training and test, and delete the ID column.

# In[862]:


# Split into train and test again and delete ID
train = combined[:ntrain]
test = combined[ntrain:]

print(train.shape)
print(test.shape)

del train["Id"]
del test["Id"]


# ## 10. Checking for significant outliers

# One last step before I run my models: I check for significant outliers in my residuals that may influence my results and drop these from the dataset.

# In[863]:


# Check for outliers
import statsmodels.api as sm

ols = sm.OLS(endog = y_target, exog = train)
fit = ols.fit()
otest = fit.outlier_test()['bonf(p)']

outliers = list(otest[otest<1e-3].index) 

outliers


# In[864]:


# Drop outliers from train
for index in sorted(outliers, reverse=True):
    train = train.drop([index])


# In[865]:


# Drop outliers from y_target (SalesPrice)
y_target = np.delete(y_target, outliers)


# ## 11. Modeling and Prediction

# Now I am finally ready to model. As previously stated, I am limited to using only regression models for this particular project, so I use a function to check my local RMSLE and then create a Lasso, Ridge, and Elastic Regression model to compare results.
# 
# Note: A full linear regression model was initially tested, but discarded due to a very high RMSE.

# In[4]:


# Import libraries for modeling
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
from vecstack import stacking


# In[6]:


# Local validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_target, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[7]:


# Lasso model
lasso = make_pipeline(RobustScaler(), LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[870]:


# Ridge model
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60]))
score = rmsle_cv(ridge)
print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[871]:


# Elastic net model
ENet = make_pipeline(RobustScaler(), ElasticNetCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# I want to check what variables my lasso model picked, so I fit it and then plot the top coefficients.

# In[872]:


# Fit lasso to train and y_target
lasso2 = lasso.fit(train, y_target)


# In[873]:


# Plot feature importance in lasso model
coef = pd.Series(lasso2.steps[1][1].coef_, index = train.columns)

imp_coef = pd.concat([coef.sort_values().head(15),
                     coef.sort_values().tail(15)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


# I do the same for Ridge and Elastic.

# In[874]:


# Plotting feature importance in Ridge Model

ridge2 = ridge.fit(train, y_target)
coef = pd.Series(ridge2.steps[1][1].coef_, index = train.columns)

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model")


# In[875]:


# Plotting feature importance in Elastic Model

ENet2 = ENet.fit(train, y_target)
coef = pd.Series(ENet2.steps[1][1].coef_, index = train.columns)

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Elastic Net Model")


# The three models have similar coefficients, but some differences, so it may help to combine the three instead of just relying on one. So I use the vecstack library to stack my three models and fit a 2nd level model. 

# In[876]:


# Select models for stacking
models = [lasso, ridge, ENet]
    
# Compute stacking features
S_train, S_test = stacking(models, train, y_target, test, 
    regression = True, metric = mean_absolute_error, n_folds = 4, 
    shuffle = True, random_state = 0, verbose = 2)

# Initialize 2nd level model
model = make_pipeline(RobustScaler(), ElasticNetCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], l1_ratio=.9, random_state=3))
    
# Fit 2nd level model
model = model.fit(S_train, y_target)

# Predict
y_pred = np.expm1(model.predict(S_test))


# In[877]:


# Test stacked model RSME
from sklearn.model_selection import cross_val_predict

score_preds = cross_val_predict(model, X=S_train, y=y_target)
print("stacked RMSE = ", np.sqrt(np.mean((y_target - score_preds)**2)))


# I can see that my local RMSE, 0.1029, for my stacked model is indeed lower than the local RMSEs for my three regression models individually. Therefore, I proceed with my stacked model and use it to generate my final predictions and submission.

# In[878]:


# Check my Sales Price predictions before submitting
y_pred


# In[791]:


# Create submission
submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = y_pred
submission.to_csv('submission_v16.csv',index=False)


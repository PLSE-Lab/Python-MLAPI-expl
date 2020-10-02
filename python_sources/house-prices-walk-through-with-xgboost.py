#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# regular expressions
import re 

# math and data utilities
import numpy as np
import pandas as pd
import scipy.stats as ss
import itertools as it

# ML and statistics libraries
import xgboost as xgb
import scipy.stats as stats
import sklearn.preprocessing as pre
from sklearn import model_selection
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC

# visualization libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Set-up default visualization parameters
mpl.rcParams['figure.figsize'] = [10,6]
viz_dict = {
    'axes.titlesize':18,
    'axes.labelsize':16,
}
sns.set_context("notebook", rc=viz_dict)
sns.set_style("darkgrid")
sns.set_palette(sns.color_palette("GnBu_r", 19))


# In[ ]:


# for kaggle path names
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Exploratory Data Analysis of Home Prices 
# 
# This first section will focus primarily on EDA and missing value imputation. I'm going to hide most of the code, so that it doesn't detract attention from the analysis. However, I will show code for the imputation of missing values, as I think many readers would benefit from it. My goal is to outline a process for building an intuition as to what variables are at play, and which have the greatest impact on sale price. I also want to evaluate relationships between variables because it will give insight on how we might intelligently fill missing data. After this analysis is complete, I will move on to predictive modeling.
# 
# Because predicting home prices is a regression problem, I'll be focusing my analysis on the following ideas (in no particular order):
#  - __Variable Distributions__ - We will want to make sure that our continuous variables are normally distributed for the purposes of having well formed data to feed to our regression model. This includes our target variable, __SalePrice__.
#  - __Finding and Dealing with Missing Data__ - In this final phase, we will analyze what variables have missing values, why they are missing and how we might deal with the missing data. For example, we will see that the variable __PoolQC__ which describes pool quality and condition has the majority of its entries missing. Should we keep the variable? Perhaps it is missing data, because it is associated with homes that don't have pools. Or should we drop it? Or create a new variable that simply describes "Has Pool" in a yes or no sort of way. 
#  - __Linearity of Continuous Variables__. If our variables have linear relationships with Sale Price, then when we get new, unseen data on for sale homes, the final sale price will be easier to predict. If our data does not have a linear relationship, then we must either transform the data, or consider non-linear regression models for prediction.
#  - __Strength of association.__ Provided that a variable is related to Sale Price, how strong is that relationship? In other words, is a particular variable worth keeping or not? Often this can only be answered by making predictions with AND without the variable in question, then checking the accuracy of the prediction. Our analysis will simply give us an idea of what variables to experiment with.
#  - __ANOVA F-Test of categorical variables against sale price.__ Unlike continuous variables, we can not necessarily determine traditional linear relationships between categories and continuous data. For example, we might easily be able to say "The more little Timmy studies (variable X) the better his test scores (variable Y)". However, let's say little Timmy has 3 math books to choose from and must study for an upcoming test. Now, variable X has possible values of book A, book B, or book C, while variable Y is still his test score. Because the books have no underlying rank or order, we cannot give them meaning through numbers. Without order, linear relationships are impossible. Instead we will use the idea of variance of outcome, based on textbook. Through ANOVA, we may look at the outcome of Timmy's test scores after using each of three books. While there may be differences between his test scores, we must dig deep to find out if those differences can really be attributed to his choice of study material. 
# 
# 

# In[ ]:


train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
train_df.head()


# In[ ]:


# Roughly same amount of samples in both training and testing data
test_df.shape, train_df.shape


# ## Warmup 1: Distribution of Target Variable:

# In[ ]:


sigma = train_df.SalePrice.std()
mu = train_df.SalePrice.mean()
med = train_df.SalePrice.median()
mode = train_df.SalePrice.mode().to_numpy()

sns.distplot(train_df.SalePrice)
plt.axvline(mode, linestyle='--', color='green', label='mode')
plt.axvline(med, linestyle='--', color='blue', label='median')
plt.axvline(mu, linestyle='--', color='red', label='mean')
plt.legend()


# Our response variable appears to be right skewed, so is a candidate for log transformation. Below we see that log(y+1) provides a nice distribution. We will keep this distribution in mind for use in a `TransformedTargetRegressor` when it is time to make predictions.
# 
# testing a log transformation supplies a much more normal distribution:

# In[ ]:


sns.distplot(np.log1p(train_df.SalePrice), kde=True)


# ## Warmup 2: Checking Basic Intuitions
# 
# As our second warm up exercise, let's check our intuition that it's not just quantity but quality that counts toward final sale price:

# In[ ]:


fig, axes = plt.subplots(2,2, figsize=(12,12))
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallQual', data=train_df, ax=axes[0,0])
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallCond', data=train_df, ax=axes[0,1])
sns.boxplot(x='OverallQual', y='SalePrice', data=train_df, ax=axes[1,0])
sns.boxplot(x='OverallCond', y='SalePrice', data=train_df, ax=axes[1,1])
plt.tight_layout()


# It appears that sale price is graduated not only by above ground living area but also by quality of material and finish (described by __OverallQual__). Surprisingly, it appears that overall condition (__OverallCond__) is rather well mixed. The box plots do show negative skew (towards lower price) in below average condition homes and positive skew (towards higher price) in above average condition homes, but nothing to write home about. There are also a large number of outliers in the category 5 (average condition) of __OverallCond__. This may contribute to the disorder.
# 
# Also notice the two outliers, homes with relatively large square footage, plus high quality materials, above average condition which sold ___well___ below average. We will discount these outliers for purposes of making predictions.

# In[ ]:


drop_mask = (train_df.GrLivArea > 4000) & (train_df.SalePrice < 200000)
drop_idx = train_df[drop_mask].index
train_df = train_df.drop(drop_idx)


# In[ ]:


# verify outliers are dropped:
sns.relplot(x='GrLivArea', y='SalePrice', hue='OverallQual',data=train_df)


# In a while, we will run through some of the other most intuitive predictors of home sale prices, looking for correlation to our response variable (sale price). We will use both standard univariate correlation, and later residual plots to test our assumptions of linearity. Before all of that, though, I'd like to address our missing data and figure out a bit more about what our data has to tell us.

# ## Checking For and Dealing With Missing Data
# 
# Our next round of analysis will focus on our missing data. We want to know why the data is missing, and how we might be able to fill in the blanks. For example, data might be missing for a home's basement simply because the home doesn't have a basement. The data dictionary that came with this dataset shows that, for most variables, the value "NA" signifies that the home does not have the feature in question. It may simply come down to mis-recorded values. In those cases we simply need to change "NaN" to "NA" (or 0, if we've already label encoded the variable).
# 
# __Warning: This section is long and tedious.__ There is A LOT of missing data in both our training and testing sets. This data cannot be ignored if we wish to get really good results. This part of the process is necessary. I encourage you to focus on the following thought process:
# 
# 1. look at a missing data point for a specific variable. 
# 2. See if the variable associated with that missing data is connected to a second variable.
# 3. Determine if the missing data point can be accounted for by the related variable. 
# 
# You will see many examples of this process below. 
# 
# ### Missing Data in Training Set:
# 
# First we will address the gaps in our missing training data. 

# In[ ]:


# A function to vizualize missing data in a DataFrame
def viz_missing(df):

    missing = pd.DataFrame({
        'Missing':df.isnull().sum(),
        '% Missing': df.isnull().sum()/len(df)
    })
    missing = missing[missing['% Missing'] > 0].sort_values(by='Missing', ascending=False)
    sns.barplot(x=missing.index, y='% Missing', data=missing)
    plt.xticks(rotation=45)
    plt.show()

viz_missing(train_df)


# __FireplaceQu__ (Fireplace Quality) is the first variable with less than 50 percent missing data. The __Fireplaces__ (# of fireplaces variable)  has a possible value of "NA - No Fireplace". Perhaps these entries are associated with the missing data in the quality description.

# In[ ]:


nofire_na = train_df.loc[train_df.FireplaceQu.isna() &
                         (train_df.Fireplaces == 0)].shape[0]


# In[ ]:


train_df.FireplaceQu.isna().sum(), nofire_na


# So indeed, all 690 instances of missing data in __FireplaceQu__ are associated with homes that don't have fireplaces. According to the variable description, these values should be "NA" not "NaN". We will fix that now:

# In[ ]:


train_df.FireplaceQu.fillna('NA', inplace=True)


# I'd like to do the same check on the variable __PoolQC__. My educated guess is that a pool adds significant value to a home, but with so much missing data, it's going to be hard to determine if my intuition is correct.

# In[ ]:


pool_na = train_df.loc[train_df.PoolQC.isna() & (train_df.PoolArea == 0)].shape[0]
pool_na, train_df.PoolQC.isna().sum()


# Note that there is no missing data for the variable __PoolArea__, and there are exactly 1453 entries where pool area is 0 square feet. This matches our 1453 missing entries for __PoolQC__. According to the data dictionary, the variable for pool quality and condition should be "NA" for those homes that have no pool. Since this is not the case, we will fill in those missing entries with "NA".

# In[ ]:


train_df.PoolQC.fillna("NA", inplace=True)


# In[ ]:


viz_missing(train_df)


# In[ ]:


train_df.MiscFeature.value_counts()


# In[ ]:


train_df.MiscFeature.fillna("NA", inplace=True)


# 49 homes have sheds, two have a second garage and on has a tennis court. These are very different features, and so this variable may not be particularly useful. A shed will obviously add less value to a home than a tennis court. It might be worth one-hot encoding variables such as these so that their data can be made available to a machine learning algorithm. Or, alternatively, we can extract these features and add them as their own variables. As it stands, I believe most algorithms would have a hard time with data such as this. We'll get to that when we are done dealing with our missing data.
# 
# For our two remaining continuous variable, __LotFrontage__ and __MasVnrArea__ we will fill the missing data points with the median (chosen over mean because it is more robust to outliers in the data.

# In[ ]:


train_df.MasVnrArea.fillna(train_df.MasVnrArea.median(), inplace=True)
train_df.LotFrontage.fillna(train_df.LotFrontage.median(), inplace=True)
viz_missing(train_df)


# In[ ]:


for var in ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']:
    print(train_df[var].isna().sum())


# It appears that all variables which describe a home's garage have the same amount of missing data. This implies to me that the homes associated with these missing data points do not have a garage. Each of these variables should read "NA" if there is no garage with the home. The only exception is __GarageYrBlt__, which we will have to figure something else out for.

# In[ ]:


for var in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train_df[var].fillna("NA", inplace=True)
    


# In[ ]:


viz_missing(train_df)


# In[ ]:


for var in ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']:
    print(train_df[var].isnull().sum())


# The values don't quite match, so let's take a closer look at the data, just in case the missing values are dissociated from one another. 

# In[ ]:


# Are all of the missing values associated with one another? 
train_df.loc[train_df.BsmtExposure.isna(), ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']]


# With the exception of entry 949, missing basement data can also be chocked up to the home having no basement. Again, we will fill in the missing data with 'NA'.

# In[ ]:


for var in ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']:
    train_df[var].fillna('NA', inplace=True)


# In[ ]:


train_df[['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']].isna().sum()


# In[ ]:


train_df.BsmtFinType2.value_counts()


# In[ ]:


viz_missing(train_df)


# For the variable __GarageYrBlt__, for any missing data points, I will assume that the garage was built in the same year as the home. For __MasVnrType__ and __Electrical__, I will fill the missing values with the mode of the existing data. The final two variables __Alley__ and __Fence__ I intend to drop completely from the analysis.

# In[ ]:


train_df.loc[train_df.GarageYrBlt.isna(), 'GarageYrBlt'] = train_df.YearBuilt
train_df.MasVnrType.fillna('None', inplace=True) # fill with mode
train_df.Electrical.fillna('Sbrkr', inplace=True)


# In[ ]:


train_df = train_df.drop(['Alley', 'Fence'], axis=1)


# ### Missing Data in Test Set

# Next, we want to address any gaps in the competition test data. These gaps will need to be filled before we can make good predictions, so let's go ahead and get it out of the way, too. 
# 
# From the below visualization of missing test data, we see that there is actually a more diverse range of missing values than in the training data. This means we may have to slightly tweak how we fill the gaps. 

# In[ ]:


viz_missing(test_df)


# This dataset suffers many of the same problems that the training data did when it comes to home features. Namely, the data dictionary specifies a value of 'NA' or 'None' when the home doesn't have the feature in question. However, when I inspect the dataset, there are a lot of 'NaN' values instead. So we'll go through the above variables and determine the answer to 'Does this feature exist in this home?'. If the feature does not exist, we will insert a value of 'NA' or 'None'. If it turns out the feature does exist, we will fill the missing value with either the mode (for categorical data) or the median value (for continuous data).

# In[ ]:


m = test_df.PoolQC.mode()[0]

# Home has a pool, fill in the quality with the mode of existing data
test_df.loc[test_df.PoolQC.isna() & (test_df.PoolArea > 0), 'PoolQC'] = m

#Home has no pool, so instead fill missing value with 'NA'
test_df.loc[test_df.PoolQC.isna() & (test_df.PoolArea == 0), ['PoolQC']] = 'NA'

# And the same for some other variables. 
test_df.MiscFeature.fillna('NA', inplace=True)
test_df.FireplaceQu.fillna('NA', inplace=True)
test_df.MasVnrArea.fillna(test_df.MasVnrArea.median(), inplace=True)
test_df.LotFrontage.fillna(test_df.LotFrontage.median(), inplace=True)

# Drop these two columns, just as we did for the training data.
test_df.drop(['Alley', 'Fence'], axis=1, inplace=True)


# In[ ]:


viz_missing(test_df)


# We will use total basement square footage to determine why a a given categorical data is missing information. If for example __GarageCond__ is missing information, and we see that basement area is 0, then it makes sense to fill __GarageCond__ with 'NA' (no garage) for that data point. However, if the missing entry is accompanied by a basement area greater than 0, then we will fill the missing data with the mode for that category. 
# 
# In order for this technique to be effective, it requires that we first take care of any missing __GarageArea__ data points first.

# In[ ]:


garage_vals = ['GarageCond', 'GarageQual', 'GarageFinish', 'GarageType']
test_df.loc[test_df.GarageArea.isna(), garage_vals] 


# The single missing garage area data point is accompanied by a __GarageType__ of 'Detchd'. I'll fill those blanks with the mode of each category.

# In[ ]:


# remove garage type so existing value isn't overwritten
g_type = garage_vals.pop(3)

for g in garage_vals:
    mode = test_df[g].mode()[0]
    test_df.loc[test_df.GarageArea.isna(), g] = mode

# Fill single entry of missing GarageArea with median    
med = test_df.GarageArea.median()
test_df.loc[test_df.GarageArea.isna(), 'GarageArea']  = med

# add garage type back to list    
garage_vals.append(g_type)    

# Check that missing data was filled. Should see empty dataframe    
test_df.loc[test_df.GarageArea.isna(), garage_vals]


# Now that the single instance of missing garage area is taken care of, we can move to the strategy we discussed above for th rest of the missing data:

# In[ ]:


for g in garage_vals:
    
    # For any missing value where there is proof a garage exists, fill with mode.
    mode = test_df[g].mode()[0]
    test_df.loc[test_df[g].isna() & (test_df.GarageArea > 0), g] = mode
    
    # For any missing value where no garage exists, fill with 'NA'
    test_df.loc[test_df[g].isna() & (test_df.GarageArea == 0), g] = 'NA'

# If Garage Year Built is missing, fill it with the year the home was built
years = test_df.loc[test_df.GarageYrBlt.isna(), 'YearBuilt']
test_df.loc[test_df.GarageYrBlt.isna(), 'GarageYrBlt'] = years


# We will use the same logic as with garage variables here with basement variables. Again, since we are basing our imputation off of total basement square footage, we should make sure that this data is in place and correct.

# In[ ]:


bsmt_vars = ['BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1']
bsmt_sf = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
test_df.loc[test_df.TotalBsmtSF.isna(), bsmt_vars+bsmt_sf]


# This time, it looks like there is zero information about this home's basement. We will set all of its area descriptors to 0, and all its qualitative variables to 'NA'. 

# In[ ]:


test_df.loc[test_df.TotalBsmtSF.isna(), bsmt_sf] = 0


# In[ ]:


# check to see if all area variables were set to zero
test_df.loc[:, bsmt_sf].isna().sum()


# In[ ]:


# Fill blanks with same logic as above (garage variables)
for var in bsmt_vars:
    mode = test_df[var].mode()[0]
    test_df.loc[test_df[var].isnull() & (test_df.TotalBsmtSF == 0), var] = 'NA'
    test_df.loc[test_df[var].isnull() & (test_df.TotalBsmtSF > 0), var] = mode

# Assume type 2 finish is the same as type 1 finish if missing data
finish = test_df.loc[test_df.BsmtFinType2.isna(), 'BsmtFinType1']
test_df.loc[test_df.BsmtFinType2.isna(), 'BsmtFinType2'] = finish


# We just filled in a lot of mising data. Let's take a look at what we have left:

# In[ ]:


viz_missing(test_df)


# The variable __MasVnrType__ describes the kind veneer a home has. It is associated with a second variable __MasVnrArea__ which describes the square footage of surface covered by veneer. This second variable is not missing, so we can use the same logic check as before to see if we should fill it with 'None' or with the most common value.

# In[ ]:


test_df.loc[test_df.MasVnrType.isna(), ['MasVnrArea']]


# In[ ]:


# All but one instances don't have any veneer. Fill single value with mode:
mode = test_df.MasVnrType.mode()[0]
test_df.loc[(test_df.MasVnrType.isna() & (test_df.MasVnrArea > 0)), 'MasVnrType'] = mode


# In[ ]:


test_df.loc[test_df.MasVnrType.isna(), ['MasVnrType']] = 'None'


# In[ ]:


viz_missing(test_df)


# If we assume that any given neighborhood has a single zoning category, we can fill in missing __MSZoning__ data by looking at the associated neighborhood. Let's check this assumption on the neighborhoods associated with our missing data:

# In[ ]:


test_df.loc[test_df.MSZoning.isna(), 'Neighborhood']


# In[ ]:


test_df.loc[test_df.Neighborhood == 'IDOTRR', 'MSZoning'].value_counts()


# So it looks like the neighborhood IDOTRR (Iowa DOT and Rail Road) has mixed zoning, but is primarily zoned as RM (Residential Medium Density). We will use this value to fill in the blanks. Let's also check the Mitchel neighborhood.

# In[ ]:


test_df.loc[test_df.Neighborhood == 'Mitchel', 'MSZoning'].value_counts()


# Again we see mixed zoning, but this time all zones are residential and the preponderance of homes fall in low density residential areas. That's what we will use to fill that blank.

# In[ ]:


mask1 = (test_df.Neighborhood == 'IDOTRR') & (test_df.MSZoning.isna())
mask2 = (test_df.Neighborhood == 'Mitchel') & (test_df.MSZoning.isna())
test_df.loc[mask1, 'MSZoning'] = 'RM'
test_df.loc[mask2, 'MSZoning'] = 'RL'


# In[ ]:


viz_missing(test_df)


# In[ ]:


# most of the remaining variables are categorical
rem_vars = ['Utilities', 'Functional', 'Exterior1st', 
        'Exterior2nd', 'KitchenQual', 'SaleType']

for var in rem_vars:
    mode = test_df[var].mode()[0]
    test_df.loc[test_df[var].isna(), var] = mode


# For the missing __GarageCars__ data point, let's look at the square footage of the garage. A typical 2-car garage is about 400 square feet or more. Below, we see that this particular garage is 480 square feet, so we will assign it as a 2 car garage. 

# In[ ]:


test_df.loc[test_df.GarageCars.isna(), 'GarageArea']


# In[ ]:


test_df.loc[test_df.GarageCars.isna(), 'GarageCars'] = 2


# Now, for the missing basement bathroom data, we will check to see if the associated basements have any finish. If not, we will assume that the number of bathrooms are zero. Otherwise, we will use the median number. 
# 
# Below we see that neither of the basements in question have any finish, so we can assume they don't have any bathrooms:

# In[ ]:


bsmt_bths = ['BsmtFullBath', 'BsmtHalfBath']
mask = (test_df.BsmtFullBath.isna() | test_df.BsmtHalfBath.isna())
test_df.loc[mask, ['BsmtFinSF1']+bsmt_bths]


# In[ ]:


test_df.loc[mask, bsmt_bths] = test_df.loc[mask, bsmt_bths].fillna(0)


# In[ ]:


# Did we fill all missing values? Should see 0
test_df.isna().sum().sum()


# Our task of filling in the vast amounts of missing data is now concluded. For me, this was time consuming and not very rewarding in the short-term. However, I hope that my diligence pays off in good predictions later. 

# # Feature Engineering
# 
# In order to make good predictions, our variables should describe the underlying factors that influence home prices. Sometimes, the data we are handed can be improved by taking a new view. For example, a given variable like __MisFeature__ may provide good information in a 'between the lines' sort of way. It has the following possible values:
# 
#        Elev	Elevator
#        Gar2	2nd Garage (if not described in garage section)
#        Othr	Other
#        Shed	Shed (over 100 SF)
#        TenC	Tennis Court
#        NA	  None
# 
# From this variable we can ascertain whether or not a home has a large shed, or even a tennis court. But a large shed and a tennis court are two entirely different entities. A tennis court almost certainly adds more value to a home than a shed or a second garage. To bring that information to the forefront, we can split __MiscFeature__ into separate variables like __HasShed__, __HasTennis__, etc. While a predictive algorithm often can pick up on underlying information on its own, bringing these variables to the forefront make it more likely the model will succeed.
# 
# Below, I am adding some features to the data that I think may be useful:

# In[ ]:


# Create variables that signify home has ____ feature.

# Training
train_df['cond*qual'] = (train_df['OverallCond'] * train_df['OverallQual']) / 100.0
train_df['TotIntArea'] = train_df['TotalBsmtSF'] + train_df['GrLivArea']
train_df['AgeSold'] = train_df['YrSold'] - train_df['YearBuilt']
train_df['Total_Bathrooms'] = train_df['FullBath']                                + 0.5 * train_df['HalfBath']                                + train_df['BsmtFullBath']                                + 0.5 * train_df['BsmtHalfBath']
train_df['HasShed'] = train_df.MiscFeature.apply(lambda x: 1 if x == 'Shed' else 0)
train_df['HasTennis'] = train_df.MiscFeature.apply(lambda x: 1 if x == 'TenC' else 0)
train_df['HasGar2'] = train_df.MiscFeature.apply(lambda x: 1 if x == 'Gar2' else 0)
train_df['HasPool'] = train_df.PoolArea.apply(lambda x: 1 if x > 0 else 0)
train_df['HasDeck'] = train_df.WoodDeckSF.apply(lambda x: 1 if x > 0 else 0)
train_df['IsNew'] = train_df.YearBuilt.apply(lambda x: 1 if x > 2000 else 0)
train_df['IsOld'] = train_df.YearBuilt.apply(lambda x: 1 if x < 1946 else 0)


# Test
test_df['cond*qual'] = (test_df['OverallCond'] * test_df['OverallQual']) / 100.0
test_df['TotIntArea'] = test_df['TotalBsmtSF'] + test_df['GrLivArea']
test_df['AgeSold'] = test_df['YrSold'] - test_df['YearBuilt']
test_df['Total_Bathrooms'] = test_df['FullBath']                                + 0.5 * test_df['HalfBath']                                + test_df['BsmtFullBath']                                + 0.5 * test_df['BsmtHalfBath']
test_df['HasShed'] = test_df.MiscFeature.apply(lambda x: 1 if x == 'Shed' else 0)
test_df['HasTennis'] = test_df.MiscFeature.apply(lambda x: 1 if x == 'TenC' else 0)
test_df['HasGar2'] = test_df.MiscFeature.apply(lambda x: 1 if x == 'Gar2' else 0)
test_df['HasPool'] = test_df.PoolArea.apply(lambda x: 1 if x > 0 else 0)
test_df['HasDeck'] = test_df.WoodDeckSF.apply(lambda x: 1 if x > 0 else 0)
test_df['IsNew'] = test_df.YearBuilt.apply(lambda x: 1 if x > 2000 else 0)
test_df['IsOld'] = test_df.YearBuilt.apply(lambda x: 1 if x < 1946 else 0)


# In[ ]:


porch_area = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
train_df['HasPorch'] = train_df.loc[:, porch_area]                             .sum(axis=1)                             .apply(lambda x: 1 if x > 0 else 0)

test_df['HasPorch'] = test_df.loc[:, porch_area]                             .sum(axis=1)                             .apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


test_df.drop(['MiscFeature'], axis=1, inplace=True)
train_df.drop(['MiscFeature'], axis=1, inplace=True)


# In[ ]:


train_df[['HasPorch']+porch_area].sort_values(by='HasPorch')


# In[ ]:


# Show our new dataframe with added features
train_df.head()


# # Strength of Relationships Between Variables
# 
# At this point, we have filled in all of our missing data, and come up with some new ways to describe our homes in question. With a full dataset, we are almost ready to apply deeper statistical analysis of our data. However, in order to model our data's relationships, we must first convert our non-numeric data to number format.  
# 
# We will split our variables into their three types: __continuous, ordinal and nominal__.
#  - __Continuous__ : variables whose values are on a continuous range, typically with an infinite number of possible values. Examples include a child's height, speed, or depth. 
#  - __Ordinal__ : variables whose values are discrete and finite, but which have underlying order. A variable that classifies disease severity into 'mild', 'moderate' and 'severe' is one such example. Only three values, non-continuous, but with an underlying rank.
#  - __Nominal__ : variables whose values are discrete categories with no underlying order. One such example might be the variable sex, which contains the categories 'Male' and 'Female'. 

# In[ ]:


# a selection of variables that I expect to add to home value and/or curbside appeal, by variable type
exp_ordinal = ['ExterQual', 'BsmtQual', 'GarageQual', 
               'KitchenQual', 'FireplaceQu', 'PoolQC',  
               'OverallQual', 'ExterCond', 'BsmtCond', 
               'GarageCond', 'OverallCond', 'HeatingQC',
               'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
               'GarageCars', 'GarageFinish', 'BedroomAbvGr', 
               'KitchenAbvGr', 'HouseStyle', 'TotRmsAbvGrd',
               'YearBuilt', 'MoSold', 'YrSold', 'MasVnrArea', 
               'Utilities', 'BsmtFullBath', 'BsmtHalfBath', 
               'FullBath', 'HalfBath', 'CentralAir', 
               'YearRemodAdd','LotShape', 'Functional']

exp_nominal = ['Neighborhood', 'MSZoning', 'Condition1', 
               'Condition2', 'RoofStyle', 
               'RoofMatl', 'Exterior1st', 'Exterior2nd', 
               'MasVnrType', 'Foundation', 'Electrical', 
               'SaleType', 'SaleCondition', 'HasShed',
               'HasTennis', 'HasGar2', 'HasPorch', 
               'HasDeck', 'HasPool', 'GarageType', 
               'LotConfig', 'PavedDrive', 'IsOld', 'IsNew']

exp_contin  = ['LotArea', 'LotFrontage', 'GrLivArea', 
               'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2', 
               'TotalBsmtSF', 'BsmtUnfSF', '1stFlrSF', 
               '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
               'PoolArea', 'MiscVal', 'TotIntArea', 'SalePrice']


# ### Converting Ordinal Variables to Numbers:
# 
# Almost all "Quality and Condition" variables share the same letter code scheme in the data: 
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# 
# Some variables like __BsmtCond__ (Basement Condition) have an additional category (NA - No Basement). We will encode these as integers from 0 to 5 in increasing quality or condition.
# 
# __Note:__ It should be noted that this same functionality can be achieved by using `pd.factorize()`, but that pandas does not know the order of the ordinal data, so the labels will be random. This means that any machine learning algorithm will not be able to harness this natural order for use in predicting sale prices. 

# In[ ]:


qual_map = {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
qual_vars = ['ExterQual', 'ExterCond', 'BsmtQual', 
             'BsmtCond', 'HeatingQC', 'KitchenQual', 
             'FireplaceQu', 'GarageQual', 'GarageCond',
             'PoolQC']

for col in qual_vars:
    train_df[col] = train_df[col].map(qual_map)
    test_df[col] = test_df[col].map(qual_map)
    
# Make sure we see all numeric data:    
train_df[qual_vars]


# Next, let's take care of some of the basement and garage qualitative variables that don't follow the above mold. Each category needs to be encoded in it's appropriate order/rank. The following mappings were taken from the data dictionary that accompanied the data:

# In[ ]:


# Maps for Ordinal varibles:

BsmtExposure_map = {
    'Gd':4,
    'Av':3,
    'Mn':2,
    'No':1,
    'NA':0
}

BsmtFinType1_map = {
    'NA': 0,
    'Unf': 1,
    'LwQ': 2,
    'Rec': 3,
    'BLQ': 4,
    'ALQ': 5,
    'GLQ': 6
}

BsmtFinType2_map = {
    'NA': 0,
    'Unf': 1,
    'LwQ': 2,
    'Rec': 3,
    'BLQ': 4,
    'ALQ': 5,
    'GLQ': 6
}

GarageFinish_map = {
    'NA': 0,
    'Unf': 1,
    'RFn': 2,
    'Fin': 3
}

Functional_map = {
    'Sal': 0,
    'Sev': 1,
    'Maj2': 3,
    'Maj1': 4,
    'Mod': 5,
    'Min2': 6,
    'Min1': 7,
    'Typ': 8
}

Utilities_map = {
    'ELO': 0,
    'NoSeWa': 1,
    'NoSewr': 2,
    'AllPub': 3
}

CentralAir_map = {
    'N': 0,
    'Y': 1
}

LotShape_map = {
    'IR3': 0,
    'IR2': 1,
    'IR1': 2,
    'Reg': 3
}

HouseStyle_map = {
    '1Story': 1,
    '1.5Unf': 2,
    '1.5Fin': 3,
    '2Story': 4,
    '2.5Unf': 5,
    '2.5Fin': 6,
    'SFoyer': 7,
    'SLvl': 8
}


# In[ ]:


train_df.BsmtExposure = train_df.BsmtExposure.map(BsmtExposure_map)
train_df.BsmtFinType1 = train_df.BsmtFinType1.map(BsmtFinType1_map)
train_df.BsmtFinType2 = train_df.BsmtFinType2.map(BsmtFinType2_map)     
train_df.GarageFinish = train_df.GarageFinish.map(GarageFinish_map)
train_df.Functional   = train_df.Functional.map(Functional_map)
train_df.Utilities    = train_df.Utilities.map(Utilities_map)
train_df.CentralAir   = train_df.CentralAir.map(CentralAir_map)
train_df.LotShape     = train_df.LotShape.map(LotShape_map)
train_df.HouseStyle   = train_df.HouseStyle.map(HouseStyle_map)

test_df.BsmtExposure = test_df.BsmtExposure.map(BsmtExposure_map)
test_df.BsmtFinType1 = test_df.BsmtFinType1.map(BsmtFinType1_map)
test_df.BsmtFinType2 = test_df.BsmtFinType2.map(BsmtFinType2_map)     
test_df.GarageFinish = test_df.GarageFinish.map(GarageFinish_map)
test_df.Functional   = test_df.Functional.map(Functional_map)
test_df.Utilities    = test_df.Utilities.map(Utilities_map)
test_df.CentralAir   = test_df.CentralAir.map(CentralAir_map)
test_df.LotShape     = test_df.LotShape.map(LotShape_map)
test_df.HouseStyle   = test_df.HouseStyle.map(HouseStyle_map)


# In[ ]:


train_df.BsmtExposure


# ## Looking at Relationships Between Variables:
# 
# Now, we must consider the association between our predictor variables and our response variable. Because we are dealing with a continuous response variable, but have some categorical features, we will have to think carefully about how to measure association. 
# 
# ### Broad Overview by Heatmap (Ordinal/Ranked and Continuous Data):
# 
# We will work first with our ranked/ordinal and continuous variables, as traditional "correlation" is easier to see with these types of variables. This is because correlation, by definition, is a measure of the strength of a linear relationship. Linear relationships only occur among data that can be measured continuously, or discrete data whose levels are evenly spaced in their impact. For the ordinal data, I propose either using Kendall's $\tau$ or descritizing sale prices and using Goodman and Kruskal's $\gamma$. Since Kendall's $\tau$ is included in the scikit learn library, we need not write our own method. 
# 
# First, let's take a high level look of all of our candidate variables' relationships with heatmaps:

# In[ ]:


fig, axes = plt.subplots(3, 1, figsize=(10,30))

#Too many variables for a single heatmap of all ordinal data
ord1 = exp_ordinal[0:17] + ['SalePrice']
ord2 = exp_ordinal[17:] + ['SalePrice']

corr1 = train_df[ord1].corr(method='kendall')
corr2 = train_df[ord2].corr(method='kendall')
corr3 = train_df[exp_contin].corr()

sns.heatmap(corr1, annot=True, ax=axes[0], fmt='0.2f').set_title('Ordinal Data 1')
sns.heatmap(corr2, annot=True, ax=axes[1], fmt='0.2f').set_title('Ordinal Data 2')
sns.heatmap(corr3, annot=True, ax=axes[2], fmt='0.2f').set_title('Continuous Data')

plt.tight_layout()
plt.show()


# The above broad overview of correlation is interesting in that we see that almost all of the variables at play are somewhat correlated with one another. We appear to have a lot of information that is redundantly encoded in our variables. For example we would expect __TotRmsAbvGr__ would correlate to well to __BedroomAbvGr__, and we see exactly that. Sometimes it is beneficial to drop highly correlated predictor variables because many models operate much more efficiently on independent data. 
# 
# ### Isolating Predictor vs Response Variable Association:
# 
# We want our predictor variables to be highly correlated with our outcome, so we will look at those values individually below:

# In[ ]:


ord1_corr = pd.DataFrame(
    train_df[ord1].corr(method='kendall').loc['SalePrice']
)
ord1_corr.sort_values(by='SalePrice', inplace=True, ascending=False)

ord2_corr = pd.DataFrame(
    train_df[ord2].corr(method='kendall').loc['SalePrice']
)
ord2_corr.sort_values(by='SalePrice', inplace=True, ascending=False)

cont_corr = pd.DataFrame(
    train_df[exp_contin].corr().loc['SalePrice']
)
cont_corr.sort_values(by='SalePrice', inplace=True, ascending=False)


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 5))
sns.heatmap(cont_corr, annot=True, ax=ax1).set_title('Continuous')
sns.heatmap(ord1_corr, annot=True, ax=ax2).set_title('Ranked 1')
sns.heatmap(ord2_corr, annot=True, ax=ax3).set_title('Ranked 2')
plt.tight_layout()
plt.show()


# The above graphic describes each predictor variable's correlation to sale price, in descending order. Some interesting relationships stand out, one example being __GarageCars__ being moderately negatively correlated to Sale Price. We see the same interesting reverse correlation between __LotShape__ where apparently sale price tends to go up as the shape of the lot becomes more irregular. 
# 
# This view of variable relationships is somewhat naive. When one or two variables are at play, univariate analysis is more trustworthy. However, when we begin to include more variables, there are cases where the apparent (calculated) correlation and the true correlation between the two variables is very much different (sometimes even completely reversed). One such phenomenon is called "Simpson's Paradox".
# 
# Additionally, traditional __Pearson's $r$__ correlation typically used for continuous data only measure's linear relationships. 'Spearman's rho ($\rho$)', however is used to detect monotonic increase or decrease, and is therefore better suited to find both linear and non-linear relationships. In fact, if we see that  $\rho$ is much greater than $r$, it is a sign that there is a non-linear relationship between the two variables in question. 
# 
# 

# In[ ]:


def plot_response_corr(df, features, response, corr_type):
    
    cor = pd.DataFrame()
    width = len(features)*0.5
    height = 5
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Measure difference between Spearman's and Pearson's to analyze for non-linearity
    if corr_type == 's-p':
        
        cor['feature'] = features
        cor['spearman'] = [df[f].corr(df[response], 'spearman') for f in features]
        cor['pearson'] = [df[f].corr(df[response], 'pearson') for f in features]
        cor['comparison'] = cor['spearman'] - cor['pearson']
        
        sns.barplot(
            data=cor.sort_values(by='comparison'), 
            x='feature', y='comparison', ax=ax
        ).set_title('S-P Comparison')
           
    else:
        cor['feature'] = features
        cor[corr_type] = [df[f].corr(df[response], corr_type) for f in features]
        cor = cor.sort_values(corr_type)
        plt.figure(figsize=(len(features)*1.5, 5))
        axes = sns.barplot(data=cor, x='feature', y=corr_type, ax=ax)             .set_title(f'{corr_type} Association to Response Variable')        
    

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return cor 


# In[ ]:


df = plot_response_corr(train_df, exp_contin, 'SalePrice', 's-p')


# It appears that __LotArea__ in particular shows moderate difference in Spearman and Pearson measures of association. This may mean that house prices are not related to the lot size in a linear fashion. Perhaps the relationship is polynomial or exponential. 
# 
# Below, the same analysis applied to the ordinal data shows moderate difference in 

# In[ ]:


df = plot_response_corr(train_df, exp_ordinal, 'SalePrice', 's-p')


# ### Categorical Data (Nominal)
# 
# For our nominal data, quantitative association is somewhat difficult to achieve in the same way as ordinal and continuous data. We can, however, take a look at box plots for each of our interesting nominal variables. The idea is to look for separation of home values according to category. We are graphically answering questions like _"Is the average cost of a home significantly different between neighborhoods X, Y, and Z?"_.
# 
# To refresh our memories on what some of our nominal variables are:
# 
# - __Neighborhood__ - What neighborhood is the home located in.
# - __MSZoning__ -Identifies the type of zoning (agricultural, commercial, etc.)
# - __Condition1 & Condition2__ - Proximity to urban features such as railroads, parks, etc.
# 
# 
# ### Neighborhood:
# 
# I have created a boxplot of neighborhoods ordered by there average home value. Visually, we do see a rough ordering to these neighborhoods. Is this variance in sale price due to differences between neighborhoods, or is it more due to differences within individual neighborhoods (for example, is the mean home value of neigborhood X quite high compared to neighborhood Y because there is a billion dollar mansion at the end of the street, while all the other homes are relatively cheap?). This question cannot be well answered by a graph, and we will address it later. 

# In[ ]:


# for ordering the boxplots by median
grouped = train_df.loc[:,['Neighborhood', 'SalePrice']]     .groupby(['Neighborhood'])     .mean()     .sort_values(by='SalePrice')


# In[ ]:


chart = sns.boxplot(x=train_df.Neighborhood, y=train_df.SalePrice, order=grouped.index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.show()


# We will perform one way ANOVA on the data to see if this difference between neighborhoods is significant (statistically). It remains to be seen how much weight the adage "location, location, location" carries.

# ### Zoning
# 
# Our data dictionary defines __MSZoning__ as _"Identifies the general zoning classification of the sale."_, with the following possible categories:
# 
#        A	 Agriculture
#        C	 Commercial
#        FV	Floating Village Residential
#        I	 Industrial
#        RH	Residential High Density
#        RL	Residential Low Density
#        RP	Residential Low Density Park 
#        RM	Residential Medium Density
# 
# We will apply the same techniques as before to see if we can rank these zones by their average home value.

# In[ ]:


grouped = train_df.loc[:,['MSZoning', 'SalePrice']]     .groupby(['MSZoning'])     .mean()     .sort_values(by='SalePrice')

chart = sns.boxplot(x=train_df.MSZoning, y=train_df.SalePrice, order=grouped.index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.show()


# As before, there is visually some separation between the means. We will add it to the 'to-do ANOVA' list.
# 
# ## Proximity to Urban Features
# 
# The __Condition1 & Condition2__ variables describe proximity to the following features such as parks, railroads, etc:
# 
#        Artery	Adjacent to arterial street
#        Feedr	 Adjacent to feeder street	
#        Norm	  Normal	
#        RRNn	  Within 200' of North-South Railroad
#        RRAn	  Adjacent to North-South Railroad
#        PosN	  Near positive off-site feature--park, greenbelt, etc.
#        PosA	  Adjacent to postive off-site feature
#        RRNe	  Within 200' of East-West Railroad
#        RRAe	  Adjacent to East-West Railroad
# 
# We will see if we can rank these features as before, but this time we will rank both individual features and every combination of two. 

# In[ ]:


grouped = train_df.loc[:,['Condition1', 'SalePrice']]     .groupby(['Condition1'])     .mean()     .sort_values(by='SalePrice')

chart = sns.boxplot(x=train_df.Condition1, y=train_df.SalePrice, order=grouped.index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.show()


# This one doesn't look as promising to me. Again, only a test will tell. Let's go ahead and run such a test on all of these questionable categories. While we have already seen clear association between ordinal variables and __SalePrice__ we will include them in this analysis as well. 
# 
# ### Performing Analysis of Variation (ANOVA)
# 
# While visual inspection of our nominal variables by boxplot is a good start, we need to find out if there is any statistically significant differences between our variable categories when considering sale prices. In order to to answer that question, we will perform ANOVA F-testing to each variable. What this test tells us is, given a variable like __Neighborhood__, is the variance in price that we see across neighborhoods due more to variance between neighborhoods or more from variance within neighborhoods. 

# In[ ]:


def anova(df, predictor, target, alpha=0.05):
    
    # get unique, indendant treatments (categories in predictory)
    treatments = df.loc[df[predictor].notnull(), predictor].unique()
    group_values = []
    
    # group target variable by category
    for t in treatments:
        group_values.append(df.loc[df[predictor] == t, target].to_numpy())
    
    # calculate degrees of freedom and confidence level
    k = len(treatments)
    n = df[predictor].notnull().sum()
    conf = 1 - alpha
    df1 = k - 1
    df2 = n - k
    
    # calculate critical value of F-distribution
    crit_val = stats.f.ppf(conf, df1, df2)
    
    # calculate F and p-values
    f, p = stats.f_oneway(*group_values)
    
    return f, p, crit_val


# In[ ]:


# DataFrame to hold ANOVA results.
f_frame = pd.DataFrame(columns=['variable','p-value','f-value', 'critical', 'passed'])

# Perform ANOVA on each variable. Store data for plotting
for var in exp_nominal+exp_ordinal:
    f, p, c = anova(train_df, var, 'SalePrice')
    f_frame.loc[len(f_frame)] = [var, p, f, c, (f > c)]

# Plot test results
f_frame.sort_values(by='f-value', ascending=False, inplace=True)

# sns.set(rc={'figure.figsize':(15,8)})
sns.barplot(x='variable', y='f-value', data=f_frame)
plt.xticks(rotation=90)
plt.show()


# ### Summary of ANOVA
# 
# What is difficult to see in the visualization is that almost all of our variables passed their F-Test with flying colors. This signifies that there is significant variation in sale price by category within each variable. Those variables that failed the test are below. They may be candidates for dropping before we proceed with predictive modeling. 

# In[ ]:


f_frame.loc[f_frame.passed == False]


# # Summary of EDA:
# 
# We've looked at a lot. Here is a summary.
# 
# ## Relationships in the Data
# 
# ### Continuous Data
# 
# There is moderate to high correlation between sale price and the features describing lot size, above ground living area and garage size. Additionally, while the association appears to be linear in above grade living area and garage area. However, there is some indication of a non-linear relationship between lot area and sale price. It might be worth investigating, and is an indicator that the data needs to be transformed before applying machine learning. 
# 
# Some variables that did not show promising correlation to sale price:
# 
# - __BsmtFinSF2__
# - __LowQualFinSF__
# - __3SsnPorch__
# - __PoolArea__
# - __MiscVal__
# 
# ### Categorical Data (Ordinal)
# 
# Most of our variables seem to be a good fit for predictive modeling. The data that did not seem to be impactful (at least in a one vs. one comparison) is:
# 
# - __BsmtFinType2__
# - __ExterCond__
# - __BsmtHalfBath__
# - __YearSold__
# - __Utilities__
# 
# A couple notable exceptions are the low association between __SalePrice__ and __ExterCond__, which was unexpected.
# 
# Additionally, there is an inverse association with __OverallCond__. This seems to indicate that better overall condition of a home negatively impacts sale price. This is unlikely to be true, and so I will address this and other issues below in the 'caveats' section.
# 
# ### Categorical Data (Nominal)
# 
# Most of our nominal data seems to impact sale price. There are some indications that the following variables might not, however:
# 
# - __HasTennis__
# - __HasGar2__
# 
# ### Caveats
# 
# Correlation/association at a univariate level is not necessarily an indicator of true association when there are many variables at play in the data. It is important to note that things get complicated (i.e. Simpson's Paradox) when more than one variable is at play. For this reason we will not immediately dismiss the above variables that do not seem to have a strong relationship with sale price. At first, we will include them, and then if our predictions need improvement, we will experiment with removing some of them.
# 
# Second, we did not focus on the interrelations between independent predictor variables. Leaving in highly correlated variables can cause over-fitting (the phenomenon where the machine learning model knows the data it learned on very well, but cannot predict with accuracy the outcome with never before seen data). However, since we expect that the new data will have identical multicollinearity (because the training data is a random subset of all of the data), I hope that it will not cause any issues. 

# # Getting Ready to Train a Model
# 
# The model I intend to use is XGBoost.
# 
# ## Converting Nominal Data to Numbers:
# 
# This could have been done when we were converting ordinal data to numbers, however we would have lost some of the meaningful labels, so I chose to do it after analysis. 
# 
# Our nominal data has no underlying order to it, so we will make use of One-Hot encoding so that our regression model doesn't make the mistake of thinking that the order of the numbers matters. 
# 
# Some of our data, namely those features we engineered like __HasTennis__ or __HasPool__ are already one-hot encoded. We will focus our efforts on those which are not.

# In[ ]:


train_df[exp_nominal].info()


# We see that the data that needs to be One-Hot encoded is all of type 'object' in the data frame. We will perform this encoding simultaneously for both training and prediction data to ensure consistency of the final product. The reason this is something to think about is that the `OneHotEncoder` adds a single column for each category it finds under a given variable. It is quite possible that the not all unique possible values of each variable are encountered in both the training and the test sets. This would lead to two datasets with different dimensions after encoding.

# In[ ]:


train_idx = train_df.index
test_idx = test_df.index

comb_df = train_df.append(test_df)

to_encode = comb_df.select_dtypes(include='object')
encoder = pre.OneHotEncoder()
encoder.fit(to_encode)
encoded = encoder.transform(to_encode).toarray()
dum_df = pd.DataFrame(encoded, columns=encoder.get_feature_names(to_encode.columns))
comb_df = comb_df.join(dum_df)
comb_df.drop(to_encode.columns, axis=1, inplace=True)

# Split datasets back apart
train_df = comb_df.loc[train_idx].copy()
test_df = comb_df.loc[test_idx].copy()

# delete unused df
del comb_df

# verify correct dimensions (# of columns)
train_df.shape, test_df.shape


# In[ ]:


# test_df picked up a 'SalePrice' column when we joined DF's. Let's drop it.
test_df.drop(['SalePrice'], axis=1, inplace=True)


# In[ ]:


train_df.shape, test_df.shape


# ## Dropping Features With Low Correlation to Sale Price
# 
# Recall that we performed two types of testing on our predictor variables. Correlation for the continuous/ordinal data and ANOVA on the categorical data. For both sets of tests, there were variables that responded poorly to the testing. We will drop those variables in hopes that it reduces variance in our tree-based XGBoost model. The features that failed ANOVA and/or had low correlation to __SalePrice__ were:
# 
# - BsmtFinSF2
# - BsmtHalfBath
# - MiscVal
# - Utilities
# - HasGar2
# - HasTennis
# - MoSold
# - YrSold
# 
# Additionally, __ExterCond__ showed very low correlation to SalePrice, which is counter-intuitive. During ANOVA testing, this variable showed significant variance between categories, indicating that it does indeed contribute well predictions. We will leave it in.

# In[ ]:


to_drop = ['BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'Utilities', 
           'HasGar2', 'HasTennis', 'MoSold', 'YrSold']

train_df.drop(to_drop, axis=1, inplace=True)
test_df.drop(to_drop, axis=1, inplace=True)


# ## Dropping Highly Correlated Predictors:
# 
# After we created the feature __TotIntArea__ which combined total basement area and total above ground living area, we saw that our new feature had very high correlation to sale price. However, when I dropped these features I saw a decrease in score on the LB. I have chosen to leave them in for this reason. 

# # Tuning, Training and Testing the Model
# Now that all of our data is numeric, either one-hot encoded (nominal) or integer encoded (ordinal) we can train our model, tune it and test it's performance using k-fold cross validation. 

# In[ ]:


X = train_df.drop(['SalePrice'], axis=1)
Y = train_df.pop('SalePrice')
# If only there was a function called "lock"


# In[ ]:


# Split Data
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, random_state=1)


# ## Hyper-Parameter Tuning
# We will iteratively tune a couple of parameters at a time. This is a time consuming process, because for each possible parameter value in the grid, `GridSearchCV` is performing k-fold cross validation. Therefore, if you wish to reduce tuning time, you may pass `cv=3` or `cv=2` into the `GridSearchCV` object. It defaults to 5 folds. 
# 
# This operation can take 10-20 minutes per round of tuning, depending on your machine's performance.
# 
# ### Tuning `max_depth` and `min_child_weight`:
# We will tune these two parameters first, over a grid of values incremented by two. After the first round of tuning we will select the best parameters and tune again from one below to one above the first results.

# In[ ]:


# ROUND 1:
param_grid1 = {
    'max_depth':np.arange(1,11,2),
    'min_child_weight':np.arange(1,11,2)
}

# First Search
gsearch1 = model_selection.GridSearchCV(xgb.XGBRFRegressor(),
    param_grid1, 
    scoring='neg_mean_squared_log_error',
    cv=4)

gsearch1 = gsearch1.fit(x_train, y_train)


# In[ ]:


best_params = gsearch1.best_params_
best_params, -gsearch1.best_score_


# In[ ]:


# ROUND 2
# use value range one above and one below previous round's results:
a = best_params['max_depth'] - 1
b = best_params['max_depth'] + 2
c = best_params['min_child_weight'] - 1
d = best_params['min_child_weight'] + 2


param_grid2 = {
    'max_depth':range(a, b),
    'min_child_weight':range(c, d)
}

gsearch2 = model_selection.GridSearchCV(
    xgb.XGBRegressor(**best_params), 
    param_grid2, 
    scoring='neg_mean_squared_log_error', 
    cv=4
)

gsearch2 = gsearch2.fit(x_train, y_train)


# In[ ]:


best_params.update(gsearch2.best_params_)
best_params, -gsearch2.best_score_
# Small drop in performance here. 


# ### Tuning `gamma`:
# We'll now tune our regressor for gamma, holding the parameters we have already tuned fixed.

# In[ ]:


param_grid3 = {'gamma':[i/10.0 for i in range(0,6)]}

gsearch3 = model_selection.GridSearchCV(
    xgb.XGBRFRegressor(**best_params),
    param_grid3,
    scoring='neg_mean_squared_log_error', 
    cv=5
)

gsearch3 = gsearch3.fit(x_train, y_train)


# In[ ]:


best_params.update(gsearch3.best_params_)
best_params, -gsearch3.best_score_


# ### Tuning `subsamle` and `colsample_bytree`:

# In[ ]:


param_grid4 = {
    'colsample_bytree':[i/10.0 for i in range(0,11)],
    'subsample':[i/10.0 for i in range(0,11)]
}

gsearch4 = model_selection.GridSearchCV(
    xgb.XGBRFRegressor(**best_params),
    param_grid4,
    scoring='neg_mean_squared_log_error',
    cv=3   
)

gsearch4 = gsearch4.fit(x_train, y_train)


# In[ ]:


best_params.update(gsearch4.best_params_)
best_params, -gsearch4.best_score_


# ### Tuning Regularization Terms $\alpha$ and $\lambda$
# 
# According to the XGBoost documentation, these parameters are analogous to the `l1` and `l2` regularization parameters. Even though I used K-Fold CV on my training set, and the results are good on the test set, my model performed much worse on the Leader Board data. This indicates to me that overfitting may still be an issue. For that reason, I have decided to tune these two parameters.

# In[ ]:


param_grid5 = {
    'alpha':[i/10 for i in range(0,11)],
    'lambda':[i/10 for i in range(0,11)]
}

gsearch5 = model_selection.GridSearchCV(
    xgb.XGBRegressor(**best_params),
    param_grid5, 
    scoring='neg_mean_squared_log_error',
    cv=5
)

gsearch5 = gsearch5.fit(x_train, y_train)


# In[ ]:


best_params.update(gsearch5.best_params_)
best_params, -gsearch5.best_score_


# ### Tuning the number of trees and learning rate:
# 
# `n_estimators` is the parameter that dictates the number of trees to iteratively add to the model. Returns for added trees in the model diminish after a point, and so we may find an early stopping point. Because I am not using the native XGBoost library, but rather the scikit-learn wrapper, I do not have access to `early_stopping_round` which would stop adding trees automatically when no significant improvement is noted. 
# 

# In[ ]:


param_grid6 = {
    'n_estimators':np.arange(50, 450, 50),
    'learning_rate':[0.01, 0.05, 0.1, .5, 1]
}

gsearch6 = model_selection.GridSearchCV(
    xgb.XGBRegressor(**best_params),
    param_grid6, 
    scoring='neg_mean_squared_log_error',
    cv=5
)

gsearch6 = gsearch6.fit(x_train, y_train)


# In[ ]:


best_params.update(gsearch6.best_params_)
best_params, -gsearch6.best_score_


# In[ ]:


# Compare base model with tuned model: 
base_booster = xgb.XGBRegressor()
tune_booster = xgb.XGBRegressor(**best_params)

# We want to shuffle the data for the cross_val_score 
kf = model_selection.KFold(5, shuffle=True, random_state=1)


# In[ ]:


score1 = model_selection.cross_val_score(
    base_booster, 
    x_train, 
    y_train, 
    scoring='neg_mean_squared_log_error',
    cv = kf
)

score2 = model_selection.cross_val_score(
    tune_booster, 
    x_train, 
    y_train, 
    scoring='neg_mean_squared_log_error',
    cv = kf
)


# In[ ]:


# Compare the performance of tuned and untuned models. 
print(f"Base: Root Mean Log Error: {-score1.mean()} (+/-{score1.std()*2})")
print(f"Tuned: Root Mean Log Error: {-score2.mean()} (+/-{score2.std()*2})")   


# In[ ]:


# Compare CV scores with final score on unseen data:
tune_booster.fit(x_train, y_train)
pred = tune_booster.predict(x_test)
metrics.mean_squared_log_error(y_test, pred)


# ## Assessment: 
# It Looks like the cross validation score from the training data matches score on the unseen test data. This is a good indicator that overfitting will not be an issue, however it does appear to occur based on leader board results. 

# # Making Final Predictions
# 
# Now that we have trained our ensemble and pre-processed our data for prediction, all we have left to do is make final predictions on the unseen 'test' data provided by Kaggle. These predictions are what our leader board score will be based on. First we will combine our training and validation sets and re-fit our model to all available labeled data.

# In[ ]:


tune_booster.fit(X, Y)
final_preds = tune_booster.predict(test_df)


# In[ ]:


submission = pd.DataFrame({'Id':test_df.index, 'SalePrice':final_preds})
submission.to_csv('submission3.csv', index=False)


# In[ ]:


submission


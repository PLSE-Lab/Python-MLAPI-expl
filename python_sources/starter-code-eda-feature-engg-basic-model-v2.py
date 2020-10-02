#!/usr/bin/env python
# coding: utf-8

# ### Importing packages

# A few of these packages might not being used right now. Will use as I build more models

# In[ ]:


import pandas as pd

import numpy as np
from scipy import stats
from math import ceil
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# ### Data Exploration and Understanding

# In[ ]:


train_df.shape # 1460,81
test_df.shape # 1459,80


# In[ ]:


train_df.info()
#checking all non-numerical columns
for c in train_df.columns:
    col_type = train_df[c].dtype
    if col_type != 'int64' and col_type != 'float64':
        print(c)


# In[ ]:


# Plot the Correlation map to see how features are correlated with target: SalePrice
corr_matrix = train_df.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corr_matrix, vmax=0.9, square=True)


# Let us do some feature selection on the basis of the correlations above and some general understanding of the problem:
# 
# 1. If we look at the heatmap above, all white squares indicate high correlation between the corresponding variables
# 2. 'GarageCars' and 'GarageArea' show high correlation and is aligned with our intuitive thinking as well. More the area of the Garage , more the number of cars. Furthermore, both seem to have a similar (and relatively high) correlation with 'SalePrice'. This shows a clear case of multicollinearity. Thus we can remove one of them and retain the other.
# 3. Furthermore, 'TotalBsmtSF' and '1stFlrSF' show high correlation again indicating multicollinearity. We should thus remove one of these.
# 4. If we look at the correlation between 'TotalBsmtSF' and 'SalePrice', we see a white square i.e. high correlation. This indicates that TotalBsmtSF should be retained as it can help with the SalePrice prediction
# 5. Another set of variables that show high correlation are 'YearBuilt' and 'GarageYrBlt'. Let us look at two other aspects: percentage of missing values in 'GarageYrBlt' and the correlation between 'YearBuilt' and 'SalePrice'. 'GarageYrBlt' has over a 5% missing values. Also, 'YearBuilt' seems to have a decent (around 0.5) correlation with 'SalePrice'. It seems like we should retain 'YearBuilt' and let go of 'GarageYrBlt'
# 6. Also, besides 'YearBuilt', ( one of 'GarageCars' or 'GarageArea') and 'TotalBsmtSF', we should keep in mind four other variables that seem to have good correlation with 'SalePrice': 'OverallQual', 'GrLivArea', 'FullBath' and 'TotRmsAbvGrd'
# 7. However, if we look at the correlation of 'TotRmsAbvGrd' with other variables, we see a high correlation with 'GrLivArea'. To be able to make a call on which variable to remove, let us look at some more analysis.

# In[ ]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars','GarageArea' ,'TotalBsmtSF', 'FullBath', 'YearBuilt','TotRmsAbvGrd']
sns.pairplot(train_df[cols], size = 2.5)
plt.show();


# Following are the major observations from here:
# 1. GrLivArea and TotRmsAbvGrd show high linear relationship. 
# 2. Let us shift focus to GrLivArea and TotalBsmtSF show a linear relationship with almost a boundary defining the plot. This basically indicates that GrLivArea defined the higher limit for the TotalBsmtSF (Basement area). Not many houses will have basements larger than the ground floor living area.
# 3. 'SalePrice' shows almost a steep increase with 'YearBuilt', basically indicating that prices increase (almost eponentially) as the houses decrease in age. Most recent houses are highly priced.
# 
# We shall just zoom into the correlation matrix with 'SalePrice' and a few other features:
# 

# In[ ]:



#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# Thus final list of variables to be definitely considered and variables to be excluded:
# 
# 1. Variable with highest correlation with 'SalePrice' is 'OverallQual'-----> Retain 'OverallQual'
# 
# 2. High correlation between 'GarageCars' and 'GarageArea'  + High correlation between 'GarageCars' and 'SalePrice' ----> Keep 'GarageCars'; remove 'GarageArea'
# 
# 3. 'TotalBsmtSF' and '1stFlrSF' have high correlation and are equally correlated with 'SalePrice'-----> Randomly selecting 'TotalBsmtSF', remove '1stFlrSF'
# 
# 4. Of 'GarageYrBlt' and 'YearBuilt', 'YearBuilt' has lower missing values and higher correlation with 'SalePrice'-----> retain 'YearBuilt', remove 'GarageYrBlt'
# 
# 5. Strong correlation between 'TotRmsAbvGrd' and 'GrLivArea' + higher correlation between 'GrLivArea'  and 'SalePrice'-------> Keep 'GrLivArea' ; remove 'TotRmsAbvGrd'
# 
# 6. Retain 'FullBath' as we did not see correlation with any other variable but it has a significant association with 'SalePrice'

# ### Outlier Removal

# Let us first look at outliers in the numerical variables with the highest correlation to 'SalePrice'. Those are the ones that should make the most difference to 'SalePrice' predictions. 

# In[ ]:


# first variable : GrLivArea
plt.subplots(figsize=(15, 5))

plt.subplot(1, 2, 1)
g = sns.regplot(x=train_df['GrLivArea'], y=train_df['SalePrice'], fit_reg=False).set_title("Before")

# Delete outliers
plt.subplot(1, 2, 2)                                                                                
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)
g = sns.regplot(x=train_df['GrLivArea'], y=train_df['SalePrice'], fit_reg=False).set_title("After")


# In[ ]:


# Next Up: TotalBsmtSF
plt.subplots(figsize=(15, 5))

plt.subplot(1, 2, 1)
g = sns.regplot(x=train_df['TotalBsmtSF'], y=train_df['SalePrice'], fit_reg=False).set_title("Before")

# Delete outliers
plt.subplot(1, 2, 2)                                                                                
train_df = train_df.drop(train_df[(train_df['TotalBsmtSF']>3000)].index)
g = sns.regplot(x=train_df['TotalBsmtSF'], y=train_df['SalePrice'], fit_reg=False).set_title("After")


# In[ ]:


# Next Up : OverallQual
plt.subplots(figsize=(15, 5))

plt.subplot(1, 2, 1)
g = sns.regplot(x=train_df['OverallQual'], y=train_df['SalePrice'], fit_reg=False).set_title("Before")

# Delete outliers
plt.subplot(1, 2, 2)                                                                                
train_df = train_df.drop(train_df[(train_df['OverallQual']>9) & (train_df['SalePrice']>700000)].index)
g = sns.regplot(x=train_df['OverallQual'], y=train_df['SalePrice'], fit_reg=False).set_title("After")


# In[ ]:


train_df.shape # 1453,81
# 7 rows deleted


# In[ ]:


#check if there are columns that are present in train and not in test
# if there are any, we will have to drop them from train
extra_train_cols = set( train_df.columns ) - set( test_df.columns )
extra_train_cols # no columns that are present in train and not in test


# ### Investigation of Target Variable

# #### What are we doing?
# We are investigating the nature of the target or response variable here; i.e. 'SalePrice'. On the basis of our findings here, we can transform the variable.
# 
# #### Why are we doing this?
# This is to ensure the model predictions behave better. What this means is, in regression it is necessary that the residuals follow a normal distribution. Now, if the predicted values are normally distributed then the residuals are as well and vice versa. For a detailed explaination refer to this [link here](https://stats.stackexchange.com/questions/60410/normality-of-dependent-variable-normality-of-residuals).

# In[ ]:


# basic states of 'SalePrice'
train_df['SalePrice'].describe()
# here we see min= 34900 and max= 625000
# to get a better understanding, we will plot a distribution curve


# In[ ]:


#distribution plot- histogram
sns.distplot(train_df['SalePrice']).set_title("Distribution of SalePrice")

# probability plot
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)


# We see a positive skewness. This is not exactly a normal distribution. Also, the balues do not follow the linear trend here.  
# Now, if we observe a positive or a right skewness, log transformation is a good option
# 
# #### Why log?
# The non-linear trend shows 'SalePrice' has some sort of exponential relationship with the independent variables. Applying a log function on these values should give a linear trend and convert the set of values into 'normally distributed' values.
# 
# Furthermore,  for this problem, "submissions are evaluated on **Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price**. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)"
# 
# All this would just make sense if we replace the values in 'SalePrice' column by the corresponding log values

# In[ ]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

#Check the new distribution 
sns.distplot(train_df['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_df['SalePrice'])
print( '\n mean = {:.2f} and std dev = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Distribution of Log SalePrices')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()


# In[ ]:


# let us also store the target variable
y_train = train_df.SalePrice.values
train_df.head()


# ### Concatenating train and test to create total_df (aggregated dataset)

# In[ ]:


# store the unique ids of training dataset
train_ids = train_df.index    
# store the unique ids of test dataset
test_ids = test_df.index

# combine train and test datas in to one dataframe
total_df = pd.concat([train_df,test_df]).reset_index(drop=True)
#total_df.drop(['SalePrice'], axis=1, inplace=True)
print("Shape of total_df : {}".format(total_df.shape))
total_df.isnull().sum()


# Creating a list of features to be removed 
# 

# In[ ]:


feature_drop1= ['GarageYrBlt','TotRmsAbvGrd'] # will remove 1stFlrSF and GarageArea later-- after creating additional features


# In[ ]:


#removing features-- with multicollinearity or low correlation with target variable
total_df.drop(feature_drop1,
              axis=1, inplace=True)
total_df.head()


# ### Missing Value Treatment

# In[ ]:


#Checking for missing data
NAs = pd.concat([train_df.isnull().sum(), test_df.isnull().sum()], axis=1, keys=['Train', 'Test'])
NAs[NAs.sum(axis=1) > 0]


# In[ ]:


# find missing values as percentage of data length
total_na = (total_df.isnull().sum() / len(total_df)) * 100
total_na = total_na.drop(total_na[total_na == 0].index).sort_values(ascending=False)[:30]
missing_data_perc = pd.DataFrame({'Missing Ratio' :total_na})
missing_data_perc


# In[ ]:


# columns with attributes like Pool, Fence etc. marked as NaN indicate the absence of these features.
attributes_with_na = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2']

# replace 'NaN' with 'None' in these columns
for col in attributes_with_na:
    total_df[col].fillna('None',inplace=True)
    
#NAs in basement related columns will indicate no masonry veneer. Thus replacing MasVnr Area with 0
total_df.MasVnrArea.fillna(0,inplace=True)

#NAs in basement related columns will indicate no basement. Thus replacing all areas and count column with 0 
total_df.BsmtFullBath.fillna(0,inplace=True)
total_df.BsmtHalfBath.fillna(0,inplace=True)
total_df.BsmtFinSF1.fillna(0,inplace=True)
total_df.BsmtFinSF2.fillna(0,inplace=True)
total_df.BsmtUnfSF.fillna(0,inplace=True)
total_df.TotalBsmtSF.fillna(0,inplace=True)

#Similarly for Garage Cars-- fill with 0; cause if no garage, no cars can parked in it
total_df.GarageCars.fillna(0,inplace=True)   
# doing the same for GarageArea
total_df.GarageArea.fillna(0,inplace=True)   


# In[ ]:


total_df.isnull().sum()


# In[ ]:


# Let us look at perc of nulls again
# find missing values as percentage of data length
total_na_rev = (total_df.isnull().sum() / len(total_df)) * 100
total_na_rev = total_na_rev.drop(total_na_rev[total_na_rev == 0].index).sort_values(ascending=False)[:30]
missing_data_perc_rev = pd.DataFrame({'Missing Ratio' :total_na_rev})
missing_data_perc_rev


# In[ ]:


# Let us first focus on the lesser null percentages (except LoTFrontage)
# Let us see the distribution of data across these fields

# first up: Utilities
total_df.groupby(['Utilities']).size() # only one NoSeWa value and 2 nulls 
train_df.groupby(['Utilities']).size() # train data contains the 'NoSeWa'i.e. Test has no NoSeWa value
# 2 null values come from Test data
## intuitively this will not play a significant role in our model prediction
# for now let us populate the nulls with the most frequent value 'AllPub'-- can drop it later
total_df['Utilities'] = total_df['Utilities'].fillna(total_df['Utilities'].mode()[0]) 


# In[ ]:


# next is: Functional
# Similarly for Functional
#Functional : by the definition of the column, 'NA' means typical
total_df.groupby(['Functional']).size() # typ has 2717 as of now
# Since 'typ' is also the most frequent value, let us replace 'NA' with 'typ'
total_df["Functional"] = total_df["Functional"].fillna("Typ")
total_df.groupby(['Functional']).size() # typ= 2719 now


# In[ ]:


# Let us now look at: Electrical
total_df.groupby(['Electrical']).size() # this has one missing value in Train i.e. SBrKr (currently 2671)
# Let us just populate the NA with the most frequent entry
total_df['Electrical'] = total_df['Electrical'].fillna(total_df['Electrical'].mode()[0])
total_df.groupby(['Electrical']).size() # now SBrKr= 2672


# In[ ]:


# Like Electrical, KitchenQual has 1 missing value
total_df.groupby(['KitchenQual']).size() # the missing value is in Test; most frequent value is 'TA'= 1492
# Let us just replace null with 'TA'
total_df['KitchenQual'] = total_df['KitchenQual'].fillna(total_df['KitchenQual'].mode()[0])
total_df.groupby(['KitchenQual']).size() # 'TA'= 1493


# In[ ]:


# The next column is SaleType
total_df.groupby(['SaleType']).size() # one NA in Test, most frequent value is 'WD'=2525
#populating nulls with the most frequent values
total_df['SaleType'] = total_df['SaleType'].fillna(total_df['SaleType'].mode()[0])
total_df.groupby(['SaleType']).size() # 'WD'= 2526


# In[ ]:


# Doing the same thing for Exterior1st and 2nd
total_df['Exterior1st'] = total_df['Exterior1st'].fillna(total_df['Exterior1st'].mode()[0])
total_df['Exterior2nd'] = total_df['Exterior2nd'].fillna(total_df['Exterior2nd'].mode()[0])


# In[ ]:


# Moving on to the higher null percentages: MSZoninng
total_df.groupby(['MSZoning']).size() #most frequent value is 'RL'=2265
# Let us just substitute the 4 nulls with the most frequent values
total_df['MSZoning'] = total_df['MSZoning'].fillna(total_df['MSZoning'].mode()[0])
total_df.groupby(['MSZoning']).size() 


# In[ ]:


#Checking for missing data once again
NAs_again = pd.concat([total_df.isnull().sum()], axis=1)
NAs_again[NAs_again.sum(axis=1) > 0] # just one column LotFrontage-- has 486 missing values 


# In[ ]:


# function to scale a column
def norm_minmax(col):
    return (col-col.min())/(col.max()-col.min())


# In[ ]:


# By business definition, LotFrontage is the area of each street connected to the house property
# Intuitively it should be highly correlated to variables like LotArea
# It should also depend on LotShape, LotConfig
# Let us make a simple Linear regressor to get the most accurate values

# convert categoricals to dummies
#also dropping the target 'SalePrice' for now as the target currently is 'LotFrontage'
total_df_dummy = pd.get_dummies(total_df.drop('SalePrice',axis=1))
# scaling all numerical columns
for col in total_df_dummy.drop('LotFrontage',axis=1).columns:
    total_df_dummy[col] = norm_minmax(total_df_dummy[col])

frontage_train = total_df_dummy.dropna()
frontage_train_y = frontage_train.LotFrontage
frontage_train_X = frontage_train.drop('LotFrontage',axis=1)  

# fit model
lin_reg= linear_model.LinearRegression()
lin_reg.fit(frontage_train_X, frontage_train_y)

# check model results
lr_coefs = pd.Series(lin_reg.coef_,index=frontage_train_X.columns)
print(lr_coefs.sort_values(ascending=False))


# In[ ]:


# use model predictions to populate nulls
nulls_in_lotfrontage = total_df.LotFrontage.isnull()
features = total_df_dummy[nulls_in_lotfrontage].drop('LotFrontage',axis=1)
target = lin_reg.predict(features)

# fill nan values
total_df.loc[nulls_in_lotfrontage,'LotFrontage'] = target


# In[ ]:


#Checking for missing data once again
NAs_again = pd.concat([total_df.isnull().sum()], axis=1)
NAs_again[NAs_again.sum(axis=1) > 0] # just one column LotFrontage-- has 486 missing values 


# In[ ]:


train_subset=total_df[total_df['SalePrice'].notnull()]
ntrain= train_subset.shape[0]


# ### Feature Engineering or Feature Manipulation

# **OverallQual**

# In[ ]:


# Let us start with the variables having highest correlation with the target variable
# looking at OverallQual, GrLivArea, GarageCars and TotalBsmtSF

# Since it is one of the highest correlated variables with the response, we can create a quadratic variable that might be a part of the regression equation
total_df["OverallQual_2"] = total_df["OverallQual"].astype(int) ** 2
#also creating cubic
total_df["OverallQual_3"] = total_df["OverallQual"].astype(int) ** 3
# another sqrt transformation
total_df["OverallQual_sqrt"] = np.sqrt(total_df["OverallQual"].astype(int))

# OverallQual is just a categorical variable in guise of integers
# Changing OverallQual into a categorical variable
total_df['OverallQual'] = total_df['OverallQual'].astype(str)


# In[ ]:


# next variable: GrLivArea
# creating the polynomial variables from here as well
total_df["GrLivArea_2"] = total_df["GrLivArea"] ** 2
#also creating cubic
total_df["GrLivArea_3"] = total_df["GrLivArea"] ** 3
# another sqrt transformation
total_df["GrLivArea_sqrt"] = np.sqrt(total_df["GrLivArea"])


# **GrLivArea**

# In[ ]:


# let us check the distribution of GrLivArea 
#distribution and probability plots
#distribution plot- histogram
sns.distplot(total_df['GrLivArea']).set_title("Distribution of GrLivArea")

# probability plot
fig = plt.figure()
res = stats.probplot(total_df['GrLivArea'], plot=plt)


# We see a similar positive skewness in this variable. Let us create a log-transformed variable from GrLivArea

# In[ ]:


# log transformed
total_df['GrLivArea_log'] = np.log1p(total_df['GrLivArea'])


# In[ ]:


# we can also create buckets on GrLivArea
total_df['GrLivArea_Band'] = pd.cut(total_df['GrLivArea'], 6,labels=["1", "2", "3","4","5","6"])
print(total_df['GrLivArea_Band'].unique())

# since these are essential categorical variables,
# let us convert them to string
total_df['GrLivArea_Band'] = total_df['GrLivArea_Band'].astype(str)


# In[ ]:


total_df.head()


# **TotalBsmtSF**

# In[ ]:


# creating polynomial features from TotalBsmtSF
total_df["TotalBsmtSF_2"] = total_df["TotalBsmtSF"] ** 2
#also creating cubic
total_df["TotalBsmtSF_3"] = total_df["TotalBsmtSF"] ** 3
# another sqrt transformation
total_df["TotalBsmtSF_sqrt"] = np.sqrt(total_df["TotalBsmtSF"])

# log transformed variable
total_df['TotalBsmtSF_log'] = np.log1p(total_df['TotalBsmtSF'])


# In[ ]:


# also creating a 1-0 flag called 'HasBsmt' using 'TotalBsmtSF'
#if area>0 it is 'Y', else 'N'
total_df['HasBsmt'] = np.where(total_df['TotalBsmtSF']>0, 'Y', 'N')


# In[ ]:


# we can also create buckets on GrLivArea
total_df['TotalBsmtSF_Band'] = pd.cut(total_df['TotalBsmtSF'], 3,labels=["1", "2", "3"])
print(total_df['TotalBsmtSF_Band'].unique())

# since these are essential categorical variables,
# let us convert them to string
total_df['TotalBsmtSF_Band'] = total_df['TotalBsmtSF_Band'].astype(str)


# **GarageCars**

# In[ ]:


# creating polynomial features from GarageCars
total_df["GarageCars_2"] = total_df["GarageCars"] ** 2
#also creating cubic
total_df["GarageCars_3"] = total_df["GarageCars"] ** 3
# another sqrt transformation
total_df["GarageCars_sqrt"] = np.sqrt(total_df["GarageCars"])

# log transformed variable
total_df['GarageCars_log'] = np.log1p(total_df['GarageCars'])


# Let us now start creating some features from multiple base variables.

# In[ ]:


# OverallCond is again just a rating- categorical variable. let us first convert the datatype
total_df['OverallCond'] = total_df['OverallCond'].astype(str)
# use OverallQual and OverallCond to get a total home quality-- averaging both
total_df['TotalHomeQual'] = (total_df['OverallCond'].astype(int) + total_df['OverallQual'].astype(int))/2
total_df['TotalHomeQual'] = total_df['TotalHomeQual'].astype(str) # converted to string
total_df[:5]


# In[ ]:


# Adding all floors SF
total_df['AllFlrs_SF'] = total_df['TotalBsmtSF'] + total_df['1stFlrSF'] + total_df['2ndFlrSF']

# creating features with finish type fraction of basement SF
# create separate columns for area of each possible
# basement finish type
bsmt_fin_cols = ['BsmtGLQ','BsmtALQ','BsmtBLQ',
                 'BsmtRec','BsmtLwQ']

for col in bsmt_fin_cols:
    # initialise as columns of zeros
    total_df[col+'SF'] = 0

# fill remaining finish type columns
for row in total_df.index:
    fin1 = total_df.loc[row,'BsmtFinType1']
    if (fin1!='None') and (fin1!='Unf'):
        # add area (SF) to appropriate column
        total_df.loc[row,'Bsmt'+fin1+'SF'] += total_df.loc[row,'BsmtFinSF1']
        
    fin2 = total_df.loc[row,'BsmtFinType2']
    if (fin2!='None') and (fin2!='Unf'):
        total_df.loc[row,'Bsmt'+fin2+'SF'] += total_df.loc[row,'BsmtFinSF2']


# remove initial BsmtFin columns
total_df.drop(['BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2'], axis=1, inplace=True)

# already have BsmtUnf column in dataset
bsmt_fin_cols.append('BsmtUnf')

# also create features representing the fraction of the basement that is each finish type
for col in bsmt_fin_cols:
    total_df[col+'Frac'] = total_df[col+'SF']/total_df['TotalBsmtSF']
    # replace any nans with zero (for properties without a basement)
    total_df[col+'Frac'].fillna(0,inplace=True)


# In[ ]:


#creating a feature on LivingAreaSF
total_df['LivingAreaSF'] = total_df['1stFlrSF'] + total_df['2ndFlrSF'] + total_df['BsmtGLQSF'] + total_df['BsmtALQSF'] + total_df['BsmtBLQSF']


# In[ ]:


# removing all individual SF variables used to create the above features
SF_feature_drop= ['1stFlrSF','2ndFlrSF','BsmtGLQSF','BsmtALQSF','BsmtBLQSF','GarageArea']
#removing features
total_df.drop(SF_feature_drop,
              axis=1, inplace=True)
total_df.head()

total_df[:5]


# In[ ]:


# timeline related variables
# age at time of selling
total_df['age_at_selling_point']= total_df['YrSold']-total_df['YearBuilt']

# time since last remodel
total_df['time_since_remodel']= total_df['YrSold']-total_df['YearRemodAdd']

# create a flag feature whether the house was remodelled
total_df['remodelled_after']= total_df['YearRemodAdd']-total_df['YearBuilt']
total_df['HasBeenRemodelled'] = np.where(total_df['remodelled_after']>0, 'Y', 'N')

# create feature on decade of selling
total_df['DecadeSold']= (total_df['YrSold']//10)*10
# convert this to char
total_df['DecadeSold'] = total_df['DecadeSold'].astype(str)

# create feature on decade of building
total_df['DecadeBuilt']= (total_df['YearBuilt']//10)*10
# convert this to char
total_df['DecadeBuilt'] = total_df['DecadeBuilt'].astype(str)


# In[ ]:


# drop the time fields used to create above features
# removing all individual SF variables used to create the above features
time_feature_drop= ['YrSold','YearRemodAdd','remodelled_after','YearBuilt']
#removing features
total_df.drop(time_feature_drop,
              axis=1, inplace=True)
total_df.head()


# In[ ]:


#MSSubClass is a categorical variable. Let us change the data type
total_df['MSSubClass'] = total_df['MSSubClass'].astype(str)


# In[ ]:


list(total_df)


# In[ ]:


#Month sold are transformed into categorical features.
total_df['MoSold'] = total_df['MoSold'].astype(str)

# doing similar exercise for several other columns
qual_cols= ['HeatingQC','KitchenQual','FireplaceQu','GarageQual','PoolQC','ExterQual','BsmtQual','Fence','BsmtCond','GarageCond','ExterCond','GarageCond']

for c in qual_cols:
    total_df[c] = total_df[c].astype(str)


# In[ ]:


total_df.info()
#checking all non-numerical columns
for c in total_df.columns:
    col_type = total_df[c].dtype
    if col_type != 'int64' and col_type != 'float64':
        print(c)


# #### treating all object variables

# Of the above variables, a few are ordinal, others are categorical
# We need to LabelEncode the ordinal features and One-hot encode the categorical features

# In[ ]:


# create a list of ordinal variables
ordinal_variables=['HeatingQC','KitchenQual','FireplaceQu','GarageQual','PoolQC','ExterQual','BsmtQual','Fence','BsmtCond','GarageCond','ExterCond','GarageCond','OverallCond','OverallQual','TotalHomeQual']
# label encoder
le = preprocessing.LabelEncoder()

for c in ordinal_variables:
    le.fit(total_df[c])
    total_df[c] = le.transform(total_df[c])


# In[ ]:


# create a list of categorical columns for one hot encoding
cat_variables= ['MSSubClass','MSZoning','Street','Alley','LotShape','LotConfig','LandContour','BsmtExposure','BldgType','CentralAir','Condition1','Condition2','Electrical','Exterior1st','Exterior2nd','Foundation','Functional','GarageFinish','GarageType','Heating','HouseStyle','LandSlope','SaleCondition','Utilities','RoofStyle','HasBsmt','RoofMatl','MasVnrType','HasBeenRemodelled','DecadeBuilt','DecadeSold','MoSold','Neighborhood','PavedDrive','MiscFeature','GrLivArea_Band','TotalBsmtSF_Band','SaleType']

# One-Hot encoding to convert categorical columns to numeric
print('start one-hot encoding')

total_df = pd.get_dummies(total_df, prefix = cat_variables,
                         columns = cat_variables)

print('one-hot encoding done')

# dropping SalePrice
total_df.drop(['SalePrice'], axis=1, inplace=True)

# normalize the variables to values from 0 to 1
normalized_total_df = pd.DataFrame(preprocessing.normalize(total_df))


# In[ ]:


print(total_df.shape)
print(normalized_total_df.shape)
normalized_total_df.columns= list(total_df)
normalized_total_df


# In[ ]:


print(normalized_total_df.shape)
normalized_total_df.info()


# ### Modelling

# In[ ]:


# Splitting the train and test datasets
train = normalized_total_df[:ntrain]
test = normalized_total_df[ntrain:]


# In[ ]:


# metric for evaluation
def rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff**2)    
    n = len(y_pred)   
    
    return np.sqrt(sum_sq/n)


# #### Tree Based Regression

# In[ ]:


# LGBM Regression
lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

lgb_model.fit(train.values, y_train)
lgb_train_pred = lgb_model.predict(train.values)
lgb_pred = np.expm1(lgb_model.predict(test.values))


# In[ ]:


print('score=',lgb_model.score(train.values,y_train))
print('rmse=',rmse(y_train, lgb_train_pred))
print('explained variance score=',explained_variance_score(y_train, lgb_train_pred))
print('mae=',mean_absolute_error(y_train, lgb_train_pred))
print('r-squared=',r2_score(y_train, lgb_train_pred))


# In[ ]:


# XGB regressor
xgb_model = xgb.XGBRegressor(max_depth= 3, learning_rate= 0.05, n_estimators= 800, booster= 'gbtree', gamma= 0, reg_alpha= 0.1,
                  reg_lambda=0.7, max_delta_step= 0, min_child_weight=1, colsample_bytree=0.5, colsample_bylevel=0.2,
                  scale_pos_weight=1)

xgb_model.fit(train.values, y_train)
xgb_train_pred = xgb_model.predict(train.values)
xgb_pred = np.expm1(xgb_model.predict(test.values))


# In[ ]:


print('score=',xgb_model.score(train.values,y_train))
print('rmse=',rmse(y_train, xgb_train_pred))
print('explained variance score=',explained_variance_score(y_train, xgb_train_pred))
print('mae=',mean_absolute_error(y_train, xgb_train_pred))
print('r-squared=',r2_score(y_train, xgb_train_pred))


# #### KNeighbours Regression

# In[ ]:


#KNearestNeighbours
knn_model= KNeighborsRegressor()

knn_model.fit(train.values, y_train)
knn_train_pred = knn_model.predict(train.values)
knn_pred = np.expm1(knn_model.predict(test.values))


# In[ ]:


print('score=',knn_model.score(train.values,y_train))
print('rmse=',rmse(y_train, knn_train_pred))
print('explained variance score=',explained_variance_score(y_train, knn_train_pred))
print('mae=',mean_absolute_error(y_train, knn_train_pred))
print('r-squared=',r2_score(y_train, knn_train_pred))


# ### Appending predictions to test

# In[ ]:


# best model was xgb with max r-squared and min error
print("number of predictions=",xgb_pred.shape[0],"rows") # shape of predictions is equal to shape of test dataset

#hence appending
# storing in best model predictions
best_model_prediction= xgb_pred
# concat
test_df['SalePrice']= best_model_prediction


# In[ ]:


final_submission=test_df.copy()
final_submission = final_submission[['Id','SalePrice']]
print(final_submission.shape)
final_submission[:10]


# In[ ]:


final_submission.to_csv('Final_submission.csv', encoding='utf-8',header=True ,index=False)


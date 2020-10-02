#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Reading and Understanding the Data
# 
# Let us first import NumPy and Pandas and read the automobile dataset

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('max_columns', 200)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import os


# In[ ]:


# Importing train.csv
# Please make sure that the csv file is in the same folder as the python notebook otherwise this command wont work
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")


# In[ ]:


# Check the head of the dataset
df_train.head()


# Inspect the various aspects of the df_train dataframe

# In[ ]:


df_train.shape


# In[ ]:


# Prining all the columns of the dataframe
df_train.columns.values


# In[ ]:


print("{} Numerical columns, {} Categorial columns are part of the original dataset.".format(
    list(df_train.select_dtypes(include=[np.number]).shape)[1],
    list(df_train.select_dtypes(include = ['object']).shape)[1]))


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# In[ ]:


#Checking for any duplicates in the data frame
df_train.loc[df_train.duplicated()]


# ## Step 2: Cleaning the Data

# ### 2.1 Drop un-needed variables

# Dropping Id column since its a unique identifier in the dataset and does not help in the analysis.

# In[ ]:


df_train = df_train.drop('Id',axis=1)


# ### 2.2 Checking for Missing Values and Treating Them

# Need to check if there is any missing data in the dataset.

# In[ ]:


# finding all the missing data and summing them based on each column and storing it in a dataframe
total = df_train.isnull().sum().sort_values(ascending = False)
# Finding the percentage of the missing data by diving the number of missing values with total and  storing it in a dataframe
percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
# Concatinating both the above df's
df_train_missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# Printing the data
df_train_missing_data.head(20)


# Let's visualize the above as well

# In[ ]:


# setting a grid for the pot
sns.set_style("whitegrid")
# finding the no of missing values 
missing = df_train.isnull().sum()
# filtering the columns with just missing values
missing = missing[missing > 0]
# sorting the values
missing.sort_values(inplace=True)
# plotting the bar chart
missing.plot.bar()
# setting the title of the plot
plt.title('Columns with Missing values', fontsize=15)
# setting the x label
plt.xlabel('Columns')
# setting the y label
plt.ylabel('No of missing values')


# ### Inference:  

# First thing to do is get rid of the features with more than 90% missing values. For example the PoolQC's missing values are probably due to the lack of pools in some buildings, which is very logical. But replacing those (more than 90%) missing values with "no pool" will leave us with a feature with low variance, and low variance features are uniformative for machine learning models. So we drop the features with more than 80% missing values.

# In[ ]:


# removing any column which has more than 90% null values
df_train = df_train.loc[:,df_train.isnull().sum()/df_train.shape[0]*100<80]
# printing the df
print(df_train.shape)


# In[ ]:


# finding all the missing data and summing them based on each column and storing it in a dataframe
total = df_train.isnull().sum().sort_values(ascending = False)
# Finding the percentage of the missing data by diving the number of missing values with total and  storing it in a dataframe
percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
# Concatinating both the above df's
df_train_missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# Printing the data
df_train_missing_data.head(16)


# In[ ]:


NA=df_train[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','GarageYrBlt','BsmtFinType2',
'BsmtFinType1','BsmtCond', 'BsmtQual','BsmtExposure', 'MasVnrArea','MasVnrType','Electrical','FireplaceQu',
             'LotFrontage']]


# In[ ]:


NAcat=NA.select_dtypes(include='object')
NAnum=NA.select_dtypes(exclude='object')
print('We have :',NAcat.shape[1],'categorical features with missing values')
print('We have :',NAnum.shape[1],'numerical features with missing values')


# Now, upon further checking the data, we can see that there are columns having null values but as per the data description these are not values which were not captured. These basically mean that those features were not available as part of the property. Hence we will have to impute them appropriately.
# 
# - We have decided to impute such columns with a value of `No`.

# In[ ]:


# columns where NA values have meaning e.g. no garage etc.
cols_fillna = ['MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2']

# replace 'NA' with 'No' in these columns
for col in cols_fillna:
    df_train[col].fillna('No',inplace=True)


# In[ ]:


# finding all the missing data and summing them based on each column and storing it in a dataframe
total = df_train.isnull().sum().sort_values(ascending = False)
# Finding the percentage of the missing data by diving the number of missing values with total and  storing it in a dataframe
percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
# Concatinating both the above df's
df_train_missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# Printing the data
df_train_missing_data.head(5)


# Now that we have imputed the NA columns with the value of `No`, we are still left with columns which have a high percentage of null values. 
# 
# Let's check these individually and see how to treat such columns.

# #### 2.2.1 LotFrontage column check

# In[ ]:


# checking the count of different values within the column
df_train['LotFrontage'].value_counts()


# In[ ]:


# pulling the length of number of unique values in the column
num_unique_values = len(df_train['LotFrontage'].unique())
# Plotting a histogram for visualizing the data
df_train['LotFrontage'].plot.hist(bins = num_unique_values)


# Now we have decided to `impute` the values of either the `mean/ median or mode` with the null values. Hence lets check what these 3 metrics provide us.

# In[ ]:


# checking the mean of the column
print("Mean is ",df_train['LotFrontage'].mean())
# checking the mode of the column
print("Mode is ",df_train['LotFrontage'].mode())
# checking the median of the column
print("Median is ",df_train['LotFrontage'].median())


# ### Inference:
# As all 3 metrics are comparable. But since we want missing value to be imputed with an integer, taking **median** i.e. `69`.

# In[ ]:


# imputing the value of median to the null values
df_train.loc[pd.isnull(df_train['LotFrontage']),['LotFrontage']]=69


# Lets check the Distribution now

# In[ ]:


# pulling the length of number of unique values in the column
num_unique_values =  len(df_train['LotFrontage'].unique())
# Plotting a histogram for visualizing the data
df_train['LotFrontage'].plot.hist(bins = num_unique_values)


# ### Inference:
# Since the distribution has not changed much before and after the null value imputation, we should be good here.

# #### 2.2.2 GarageYrBlt column check

# In[ ]:


# checking the count of different values within the column
df_train['GarageYrBlt'].value_counts()


# In[ ]:


# pulling the length of number of unique values in the column
num_unique_values =  len(df_train['GarageYrBlt'].unique())
# Plotting a histogram for visualizing the data
df_train['GarageYrBlt'].plot.hist(bins = num_unique_values)


# In[ ]:


# checking the mean of the column
print("Mean is ",df_train['GarageYrBlt'].mean())
# checking the mode of the column
print("Mode is ",df_train['GarageYrBlt'].mode())
# checking the median of the column
print("Median is ",df_train['GarageYrBlt'].median())


# ### Inference:
# Since this column tells about in what year the garage was built, we cannot apply the mean or the median here. Hence we will impute the values with mode.

# In[ ]:


# imputing the value of mode to the null values
df_train.loc[pd.isnull(df_train['GarageYrBlt']),['GarageYrBlt']]=1980


# Lets check the Distribution now

# In[ ]:


# pulling the length of number of unique values in the column
num_unique_values =  len(df_train['GarageYrBlt'].unique())
# Plotting a histogram for visualizing the data
df_train['GarageYrBlt'].plot.hist(bins = num_unique_values)


# ### Inference:
# Since the distribution has not changed much before and after the null value imputation, we should be good here.

# In[ ]:


# finding all the missing data and summing them based on each column and storing it in a dataframe
total = df_train.isnull().sum().sort_values(ascending = False)
# Finding the percentage of the missing data by diving the number of missing values with total and  storing it in a dataframe
percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
# Concatinating both the above df's
df_train_missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# Printing the data
df_train_missing_data.head(5)


# #### 2.2.3 MasVnrArea column check

# In[ ]:


# checking the count of different values within the column
df_train['MasVnrArea'].value_counts()


# In[ ]:


# checking the mean of the column
print("Mean is ",df_train['MasVnrArea'].mean())
# checking the mode of the column
print("Mode is ",df_train['MasVnrArea'].mode())
# checking the median of the column
print("Median is ",df_train['MasVnrArea'].median())


# ### Inference:
# Since the mode and median are the same, and majority of the values are 0, we will impute the missing values as 0.

# In[ ]:


# imputing the value of median to the null values
df_train.loc[pd.isnull(df_train['MasVnrArea']),['MasVnrArea']]=0


# #### 2.2.4 Electrical column check

# In[ ]:


# checking the count of different values within the column
df_train['Electrical'].value_counts()


# Since the Majority is SBrkr, we will impute it with the same.

# In[ ]:


df_train['Electrical'] = df_train['Electrical'].fillna("SBrkr")


# In[ ]:


# finding all the missing data and summing them based on each column and storing it in a dataframe
total = df_train.isnull().sum().sort_values(ascending = False)
# Finding the percentage of the missing data by diving the number of missing values with total and  storing it in a dataframe
percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
# Concatinating both the above df's
df_train_missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# Printing the data
df_train_missing_data[df_train_missing_data.sum(axis=1)>0]


# ## 2.3 Outlier Analysis

# There are 2 types of outliers and we will treat the outliers since they can skew our dataset.
# 
# - Statistical
# - Domain specific

# Lets plot a box plot to check the outliers

# In[ ]:


# before we move forward, lets create a copy of the existing df
df_train1=df_train.copy()


# In[ ]:


# Initializing the figure
fig = plt.figure(figsize = (12,8))
# prining the boxplot
# df_train = df_train.drop(['SalePrice'],axis=1)

sns.boxplot(data=df_train1)
# setting the title of the figure
plt.title("PC Distribution", fontsize = 12)
# setting the y-label
plt.ylabel("Range")
# setting the x-label
plt.xlabel("Columns")
plt.xticks(rotation=90)

# printing the plot
plt.show()


# In[ ]:


df_train1.describe(percentiles=[.05,.25, .5, .75, .90, .95, .99])


# As we can see from the graph and table above, there are some outliers in the dataset. Lets treat these outliers. We will keep the lower quantile at 0.05 and higher quantile at 0.95.

# In[ ]:


# Finding the columns on which the outlier treatment will be performed
AllCols = df_train1.select_dtypes(exclude='object')
# Sorting the columns
AllCols = AllCols[sorted(AllCols.columns)]
# printing the columns
print(AllCols.columns)


# In[ ]:


# running a for loop to remove the outliers from each column
for i in AllCols.columns:
    # setting the lower whisker
    Q1 = df_train[i].quantile(0.05)
    # setting the upper whisker
    Q3 = df_train[i].quantile(0.95)
    # setting the IQR by dividing the upper with lower quantile
    IQR = Q3 - Q1
    # performing the outlier analysis
    df_train = df_train[(df_train[i] >= Q1-1.5*IQR) & (df_train[i] <= Q3+1.5*IQR)]


# In[ ]:


# Checking the shape of the df now
df_train.shape


# In[ ]:


# checking the different percentiles now
df_train.describe(percentiles=[.05,.25, .5, .75, .90, .95, .99])


# In[ ]:


# Initializing the figure
fig = plt.figure(figsize = (12,8))
# prining the boxplot
# df_train1 = df_train.drop(['SalePrice','LotArea'],axis=1)

sns.boxplot(data=df_train)
# setting the title of the figure
plt.title("PC Distribution", fontsize = 12)
# setting the y-label
plt.ylabel("Range")
# setting the x-label
plt.xlabel("Columns")
plt.xticks(rotation=90)

# printing the plot
plt.show()


# In[ ]:


# Let's look at the scarifice
print("Shape before outlier treatment: ",df_train1.shape)
print("Shape after outlier treatment: ",df_train.shape)

print("Percentage data removal is around {}%".format(round(100*(df_train1.shape[0]-df_train.shape[0])/df_train1.shape[0]),2))


# ## Step 3: Visualising the Data using EDA
# 
# - Here's where we'll identify if some predictors directly have a strong association with the outcome variable i.e `Sales Price`.
# 
# - We'll visualise our data using `matplotlib` and `seaborn`.

# ### 3.1 Univariate Analysis

# ### 3.1.1 Plotting the Price of all the houses in the dataset

# In[ ]:


# Initializing a figure
plt.figure(figsize=(20,8))

# Initializing a subplot
plt.subplot(1,2,1)
# setting the title of the plot
plt.title('House Price Histogram')
# Plotting a Histogram for price column
sns.distplot(df_train.SalePrice, kde=False, fit=stats.lognorm)
plt.ylabel('Frequency')

# Initializing another subplot
plt.subplot(1,2,2)
# setting the title of the plot
plt.title('House Price Box Plot')
# Plotting a boxplot for price column
sns.boxplot(y=df_train.SalePrice)

plt.show()


# In[ ]:


# Checking the various percentile values for the price column
print(df_train.SalePrice.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1]))


# In[ ]:


#skewness
print("Skewness: " + str(df_train['SalePrice'].skew()))


# #### Inference :
# 
# 1. The house price's looks to be `right-skewed` as majority of the house prices are `low` (Below 250,000).
# 2. There is a significant difference between the `mean` and the `median` of the price distribution.
# 3. `Large Standard deviation` indicates `high variance` in the house prices (85% of the prices are below 250,000, whereas the remaining 15% are between 250,000 and 755,000).
# 
# **`Note:`** There are some outliers in the Price as well but we will not remove them for now. Also the target variable is highly skewed. We will treat this later on. 

# ### 3.1.2 Visualising Numeric Variables
# 
# For Visualization, we will 1st find all the numerical columns and then make `scatterplots` for all of them.

# In[ ]:


# Finding all the numerical columns in the dataset. 
numCols = df_train.select_dtypes(include=['int64','float'])

# Sorting the columns
numCols = numCols[sorted(numCols.columns)]

# Printing the columns
print(numCols.columns)
print("Numerical features : " + str(len(numCols.columns)))


# In[ ]:


# Initializing a figure
plt.figure(figsize=(30,200))

# Dropping the price column from the plot since we dont need to plot a scatter plot for price
numCols = numCols.drop('SalePrice',axis=1)

# running a for-loop to print the scatter plots for all numerical columns
for i in range(len(numCols.columns)):
    # Creating a sub plot
    plt.subplot(len(numCols.columns),2,i+1)
    # Creating a scatter plot
    plt.scatter(df_train[numCols.columns[i]],df_train['SalePrice'])
    # Assigning a title to the plot
    plt.title(numCols.columns[i]+' vs Price')
    # Setting the y label
    plt.ylabel('Price')
    # setting the x label
    plt.xlabel(numCols.columns[i])


# printing all the plots
plt.tight_layout()


# ## Inference:
# - `1stFlrSF, 2ndFlrSF, GarageArea, GrLivArea, GarageYrBlt,LotFrontage, LotArea,OverallQual,TotalBsmtSF,WoodDeckSF` seem to be positively correlated to price
# - Majority of the values in `3SsnPorch, LowQualFinSF, MiscVal, PoolArea,ScreenPorch` are 0 hence we can take a call to delete these columns if the columns are heavily skewed.
# - `BedroomAbvGr,MoSold` seems to have less correlation with price.
# - `BsmtFinSF1, BsmtFinSF2,BsmtFullBath, BsmtHalfBath, BsmtUnfSF,Enclosed Porch, HalfBath, Fireplaces, FullBath, GarageCars, MSSubClass, MasVnrArea, OpenPorchSF,OverallCond,TotRmsAbvGrd, YearBuilt, YearRemodAdd,YrSold` seems to have some correlation with price.
# - Majority of the values in `kitchenAbvGr` are 1 hence we can take a call to delete this columns if the column is heavily skewed.

# ### 3.1.3 Visualising Categorical Variables
# 
# In order to visualize the Categorical Variables, we will make `Histograms and Boxplots`.

# In[ ]:


# Finding the categorical columns and printing the same.
categCols = df_train.select_dtypes(exclude=['int64','float64'])
# Sorting the columns
categCols = categCols[sorted(categCols.columns)]
# printing the columns
print(categCols.columns)


# In[ ]:


# Initializing a figure
plt.figure(figsize=(15,100))

# Initializing a variable for plotting multiple sub plots
n=0

# running a for-loop to print the histogram and boxplots for all categorical columns
for i in range(len(categCols.columns)):
    # Increasing the count of the variable n
    n+=1
    # Creating a 1st sub plot
    plt.subplot(len(categCols.columns),2,n)
    # Creating a Histogram as the 1st plot for the column
    sns.countplot(df_train[categCols.columns[i]])
    # assigning x label rotation for carName column for proper visibility
    if categCols.columns[i]=='Exterior1st' or categCols.columns[i]=='Exterior2nd' or categCols.columns[i]=='Neighborhood':        plt.xticks(rotation=75)
    else:
        plt.xticks(rotation=0)
    # Assigning a title to the plot
    plt.title(categCols.columns[i]+' Histogram')
    
    # Increasing the count of the variable n to plot the box plot for the same column
    n+=1
    
    # Creating a 2nd sub plot
    plt.subplot(len(categCols.columns),2,n)
    # Creating a Boxplot as the 2nd plot for the column
    sns.boxplot(x=df_train[categCols.columns[i]], y=df_train1.SalePrice)
    # Assigning a title to the plot
    plt.title(categCols.columns[i]+' vs Price')
    # assigning x label rotation for carName column for proper visibility
    if categCols.columns[i]=='Exterior1st' or categCols.columns[i]=='Exterior2nd' or categCols.columns[i]=='Neighborhood':
        plt.xticks(rotation=75)
    else:
        plt.xticks(rotation=0)
        

# printing all the plots
plt.tight_layout()


# ## Inference:
# - Majority of the values in BldgType are 1Fam i.e Single-family Detached.
# - Majority of the values in BsmtCond are TA i.e typical condition.
# - Majority of the values in BsmtExposure are No i.e No Exposure.
# - Majority of the values in BsmtFinType1 are GLQ (Good Living Quarters) and Unf (Unfinshed). The GLQ are highly priced as compared to other Ratings.
# - Majority of the values in BsmtFinType2 are Unf which means that the second basement is mostly unfinished. The ALQ rating basement has the highest quantile range.
# - Majority of the values in BsmtQual are Gd and TA which means that the average height of the basement is above 80 inches. Ex(Excellent) has the highest price range.
# - Majority of the houses have Central air conditioning and hence have the highest price range as well.
# - Conditon1 and Condition2 of the houses in majority are Normal. Artery has a good price range as compared to others except Normal condition.
# - Majority of the houses have SBrkr Electrical system.
# - Majority of the external conditions of the materials are Average/Typical. These and Good external conditions have the highest price ranges.
# - The external quality of the materials on an average is Typical. Good and Excellent condition have the highest price ranges which is as expected.
# - Exterior covering on majority of the houses are Vinyl Siding followed by Metal Siding and Hard Board. Hard Board and Vinyl Siding have the highest price range.
# - Majority of the houses have No Fireplace. If they have then they are in Good and Typical condition and hence these houses attract more prices.
# - The type of foundation is more of Poured Contrete	and Cinder Block. Poured Contrete have the highest price range.
# - The home functionality, Garage Condition, GarageQual is typical in majority of the cases. We can think of deleting these columns if they are heavily skewed.
# - The Garage Finish is majorly Unfinished but furnished garages have a higher price.
# - Majority of the GarageType are attached to the home and attract the highest house price as well.
# - Majority of the houses are Gas forced warm air furnace heated. We can think of deleting this column if it is heavily skewed.
# - Majority of the houses have Excellent Heating quality and hance have the highest price range.
# - Majority of the houses are 1 story tall followed by 2 Stories. 2Stories houses are most priced followed by 1 story houses.
# - Majority of the houses have typical Kitchen quality but the excellent quality attract the highest prices.
# - Majority of the houses are on a leveled or flat land and have a Gentle slope. We can think of deleting this column if it is heavily skewed.
# - Majority of the houses have an inside lot and are regular in shape. We can think of deleting this column if it is heavily skewed.
# - Majority of the houses are from Residential Low Density zone and these attract the highest prices as well.
# - Majority of the houses do not have Masonry veneer done and the ones which have it's Brick Common. These 2 have the highest price range as well.
# - Majority of the houese have been bought in North Ames area followed by College Creek. Northridge and Northridge Heights neighbourhood have the highest price ranges.
# - Majority of the houses have a Paved Driveway and a Paved Street. We can think of deleting this column if it is heavily skewed.
# - Majority of the houses have the roof Material as Standard Shingle and the type as Gable. We can think of deleting this column if it is heavily skewed.
# - Majority of the houses has had a Normal with Warranty Deed - Conventional Sale. We can think of deleting this column if it is heavily skewed.
# - Majority of the houses have All public utilities available (E,G,W,& S). We can think of deleting this column if it is heavily skewed.

# ## Conclusion:
# 
# Based on the EDA, we can easily drop some columns which are highly skewed since they will not help us in our model building.

# In[ ]:


# pulling all the columns which can be deleted based on skewness
cols_to_drop = ['Utilities','3SsnPorch','LowQualFinSF','MiscVal','PoolArea','ScreenPorch','KitchenAbvGr','GarageQual'
               ,'GarageCond','Functional','Heating','LandContour','LandSlope','LotConfig','MSZoning','PavedDrive',
                'RoofMatl','RoofStyle','SaleCondition','SaleType','Street']

# running the for loop to print the value counts
for i in cols_to_drop:
    print(df_train[i].value_counts(normalize=True) * 100)


# Looking at the distribution of the values in the above mentioned columns, we have taken a decision of `deleting all these columns which have a single value of >80%` since they will not help us in our model building.

# In[ ]:


# dropping the columns 
df_train = df_train.drop(['Utilities','3SsnPorch','LowQualFinSF','MiscVal','PoolArea','ScreenPorch','KitchenAbvGr','GarageQual'
               ,'GarageCond','Functional','Heating','LandContour','MSZoning','PavedDrive',
                'RoofStyle','SaleCondition','SaleType','Street','BedroomAbvGr','MoSold'],axis=1)


# In[ ]:


# checking the shape of the df now
df_train.shape


# #### Let's check the correlation coefficients to see which variables are highly correlated

# In[ ]:


# saleprice correlation matrix
corr_num = 15 #number of variables for heatmap
corrmat = df_train.corr()
cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index
corr_mat_sales = np.corrcoef(df_train[cols_corr].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)
plt.show()


# #### Inference:
# - We can see that OverAllQual, GrLivArea and GarageCars are highly correlated with Sales Price.
# 

# # Step 4: Data Preparation

# Ok, now that we have dealt with all the missing values and the uncorrelated columns, it looks like it's time for some feature engineering, the second part of our data preprocessing. We need to create feature vectors in order to get the data ready to be fed into our model as training data. This requires us to convert the categorical values into representative numbers..

# ## 4.1 Check for skewness

# First, let's take a look at our target.

# In[ ]:


# Initializing a figure
plt.figure(figsize=(20,8))

# Initializing a subplot
plt.subplot(1,2,1)
# setting the title of the plot
plt.title('House Price Histogram')
# Plotting a Histogram for price column
sns.distplot(df_train.SalePrice, kde=False, fit=stats.lognorm)
plt.ylabel('Frequency')


# Since the data is skewed, we will try to fix this with a `log transformation`.

# In[ ]:


# Checking if the log transformation normalizes the target variable
sns.distplot(np.log(df_train["SalePrice"]))


# ### Inference:
# It appears that the target, SalePrice, is very skewed and a transformation like a logarithm would make it more normally distributed. Machine Learning models tend to work much better with normally distributed targets, rather than greatly skewed targets. By transforming the prices, we can boost model performance.

# In[ ]:


# Applying the log transformation to the target variable
df_train["SalePrice"] = np.log(df_train["SalePrice"])


# Now lets check for skewness for all columns

# In[ ]:


# importing the skew library to check the skewness
from scipy.stats import skew  


# In[ ]:


# pulling the numeric columns from the dataset 
numeric_feats = df_train.dtypes[df_train.dtypes != "object"].index

skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

skewed_feats


# #### Inference:
# We can see that there are columns which have skewness in them but we will leave these for now.

# ## 4.2 Feature Engineering

# Now looking at the data dictionary, we can see that there are some columns which can be merged to create new features.
# 
# Let's do that now.

# In[ ]:


# Lets combine the floors square feet and the basement square feet to create the total sq feet
df_train['Total_sq_feet'] = (df_train['BsmtFinSF1'] + df_train['BsmtFinSF2'] +
                                     df_train['1stFlrSF'] + df_train['2ndFlrSF'])

# Lets combine all the bathrooms square feet to create the total bathroom sq feet
df_train['Total_Bathrooms_sq_feet'] = (df_train['FullBath'] + (0.5 * df_train['HalfBath']) +
                                   df_train['BsmtFullBath'] + (0.5 * df_train['BsmtHalfBath']))

# Lets combine all the porch square feet to create the total porch sq feet
df_train['Total_porch_sq_feet'] = (df_train['OpenPorchSF'] + df_train['EnclosedPorch'] + df_train['WoodDeckSF'])


# In[ ]:


# checking the shape of the df now
df_train.shape


# In[ ]:


# lets drop the columns which we used to create new features
df_train= df_train.drop(['BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','FullBath','HalfBath',
                           'BsmtFullBath','BsmtHalfBath','OpenPorchSF','EnclosedPorch','WoodDeckSF'],axis=1)


# In[ ]:


# checking the df now
df_train.head()


# In[ ]:


# checking the shape now
df_train.shape


# Now we can also see that there are 4 `YEAR columns` in the dataset. In order to handle them we will convert them as well by finding the number of years.

# In[ ]:


# pulling the list of all the year columns from the dataset
Year_cols = df_train.filter(regex='Yr|Year').columns
# running a for loop to find the max year of each column 
for i in Year_cols:
    i = df_train[i].max()
    print(i)


# Since the max values for all the year columns are the same, we will now convert the year columns by `subtracting the max year date with all the values in the 4 columns`.

# In[ ]:


# running a for loop to subtract the max year with all values
for i in Year_cols:
    df_train[i] = df_train[i].apply(lambda x: 2010 - x)


# In[ ]:


# Checking the dataset now
df_train.head()


# ## 4.3 Creating dummies

# We will now create dummy variables for all the categorical variables in order to conveert them to numerical so that the model could be built for the same.

# In[ ]:


# pulling all the categorical columns from the dataset.
categCols = df_train.select_dtypes(exclude=['int64','float64'])
# Sorting the columns
categCols = categCols[sorted(categCols.columns)]
# printing the categorical columns
print(categCols.columns)


# In[ ]:


# Defining the map function
def dummies(x,df):
    # Get the dummy variables for the categorical feature and store it in a new variable - 'dummy'
    dummy = pd.get_dummies(df[x], drop_first = True)
    for i in dummy.columns:
        dummy = dummy.rename(columns={i: x+"_"+i})
    # Add the results to the original dataframe
    df = pd.concat([df, dummy], axis = 1)
    # Drop the original category variables as dummy are already created
    df.drop([x], axis = 1, inplace = True)
    # return the df
    return df

#Applying the function to the df_train categorical columns
for i in categCols:
    df_train = dummies(i,df_train)


# In[ ]:


# checking the dataset now
df_train.head()


# As we can see all the categorical values have been expanded and representated as 0's & 1's. This step is crucial to build a robust linear regression model.

# In[ ]:


# Checking the shape of the new dataset which will be used for model building
df_train.shape


# ## Step 5: Model Building

# ### 5.1 Rescaling the Features  
# 
# It is extremely important to rescale the variables so that they have a comparable scale. If we don't have comparable scales, then some of the coefficients as obtained by fitting the regression model might be very large or very small as compared to the other coefficients.
# 
# Now, there are two common ways of rescaling:
# 
# 1. Min-Max scaling 
# 2. Standardisation (mean-0, sigma-1) 
# 
# We will use `Standardisation scaling`. In this, for all the columns the mean will be 0.

# In[ ]:


# importing the libraries
from sklearn.preprocessing import StandardScaler
# dropping the target variable and stroing the remaining in a new df
X = df_train.drop(['SalePrice'],axis=1)
# storing the target column in a new df
y = df_train['SalePrice']
# initializing the standard scalar
scaler = StandardScaler()
# scale the X df
scaler.fit(X)


# ### 5.2 Remove Multicolinearity

# Now before we move forward, we should also remove multicolinearity. 
# 
# We will do this by checking VIF and removing all the highly correlated columns since they would be redundent in our model.

# In[ ]:


# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif.loc[vif['VIF'] > 3000, :]


# #### Inference:
# Dropping the columns which have inf VIF value.

# In[ ]:


# Dropping the above columns
X=X.drop(['Exterior2nd_CBlock','BsmtUnfSF','BsmtQual_No','Total_sq_feet','Exterior1st_CBlock','BsmtCond_No'
         ,'TotalBsmtSF','GrLivArea','GarageFinish_No','GarageType_No','BsmtFinType1_No'],axis=1)


# Let's check the VIF values again

# In[ ]:


# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
# X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif.loc[vif['VIF'] > 100, :]


# #### Inference:
# Dropping the columns which have VIF value>100.

# In[ ]:


X=X.drop(['ExterCond_TA','Condition2_Norm','Exterior1st_VinylSd','Exterior2nd_VinylSd','Exterior1st_MetalSd','Exterior2nd_MetalSd'
         ,'Exterior1st_HdBoard'],axis=1)


# Let's check the VIF values again

# In[ ]:


# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
# X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ### 5.3 Splitting the Data into Training and Testing sets

# We now need to split our variable into training and testing sets. We'll perform this by importing `train_test_split` from the `sklearn.model_selection` library. It is usually a good practice to keep 70% of the data in the train dataset and the rest 30% in the test dataset which is what we will follow as well.

# In[ ]:


# importing the required libraries
from sklearn.model_selection import train_test_split

# splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 100)


# In[ ]:


# Checking the number of columns and rows in the train dataset
X_train.shape


# In[ ]:


# Checking the number of columns and rows in the test dataset
X_test.shape


# #### Model 1:
# 
# Lets first run linear regression on the dataset and check what kind of results we get.

# In[ ]:


# linear regression
lm = LinearRegression()
lm.fit(X_train, y_train)

# predict
y_train_pred = lm.predict(X_train)

# print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
print("RMSE Train {}".format(np.sqrt(mean_squared_error(y_train, y_train_pred))))
print("R2 Score Train {}".format(r2_score(y_train, y_train_pred)))
y_test_pred = lm.predict(X_test)
# print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
print("RMSE Test {}".format(np.sqrt(mean_squared_error(y_test, y_test_pred))))
print("R2 Score Test {}".format(r2_score(y_test, y_test_pred)))


# #### Inference:
# As we can see from the above our `train R2 score is 92.68 and test R2 is 87.61`.

# Lets check the different features and their respective coefficients value

# In[ ]:


# model coefficients
# liner regression model parameters
model_parameters = list(lm.coef_)
model_parameters.insert(0, lm.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))


# ## Step 6: Ridge and Lasso Regression
# 
# Since we have multiple features and a big difference between the R2 score of Train and Test set, we will try to make this better by performing Advanced Regression Techniques.
# 
# The 2 that we will use here are:
# - Ridge Regression
# - Lasso Regression

# ## 6.1 Ridge Regression

# In[ ]:


# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}

# Initializing the Ridge regression
ridge = Ridge()

# cross validation
# Setting the number of folds
folds = 5
# performing GridSearchCV on the ridge regression using the list of params
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
# Fitting the model on our Train sets
model_cv.fit(X_train, y_train) 


# In[ ]:


# Storing the results in a new df
cv_results = pd.DataFrame(model_cv.cv_results_)
# filtering out the alpha parameters which are less than 200
cv_results = cv_results[cv_results['param_alpha']<=200]
# checking the results
cv_results.head()


# Lets plot the above values so that we can better visualize the results.

# In[ ]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
# plotting the mean train scores
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
# plotting the mean test scores
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
# setting the x label
plt.xlabel('alpha')
# setting the y label
plt.ylabel('Negative Mean Absolute Error')
# setting the title
plt.title("Negative Mean Absolute Error and alpha")
# setting the legend
plt.legend(['train score', 'test score'], loc='upper left')
# showing the plot
plt.show()


# In[ ]:


# finding the best Alpha value
print ('The best value of Alpha for Ridge Regression is: ',model_cv.best_params_)


# ### HyperParameter tuning

# Since we now know that the best Alpha (Regularization term) value is 7, we will now build our model with the same.

# In[ ]:


# setting the value of alpha as 7
alpha = 7
# initializing the ridge regression with the optimized alpha value
ridge = Ridge(alpha=alpha)

# running the ridge algo on the train datasets
ridge.fit(X_train, y_train)

# Lets predict
y_train_pred = ridge.predict(X_train)
print("RMSE Train {}".format(np.sqrt(mean_squared_error(y_train, y_train_pred))))
print("R2 Score Train {}".format(r2_score(y_train, y_train_pred)))
y_test_pred = ridge.predict(X_test)
print("RMSE Test {}".format(np.sqrt(mean_squared_error(y_test, y_test_pred))))
print("R2 Score Test {}".format(r2_score(y_test, y_test_pred)))


# As we can see from the above, the R2 score of `Train set is 0.92 and test is 0.88` for Ridge Regression.

# In[ ]:


# checking the coefficient values of all the features.
ridge.coef_


# In[ ]:


# Assigning the columns to the respective coefficient values
# ridge model parameters
model_parameters = list(ridge.coef_)
model_parameters.insert(0, ridge.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))


# Let's Plot the above and find the top 10 features of Ridge regression

# In[ ]:


# pulling the coefficients and index and creating a new df
coef = pd.Series(ridge.coef_, index = X.columns).sort_values()
# filtering the top 5 positive and negative features 
ridge_imp_coef = pd.concat([coef.head(10), coef.tail(10)])
# plotting the graph
ridge_imp_coef.plot(kind = "barh")
# setting the title of the plot
plt.title("Model Coefficients")


# In[ ]:


# Converting the important feature list into a df for better understanding
ridge_imp_coef = ridge_imp_coef.to_frame('Coeff_val').reset_index()
ridge_imp_coef.columns = ['Features', 'Coeff_val']
ridge_imp_coef['Coeff_val'] = ridge_imp_coef['Coeff_val'].abs()
ridge_imp_coef = ridge_imp_coef.sort_values(by=['Coeff_val'], ascending=False)
ridge_imp_coef.head(10)


# In[ ]:


p_pred = np.expm1(ridge.predict(X))
plt.scatter(p_pred, np.expm1(y))
plt.plot([min(p_pred),max(p_pred)], [min(p_pred),max(p_pred)], c="red")


# #### Inference:
# As we can see that our predicted line is passing through almost the entire dataset.

# ## 6.2 Lasso Regression

# Now lets run Lasso Regression on the dataset.

# In[ ]:


# initializing the Lasso regression
lasso = Lasso()

# cross validation
# performing GridSearchCV on the lasso regression using the list of params
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
# Fitting the model on our Train sets
model_cv.fit(X_train, y_train) 


# In[ ]:


# Storing the results in a new df
cv_results = pd.DataFrame(model_cv.cv_results_)

# checking the results
cv_results.head()


# In[ ]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
# plotting the mean train scores
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
# plotting the mean test scores
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
# setting the x label
plt.xlabel('alpha')
# setting the y label
plt.ylabel('Negative Mean Absolute Error')
# setting the title
plt.title("Negative Mean Absolute Error and alpha")
# setting the legend
plt.legend(['train score', 'test score'], loc='upper left')
# showing the plot
plt.show()


# Since we are unable to read the above graph properly, we will convert the x-scale into log

# In[ ]:


cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')
# plotting the mean train scores
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
# plotting the mean test scores
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
# setting the x label
plt.xlabel('alpha')
# setting the y label
plt.ylabel('Negative Mean Absolute Error')
# setting the xscale into log
plt.xscale('log')
# setting the title
plt.title("Negative Mean Absolute Error and alpha")
# setting the legend
plt.legend(['train score', 'test score'], loc='upper left')
# showing the plot
plt.show()


# In[ ]:


print('The best value of Alpha for Lasso Regression is: ',model_cv.best_params_)


# ### HyperParameter tuning

# Now we will build our model with the optimized value of alpha for Lasso regression i.e 0.001

# In[ ]:


# initializing the ridge regression with the optimized alpha value
lm = Lasso(alpha=0.001)
# fitting the model on the train datasets
lm.fit(X_train, y_train)

# predict
y_train_pred = lm.predict(X_train)
print("RMSE Train {}".format(np.sqrt(mean_squared_error(y_train, y_train_pred))))
print("R2 Score Train {}".format(r2_score(y_train, y_train_pred)))
y_test_pred = lm.predict(X_test)
print("RMSE Test {}".format(np.sqrt(mean_squared_error(y_test, y_test_pred))))
print("R2 Score Test {}".format(r2_score(y_test, y_test_pred)))


# As we can see from the above, the R2 score of `Train set is 0.90 and test is 0.88` for Lasso Regression.

# In[ ]:


# checking the coefficient values of all the features.
lm.coef_


# In[ ]:


# Assigning the columns to the respective coefficient values
# lasso model parameters
model_parameters = list(lm.coef_)
model_parameters.insert(0, lm.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))


# In[ ]:


# pulling the coefficients and index and creating a new df
coef = pd.Series(lm.coef_, index = X.columns).sort_values()
# filtering the top 5 positive and negative features 
lasso_imp_coef = pd.concat([coef.head(10), coef.tail(10)])
# plotting the graph
lasso_imp_coef.plot(kind = "barh")
# setting the title of the plot
plt.title("Model Coefficients")


# In[ ]:


# Converting the important feature list into a df for better understanding
lasso_imp_coef = lasso_imp_coef.to_frame('Coeff_val').reset_index()
lasso_imp_coef.columns = ['Features', 'Coeff_val']
lasso_imp_coef['Coeff_val'] = lasso_imp_coef['Coeff_val'].abs()
lasso_imp_coef = lasso_imp_coef.sort_values(by=['Coeff_val'], ascending=False)
lasso_imp_coef.head(10)


# Lets visualize the fit after the modelling

# In[ ]:


p_pred = np.expm1(lm.predict(X))
plt.scatter(p_pred, np.expm1(y))
plt.plot([min(p_pred),max(p_pred)], [min(p_pred),max(p_pred)], c="red")


# In[ ]:


# checking how many features were dropped by lasso during modelling
print("Lasso kept",sum(coef != 0), "important features and dropped the other", sum(coef == 0),"features")


# # Final Conclusion:
# 
# Based on our regression results, below are the top 10 features which drive the Sales prices of the Houses in Australia.
# 
# - **Ridge Regression**:
#     - We can see that the Train and Test R2 value was `0.92 and 0.88` respectively.
#     - The top 10 features that drive the house prices as per Ridge regression are :
#         - Neighborhood_Crawfor
#         - Exterior1st_BrkFace
#         - Neighborhood_StoneBr
#         - BsmtQual_Fa
#         - Neighborhood_MeadowV
#         - OverallQual
#         - Neighborhood_BrDale
#         - Neighborhood_NoRidge
#         - Neighborhood_Gilbert
#         - KitchenQual_TA
#         
#         
# - **Lasso Regression**:
#     - We can see that the Train and Test R2 value was `0.90 and 0.88` respectively.
#     - The top 10 features that drive the house prices as per Lasso regression are:
#         - Neighborhood_Crawfor
#         - Exterior1st_BrkFace
#         - OverallQual
#         - Neighborhood_StoneBr
#         - Total_Bathrooms_sq_feet
#         - KitchenQual_TA
#         - TotRmsAbvGrd
#         - Foundation_PConc
#         - Fireplaces
#         - Neighborhood_NoRidge

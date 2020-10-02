#!/usr/bin/env python
# coding: utf-8

# **Exploratory Data Analysis**

# In[ ]:


#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

sns.set(style='darkgrid')


# In[ ]:


#importing data
df_train = pd.read_csv("../input/train.csv")
df_train.head()


# In[ ]:


#let's take a look at columns
print(df_train.columns)

#total number of columns
print(len(df_train.columns))


# In[ ]:


#size of the data at hand
df_train.size


# **Metadata**
# 
# Now let's create some metadata about the dataset we have at hand.
# This will include column names,data-type for each column, number of missing values etc.
# Metadata comes in handy to get a basic feel of the data so that we can plan on cleaning it accordingly and imputing missing values if necessary.
# 

# In[ ]:


metadata=pd.DataFrame(columns=['Column Name','Type','Missing Values','Unique Values'])

metadata['Column Name'] = df_train.columns

metadata.Type=df_train.dtypes
metadata.Type=list(df_train.dtypes)

misval=df_train.isna().sum()
metadata['Missing Values']=list(misval)

nun=df_train.nunique()
metadata['Unique Values']=list(nun)

metadata['Missing Values %']=metadata['Missing Values']*100/len(df_train)

metadata


# In[ ]:


#number of columns with missing data
print(len(metadata.loc[metadata['Missing Values %'] >= 1]))

#listing columns with data missing
metadata.loc[metadata['Missing Values %'] >=1]


# We have 5 columns that are having more than 30% of the values missing.
# We will be better off dropping them rather than imputing them.

# In[ ]:


#dropping columns with more than 30% of the data missing
missing_data = metadata.loc[metadata['Missing Values %'] > 0]["Column Name"]
df_train.drop(missing_data,inplace=True,axis=1)


# In[ ]:


#dropping column Id as well
df_train.drop("Id",inplace=True,axis=1)


# In[ ]:


#number of columns we are left with
print("Number of columns left in df_train {}".format(len(df_train.columns)))


# Now that missing values out of our way. Let's get going with some exploration.
# 
# Following are the steps we follow to explore the data:
# 1. Univariate Analysis
# 2. Bivariate Analysis
# 3. Multivariate Analysis
# 4. Segmented Analysis
# 5. Deriving Metrics

# *Note: Missing values are dealt with only considering EDA as the goal. For building a model, we will reconsider the treatment of missing values.*

# **Univariate Analysis**
# 
# Let's study the feature variables and target variable individually.
# 
# To get a visual understanding of the distribution of feature variables,plotting boxplots and distplots
# will be a good start.

# In[ ]:


#function to create boxplot and distribution plot for columns with numerical data
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
dimens = (20,10)
def create_distribution_plots(column_name,box_log_scale = False,dist_log_scale = False):
    fig, ax = plt.subplots(figsize=dimens,ncols = 2,nrows=1)
    ax[0].set_title('{} Univariate Analysis'.format(column_name))
    ax[0] = sns.boxplot(x=df_train[column_name],orient = 'v',ax=ax[0])
    
    ax[1].set_title('{} Univariate Analysis'.format(column_name))
    ax[1] = sns.distplot(df_train[column_name],ax=ax[1])
    
    if(box_log_scale):
        ax[0].set_yscale('log')
        
    if(dist_log_scale):
        ax[1].set_yscale('log')


# In[ ]:


# LotArea
create_distribution_plots('LotArea',box_log_scale = True)


# Around 50% of the houses have lot area of ~10,000 sq.feet with few outliers having Lot area beyond ~100,000 sq.feet.
# 
# Similarly distplot shows peak around 10,000 sq.feet.

# In[ ]:


# BsmtFinSF1

create_distribution_plots('BsmtFinSF1')


# Majority of the houses have BsmntFinSF1 between 0-1000 sq.feet.

# In[ ]:


#BsmtFinSF2
create_distribution_plots('BsmtFinSF2')


# Most of the houses have BsmtFinSF2 area have 0 sq.feet.

# In[ ]:


#BsmtFinSF2

create_distribution_plots('BsmtUnfSF')


# Around 75% of houses have Unfinished Basement Area less than 1000 sq. feet.

# In[ ]:


#TotalBsmtSF

create_distribution_plots('TotalBsmtSF')


# Except some outliers almost all houses have their Basement Area within 0-2000 sq.feet range with peak at 1000 sq.feet.

# In[ ]:


#1stFlrSF

create_distribution_plots('1stFlrSF')


# 1stFlrSF shows distribution similar to TotalBsmtSF. Most of the houses have First Floor area ranging between 0-2000 sq.feet with second quartile around ~1100 sq.feet

# In[ ]:


#2ndFlrSF
create_distribution_plots('2ndFlrSF')


# With peak around 0 sq.feet,houses that have a second floor have it's area ranging from ~400 sq. feet to ~1500 sq.feet with outliers going beyond ~1500 sq.feet

# In[ ]:


#GrLivArea
create_distribution_plots('GrLivArea')


# The GrLivArea ranges from around ~300 sq.feet to ~3500 sq. feet.

# In[ ]:


#GarageArea

create_distribution_plots('GarageArea')


# 50% of the houses have have garage area of 500 sq.feet as shown by the peak in the distplot.

# In[ ]:


#WoodDeckSF
create_distribution_plots('WoodDeckSF')


# In[ ]:


#OpenPorchSF

create_distribution_plots('OpenPorchSF')


# In[ ]:


#EnclosedPorch

create_distribution_plots('EnclosedPorch')


# Almost all houses have no EnclosedPorch with some exceptions ofcourse

# In[ ]:


#3SsnPorch
create_distribution_plots('3SsnPorch')


# Similarly most houses have no 3SsnPorch.

# In[ ]:


#ScreenPorch

create_distribution_plots('ScreenPorch')


# Apart from few exceptions, ScreenPorch is not present for most of the houses

# In[ ]:


#PoolArea

create_distribution_plots('PoolArea')


# Almost all houses have no pools. Finally let's take a look at variable of our interest,the target variable SalePrice.
# 

# In[ ]:


#SalePrice

create_distribution_plots('SalePrice')


# SalesPrice has got a wide range with a normal distribution around ~150,000 as median.

# Now that we saw all the numerical variables and their distributions we got a fair idea of how these individual variables are distributed.
# Futher now let's take a look at categorical variables and understand their frequency with seaborn's countplot. This will help us in understanding what values of those categorical variables are occuring the most.

# In[ ]:


#MSSubClass

sns.countplot(df_train['MSSubClass'])


# Maximum of the houses (500 +) have MSSubClass as 20 with MSSubClass as 60 being second in line

# In[ ]:


#OverallQual
sns.countplot(df_train['OverallQual'])


# Most number of houses have rating of 5 when it comes to Material and Quality with Rating of 6 and 7 as second most and third most respectively.

# In[ ]:


#OverallCond
sns.countplot(df_train['OverallCond'])


# Overall condition of maximum houses can be rated as 5,with 6 and 7 next in line

# In[ ]:


#BedroomAbvGr,TotRmsAbvGrd,KitchenAbvGr,Fireplaces
fig, ax = plt.subplots(figsize=(10,10),ncols = 2,nrows=2)
ax[0][0] = sns.countplot(df_train['BedroomAbvGr'],ax=ax[0][0])

ax[0][1] = sns.countplot(df_train['TotRmsAbvGrd'],ax=ax[0][1])

ax[1][0] = sns.countplot(df_train['KitchenAbvGr'],ax=ax[1][0])

ax[1][1] = sns.countplot(df_train['Fireplaces'],ax=ax[1][1])


# In[ ]:


#FullBath,HalfBath,BsmtFullBath,BsmtHalfBath
fig, ax = plt.subplots(figsize=(10,10),ncols = 2,nrows=2)
ax[0][0] = sns.countplot(df_train['FullBath'],ax=ax[0][0])

ax[0][1] = sns.countplot(df_train['HalfBath'],ax=ax[0][1])

ax[1][0] = sns.countplot(df_train['BsmtFullBath'],ax=ax[1][0])

ax[1][1] = sns.countplot(df_train['BsmtHalfBath'],ax=ax[1][1])


# There are few features like 'YearBuilt' that we can bin them to get better insight into them.

# In[ ]:


bins = [1872, 1900,1930,1960, 1990, 2000, 2005,2010]
labels = ['before 1900','1900-30','1930-60', '1960-90', '1990-2000', '2000-05', '2005-10']
df_train['YearBuiltBins'] = pd.cut(df_train['YearBuilt'], bins, labels=labels)


# In[ ]:


#lets plot the countplot for this newly created binned feature
fig, ax = plt.subplots(figsize=(10,5))
ax = sns.countplot(df_train['YearBuiltBins'],ax=ax)


# In[ ]:


#YrSold
sns.countplot(df_train['YrSold'])


# In[ ]:


#MoSold
sns.countplot(df_train['MoSold'])


# ### Bivariate Analysis
# 
# Let's analyse these variables in relation to the target variable 'SalePrice'
# 

# In[ ]:


#let's first plot scatter plot of the numerical variables with Sale Price
numerical_features = ['LotArea','TotalBsmtSF','GrLivArea','PoolArea','GarageArea', 'WoodDeckSF','1stFlrSF', '2ndFlrSF','SalePrice']

sns.pairplot(df_train[numerical_features])


# One thing we can see is  GrLivArea vs 1stFlrSF where almost a straight line is formed and all other points lie above that line. It shows that GrLivArea is atleast as much as 1stFlrSF or more than that but never less than that.
# 
# If we take a look at the scatter plots for * vs SalePrice we can notice following :
# 
# - GrLivArea, 2ndFlrSF, 1stFlrSF,GarageArea are almost linearly related to SalePrice.
# 
# - **SalePrice vs PoolArea**
#   - PoolArea is 0 for almost all data points.Apart from few outliers it does not affect SalePrice much.Same is represented by the straight line created by data-points at PoolArea = 0 sq. feet.
#   
# - Data Points create a straight line at 0 for (2ndFlrSF,PoolArea,WoodDeckSF) vs Sale Price.In each case it represents that quite a bit of datapoints for (2ndFlrSF,PoolArea,WoodDeckSF) have a value of 0.
# - LotArea does not play a significant role in influencing SalePrice. Rather Livable Area Above Grade,GarageArea, 1st Floor Area,2nd Floor Area,Basements tend to influence SalePrice better than Lot Area.

# In[ ]:


print(df_train['PoolArea'].value_counts())
sns.boxplot(df_train['PoolArea'])


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(x=df_train['YearBuiltBins'], y =df_train['SalePrice'])


# As expected SalePrice is highest for houses built recently. Let's create a feature called 'Age' of the house and see how it relates

# In[ ]:


sns.barplot(x= df_train['YrSold'],y=df_train['SalePrice'])


# Though we can guess how OverallQual and OverallCond must be related to SalePrice,let's look at the relationship with a barplot.We will analyse the nature of this relationship with seaborn's lineplot.

# In[ ]:


sns.barplot(x= 'OverallQual',y='SalePrice',data=df_train)
sns.lineplot(x= 'OverallQual',y='SalePrice',data=df_train)


# This was expected. The SalePrice increases with increase in OverAllQual and the increase is exponential.

# In[ ]:


fig, ax = plt.subplots(figsize=(20,20))
plot = sns.catplot(x= 'OverallQual',y='SalePrice',data=df_train,ax=ax,kind='swarm',sharey=False)
plt.close(plot.fig)


# As the plot shows, for OverallQual of 4,5,6,7,8 most of the meat is in $100,000-200,000. For OverallQual of 9,10 SalePrice is pretty high. Most of houses with OverallQual of 9,10 have SalePrice from 300,000-500,000.
# 

# In[ ]:


sns.barplot(x= 'OverallCond',y='SalePrice',data=df_train)
sns.lineplot(x= 'OverallCond',y='SalePrice',data=df_train)


# In[ ]:


fig, ax = plt.subplots(figsize=(20,20))
plot = sns.catplot(x= 'OverallCond',y='SalePrice',data=df_train,ax=ax,kind='swarm',sharey=False)
plt.close(plot.fig)


# For OverallCond of 5, though most of the meat is in the $100,000-300,000 range. We can explore the outliers in the Multivariate Analysis section.

# #### Binning GrLivArea

# In[ ]:



#Let's bin the GrLivArea.
bins = [300,500,700,1000,1500,2000,2500,3000, 3500,4000,4500,5000,6000]

labels = ['< 500','300-500 ','500-1000', '1000-1500', '1500-2000', '2000-2500', '2500-3000','3000-3500','3500-4000','4000-4500','4500-5000','5000+']

df_train['GrLivAreaBins'] = pd.cut(df_train['GrLivArea'], bins, labels=labels)



# #### Deriving Age
# Let's create a variable Age which basically Age of House when sold given by YrSold - YrBuilt

# In[ ]:


df_train['Age'] = df_train['YrSold'] - df_train['YearBuilt']

#Newly derived Age feature                                             
df_train[['YrSold','YearBuilt','Age']]


# In[ ]:


#Let's further create bins for Age.
#Considering -1 as well in bin so that houses with YrSold and YearBuilt same will have Age as 0 and must be accomodated in some bin.
bins = [-1,0, 10,20,30, 40, 50, 60,80,100,120]
labels = ['0','0-10 ','10-20', '20-30', '30-40', '40-50', '50-60','60-80','80-100','100+']
df_train['AgeBins'] = pd.cut(df_train['Age'], bins, labels=labels)


# ### Multivariate Analysis
# 

# In[ ]:


fig, ax = plt.subplots(figsize=(20,20))
sns.barplot(x='AgeBins',y='SalePrice',hue='OverallQual',data=df_train,ax=ax)


# Houses with Age between 10-20 years with OverallQual 10 have sold with highest SalePrice.We will analyse it further as to why it is so.

# In[ ]:


fig, ax = plt.subplots(figsize=(20,20))
sns.barplot(x='AgeBins',y='SalePrice',hue='GrLivAreaBins',data=df_train,ax=ax)


# This barplot makes it clear why Houses with 10-20 years of Age sold at higher SalePrice than Houses with any other Age because the Houses with 10-20 years of Age had higher GrLivArea (4000-4500) than any other AgeBin

# Coming to OverallQual vs SalePrice with GrLivAreaBins as hue

# In[ ]:


fig, ax = plt.subplots(figsize=(20,20))
plot = sns.catplot(x= 'OverallQual',y='SalePrice',hue='GrLivAreaBins',data=df_train,ax=ax,kind='swarm',sharey=False)
plt.close(plot.fig)


# This is an interesting plot. Following things can be noticed from the Beeswarms here:
# 
# 1. Most houses with OverallQual of 4,5 have GrLivArea in the range of 500-1000 and 1000-1500 sq. feet
# 2. For OverallQual of 6 there is good mix of houses with GrLivArea in the range of 500-1000,1000-1500 and 1500-2000 sq. feet.
# 3. Most interesting thing here is that two houses with GrLivArea on the higher end (4500-5000 and 5000+ sq. feet) are having SalePrice less than $200,000. This is worth exploring further.

# In[ ]:


df_train.loc[(df_train['OverallQual'] ==10) & (df_train['SalePrice']  < 200000)][['GrLivArea','OverallCond','OverallQual','SaleCondition']]


# Also let's add 1stFlrSF and 2ndFlrSF to the mix since both of them are linearly related to SalePrice as evident from the pairplot in Bivariate Analysis section.

# In[ ]:


df_train.loc[(df_train['OverallQual'] ==10) & (df_train['SalePrice']  < 200000)][['GrLivArea','OverallCond','OverallQual','SaleCondition','1stFlrSF','2ndFlrSF']]


# From the above slice of df_train we can see that 1stFlrSF and 2ndFlrSF both have fair values.
# 
# Inspite of high GrLivArea and considerably high 1stFlrSF and 2ndFlrSF,SalePrice is still low.
# 
# Above dataframe shows that both the houses have OverallCond of 5 which is average. 
# 
# Also the houses were not complete during the time of sale as shown by SaleCondition=Partial. 
# 
# These two factors i.e OverallCond = 5 and SaleCondition= Partial must have contributed to low SalePrice inspite of high GrLivArea,1stFlrSF and 2ndFlrSF.

#  Further let's analyse outliers for  OverallCond of 5 since it's distribution is more spread out,as we saw in the Bivariate Analysis Section.

# In[ ]:


fig, ax = plt.subplots(figsize=(20,20))
plot = sns.catplot(x= 'OverallCond',y='SalePrice',data=df_train,ax=ax,kind='swarm',hue='GrLivAreaBins',sharey=False)
plt.close(plot.fig)


# In[ ]:


df_train.loc[(df_train['OverallCond'] == 5) & (df_train['SalePrice']  > 700000)][['GrLivArea','OverallCond','OverallQual','1stFlrSF','2ndFlrSF','GarageArea']]


# The outlier with OverallCond of 5 and SalePrice more than $700,000, has GrLivArea of 4476 sq.feet and OverQual of 10.
# 
# Also values  for 1stFlrSF,2ndFlrSF and GarageArea is fairly high.
# 
# Thus it is justified that the outlier has high SalePrice.

# Finally with a heatmap we will get overall idea of the correlation between features.
# 

# In[ ]:


fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(df_train.corr(),ax=ax)


# Few things can be noticed here:
# 
# 1. Positive Correlation
#     1. TotRmsAbvGrd and GrLivArea
#     2. TotalBsmntSF and 1stFLRSF
#     3. GarageArea and GarageCars
#   
#     
# 2. Negative Correlation
#    1. BsmtFullBath and BsmntUnfSF
#    2. BsmntFinSF1 and BsmntUnfSF
#    3. EnclosedPorch and YearBuilt
#    4. YearBuilt and Age
#    5. SalePrice and Age
#  
#   
#   

# We have to watch out for these features as they may contribute to multicollinearity while model building.

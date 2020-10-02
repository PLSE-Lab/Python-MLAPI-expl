#!/usr/bin/env python
# coding: utf-8

# <h6>Before starting <span class="label label-danger">IMPORTANT</span></h6>
# Throughout this project, I will be introducing some **reproducible functions** that can be used in any machine learning project to automate the process:
# * Data visualization: Functions `msv1`and `msv2`to visualize missing values.
# * Machine learning: Function `r_reg` does regression with regularization (Ridge, Lasso).
# 
# For more information, check out my [regression github repository](https://github.com/Amiiney/regression).
# ***
# <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">PROJECT CONTENT</h3>
#      
# > ### 1. EXPLORATORY DATA ANALYSIS
# 
# > ### 2. DATA CLEANING
# 
# > ### 3. FEATURE ENGINEERING
# 
# > ### 4. ENCODING CATEGORICAL FEATURES
# 
# > ### 5. DETECTING OUTLIERS
# 
# > ### 6. MACHINE LEARNING

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import warnings
warnings.simplefilter(action='ignore')

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import math
import sklearn.model_selection as ms
import sklearn.metrics as sklm


# ## Aim:
# ***
# The aim of this competition is to predict the sale price of residential homes in Ames, Iowa. We will practice feature engineering and regression algorithms to achieve the lowest prediction error (RMSE is the metric used in this competition).

# # 1- Exploratory data analysis
# ***
# Before starting, I would like to mention that I will be using alphabets to name my datasets to make my workflow easier:
#         
#          a=train
#          b=test
#          c= combined dataset (train+test)
# 
# We read and open our train and test datasets

# In[ ]:


a = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
b = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


#Use this code to show all the 163 columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# 
# Here is a glimpse of what we will be dealing with:
# * Many features, many missing values and one target feature: "SalePrice" which is the price of the houses we are supposed to predict

# In[ ]:


a.head()


# In[ ]:


print('The shape of our training set: %s houses and %s features'%(a.shape[0],a.shape[1]))
print('The shape of our training set: %s houses and %s features'%(b.shape[0],b.shape[1]))
print('The testing set has 1 feature less than the training set, which is SalePrice, the target to predict  ')


# Let's have a look first at the correlation between numerical features and the target "SalePrice", in order to have a first idea of the connections between features. Just by looking at the heatmap below we can see many dark colors, many features have high correlation with the target.

# In[ ]:


num=a.select_dtypes(exclude='object')
numcorr=num.corr()
f,ax=plt.subplots(figsize=(17,1))
sns.heatmap(numcorr.sort_values(by=['SalePrice'], ascending=False).head(1), cmap='Blues')
plt.title(" Numerical features correlation with the sale price", weight='bold', fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold', color='dodgerblue', rotation=0)


plt.show()


# To have a better idea, we sort the features according to their correlation with the sale price

# In[ ]:


Num=numcorr['SalePrice'].sort_values(ascending=False).head(10).to_frame()

cm = sns.light_palette("cyan", as_cmap=True)

s = Num.style.background_gradient(cmap=cm)
s


# Interesting! The **overall quality**, **the living area, basement area, garage cars and garage area** have the highest correlation values with the sale price, which is logical, better quality and bigger area = Higher price.
# * Also some features such as, **full bath** or **1st floor surface** have a higher correlation, those are luxury features, more luxury = Higher price.
# * and **Year built**, the newer buildings seem to have higher sale prices.
# 
# 
# > **Example of a strong correlation between 2 numerical features: Sale price and ground living area**

# In[ ]:


plt.figure(figsize=(15,6))
plt.scatter(x=a['GrLivArea'], y=a['SalePrice'], color='crimson', alpha=0.5)
plt.title('Ground living area/ Sale price', weight='bold', fontsize=16)
plt.xlabel('Ground living area', weight='bold', fontsize=12)
plt.ylabel('Sale price', weight='bold', fontsize=12)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()


# Let's dig in more into the data, those are just the numerical features. I assume that categorical features will be very important, for example, the neighborhood feature will be important, maybe the most important, given that good locations nowadays cost good money.
# > Example of categorical features: **Neighborhood**

# In[ ]:


# Figure Size
fig, ax = plt.subplots(figsize=(9,6))

# Horizontal Bar Plot
title_cnt=a.Neighborhood.value_counts().sort_values(ascending=False).reset_index()
mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color=sns.color_palette('Reds',len(title_cnt)))




# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.2)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Most frequent neighborhoods',weight='bold',
             loc='center', pad=10, fontsize=16)
ax.set_xlabel('Count', weight='bold')


# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+1, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')


plt.show()
# Show Plot
plt.show()


# In[ ]:


# Figure Size
fig, ax = plt.subplots(figsize=(9,6))

# Horizontal Bar Plot
title_cnt=a.BldgType.value_counts().sort_values(ascending=False).reset_index()
mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color=sns.color_palette('Greens',len(title_cnt)))




# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.2)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Building type: Type of dwelling',weight='bold',
             loc='center', pad=10, fontsize=16)
ax.set_xlabel('Count', weight='bold')


# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+1, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')


plt.show()


# * But before going any futher, we start by cleaning the data from missing values. I set the threshold to 80% (red line), all columns with more than 80% missing values will be dropped.

# In[ ]:


na = a.shape[0]
nb = b.shape[0]
y_train = a['SalePrice'].to_frame()
#Combine train and test sets
c1 = pd.concat((a, b), sort=False).reset_index(drop=True)
#Drop the target "SalePrice" and Id columns
c1.drop(['SalePrice'], axis=1, inplace=True)
c1.drop(['Id'], axis=1, inplace=True)
print(f"Total size is {c1.shape}")


# # 2- Data cleaning
# ***
#    > ### 2.1 Features with >80% missing values
#    
# First thing to do is get rid of the features with more than 80% missing values *(figure below)*. For example the PoolQC's missing values are probably due to the lack of pools in some buildings, which is very logical. But replacing those (more than 80%) missing values with "no pool" will leave us with a feature with low variance, and low variance features are uniformative for machine learning models. So we drop the features with more than 80% missing values.

# In[ ]:


def msv1(data, thresh=20, color='black', edgecolor='black', width=15, height=3):
    """
    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
    """
    
    plt.figure(figsize=(width,height))
    percentage=(data.isnull().mean())*100
    percentage.sort_values(ascending=False).plot.bar(color=color, edgecolor=edgecolor)
    plt.axhline(y=thresh, color='r', linestyle='-')
    plt.title('Missing values percentage per column', fontsize=20, weight='bold' )
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh+12.5, f'Columns with more than {thresh}% missing values', fontsize=12, color='crimson',
         ha='left' ,va='top')
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh - 5, f'Columns with less than {thresh} missing values', fontsize=12, color='green',
         ha='left' ,va='top')
    plt.xlabel('Columns', size=15, weight='bold')
    plt.ylabel('Missing values percentage')
    plt.yticks(weight ='bold')
    
    return plt.show()


# In[ ]:


msv1(c1, 20, color=('black', 'deeppink'))


# * Good news! Most of the features are clean from missing values
# 
# * We combine first the train and test datasets to run all the data munging and feature engineering on both of them.

# In[ ]:


c=c1.dropna(thresh=len(c1)*0.8, axis=1)
print(f"We dropped {c1.shape[1]-c.shape[1]} features in the combined set")


# Before cleaning the data, we zoom at the features with missing values, those missing values won't be treated equally. Some features have barely 1 or 2 missing values, we will use the forward fill method to fill them.

# In[ ]:


allna = (c.isnull().sum() / len(c))*100
allna = allna.drop(allna[allna == 0].index).sort_values()

def msv2(data, width=12, height=8, color=('silver', 'gold','lightgreen','skyblue','lightpink'), edgecolor='black'):
    """
    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
    """
    fig, ax = plt.subplots(figsize=(width, height))

    allna = (data.isnull().sum() / len(data))*100
    tightout= 0.008*max(allna)
    allna = allna.drop(allna[allna == 0].index).sort_values().reset_index()
    mn= ax.barh(allna.iloc[:,0], allna.iloc[:,1], color=color, edgecolor=edgecolor)
    ax.set_title('Missing values percentage per column', fontsize=15, weight='bold' )
    ax.set_xlabel('Percentage', weight='bold', size=15)
    ax.set_ylabel('Features with missing values', weight='bold')
    plt.yticks(weight='bold')
    plt.xticks(weight='bold')
    for i in ax.patches:
        ax.text(i.get_width()+ tightout, i.get_y()+0.1, str(round((i.get_width()), 2))+'%',
            fontsize=10, fontweight='bold', color='grey')
    return plt.show()


# In[ ]:


msv2(c)


# In[ ]:


print(f'The shape of the combined dataset after dropping features with more than 80% M.V. {c.shape}')


# We isolate the missing values from the rest of the dataset to have a good idea of how to treat them

# In[ ]:


NA=c[allna.index.to_list()]


# We split them to:
# * Categorical features
# * Numerical features

# In[ ]:


NAcat=NA.select_dtypes(include='object')
NAnum=NA.select_dtypes(exclude='object')
print(f'We have :{NAcat.shape[1]} categorical features with missing values')
print(f'We have :{NAnum.shape[1]} numerical features with missing values')


# So, 18 categorical features and 10 numerical features to clean.
# * We start with the numerical features, first thing to do is have a look at them to learn more about their distribution and decide how to clean them:
# - Most of the features are going to be filled with 0s because we assume that they don't exist, for example GarageArea, GarageCars with missing values are simply because the house lacks a garage.
# - GarageYrBlt: Year garage was built can't be filled with 0s, so we fill with the median (1980).
# 
# > ### 2.2 Numerical features:

# In[ ]:


NAnum.head()


# In[ ]:


#MasVnrArea: Masonry veneer area in square feet, the missing data means no veneer so we fill with 0
c['MasVnrArea']=c.MasVnrArea.fillna(0)
#LotFrontage has 16% missing values. We fill with the median
c['LotFrontage']=c.LotFrontage.fillna(c.LotFrontage.median())
#GarageYrBlt:  Year garage was built, we fill the gaps with the median: 1980
c['GarageYrBlt']=c["GarageYrBlt"].fillna(1980)


# > ### 2.3 Categorical features:
# 
# And we have 18 Categorical features with missing values:
# * Some features have just 1 or 2 missing values, so we will just use the forward fill method because they are obviously values that can't be filled with 'None's
# * Features with many missing values are mostly basement and garage related (same as in numerical features) so as we did with numerical features (filling them with 0s), we will fill the categorical missing values with "None"s assuming that the houses lack basements and garages.

# In[ ]:


NAcat.head()


# >> Number of missing values per column:

# In[ ]:


NAcat1= NAcat.isnull().sum().to_frame().sort_values(by=[0]).T
cm = sns.light_palette("lime", as_cmap=True)

NAcat1 = NAcat1.style.background_gradient(cmap=cm)
NAcat1


# The table above helps us to locate the categorical features with few missing values.
# * We start our cleaning with the features having just few missing value (1 to 4):  We fill the gap with forward fill method:
# 
# 

# In[ ]:


ffill_cols = ['Electrical', 'SaleType', 'KitchenQual', 'Exterior1st',
             'Exterior2nd', 'Functional', 'Utilities', 'MSZoning']

def filling_NA(data, columns, METHOD='ffill'):
    fill_cols = columns
    
    for col in data[fill_cols]:
        data[col]= data[col].fillna(method=METHOD)
    
    return data


# In[ ]:


msv2(c)


# In[ ]:


d=filling_NA(c, ffill_cols)


# In[ ]:


msv2(d)


# In[ ]:


fill_cols = ['Electrical', 'SaleType', 'KitchenQual', 'Exterior1st',
             'Exterior2nd', 'Functional', 'Utilities', 'MSZoning']

for col in c[fill_cols]:
    c[col] = c[col].fillna(method='ffill')


# * We dealt already with small missing values or values that can't be filled with "0" such as Garage year built.
# * The rest of the features are mostly basement and garage related with 100s of missing values, we will just fill 0s in the numerical features and 'None' in categorical features, assuming that the houses don't have basements, full bathrooms or garage.

# In[ ]:


#Categorical missing values
NAcols=c.columns
for col in NAcols:
    if c[col].dtype == "object":
        c[col] = c[col].fillna("None")


# In[ ]:


#Numerical missing values
for col in NAcols:
    if c[col].dtype != "object":
        c[col]= c[col].fillna(0)


# In[ ]:


c.isnull().sum().sort_values(ascending=False).head()


# We finally end up with a clean dataset, next thing to do: **Create new features.**

# # 3- Feature engineering:
# ***
# Since the area is a very important variable, we will create a new feature "**TotalArea**" that sums the area of all the floors and the basement.
# * **Bathrooms**: All the bathroom in the ground floor
# * **Year average**: The average of the sum of the year the house was built and the year the house was remodeled
# 
# 

# In[ ]:


c['TotalArea'] = c['TotalBsmtSF'] + c['1stFlrSF'] + c['2ndFlrSF'] + c['GrLivArea'] +c['GarageArea']

c['Bathrooms'] = c['FullBath'] + c['HalfBath']*0.5 

c['Year average']= (c['YearRemodAdd']+c['YearBuilt'])/2


# Feature engineering is very important to improve the model's performance, I will start in this kernel just with the TotalArea, Bathrooms and average year features and will keep updating the kernel by creating new features.
# * ** This part of the kernel is not finished yet.**

# # 4- Encoding categorical features:
# ***
# > ### 4.1 Numerical features:
# 
# We start with numerical features that are actually categorical, for example "Month sold", the values are from 1 to 12, each number is assigned to a month November is number 11 while March is number 3. 11 is just the order of the months and not a given value, so we convert the "Month Sold" feature to categorical

# In[ ]:


#c['MoSold'] = c['MoSold'].astype(str)
c['MSSubClass'] = c['MSSubClass'].apply(str)
c['YrSold'] = c['YrSold'].astype(str)


# > ### 4.2 One hot encoding:

# In[ ]:


cb=pd.get_dummies(c)
print(f"the shape of the original dataset {c.shape}")
print(f"the shape of the encoded dataset {cb.shape}")
print(f"We have {cb.shape[1]- c.shape[1]} new encoded features")


# We are done with the cleaning and feature engineering. Now, we split the combined dataset to the original train and test sets

# In[ ]:


Train = cb[:na]  #na is the number of rows of the original training set
Test = cb[na:] 


# # 5- Outliers detection:
# ***
# > ### 5.1 Outliers visualization:
# 
# This part of the kernel will be a little bit messy. I didn't want to deal with the outliers in the combined dataset to keep the shape of the original train and test datasets. Dropping them would shift the location of the rows.
# * If you know a better solution to this, I will be more than happy to read your recommandations.
# 
# * OK. So we go back to our original train dataset to visualize the important features / Sale price scatter plot to find outliers

# In[ ]:


fig = plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((3,2),(0,0))
plt.scatter(x=a['GrLivArea'], y=a['SalePrice'], color=('yellowgreen'), alpha=0.5)
plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Ground living Area- Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(0,1))
plt.scatter(x=a['TotalBsmtSF'], y=a['SalePrice'], color=('red'),alpha=0.5)
plt.axvline(x=5900, color='r', linestyle='-')
plt.title('Basement Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(1,0))
plt.scatter(x=a['1stFlrSF'], y=a['SalePrice'], color=('deepskyblue'),alpha=0.5)
plt.axvline(x=4000, color='r', linestyle='-')
plt.title('First floor Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(1,1))
plt.scatter(x=a['MasVnrArea'], y=a['SalePrice'], color=('gold'),alpha=0.9)
plt.axvline(x=1500, color='r', linestyle='-')
plt.title('Masonry veneer Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(2,0))
plt.scatter(x=a['GarageArea'], y=a['SalePrice'], color=('orchid'),alpha=0.5)
plt.axvline(x=1230, color='r', linestyle='-')
plt.title('Garage Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(2,1))
plt.scatter(x=a['TotRmsAbvGrd'], y=a['SalePrice'], color=('tan'),alpha=0.9)
plt.axvline(x=13, color='r', linestyle='-')
plt.title('TotRmsAbvGrd - Price scatter plot', fontsize=15, weight='bold' )
plt.show()


# The outliers are the points in the right that have a larger area or value but a very low sale price. We localize those points by sorting their respective columns
# 
# * Interesting! The outlier in "basement" and "first floor" features is the same as the first outlier in ground living area: **The outlier with index number 1298. **
# 
# > ### 5.2 Outliers localization:
# 
# We sort the columns containing the outliers shown in the graph, we will use the function *head()* to show the outliers: ***head(number of outliers or dots shown in each plot)***

# In[ ]:


a['GrLivArea'].sort_values(ascending=False).head(2)


# In[ ]:


a['TotalBsmtSF'].sort_values(ascending=False).head(1)


# In[ ]:


a['MasVnrArea'].sort_values(ascending=False).head(1)


# In[ ]:


a['1stFlrSF'].sort_values(ascending=False).head(1)


# In[ ]:


a['GarageArea'].sort_values(ascending=False).head(4)


# In[ ]:


a['TotRmsAbvGrd'].sort_values(ascending=False).head(1)


# 
# We can safely remove those points.

# In[ ]:


#train=Train[(Train['GrLivArea'] < 4600) & (Train['MasVnrArea'] < 1500) & (Train['GarageArea'] < 1240)
#           & (Train['TotRmsAbvGrd'] < 13)]

#print('We removed ',Train.shape[0]- train.shape[0],'outliers')


# In[ ]:


train=Train[(Train['GrLivArea'] < 4600) & (Train['MasVnrArea'] < 1500)]

print(f'We removed {Train.shape[0]- train.shape[0]} outliers')


# We do the same thing with "SalePrice" column, we localize those outliers and make sure they are the right outliers to remove. 
# * They both have the same price range as the detected outliers. So, we can safely drop them.

# In[ ]:


target=a[['SalePrice']]
target.loc[1298]


# In[ ]:


target.loc[523]


# We gather all the outliers index positions and drop them from the target dataset

# In[ ]:


#train=Train.copy()
#pos=[30,   88,  142,  277,  308,  328,  365,  410,  438,  462,  495,
#        523,  533,  581,  588,  628,  632,  681,  688,  710,  714,  728,
#        774,  812,  874,  898,  916,  935,  968,  970, 1062, 1168, 1170,
#        1181, 1182, 1298, 1324, 1383, 1423, 1432, 14]
#target.drop(target.index[pos], inplace=True)
#train.drop(target.index[pos], inplace=True)


# In[ ]:


#pos = [1298,523, 297, 581, 1190, 1061, 635, 197,1328, 495, 583, 313, 335, 249, 706]
pos = [1298,523, 297]
target.drop(target.index[pos], inplace=True)


# *P.S. I didn't drop all the outliers because dropping all of them led to a worst RMSE score. More investigation is needed to filter those outliers.*

# In[ ]:


print('We make sure that both train and target sets have the same row number after removing the outliers:')
print( 'Train: ',train.shape[0], 'rows')
print('Target:', target.shape[0],'rows')


# In[ ]:


plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
plt.scatter(x=a['GrLivArea'], y=a['SalePrice'], color=('orchid'), alpha=0.5)
plt.title('Area-Price plot with outliers',weight='bold', fontsize=18)
plt.axvline(x=4600, color='r', linestyle='-')
#first row sec col
ax1 = plt.subplot2grid((1,2),(0,1))
plt.scatter(x=train['GrLivArea'], y=target['SalePrice'], color='navy', alpha=0.5)
plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Area-Price plot without outliers',weight='bold', fontsize=18)
plt.show()


# > ## Log transform skewed numeric features:
# 
# We want our skewness value to be around 0 and kurtosis less than 3. For more information about skewness and kurtosis,I recommend reading [this article.](https://codeburst.io/2-important-statistics-terms-you-need-to-know-in-data-science-skewness-and-kurtosis-388fef94eeaa)
# 
# Here are two examples of skewed features: Ground living area and 1st floor SF. We will apply **np.log1p** to the skewed variables.

# In[ ]:


print("Skewness before log transform: ", a['GrLivArea'].skew())
print("Kurtosis before log transform: ", a['GrLivArea'].kurt())


# In[ ]:


from scipy.stats import skew

#numeric_feats = c.dtypes[c.dtypes != "object"].index

#skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
#skewed_feats = skewed_feats[skewed_feats > 0.75]
#skewed_feats = skewed_feats.index

#train[skewed_feats] = np.log1p(train[skewed_feats])


# In[ ]:


print(f"Skewness after log transform: {train['GrLivArea'].skew()}")
print(f"Kurtosis after log transform: {train['GrLivArea'].kurt()}")


# In[ ]:


plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,10))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((2,2),(0,0))
sns.distplot(a.GrLivArea, color='plum')
plt.title('Before: Distribution of GrLivArea',weight='bold', fontsize=18)
#first row sec col
ax1 = plt.subplot2grid((2,2),(0,1))
sns.distplot(a['1stFlrSF'], color='tan')
plt.title('Before: Distribution of 1stFlrSF',weight='bold', fontsize=18)


ax1 = plt.subplot2grid((2,2),(1,0))
sns.distplot(train.GrLivArea, color='plum')
plt.title('After: Distribution of GrLivArea',weight='bold', fontsize=18)
#first row sec col
ax1 = plt.subplot2grid((2,2),(1,1))
sns.distplot(train['1stFlrSF'], color='tan')
plt.title('After: Distribution of 1stFlrSF',weight='bold', fontsize=18)
plt.show()


# Last thing to do before Machine Learning is to log transform the target as well, as we did with the skewed features.
# 
# *P.S. Log transoform is only applied on the target in this version, not on the features. I will be applying the log transoform on the features in future versions of this kernel*

# In[ ]:


print(f"Skewness before log transform: {target['SalePrice'].skew()}")
print(f"Kurtosis before log transform: {target['SalePrice'].kurt()}")


# In[ ]:


#log transform the target:
target["SalePrice"] = np.log1p(target["SalePrice"])


# In[ ]:


plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
plt.hist(a.SalePrice, bins=10, color='mediumpurple',alpha=0.5)
plt.title('Sale price distribution before normalization',weight='bold', fontsize=18)
#first row sec col
ax1 = plt.subplot2grid((1,2),(0,1))
plt.hist(target.SalePrice, bins=10, color='darkcyan',alpha=0.5)
plt.title('Sale price distribution after normalization',weight='bold', fontsize=18)
plt.show()


# In[ ]:


print(f"Skewness after log transform: {target['SalePrice'].skew()}")
print(f"Kurtosis after log transform: {target['SalePrice'].kurt()}")


# The skewness and kurtosis values look fine after log transform. We can now move forward to Machine Learning.
# 
# *P.S.To get our original SalePrice values back, we will apply **np.expm1** at the end of the study to cancel the log1p transformation after training and testing the models.*

# # 6- Machine Learning:
# ***
# > ### 6.1 Preprocessing
# 
# We start machine learning by setting the features and target:
# * Features: x
# * Target: y

# In[ ]:


x=train
y=np.array(target)


# Then, we split them to train and test sets

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=0)


# We use RobustScaler to scale our data because it's powerful against outliers, we already detected some but there must be some other outliers out there, I will try to find them in future versions of the kernel

# In[ ]:


from sklearn.preprocessing import RobustScaler
scaler= RobustScaler()
# transform "x_train"
x_train = scaler.fit_transform(x_train)
# transform "x_test"
x_test = scaler.transform(x_test)
#Transform the test set
X_test= scaler.transform(Test)


# We first start by trying the very basic regression model: Linear regression. 
# * We use 5- Fold cross validation for a better error estimate:
#  
# > ### 6.2 Linear regression
# 

# In[ ]:


#from sklearn.linear_model import LinearRegression

#lreg=LinearRegression()
#MSEs=ms.cross_val_score(lreg, x, y, scoring='neg_mean_squared_error', cv=5)
#meanMSE=np.mean(MSEs)
#print(meanMSE)
#print('RMSE = '+str(math.sqrt(-meanMSE)))


# * Our goal is to minimize the error, we use regularization methods: Ridge, Lasso and ElasticNet, in order to lower the squared error.
# 
# > ### The metric RMSE:
# The metric in this competition is RMSE. We will create a function ```score()``` (it takes one input parameter: Our predictions) to compute the RMSE score with all the models that will be used.

# In[ ]:


def score(y_pred):
    return str(math.sqrt(sklm.mean_squared_error(y_test, y_pred)))


# > ### 6.3 Regularization: 

# >> ## Ridge regression:
# * Minimize squared error + a term **alpha** that penalizes the error
# * We need to find a value of **alpha** that minimizes the train and test error (avoid overfitting)

# In[ ]:


import sklearn.model_selection as GridSearchCV
from sklearn.linear_model import Ridge

ridge=Ridge()
parameters= {'alpha':[x for x in range(1,101)]}

ridge_reg=ms.GridSearchCV(ridge, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)
ridge_reg.fit(x_train,y_train)
print(f"The best value of Alpha is: {ridge_reg.best_params_}")
print(f"The best score achieved with Alpha=11 is: {math.sqrt(-ridge_reg.best_score_)}")
ridge_pred=math.sqrt(-ridge_reg.best_score_)


# In[ ]:


ridge_mod=Ridge(alpha=15)
ridge_mod.fit(x_train,y_train)
y_pred_train=ridge_mod.predict(x_train)
y_pred_test=ridge_mod.predict(x_test)

print(f'Root Mean Square Error train =  {str(math.sqrt(sklm.mean_squared_error(y_train, y_pred_train)))}')
print(f'Root Mean Square Error test =  {score(y_pred_test)}')   


# * Next we try Lasso regularization: Similar procedure as ridge regularization but Lasso tends to have a lot of 0 entries in it and just few nonzeros (easy selection). In other words, lasso drops the uninformative features and keeps just the important ones.
# * As with Ridge regularization, we need to find the **alpha** parameter that penalizes the error

# >> ## Lasso regression

# In[ ]:


from sklearn.linear_model import Lasso

parameters= {'alpha':[0.0001,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}


lasso=Lasso()
lasso_reg=ms.GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)
lasso_reg.fit(x_train,y_train)

print(f'The best value of Alpha is: {lasso_reg.best_params_}')


# In[ ]:


lasso_mod=Lasso(alpha=0.0009)
lasso_mod.fit(x_train,y_train)
y_lasso_train=lasso_mod.predict(x_train)
y_lasso_test=lasso_mod.predict(x_test)

print(f'Root Mean Square Error train  {str(math.sqrt(sklm.mean_squared_error(y_train, y_lasso_train)))}')
print(f'Root Mean Square Error test  {score(y_lasso_test)}')


# * We check next, the important features that our model used to make predictions
# * The number of uninformative features that were dropped. Lasso give a 0 coefficient to the useless features, we will use the coefficient given to the important feature to plot the graph

# In[ ]:


coefs = pd.Series(lasso_mod.coef_, index = x.columns)

imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh", color='yellowgreen')
plt.xlabel("Lasso coefficient", weight='bold')
plt.title("Feature importance in the Lasso Model", weight='bold')
plt.show()


# Nice! The most important feature is the new feature we created "**TotalArea**". 
# * Other features such as neighborhood or overall quality are among the main important features.

# In[ ]:



print(f"Lasso kept {sum(coefs != 0)} important features and dropped the other  {sum(coefs == 0)} features")


# Next, we try ElasticNet. A regressor that combines both ridge and Lasso.
# We use cross validation to find:
# * Alpha
# * Ratio between Ridge and Lasso, for a better combination of both

# >> ## ElasticNet

# In[ ]:


from sklearn.linear_model import ElasticNetCV

#alphas = [10,1,0.1,0.01,0.001,0.002,0.003,0.004,0.005,0.00054255]
#l1ratio = [0.1, 0.3,0.5, 0.9, 0.95, 0.99, 1]

#elastic_cv = ElasticNetCV(cv=5, max_iter=1e7, alphas=alphas,  l1_ratio=l1ratio)

#elasticmod = elastic_cv.fit(x_train, y_train.ravel())
#ela_pred=elasticmod.predict(x_test)
#print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, ela_pred))))
#print(elastic_cv.alpha_)


# I tried numberers that round alpha=0.0005 and found out that 0.0005425 gives the best score, so we continue with alpha=0.0005425

# In[ ]:


from sklearn.linear_model import ElasticNetCV

alphas = [0.000542555]
l1ratio = [0.1, 0.3,0.5, 0.9, 0.95, 0.99, 1]

elastic_cv = ElasticNetCV(cv=5, max_iter=1e7, alphas=alphas,  l1_ratio=l1ratio)

elasticmod = elastic_cv.fit(x_train, y_train.ravel())
ela_pred=elasticmod.predict(x_test)
print(f'Root Mean Square Error test = {score(ela_pred)}')
print(elastic_cv.alpha_)
print(elastic_cv.l1_ratio_)


# <h1>REGULARIZATION RECAP <span class="label label-danger">Function</span></h1>
# ***
# In regularization we worked with 3 algorithms: Ridge (L2), Lasso (L1) and ElasticNet that is a combination of both L2 and L1 regressors.
# Before moving to the next section of this work, I would like to introduce a function that does all the work we did above in details **just with one line of code.** The function does all the regression pipeline:
# 
# 
# 
# 1.  Split the data to train/test
# 1.  Scale the data
# 1.  Gridsearch for the best hyperparameters
# 1.  Predict the target
# 1.  Evaluate the prediction
# 
# 
# The function takes as input parameters:
# - x: the features
# - y: the target
# - modelo: Ridge(default), Lasso, ElasticNetCV
# - scaler: RobustScaler(default), MinMaxScaler, StandardScaler
# 
# 
# <h4>In future versions, I will include more input parameters in this function to make it more flexible such as: The personalization of the hyperparameters search. <span class="label label-info">WORK IN PROGRESS</span></h4>
# 
# 
# 

# In[ ]:


def regularization(x,y,modelo=Ridge, scaler=RobustScaler):
    """"
    Function to automate regression with regularization techniques:
    
    -x: the features
    -y: the target
    -modelo: Ridge(default), Lasso, ElasticNetCV
    -scaler: RobustScaler(default), MinMaxSclaer, StandardScaler
    
    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
    Contact: amineyamlahi@gmail.com
    """
    #Split the data to train/test
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=0)
    
    #Scale the data. RobustSclaer default
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    
    scaler= scaler()
    # transform "x_train"
    x_train = scaler.fit_transform(x_train)
    # transform "x_test"
    x_test = scaler.transform(x_test)
    #Transform the test set
    X_test= scaler.transform(Test)
    
    if modelo != ElasticNetCV:
        if modelo == Ridge:
            parameters= {'alpha':[x for x in range(1,101)]}
        elif modelo == Lasso:
            parameters= {'alpha':[0.0001,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}
            
        model=modelo()
            
        model=ms.GridSearchCV(model, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)
        model.fit(x_train,y_train)
        y_pred= model.predict(x_test)

        #print("The best value of Alpha is: ",model.best_params_)
        print("The best RMSE score achieved with %s is: %s " %(model.best_params_,
                  score(y_pred)))
    elif modelo == ElasticNetCV:
        alphas = [0.000542555]
        l1ratio = [0.1, 0.3,0.5, 0.9, 0.95, 0.99, 1]

        elastic_cv = ElasticNetCV(cv=5, max_iter=1e7, alphas=alphas,  l1_ratio=l1ratio)

        elasticmod = elastic_cv.fit(x_train, y_train.ravel())
        ela_pred=elasticmod.predict(x_test)
        print("The best RMSE score achieved with alpha %s and l1_ratio %s is: %s "
              %(elastic_cv.alpha_,elastic_cv.l1_ratio_, score(ela_pred)))
        
            

  


# In[ ]:


regularization(x,y,Ridge)


# In[ ]:


regularization(x,y, Lasso)


# In[ ]:


regularization(x,y, ElasticNetCV)


# > ### 6.4 XGB and ExtraTrees regressors:

# We will try other kind of regressors, such as XGBRegressor and ExtraTreesRegressor

# In[ ]:


from xgboost.sklearn import XGBRegressor

#xg_reg = XGBRegressor()
#xgparam_grid= {'learning_rate' : [0.01],'n_estimators':[2000, 3460, 4000],
#                                     'max_depth':[3], 'min_child_weight':[3,5],
#                                     'colsample_bytree':[0.5,0.7],
#                                     'reg_alpha':[0.0001,0.001,0.01,0.1,10,100],
#                                    'reg_lambda':[1,0.01,0.8,0.001,0.0001]}

#xg_grid=GridSearchCV(xg_reg, param_grid=xgparam_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#xg_grid.fit(x_train,y_train)
#print(xg_grid.best_estimator_)
#print(xg_grid.best_score_)


# The gridSearch above tunes the hyperparamaters, but it takes forever to run. I copy the best estimator results to the model below. Feel free to uncomment and check it out.

# In[ ]:


xgb= XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.5, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=3, min_child_weight=0, missing=None, n_estimators=4000,
             n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
             reg_alpha=0.0001, reg_lambda=0.01, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
xgmod=xgb.fit(x_train,y_train)
xg_pred=xgmod.predict(x_test)
print(f'Root Mean Square Error test = {score(xg_pred)}')


# > ### 6.5 ENSEMBLE METHODS:
# >> ## VOTING REGRESSOR:
# * A voting regressor is an ensemble meta-estimator that fits base regressors each on the whole dataset. It, then, averages the individual predictions to form a final prediction.
# 
# * After running the regressors, we combine them first with voting regressor in order to get a better model

# In[ ]:


from sklearn.ensemble import VotingRegressor

vote_mod = VotingRegressor([('Ridge', ridge_mod), ('Lasso', lasso_mod), ('Elastic', elastic_cv), 
                            ('XGBRegressor', xgb)])
vote= vote_mod.fit(x_train, y_train.ravel())
vote_pred=vote.predict(x_test)

print(f'Root Mean Square Error test = {score(vote_pred)}')


# >> ## STACKING REGRESSOR:

# We stack all the previous models, including the votingregressor with XGBoost as the meta regressor:

# In[ ]:


from mlxtend.regressor import StackingRegressor


stregr = StackingRegressor(regressors=[elastic_cv,ridge_mod, lasso_mod, vote_mod], 
                           meta_regressor=xgb, use_features_in_secondary=True
                          )

stack_mod=stregr.fit(x_train, y_train.ravel())
stacking_pred=stack_mod.predict(x_test)

print(f'Root Mean Square Error test = {score(stacking_pred)}')


# * Last thing to do is average our regressors and fit them on the testing dataset

# >> ## Averaging Regressors

# In[ ]:


#We coefficients were assigned manually
final_test=(0.3*vote_pred+0.5*stacking_pred+ 0.2*y_lasso_test)
print(f'Root Mean Square Error test=  {score(final_test)}')


# Averaging the 3 best models: Stacking, Voting and Lasso gave the best results: **The lowest RMSE**
# * The coefficients assigned to the 3 models were tested manually, the models combination above gave the best RMSE score

# > ### 6.6 Fit the model on test data
# 
# Now, we fit the models on the test data and then submit it to the competition
# 
# * We apply **np.expm1** to cancel the **np.logp1** *(we did previously in data processing)* and convert the numbers to their original form

# In[ ]:


#VotingRegressor to predict the final Test
vote_test = vote_mod.predict(X_test)
final1=np.expm1(vote_test)

#StackingRegressor to predict the final Test
stack_test = stregr.predict(X_test)
final2=np.expm1(stack_test)

#LassoRegressor to predict the final Test
lasso_test = lasso_mod.predict(X_test)
final3=np.expm1(lasso_test)


# Blending and submitting the **FINAL AVERAGE OF 3 REGRESSORS**

# In[ ]:


#Submission of the results predicted by the average of Voting/Stacking/Lasso
final=(0.2*final1+0.6*final2+0.2*final3)

final_submission = pd.DataFrame({
        "Id": b["Id"],
        "SalePrice": final
    })
final_submission.to_csv("final_submission.csv", index=False)
final_submission.head()


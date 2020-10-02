#!/usr/bin/env python
# coding: utf-8

# ## Author: Caio Avelino
# * [LinkedIn](https://www.linkedin.com/in/caioavelino/)
# * [Kaggle](https://www.kaggle.com/avelinocaio)

# ## Project Phases:
# > 
# * **0) Libraries and Data Loading**
# * **1) Split variables in numerical (discrete, continuous) and categorical (ordinal, nominal)**
# * **2) Remove outliers with Low Price**
# * **3) Replace NaN's**
# * **4) Exploratory Analysis and Data Cleaning**
# * **5) Train Model**
# * **6) Stacking**
# * **7) Voting**
# * **8) Submission**

# > You will find an overview of what is done at the beginning of each part.

# # 0-Libraries and Data Loading

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.special import boxcox1p
import warnings

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingRegressor

warnings.filterwarnings("ignore") # ignoring annoying warnings


# In[ ]:


test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test["SalePrice"] = np.nan # we don't have target values for the test


# # 1-Split variables in numerical (discrete, continuous) and categorical (ordinal, nominal)

# > The types of the variables can be defined based on the data description in the competition.

# ![Variable Types](http://survivestatistics.com/wp-content/uploads/2016/07/variables3.jpg)

# [Image Source](http://survivestatistics.com/variables/) 

# In[ ]:


num_discrete = ['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd',
                'Fireplaces','GarageCars','GarageYrBlt','YearBuilt','YearRemodAdd','YrSold','MoSold']

num_continuous = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF',
                  '2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
                  '3SsnPorch','ScreenPorch','PoolArea','MiscVal','SalePrice']

cat_ordinal = ['LotShape','Utilities','LandSlope','OverallQual','OverallCond','ExterQual','ExterCond',
               'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir',
               'Electrical','KitchenQual','Functional','FireplaceQu','GarageFinish','GarageQual','GarageCond',
               'PavedDrive','PoolQC','Fence']

cat_nominal = ['MSSubClass','MSZoning','Street','Alley','LandContour','LotConfig','Neighborhood',
               'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
               'Exterior2nd','MasVnrType','Foundation','Heating','GarageType','MiscFeature',
               'SaleType','SaleCondition']


# # 2-Remove outliers with Low Price

# > We can define as outliers all the values with z score > 3 or z score < 3.
# The number of outliers of the features with, at least, weak correlation with **Sale Price** and a low % of zeros will be presented. 
# Then, the scatterplots of the variables will be plotted to detect the outliers with Low Price.

# In[ ]:


# defining a z score function
def z_score(df): 
    return (df-df.mean())/df.std(ddof=0)


# In[ ]:


# Let's take a look at the number of outliers of the variables with correlation > 0.3 and % of zeros < 30%

idx = []
outliers = []
corrs = []
zeros = []

for i in num_continuous:
    if str(train[i].dtype) != 'object':
        idx.append(i)
        outliers.append(list(abs(z_score(train[i])) > 3).count(True))
        corrs.append(train.SalePrice.corr(train[i]))
        zeros.append(len(train[i][train[i] == 0])/len(train[i]))
        
outs = pd.DataFrame({'# Outliers': outliers, 
                     'Feature': idx, 
                     'Corr': corrs, 
                     '% Zeros': zeros}).sort_values(ascending = False, 
                                                 by = '# Outliers')

outs = outs[outs["# Outliers"] > 0]
outs = outs[abs(outs["Corr"]) > 0.3].reset_index(drop=True)
outs = outs[outs["% Zeros"] < 0.3].reset_index(drop=True)
outs


# In[ ]:


# defining a function to plot the correlation of the variables shown above with SalePrice, so we can delete Low Price's outliers

def plot_outliers():
    
    fig = plt.figure(figsize=(15,15), constrained_layout=True)

    gs = GridSpec(3,2,figure=fig)

    rows = [0,0,1,1,2,2]
    columns = [0,1,0,1,0,1]

    colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

    counter = 0

    for i in outs.Feature:
        sns.scatterplot(y=train.SalePrice, x=train[i], 
                      ax=fig.add_subplot(gs[rows[counter],columns[counter]]),
                      color=colors[counter])
        counter = counter + 1

    fig.show()


# In[ ]:


plot_outliers()


# > As we can see there are two outliers with low price for **GrLivArea**, let's remove them.

# In[ ]:


train.shape


# In[ ]:


# removing GrLivArea outliers with Low Price
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)


# In[ ]:


# plotting again
plot_outliers()


# > We can see another strange outlier for **Lot Frontage**. It will be deleted as well.

# In[ ]:


train = train.drop(train[(train['LotFrontage'] > 300) & (train['SalePrice'] < 300000)].index)


# In[ ]:


plot_outliers()


# In[ ]:


train.shape


# > Ok, now it's enough. We can't delete a lot of outliers, because it can affect the results.

# > Let's concatenate train and test sets into one, so we can analyze everything and fill NaN's based on all dataset.

# In[ ]:


dataset = pd.concat([train,test],axis=0).reset_index(drop=True)
dataset = dataset.fillna(np.nan)


# # 3-Replace NaN's

# > First, let's map all the missing data and then fill them based on median (if continuous) or mode (otherwise) or NA (if written in description).

# In[ ]:


# defining a function to map the # and % of NaN's for the features
def show_null(df):
    null_columns = (df.isnull().sum(axis = 0)/len(df)).sort_values(ascending=False).index
    null_data = pd.concat([df.isnull().sum(axis = 0),
                           (df.isnull().sum(axis = 0)/len(df)).sort_values(ascending=False),
                           df.loc[:, df.columns.isin(list(null_columns))].dtypes]
                          , axis=1)
    null_data = null_data.rename(columns={0: '#', 
                                          1: '% null', 
                                          2: 'type'}).sort_values(ascending=False, by = '% null')
    null_data = null_data[null_data["#"]!=0]
    return null_data

show_null(dataset)


# In[ ]:


# Ordinals - Replacing NaN's

dataset.PoolQC[dataset.PoolQC.isnull() == True] = 'NA'
dataset.Fence[dataset.Fence.isnull() == True] = 'NA'
dataset.FireplaceQu[dataset.FireplaceQu.isnull() == True] = 'NA'
dataset.GarageCond[dataset.GarageCond.isnull() == True] = 'NA'
dataset.GarageQual[dataset.GarageQual.isnull() == True] = 'NA'
dataset.GarageFinish[dataset.GarageFinish.isnull() == True] = 'NA'
dataset.BsmtExposure[dataset.BsmtExposure.isnull() == True] = 'NA'
dataset.BsmtCond[dataset.BsmtCond.isnull() == True] = 'NA'
dataset.BsmtQual[dataset.BsmtQual.isnull() == True] = 'NA'
dataset.BsmtFinType2[dataset.BsmtFinType2.isnull() == True] = 'NA'
dataset.BsmtFinType1[dataset.BsmtFinType1.isnull() == True] = 'NA'
dataset.Electrical[dataset.Electrical.isnull() == True] = stats.mode(train.Electrical)[0][0]
dataset.Functional[dataset.Functional.isnull() == True] = stats.mode(train.Functional)[0][0]
dataset.KitchenQual[dataset.KitchenQual.isnull() == True] = stats.mode(train.KitchenQual)[0][0]
dataset.Utilities[dataset.Utilities.isnull() == True] = stats.mode(train.Utilities)[0][0]


# In[ ]:


# Nominals - Replacing NaN's

dataset.MiscFeature[dataset.MiscFeature.isnull() == True] = 'NA'
dataset.Alley[dataset.Alley.isnull() == True] = 'NA'
dataset.GarageType[dataset.GarageType.isnull() == True] = 'NA'
dataset.MasVnrType[dataset.MasVnrType.isnull() == True] = 'NA'
dataset.MSZoning[dataset.MSZoning.isnull() == True] = stats.mode(train.MSZoning)[0][0]
dataset.SaleType[dataset.SaleType.isnull() == True] = stats.mode(train.SaleType)[0][0]
dataset.Exterior1st[dataset.Exterior1st.isnull() == True] = stats.mode(train.Exterior1st)[0][0]
dataset.Exterior2nd[dataset.Exterior2nd.isnull() == True] = stats.mode(train.Exterior2nd)[0][0]


# In[ ]:


# Some Numericals - Replacing NaN's

dataset.BsmtFullBath[dataset.BsmtFullBath.isnull() == True] = stats.mode(train.BsmtFullBath)[0][0]
dataset.BsmtHalfBath[dataset.BsmtHalfBath.isnull() == True] = stats.mode(train.BsmtHalfBath)[0][0]
dataset.GarageCars[dataset.GarageCars.isnull() == True] = stats.mode(train.GarageCars)[0][0]
dataset.GarageArea[dataset.GarageArea.isnull() == True] = np.median(train.
                                                                    GarageArea[train.GarageArea.isnull() == False])
dataset.TotalBsmtSF[dataset.TotalBsmtSF.isnull() == True] = np.median(train.
                                                                      TotalBsmtSF[train.TotalBsmtSF.isnull() == False])
dataset.BsmtFinSF1[dataset.BsmtFinSF1.isnull() == True] = np.median(train.
                                                                    BsmtFinSF1[train.BsmtFinSF1.isnull() == False])
dataset.BsmtFinSF2[dataset.BsmtFinSF2.isnull() == True] = np.median(train.
                                                                    BsmtFinSF2[train.BsmtFinSF2.isnull() == False])
dataset.BsmtUnfSF[dataset.BsmtUnfSF.isnull() == True] = np.median(train.
                                                                  BsmtUnfSF[train.BsmtUnfSF.isnull() == False])
dataset.LotFrontage[dataset.LotFrontage.isnull() == True] = np.median(train.
                                                                    LotFrontage[train.LotFrontage.isnull() == False])


# > Special cases:

# In[ ]:


# MasVnrArea

dataset[dataset.MasVnrArea.isnull() == True].MasVnrType


# > So, we can see that the problem is not lack of information for MasVnrArea, but the NaN values happen just because there is no Masonry veener. Therefore we can define them as 0.

# In[ ]:


dataset.MasVnrArea[dataset.MasVnrArea.isnull() == True] = 0


# In[ ]:


# GarageYrBlt

dataset[["GarageType",
          "GarageFinish",
          "GarageQual",
          "GarageCond",
         "GarageYrBlt"]][dataset.GarageYrBlt.isnull() == True].dropna()


# > So, we can see that the problem is not lack of information, but the NaN values happen just because there is no garage. Therefore, we can define them as NA.

# In[ ]:


dataset.GarageYrBlt[dataset.GarageYrBlt.isnull() == True] = 'NA'


# In[ ]:


# Let's see again the # and % of NaN's 
show_null(dataset)

# SalePrice is ok, because NaN values are from test set.


# # 4-Exploratory Analysis and Data Cleaning

# > Here we will analyze each feature for Discrete, Continuous, Ordinal and Nominal variables.

# ## Discrete Variables

# In[ ]:


# To remember
num_discrete = ['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd',
                'Fireplaces','GarageCars','GarageYrBlt','YearBuilt','YearRemodAdd','YrSold','MoSold']


# > We will analyze each discrete variable individually and make decisions based on correlation with SalePrice, lack of information in each category and so on.

# In[ ]:


# defining a function to plot boxplot and stripplot for non continuous variables
def make_discrete_plot(feature, rotation1, rotation2):
    fig = plt.figure(figsize=(20,8))
    gs = GridSpec(1,2)
    sns.boxplot(y=dataset.SalePrice, x=dataset[feature], ax=fig.add_subplot(gs[0,0]))
    plt.xticks(rotation = rotation1)
    sns.stripplot(y=dataset.SalePrice, x=dataset[feature], ax=fig.add_subplot(gs[0,1]))
    plt.xticks(rotation = rotation2)
    fig.show()


# ### BsmtFullBath, BsmtHalfBath, FullBath, HalfBath

# > Number of bathrooms.

# In[ ]:


# Those variables tell almost the same information, so let's add them.
dataset["Baths"] = dataset.BsmtFullBath + 0.5*dataset.BsmtHalfBath + dataset.FullBath + 0.5*dataset.HalfBath


# In[ ]:


make_discrete_plot("Baths",0,0)


# In[ ]:


# there few values greater than 4, so let's put them together with 3.5
dataset.Baths = dataset.Baths.apply(lambda x: 3.5 if x > 3.5 else x)


# In[ ]:


make_discrete_plot("Baths",0,0)


# ### BedroomAbvGr

# > Bedrooms above grade.

# In[ ]:


make_discrete_plot("BedroomAbvGr",0,0)


# In[ ]:


# there few values greater than 5, so let's put them together with 5
dataset.BedroomAbvGr = dataset.BedroomAbvGr.apply(lambda x: 5 if x > 5 else x)


# In[ ]:


make_discrete_plot("BedroomAbvGr",0,0)


# In[ ]:


# change type to category
dataset.BedroomAbvGr = dataset.BedroomAbvGr.astype(str)


# ### KitchenAbvGr

# > Kitchens above grade.

# In[ ]:


make_discrete_plot("KitchenAbvGr",0,0)


# In[ ]:


# there few values equal to 0 or 3, so let's put them together with other classes
dataset.KitchenAbvGr = dataset.KitchenAbvGr.apply(lambda x: 1 if x==0 else(2 if x==3 else x))


# In[ ]:


make_discrete_plot("KitchenAbvGr",0,0)


# In[ ]:


# change type to category
dataset.KitchenAbvGr = dataset.KitchenAbvGr.astype(str)


# ### TotRmsAbvGrd

# > Total rooms above grade (does not include bathrooms).

# In[ ]:


make_discrete_plot("TotRmsAbvGrd",0,0)


# In[ ]:


# there few values equal to 2 or greater than 11, so let's put them together with other classes
dataset.TotRmsAbvGrd = dataset.TotRmsAbvGrd.apply(lambda x: 3 if x==2 else(11 if x>11 else x))


# In[ ]:


make_discrete_plot("TotRmsAbvGrd",0,0)


# ### Fireplaces

# > Number of fireplaces.

# In[ ]:


make_discrete_plot("Fireplaces",0,0)


# In[ ]:


# there few values greater than 2, so let's flag
dataset.Fireplaces = dataset.Fireplaces.apply(lambda x: 1 if x>1 else x)


# In[ ]:


make_discrete_plot("Fireplaces",0,0)


# ### GarageCars

# > Size of garage in car capacity.

# In[ ]:


make_discrete_plot("GarageCars",0,0)


# In[ ]:


# there few values greater than 3, so let's put them together with other classes
dataset.GarageCars = dataset.GarageCars.apply(lambda x: 3 if x>3 else x)


# In[ ]:


make_discrete_plot("GarageCars",0,0)


# ### GarageYrBlt

# > Year garage was built.

# In[ ]:


plt.figure(figsize=(20,8))
sns.kdeplot(dataset["GarageYrBlt"][dataset["GarageYrBlt"] != 'NA'])
plt.show()


# In[ ]:


dataset.GarageYrBlt[dataset["GarageYrBlt"] != 'NA'].sort_values(ascending= False).head(3) 
# this is strange, probably 2207 is 2007


# In[ ]:


dataset.GarageYrBlt[dataset.GarageYrBlt == 2207] = 2007


# In[ ]:


# Let's transform those values into categories
min(dataset.GarageYrBlt[dataset["GarageYrBlt"] != 'NA'].values)


# In[ ]:


bins = [1890, 1920, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2011]
names = ['-1920', '1920-1940', '1940-1950', '1950-1960', '1960-1970',
         '1970-1980', '1980-1990', '1990-2000', '2000+']

dataset.GarageYrBlt[dataset["GarageYrBlt"] != 'NA'] = pd.cut(dataset.GarageYrBlt[dataset["GarageYrBlt"] != 'NA'], 
                                                             bins, 
                                                             labels=names)


# In[ ]:


# change type from category to str
dataset.GarageYrBlt = dataset.GarageYrBlt.astype(str)


# In[ ]:


make_discrete_plot("GarageYrBlt",20,20)


# ### YearRemodAdd

# > Remodel date (same as construction date if no remodeling or additions).

# In[ ]:


plt.figure(figsize=(20,8))
sns.kdeplot(dataset["YearRemodAdd"])
plt.show()


# In[ ]:


# Let's transform those values into categories
print(min(dataset.YearRemodAdd[dataset["YearRemodAdd"] != 'NA'].values), ',',
      max(dataset.YearRemodAdd[dataset["YearRemodAdd"] != 'NA'].values))


# In[ ]:


bins = [1949, 1960, 1970, 1980, 1990, 2000, 2011]
names = ['1950-1960', '1960-1970',
         '1970-1980', '1980-1990', '1990-2000', '2000+']

dataset.YearRemodAdd = pd.cut(dataset.YearRemodAdd, bins, labels=names)


# In[ ]:


# change type from category to str
dataset.YearRemodAdd = dataset.YearRemodAdd.astype(str)


# In[ ]:


make_discrete_plot("YearRemodAdd",20,20)


# ### YearBuilt

# > Original construction date.

# In[ ]:


plt.figure(figsize=(20,8))
sns.kdeplot(dataset["YearBuilt"][dataset["YearBuilt"] != 'NA'])
plt.show()


# In[ ]:


# Let's transform those values into categories
print(min(dataset.YearBuilt[dataset["YearBuilt"] != 'NA'].values), ',',
      max(dataset.YearBuilt[dataset["YearBuilt"] != 'NA'].values))


# In[ ]:


bins = [1870, 1920, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2011]
names = ['-1920', '1920-1940', '1940-1950', '1950-1960', '1960-1970',
         '1970-1980', '1980-1990', '1990-2000', '2000+']

dataset.YearBuilt = pd.cut(dataset.YearBuilt, bins, labels=names)


# In[ ]:


# change type from category to str
dataset.YearBuilt = dataset.YearBuilt.astype(str)


# In[ ]:


make_discrete_plot("YearBuilt",20,20)


# ### YrSold

# > Year Sold (YYYY).

# In[ ]:


make_discrete_plot("YrSold",0,0)


# In[ ]:


# change type from int to str
dataset.YrSold = dataset.YrSold.astype(str)


# ### MoSold

# > Month Sold (MM).

# In[ ]:


make_discrete_plot("MoSold",0,0)


# In[ ]:


# change type from int to str
dataset.MoSold = dataset.MoSold.astype(str)


# ## Continuous Variables

# > Here we are going to analyze correlation of each feature with SalePrice, see skewness for linear and boxcox transformations (with different lambdas) e apply the better one. If skewness continues high, we will bin the variable into categories or flag (0 and 1). And if there are missing values, we will drop the column.

# In[ ]:


# to remember
num_continuous = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                  '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF',
                  'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']


# In[ ]:


# defining function to plot scatterplot for continuous variables with SalePrice.
def make_continuous_plot(feature):
    
    fig = plt.figure(figsize=(18,15))
    gs = GridSpec(2,2)
    
    j = sns.scatterplot(y=np.log1p(dataset['SalePrice']), 
                        x=boxcox1p(dataset[feature], 0.15), ax=fig.add_subplot(gs[0,1]), palette = 'blue')

    plt.title('BoxCox 0.15\n' + 'Corr: ' + str(np.round(np.log1p(dataset['SalePrice']).corr(boxcox1p(dataset[feature], 0.15)),2)) + ', Skew: ' +
               str(np.round(stats.skew(boxcox1p(dataset[feature], 0.15)),2)))
    
    j = sns.scatterplot(y=np.log1p(dataset['SalePrice']), 
                        x=boxcox1p(dataset[feature], 0.25), ax=fig.add_subplot(gs[1,0]), palette = 'blue')

    plt.title('BoxCox 0.25\n' + 'Corr: ' + str(np.round(np.log1p(dataset['SalePrice']).corr(boxcox1p(dataset[feature], 0.25)),2)) + ', Skew: ' +
               str(np.round(stats.skew(boxcox1p(dataset[feature], 0.25)),2)))
    
    j = sns.scatterplot(y=np.log1p(dataset['SalePrice']), 
                        x=boxcox1p(dataset[feature], 0.35), ax=fig.add_subplot(gs[1,1]), palette = 'blue')

    plt.title('BoxCox 0.35\n' + 'Corr: ' + str(np.round(np.log1p(dataset['SalePrice']).corr(boxcox1p(dataset[feature], 0.35)),2)) + ', Skew: ' +
               str(np.round(stats.skew(boxcox1p(dataset[feature], 0.35)),2)))
    
    j = sns.scatterplot(y=np.log1p(dataset['SalePrice']), 
                        x=dataset[feature], ax=fig.add_subplot(gs[0,0]), color = 'red')

    plt.title('Linear\n' + 'Corr: ' + str(np.round(np.log1p(dataset['SalePrice']).corr(dataset[feature]),2)) + ', Skew: ' + 
               str(np.round(stats.skew(dataset[feature]),2)))
    
    fig.show()


# ### LotFrontage

# > Linear feet of street connected to property.

# In[ ]:


make_continuous_plot('LotFrontage')


# In[ ]:


dataset.LotFrontage = boxcox1p(dataset.LotFrontage, 0.35)


# ### LotArea

# > Area of the Lot.

# In[ ]:


make_continuous_plot('LotArea')


# In[ ]:


dataset.LotArea = boxcox1p(dataset.LotArea, 0.15)


# ### MasVnrArea

# > Masonry veneer area in square feet.

# In[ ]:


make_continuous_plot('MasVnrArea')


# In[ ]:


dataset.MasVnrArea = boxcox1p(dataset.MasVnrArea, 0.15)


# ### BsmtFinSF1 and BsmtFinSF2

# > Types 1 and 2 finished square feet.

# In[ ]:


make_continuous_plot('BsmtFinSF1')


# In[ ]:


# Let's transform those values into categories, because with boxcox 0.35 the correlation decreases a lot
print(min(dataset.BsmtFinSF1.values), ',',
      max(dataset.BsmtFinSF1.values))


# In[ ]:


bins = [-1, 250, 500, 750, 1000, 1250, 4011]
names = ['-250', '250-500', '500-750', '750-1000',
         '1000-1250', '1250-4011']

dataset.BsmtFinSF1 = pd.cut(dataset.BsmtFinSF1, bins, labels=names)


# In[ ]:


# change type from category to str
dataset.BsmtFinSF1 = dataset.BsmtFinSF1.astype(str)


# In[ ]:


make_discrete_plot('BsmtFinSF1',0,0)


# In[ ]:


make_continuous_plot('BsmtFinSF2')


# In[ ]:


# since there is no correlation at all, I'll flag this feature.
dataset.BsmtFinSF2 = dataset.BsmtFinSF2.apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


make_discrete_plot('BsmtFinSF2',0,0)


# ### BsmtUnfSF

# > Unfinished square feet of basement area.

# In[ ]:


make_continuous_plot('BsmtUnfSF')


# In[ ]:


# Let's transform those values into categories
print(min(dataset.BsmtUnfSF.values), ',',
      max(dataset.BsmtUnfSF.values))


# In[ ]:


bins = [-1, 250, 500, 750, 1000, 1250, 2400]
names = ['-250', '250-500', '500-750', '750-1000',
         '1000-1250', '1250-2400']

dataset.BsmtUnfSF = pd.cut(dataset.BsmtUnfSF, bins, labels=names)


# In[ ]:


# change type from category to str
dataset.BsmtUnfSF = dataset.BsmtUnfSF.astype(str)


# In[ ]:


make_discrete_plot('BsmtUnfSF',0,0)


# ### TotalBsmtSF

# > Total square feet of basement area.

# In[ ]:


make_continuous_plot('TotalBsmtSF')


# ### 1stFlrSF and 2ndFlrSF

# > First and Second Floors square feet.

# In[ ]:


# Let's add another variable, with the sum of these others
dataset["FlrSF"] = dataset["1stFlrSF"] + dataset["2ndFlrSF"] + dataset["TotalBsmtSF"]


# In[ ]:


make_continuous_plot('FlrSF')


# In[ ]:


make_continuous_plot('1stFlrSF')


# In[ ]:


make_continuous_plot('2ndFlrSF')


# In[ ]:


dataset.FlrSF = boxcox1p(dataset.FlrSF, 0.35)
dataset["1stFlrSF"] = boxcox1p(dataset["1stFlrSF"], 0.15)


# In[ ]:


# Let's transform values of 2ndFlrSF into categories
print(min(dataset['2ndFlrSF'].values), ',',
      max(dataset['2ndFlrSF'].values))


# In[ ]:


bins = [-1, 250, 500, 750, 1000, 1250, 2100]
names = ['-250', '250-500', '500-750', '750-1000',
         '1000-1250', '1250-2100']

dataset['2ndFlrSF'] = pd.cut(dataset['2ndFlrSF'], bins, labels=names)


# In[ ]:


# change type from category to str
dataset['2ndFlrSF'] = dataset['2ndFlrSF'].astype(str)


# In[ ]:


make_discrete_plot('2ndFlrSF',0,0)


# ### LowQualFinSF

# > Low quality finished square feet (all floors).

# In[ ]:


make_continuous_plot('LowQualFinSF')


# In[ ]:


# since there is no correlation, I'll flag it
dataset.LowQualFinSF = dataset.LowQualFinSF.apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


make_discrete_plot('LowQualFinSF',0,0)


# ### GrLivArea

# > Above grade (ground) living area square feet.

# In[ ]:


make_continuous_plot('GrLivArea')


# In[ ]:


dataset.GrLivArea = boxcox1p(dataset.GrLivArea,0.15)


# ### GarageArea

# > Area of the garage.

# In[ ]:


make_continuous_plot('GarageArea')


# ### WoodDeckSF

# > Wood deck area in square feet.

# In[ ]:


make_continuous_plot('WoodDeckSF')


# In[ ]:


dataset.WoodDeckSF = boxcox1p(dataset.WoodDeckSF,0.15)


# ### OpenPorchSF, EnclosedPorch, 3SsnPorch and ScreenPorch

# > Porch areas in square feet.

# In[ ]:


# These variables can be summed into one
dataset["PorchSF"] = dataset["OpenPorchSF"] + dataset["EnclosedPorch"] + dataset["3SsnPorch"] + dataset["ScreenPorch"]
dataset = dataset.drop(columns=["OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"])


# In[ ]:


make_continuous_plot('PorchSF')


# In[ ]:


dataset.PorchSF = boxcox1p(dataset.PorchSF, 0.35)


# ### PoolArea

# > Pool area in square feet.

# In[ ]:


make_continuous_plot('PoolArea')


# In[ ]:


# since there is no correlation, I'll flag it
dataset.PoolArea = dataset.PoolArea.apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


make_discrete_plot('PoolArea',0,0)


# ### MiscVal

# > $Value of miscellaneous feature.

# In[ ]:


make_continuous_plot('MiscVal')


# In[ ]:


dataset = dataset.drop(columns=["MiscVal"])


# ## Ordinal Variables

# > Here, we will first change the strings by integers, because the variables are ordinal, we can't get dummies unless the correlation with SalePrice is very low. We are going to make decisions based on correlation, missing values and so on.

# In[ ]:


# To remember

cat_ordinal = ['LotShape','Utilities','LandSlope','OverallQual','OverallCond','ExterQual','ExterCond',
               'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir',
               'Electrical','KitchenQual','Functional','FireplaceQu','GarageFinish','GarageQual','GarageCond',
               'PavedDrive','PoolQC','Fence']


# In[ ]:


# replace strings for integers, the details are in the description of the competition

dict_ = {"PoolQC": {"NA": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
         "Fence": {"NA": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
         "FireplaceQu": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
         "GarageCond": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
         "GarageQual": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
         "GarageFinish": {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3},
         "BsmtExposure": {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
         "BsmtCond": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
         "BsmtQual": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
         "BsmtFinType2": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
         "BsmtFinType1": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
         "Electrical": {"Mix": 0, "FuseP": 1, "FuseF": 2, "FuseA": 3, "SBrkr": 4},
         "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
         "KitchenQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
         "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
         "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
         "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
         "ExterQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
         "ExterCond": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
         "HeatingQC": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
         "CentralAir": {"N": 0, "Y": 1},
         "FireplaceQu": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
         "PavedDrive": {"N": 0, "P": 1, "Y": 2}}

dataset.replace(dict_, inplace=True)


# In[ ]:


dataset[cat_ordinal].dtypes


# In[ ]:


# one of the variables didn't change to integer, so let's change it
dataset.BsmtCond = dataset.BsmtCond.astype(int)


# ### LotShape
# 

# > General shape of property.

# In[ ]:


make_discrete_plot('LotShape',0,0)


# In[ ]:


# there few values = 0, so let's put them together with 1
dataset.LotShape = dataset.LotShape.apply(lambda x: 1 if x == 0 else x)


# In[ ]:


make_discrete_plot('LotShape',0,0)


# ### Utilities

# > Type of utilities available.

# In[ ]:


make_discrete_plot('Utilities',0,0)


# In[ ]:


# of course we need to drop this, just one point in 1
dataset = dataset.drop(columns='Utilities')


# ### LandSlope
# 

# > Slope of property.

# In[ ]:


make_discrete_plot('LandSlope',0,0)


# In[ ]:


# this variable can make more sense as flag, having gentle slope or not
dataset.LandSlope = dataset.LandSlope.apply(lambda x: 1 if x == 2 else 0)


# In[ ]:


make_discrete_plot('LandSlope',0,0)


# ### OverallQual
# 

# > Rates the overall material and finish of the house.

# In[ ]:


make_discrete_plot('OverallQual',0,0)


# In[ ]:


# there are few values < 2, so let's put them together with 2
dataset.OverallQual = dataset.OverallQual.apply(lambda x: 2 if x == 1 else x)


# In[ ]:


make_discrete_plot('OverallQual',0,0)


# ### OverallCond
# 

# > Rates the overall condition of the house.

# In[ ]:


make_discrete_plot('OverallCond',0,0)


# In[ ]:


# there are few values <= 2, so let's put them together with 3
dataset.OverallCond = dataset.OverallCond.apply(lambda x: 3 if x < 3 else x)


# In[ ]:


make_discrete_plot('OverallCond',0,0)


# ### ExterQual
# 

# > Evaluates the quality of the material on the exterior.

# In[ ]:


make_discrete_plot('ExterQual',0,0)


# ### ExterCond
# 

# > Evaluates the present condition of the material on the exterior.

# In[ ]:


make_discrete_plot('ExterCond',0,0)


# In[ ]:


# there are few values < 1, so let's put them together w1 and 0, and 4 with 3 and 2, as 1
dataset.ExterCond = dataset.ExterCond.apply(lambda x: 0 if x == 1 else x)
dataset.ExterCond = dataset.ExterCond.apply(lambda x: 1 if x >= 2 else x)


# In[ ]:


make_discrete_plot('ExterCond',0,0)


# ### BsmtQual
# 

# > Evaluates the height of the basement.

# In[ ]:


make_discrete_plot('BsmtQual',0,0)


# ### BsmtCond
# 

# > Evaluates the general condition of the basement.

# In[ ]:


make_discrete_plot('BsmtCond',0,0)


# In[ ]:


# there are few values = 1, so let's put them together with 0
dataset.BsmtCond = dataset.BsmtCond.apply(lambda x: 0 if x == 1 else x)


# In[ ]:


make_discrete_plot('BsmtCond',0,0)


# ### BsmtExposure
# 

# > Refers to walkout or garden level walls.

# In[ ]:


make_discrete_plot('BsmtExposure',0,0)


# ### BsmtFinType1
# 

# > Rating of basement finished area.

# In[ ]:


make_discrete_plot('BsmtFinType1',0,0)


# In[ ]:


# not many difference here between categories, so let's turn into string and later get dummies
dataset.BsmtFinType1 = dataset.BsmtFinType1.astype(str)


# ### BsmtFinType2
# 

# > Quality of second finished area (if present).

# In[ ]:


make_discrete_plot('BsmtFinType2',0,0)


# In[ ]:


# not many difference here between categories, so let's turn into string and later get dummies
dataset.BsmtFinType2 = dataset.BsmtFinType2.astype(str)


# ### HeatingQC
# 

# > Heating quality and condition.

# In[ ]:


make_discrete_plot('HeatingQC',0,0)


# In[ ]:


# There are few values = 0, so I will put together with 1
dataset.HeatingQC = dataset.HeatingQC.apply(lambda x: 1 if x == 0 else x)


# In[ ]:


make_discrete_plot('HeatingQC',0,0)


# ### CentralAir
# 

# > If Central air conditioning is present.

# In[ ]:


make_discrete_plot('CentralAir',0,0)


# ### Electrical
# 

# > Electrical system.

# In[ ]:


make_discrete_plot('Electrical',0,0)


# In[ ]:


# There are few values < 2, so I will put together with 2
dataset.Electrical = dataset.Electrical.apply(lambda x: 2 if x < 2 else x)


# In[ ]:


make_discrete_plot('Electrical',0,0)


# ### KitchenQual
# 

# > Kitchen quality.

# In[ ]:


make_discrete_plot('KitchenQual',0,0)


# ### Functional
# 

# > Home functionality (Assume typical unless deductions are warranted).

# In[ ]:


make_discrete_plot('Functional',0,0)


# In[ ]:


# There are few values = 1, so I will put together with 2
dataset.Functional = dataset.Functional.apply(lambda x: 2 if x < 2 else x)


# In[ ]:


make_discrete_plot('Functional',0,0)


# In[ ]:


# not many difference here between categories, so let's turn into string and later get dummies
dataset.Functional = dataset.Functional.astype(str)


# ### FireplaceQu
# 

# > Fireplace quality.

# In[ ]:


make_discrete_plot('FireplaceQu',0,0)


# ### GarageFinish
# 

# > Interior finish of the garage.

# In[ ]:


make_discrete_plot('GarageFinish',0,0)


# ### GarageQual
# 

# > Garage Quality.

# In[ ]:


make_discrete_plot('GarageQual',0,0)


# In[ ]:


# There are few values > 4, so I will put together with 4 and 1 with 0
dataset.GarageQual = dataset.GarageQual.apply(lambda x: 0 if x == 1 else x)
dataset.GarageQual = dataset.GarageQual.apply(lambda x: 4 if x == 5 else x)


# In[ ]:


make_discrete_plot('GarageQual',0,0)


# ### GarageCond
# 

# > Garage condition.

# In[ ]:


make_discrete_plot('GarageCond',0,0)


# In[ ]:


# There are few values > 3, so I will put together with 4 and 1 with 0
dataset.GarageCond = dataset.GarageCond.apply(lambda x: 0 if x == 1 else x)
dataset.GarageCond = dataset.GarageCond.apply(lambda x: 3 if x > 3 else x)


# In[ ]:


make_discrete_plot('GarageCond',0,0)


# In[ ]:


# not many difference here between categories, so let's turn into string and later get dummies
dataset.GarageCond = dataset.GarageCond.astype(str)


# ### PavedDrive
# 

# > Type of Paved driveway.

# In[ ]:


make_discrete_plot('PavedDrive',0,0)


# ### PoolQC
# 

# > Pool quality.

# In[ ]:


make_discrete_plot('PoolQC',0,0)


# In[ ]:


# not so many points for values > 0
dataset = dataset.drop(columns='PoolQC')


# ### Fence

# > Fence quality.

# In[ ]:


make_discrete_plot('Fence',0,0)


# In[ ]:


# not many difference here between categories, so let's turn into string and later get dummies
dataset.Fence = dataset.Fence.astype(str)


# ## Nominal Variables

# > Here we will analyze correlation with the boxplots and missing values. Clustering information when necessary from categories, decisions will be made to drop, flag or keep the column.

# In[ ]:


# To remember

cat_nominal = ['MSSubClass','MSZoning','Street','Alley','LandContour','LotConfig','Neighborhood',
               'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
               'Exterior2nd','MasVnrType','Foundation','Heating','GarageType','MiscFeature',
               'SaleType','SaleCondition']


# ### MSSubClass
# 

# > The building class.

# In[ ]:


make_discrete_plot('MSSubClass',0,0)


# In[ ]:


# this variable has integer values, but in fact they need to be categories as strings
dataset.MSSubClass = dataset.MSSubClass.astype(str)


# ### MSZoning
# 

# > The general zoning classification.

# In[ ]:


make_discrete_plot('MSZoning',0,0)


# ### Street
# 

# > Type of road access.

# In[ ]:


make_discrete_plot('Street',0,0)


# In[ ]:


# few values for Grvl
dataset = dataset.drop(columns='Street')


# ### Alley
# 

# > Type of alley access.

# In[ ]:


make_discrete_plot('Alley',0,0)


# ### LandContour
# 

# > Flatness of the property.

# In[ ]:


make_discrete_plot('LandContour',0,0)


# ### LotConfig
# 

# > Lot configuration.

# In[ ]:


make_discrete_plot('LotConfig',0,0)


# In[ ]:


# few values = FR3, so I'll decide to put together with FR2
dataset.LotConfig = dataset.LotConfig.apply(lambda x: 'FR2' if x == 'FR3' else x)


# In[ ]:


make_discrete_plot('LotConfig',0,0)


# ### Neighborhood
# 

# > Physical locations within Ames city limits.

# In[ ]:


make_discrete_plot('Neighborhood',45,45)


# ### Condition1
# 

# > Proximity to main road or railroad.

# In[ ]:


make_discrete_plot('Condition1',0,0)


# In[ ]:


# I'll cluster RRs
dataset.Condition1 = dataset.Condition1.apply(lambda x: 'RRs' if (x == 'RRNe' or x == 'RRAe' or x == 'RRNn' or x == 'RRAn') else x)


# In[ ]:


make_discrete_plot('Condition1',0,0)


# ### Condition2
# 

# > Proximity to main road or railroad (if a second is present).

# In[ ]:


make_discrete_plot('Condition2',0,0)


# In[ ]:


# most of the values are in Norm
dataset = dataset.drop(columns='Condition2')


# ### BldgType
# 

# > Type of dwelling.

# In[ ]:


make_discrete_plot('BldgType',0,0)


# ### HouseStyle
# 

# > Style of dwelling.

# In[ ]:


make_discrete_plot('HouseStyle',0,0)


# ### RoofStyle
# 

# > Type of roof.

# In[ ]:


make_discrete_plot('RoofStyle',0,0)


# In[ ]:


# few values from Gambrel to Shed, so I'll decide to put together = Other
dataset.RoofStyle = dataset.RoofStyle.apply(lambda x: 'Other' if (x == 'Gambrel' or x == 'Mansard' or x == 'Flat' or x == 'Shed') else x)


# In[ ]:


make_discrete_plot('RoofStyle',0,0)


# ### RoofMatl
# 

# > Roof material.

# In[ ]:


make_discrete_plot('RoofMatl',0,0)


# In[ ]:


# most of the values are in CompShg, let's drop it
dataset = dataset.drop(columns='RoofMatl')


# ### Exterior1st
# 

# > Exterior covering on house.

# In[ ]:


make_discrete_plot('Exterior1st',25,25)


# In[ ]:


# few values from BrkComm to CBlock, so I'll decide to put together = Other
dataset.Exterior1st = dataset.Exterior1st.apply(lambda x: 'Other' if (x == 'BrkComm' or x == 'AsphShn' or x == 'Stone' or x == 'ImStucc' or x== 'CBlock') else x)


# In[ ]:


make_discrete_plot('Exterior1st',25,25)


# ### Exterior2nd
# 

# > Exterior covering on house (if more than one material).

# In[ ]:


make_discrete_plot('Exterior2nd',25,25)


# In[ ]:


# similar behavior comparing to Exterior1st
dataset = dataset.drop(columns = 'Exterior2nd')


# ### MasVnrType
# 

# > Masonry veneer type.

# In[ ]:


make_discrete_plot('MasVnrType',0,0)


# ### Foundation
# 

# > Type of foundation.

# In[ ]:


make_discrete_plot('Foundation',0,0)


# In[ ]:


# few values for Wood and Stone, so I'll decide to put together = Other
dataset.Foundation = dataset.Foundation.apply(lambda x: 'Other' if (x == 'Wood' or x == 'Stone') else x)


# In[ ]:


make_discrete_plot('Foundation',0,0)


# ### Heating
# 

# > Type of heating.

# In[ ]:


make_discrete_plot('Heating',0,0)


# In[ ]:


# few values for every category except GasA
dataset.Heating = dataset.Heating.apply(lambda x: 1 if x == 'GasA' else 0)


# In[ ]:


make_discrete_plot('Heating',0,0)


# ### GarageType
# 

# > Garage type.

# In[ ]:


make_discrete_plot('GarageType',0,0)


# ### MiscFeature
# 

# > Miscellaneous feature not covered in other categories.

# In[ ]:


make_discrete_plot('MiscFeature',0,0)


# In[ ]:


# few values for every category except NA
dataset.MiscFeature = dataset.MiscFeature.apply(lambda x: 0 if x == 'NA' else 1)


# In[ ]:


make_discrete_plot('MiscFeature',0,0)


# ### SaleType
# 

# > Type of sale.

# In[ ]:


make_discrete_plot('SaleType',0,0)


# In[ ]:


# few values for every category except WD and New
dataset.SaleType = dataset.SaleType.apply(lambda x: x if (x == 'WD' or x == 'New') else 'Other')


# In[ ]:


make_discrete_plot('SaleType',0,0)


# ### SaleCondition

# > Condition of Sale.

# In[ ]:


make_discrete_plot('SaleCondition',0,0)


# In[ ]:


# few values for AdjLand, Alloca and Family
dataset.SaleCondition = dataset.SaleCondition.apply(lambda x: x if (x == 'Normal' or x == 'Abnormal' or x == 'Partial') else 'Other')


# In[ ]:


make_discrete_plot('SaleCondition',0,0)


# # 5-Train Model

# > Here we are going to get dummies for categorical variables, split train and test sets, analyze skewness if yet present for both sets, scale the data (Robust is better for outliers) and, finally, train the model for:
# * Lasso
# * ElasticNet
# * Kernel Ridge
# * Gradient Boosting Regressor
# * XGBoost
# * Light Gradient Boosting

# > The tuning parameters were obtained from GridSearchCV.

# In[ ]:


dataset = dataset.reset_index(drop=True)
dataset.shape


# In[ ]:


# splitting data 
X = dataset.loc[:,dataset.columns.difference(['SalePrice', 'Id'])]
Y = dataset.SalePrice

X = pd.get_dummies(X) # getting dummies


# > Analyzing skewness and replacing column with the best transformation (boxcox with different lambdas, log or keep the column as it is).

# ![Skewness](https://www.oreilly.com/library/view/statistical-inference-a/9781118309803/images/c03/nfg005.gif)

# [Image Source](https://www.oreilly.com/library/view/statistical-inference-a/9781118309803/c03anchor-8.html)

# In[ ]:


skewness = list(abs(stats.skew(X)) > 0.7)
counter = 0
lambdas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
for i in X.columns:
    if skewness[counter] == True:
        idx = np.argmin([abs(stats.skew(boxcox1p(X[i], 0.1))),
                         abs(stats.skew(boxcox1p(X[i], 0.15))),
                         abs(stats.skew(boxcox1p(X[i], 0.2))),
                         abs(stats.skew(boxcox1p(X[i], 0.25))),
                         abs(stats.skew(boxcox1p(X[i], 0.3))),
                         abs(stats.skew(boxcox1p(X[i], 0.35))),
                         abs(stats.skew(boxcox1p(X[i], 0.4))),
                         abs(stats.skew(boxcox1p(X[i], 0.45))),
                         abs(stats.skew(np.log1p(X[i]))),
                         abs(stats.skew(X[i]))])
        if idx < 8:
            X[i] = boxcox1p(X[i], lambdas[idx])
        if idx == 8:
            X[i] = np.log1p(X[i])
    counter = counter + 1


# > Scaling with RobustScaler (better for treating outliers). You can see the types of Scaling in the link below.
# * [Types of Scaling](http://benalexkeen.com/feature-scaling-with-scikit-learn/)

# In[ ]:


scaler = RobustScaler()
X = pd.DataFrame(scaler.fit_transform(X))


# In[ ]:


# train and test
X_train = X.iloc[0:1457].as_matrix()
Y_train = Y.iloc[0:1457].as_matrix()
X_test = X.iloc[1457:].as_matrix()
test_ids = dataset.Id.iloc[1457:]


# > Same skewness analysis for target variable. 

# In[ ]:


lambdas = [0.15, 0.25, 0.35, 0.45]
idx = np.argmin([abs(stats.skew(boxcox1p(Y_train, 0.15))),
                 abs(stats.skew(boxcox1p(Y_train, 0.25))),
                 abs(stats.skew(boxcox1p(Y_train, 0.35))),
                 abs(stats.skew(boxcox1p(Y_train, 0.45))),
                 abs(stats.skew(np.log1p(Y_train)))])
if idx < 4:
    Y_train = boxcox1p(Y_train, lambdas[idx])
if idx == 4:
            Y_train = np.log1p(Y_train)


# In[ ]:


idx


# > For the target variable, the log transformation is better.

# ## Lasso

# [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)

# In[ ]:


lasso = Lasso(alpha= 0.0005) # alpha was obtained with GridSearchCV


# ## Elastic Net

# [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)

# In[ ]:


elastic = ElasticNet(alpha=0.0005, l1_ratio=.9)  # parameters was obtained with GridSearchCV


# ## Kernel Ridge

# [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html)

# > GridSearch for Ridge.

# In[ ]:


"""

k_ridge = KernelRidge()

param_grid = {'alpha': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
              'kernel':['polynomial'], 
              'degree':[2,3,4,5,6,7,8],
              'coef0':[0,1,1.5,2,2.5,3,3.5,10]}

k = GridSearchCV(k_ridge, 
                 param_grid = param_grid, 
                 cv = 10, 
                 scoring = "neg_mean_squared_error", 
                 n_jobs = -1, 
                 verbose = 1)

k.fit(X_train,Y_train)

k_best = k.best_estimator_

k.best_score_

"""


# In[ ]:


k_ridge = KernelRidge(alpha=0.1, coef0=2.5, degree=3, gamma=None, kernel='polynomial',
            kernel_params=None)


# ## GBR

# [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

# > GridSearch for GBR.

# In[ ]:


"""

GBMR = GradientBoostingRegressor()

GBMR_param_grid = {'loss': ['huber'],
                   'n_estimators':[3000,3300], 
                   'learning_rate':[0.01],
                   'max_depth':[3,5], 
                   'max_features':[18,20],
                   'min_samples_leaf':[2,3], 
                   'min_samples_split':[3,5]}

gsGBMR = GridSearchCV(GBMR, 
                      param_grid = GBMR_param_grid, 
                      cv = 5, 
                      scoring = "neg_mean_squared_error", 
                      n_jobs = -1, 
                      verbose = 1)

gsGBMR.fit(X_train,Y_train)

GBMR_best = gsGBMR.best_estimator_

gsGBMR.best_score_

"""


# In[ ]:


g_boost = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse',
                          init=None, learning_rate=0.01, loss='huber',
                          max_depth=3, max_features=18, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=3, min_samples_split=5,
                          min_weight_fraction_leaf=0.0, n_estimators=3300,
                          n_iter_no_change=None,random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)


# ## XGBoost

# [Documentation](https://dask-ml.readthedocs.io/en/stable/modules/generated/dask_ml.xgboost.XGBRegressor.html)

# > GridSearch for XGBoost.

# In[ ]:


"""

XGBMR = XGBRegressor()

XGBMR_param_grid = {'learning_rate': [0.001,0.01,0.1], 
                  'max_depth': [3,4,7],
                  'n_estimators': [3300,4000], 
                  'gamma': [0],
                  'subsample': [0.3,0.5,0.8]}

gsXGBMR = GridSearchCV(XGBMR, 
                      param_grid = XGBMR_param_grid, 
                      cv = 5, 
                      scoring = "neg_mean_squared_error", 
                      n_jobs = -1, 
                      verbose = 1)

gsXGBMR.fit(X_train,Y_train)

XGBMR_best = gsXGBMR.best_estimator_

gsXGBMR.best_score_

"""


# In[ ]:


xg_boost = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=4, min_child_weight=1, missing=None, n_estimators=3300,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=0.3, verbosity=1)


# ## Light GBM

# [Documentation](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor)

# > GridSearch for Light GBM.

# In[ ]:


'''

LGBMR = LGBMRegressor()

LGBMR_param_grid = {'objective':['regression','dart','goss','rf'],
                    'num_leaves':[7],
                    'learning_rate':[0.01], 
                    'n_estimators': [3300],
                    'max_depth':[4], 
                    'max_bin': [65],
                    'bagging_fraction':[0.6],
                    'bagging_freq':[9], 
                    'feature_fraction':[0.1],
                    'feature_fraction_seed':[1],
                    'bagging_seed':[14],
                    'min_data_in_leaf':[5], 
                    'min_sum_hessian_in_leaf':[5],
                    'colsample_bytree':[0],
                    'reg_alpha':[0.2],
                    'reg_lambda':[0.1]}

gsLGBMR = GridSearchCV(LGBMR, 
                      param_grid = LGBMR_param_grid, 
                      cv = 10, 
                      scoring = "neg_mean_squared_error", 
                      n_jobs = -1, 
                      verbose = 1)

gsLGBMR.fit(X_train,Y_train)

LGBMR_best = gsLGBMR.best_estimator_

gsLGBMR.best_score_

'''


# In[ ]:


lgbm = LGBMRegressor(bagging_fraction=0.6, bagging_freq=9, bagging_seed=14,
              boosting_type='gbdt', class_weight=None, colsample_bytree=0,
              feature_fraction=0.1, feature_fraction_seed=1,
              importance_type='split', learning_rate=0.01, max_bin=65,
              max_depth=4, min_child_samples=20, min_child_weight=0.001,
              min_data_in_leaf=5, min_split_gain=0.0, min_sum_hessian_in_leaf=5,
              n_estimators=3300, n_jobs=-1, num_leaves=7,
              objective='regression', random_state=None, reg_alpha=0.2,
              reg_lambda=0.1, silent=True, subsample=1.0,
              subsample_for_bin=200000, subsample_freq=0)


# # 6-Stacking

# [Documentation](http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/)

# > Here I decided to choose some models to be regressors and lasso to be the meta regressor. Lasso seems to be a good model for this competition, thats why it will be the meta.

# ![Stacking Method](https://miro.medium.com/max/1183/0*GHYCJIjkkrP5ZgPh.png)

# [Image Source](https://medium.com/@rrfd/boosting-bagging-and-stacking-ensemble-methods-with-sklearn-and-mlens-a455c0c982de)

# In[ ]:


stacking = StackingRegressor(regressors=(elastic, g_boost, k_ridge),
                             meta_regressor = lasso)

param_grid = {} 

stack = GridSearchCV(stacking, 
                   param_grid = param_grid,
                   cv = 10, 
                   scoring = "neg_mean_squared_error",
                   n_jobs = 5, 
                   verbose = 1)

stack.fit(X_train,Y_train)

s_best = stack.best_estimator_

stack.best_score_


# In[ ]:


stacking = s_best


# # 7-Voting

# [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)

# > A voting regressor is an ensemble meta-estimator that fits base regressors each on the whole dataset. It, then, averages the individual predictions to form a final prediction.

# > Here I used the other models not yet trained to be combined with the Stacking Regressor and make a powerful one, using the Voting Regressor.

# In[ ]:


voting = VotingRegressor(estimators=[('xgboost', xg_boost), 
                                     ('lgbm', lgbm),
                                     ('stacking', stacking)])

v_param_grid = {} # tuning voting parameter

gsV = GridSearchCV(voting, 
                   param_grid = v_param_grid,
                   cv = 10, 
                   scoring = "neg_mean_squared_error",
                   n_jobs = 5, 
                   verbose = 1)

gsV.fit(X_train,Y_train)

v_best = gsV.best_estimator_

gsV.best_score_


# In[ ]:


voting = v_best


# # 8-Submission

# > I'll use the best models and put weight on their predictions.

# In[ ]:


# Voting
y_vote = np.expm1(voting.predict(X_test))

# Lasso
lasso.fit(X_train,Y_train)
y_lasso = np.expm1(lasso.predict(X_test))


# In[ ]:


y_pred =(0.6*y_vote + 0.4*y_lasso)


# In[ ]:


submission = pd.DataFrame(test_ids.values, columns = ["Id"])
submission['SalePrice'] = list(y_pred)
submission.to_csv('submission.csv',index=False)


# ### If you made it this so far, let me know if you have questions, suggestions or critiques to improve the models. Thanks a lot!

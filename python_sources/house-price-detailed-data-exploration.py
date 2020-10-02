#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook is the second of series of three to train myself in facing regression problems. 
# 
# [In the previous notebook](https://www.kaggle.com/lucabasa/house-price-cleaning-without-dropping-features "house price: cleaning without dropping features"), I had performed a general exploration and a general cleaning of the missing values. However, I have to admit I took too many steps in the preprocessing of the data because not only I have grouped sparse classes as if it is necessary, but also encoded them with numbers. The side effect is introducing an ordering to categories that are not supposed to have one. 
# 
# The plan of this notebook, following the same division of the features in "rooms" introduced in [part one](https://www.kaggle.com/lucabasa/house-price-cleaning-without-dropping-features), is:
# 
# * Look at the correlation with the target and between features
# * Propose possible transformations
# * Select the features to train my models (which will be done in the next notebook)
# 
# Another decision I took in this notebook regards the ordinal features, common in the *quality* and *condition* features. Here, an ordering is necessary but I want to propose a different way to encode these features with numerical values. For example, the difference between *Fair* and *Good* is not the necessarily the same of the one between *Good* and *Excellent*. Moreover, every feature can (and will) have different scales.
# 
# As usual, the notebook is published with the double goal of inspiring new ideas and receiving feedbacks to get better all together.
# 
# Enjoy the read.

# In[ ]:


# standard
import pandas as pd
import numpy as np

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# ---------- DF IMPORT -------------
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
combine = [df_train, df_test]
df_train.name = "Train"
df_test.name = "Test"
pd.set_option("display.max_columns", None)


# # Previously, on HousePricing
# 
# This is one huge cell that contains every meaningful action to prepare the data.

# In[ ]:


#Function to check how a categorical variable segments the target
def segm_target(var):
    count = df_train[[var, "SalePrice"]].groupby([var], as_index=True).count()
    count.columns = ['Count']
    mean = df_train[[var, "SalePrice"]].groupby([var], as_index=True).mean()
    mean.columns = ['Mean']
    ma = df_train[[var, "SalePrice"]].groupby([var], as_index=True).max()
    ma.columns = ['Max']
    mi = df_train[[var, "SalePrice"]].groupby([var], as_index=True).min()
    mi.columns = ['Min']
    median = df_train[[var, "SalePrice"]].groupby([var], as_index=True).median()
    median.columns = ['Median']
    std = df_train[[var, "SalePrice"]].groupby([var], as_index=True).std()
    std.columns = ['Std']
    df = pd.concat([count, mean, ma, mi, median, std], axis=1)
    fig, ax = plt.subplots(1,2, figsize=(12, 5))
    sns.boxplot(var,"SalePrice",data=df_train, ax=ax[0])
    sns.boxplot(var,"LogPrice",data=df_train, ax=ax[1])
    fig.show()
    return df

#Function to check correlation with target of a number of numerical features
def corr_target(*arg):
    print(df_train[[f for f in arg]].corr())
    num = len(arg) - 1
    rows = int(num/2) + (num % 2 > 0)
    arg = list(arg)
    target = arg[-1]
    del arg[-1]
    y = df_train[target]
    fig, ax = plt.subplots(rows, 2, figsize=(12, 5 * (rows)))
    i = 0
    j = 0
    for feat in arg:
        x = df_train[feat]
        if (rows > 1):
            sns.regplot(x=x, y=y, ax=ax[i][j])
            j = (j+1)%2
            i = i + 1 - j
        else:
            sns.regplot(x=x, y=y, ax=ax[i])
            i = i+1
    fig.show()


# In[ ]:


# Cell containing all the pre-processing performed in the previous notebook by just following the documentation
# (and some common sense)
for df in combine:
    #LotFrontage
    df.loc[df.LotFrontage.isnull(), 'LotFrontage'] = 0
    #Alley
    df.loc[df.Alley.isnull(), 'Alley'] = "NoAlley"
    #MSSubClass
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    #MissingBasement
    fil = ((df.BsmtQual.isnull()) & (df.BsmtCond.isnull()) & (df.BsmtExposure.isnull()) &
          (df.BsmtFinType1.isnull()) & (df.BsmtFinType2.isnull()))
    fil1 = ((df.BsmtQual.notnull()) | (df.BsmtCond.notnull()) | (df.BsmtExposure.notnull()) |
          (df.BsmtFinType1.notnull()) | (df.BsmtFinType2.notnull()))
    df.loc[fil1, 'MisBsm'] = 0
    df.loc[fil, 'MisBsm'] = 1
    #BsmtQual
    df.loc[fil, 'BsmtQual'] = "NoBsmt" #missing basement
    #BsmtCond
    df.loc[fil, 'BsmtCond'] = "NoBsmt" #missing basement
    #BsmtExposure
    df.loc[fil, 'BsmtExposure'] = "NoBsmt" #missing basement
    #BsmtFinType1
    df.loc[fil, 'BsmtFinType1'] = "NoBsmt" #missing basement
    #BsmtFinType2
    df.loc[fil, 'BsmtFinType2'] = "NoBsmt" #missing basement
    #FireplaceQu
    df.loc[(df.Fireplaces == 0) & (df.FireplaceQu.isnull()), 'FireplaceQu'] = "NoFire" #missing
    #MisGarage
    fil = ((df.GarageYrBlt.isnull()) & (df.GarageType.isnull()) & (df.GarageFinish.isnull()) &
          (df.GarageQual.isnull()) & (df.GarageCond.isnull()))
    fil1 = ((df.GarageYrBlt.notnull()) | (df.GarageType.notnull()) | (df.GarageFinish.notnull()) |
          (df.GarageQual.notnull()) | (df.GarageCond.notnull()))
    df.loc[fil1, 'MisGarage'] = 0
    df.loc[fil, 'MisGarage'] = 1
    #GarageYrBlt
    df.loc[df.GarageYrBlt > 2200, 'GarageYrBlt'] = 2007 #correct mistake
    df.loc[fil, 'GarageYrBlt'] = 0
    #GarageType
    df.loc[fil, 'GarageType'] = "NoGrg" #missing garage
    #GarageFinish
    df.loc[fil, 'GarageFinish'] = "NoGrg" #missing
    #GarageQual
    df.loc[fil, 'GarageQual'] = "NoGrg" #missing
    #GarageCond
    df.loc[fil, 'GarageCond'] = "NoGrg" #missing
    #Fence
    df.loc[df.Fence.isnull(), 'Fence'] = "NoFence" #missing fence


# In[ ]:


# We know already we will need to deal with homoscedasticity (written without checking the real spelling)
df_train["LogPrice"] = np.log1p(df_train["SalePrice"])
df_train.LogPrice.hist(bins = 100)


# # Data Analysis
# 
# I will have a look to all the features divided following the grouping proposed in the previous notebook so that it is easier for me to take decisions out of it.
# 
# ## Zone, Street and access
# 
# * **Numerical Features**: LotFrontage, LotArea
# * **Categorical Features**: MSZoning, Street, Alley, LotShape, LandContour, LotConfig, LandSlope, Neighborhood, Condition1, Condition2

# In[ ]:


corr_target('LotFrontage', 'LotArea', 'LogPrice')


# In[ ]:


# I am curious to see if the sum of the two is better (using the sqrt of the Area because it makes more sense)
df_train['LotFrontTot'] = df_train['LotFrontage'] + np.sqrt(df_train['LotArea'])

print(df_train[['LotFrontTot', 'LotFrontage','LotArea']].corr())
print("_"*40)
corr_target('LotFrontTot', 'LogPrice')


# In[ ]:


var = "MSZoning"
segm_target(var)


# In[ ]:


# Trying a grouping to handle sparse classes
df_train.loc[(df_train.MSZoning == 'RH') | (df_train.MSZoning == 'RM'), 'MSZoningGroup'] = 'ResMedHig'
df_train.loc[(df_train.MSZoning == 'FV'), 'MSZoningGroup'] = 'Vil'
df_train.loc[(df_train.MSZoning == 'RL')| (df_train.MSZoning == 'C (all)'), 'MSZoningGroup'] = 'ResLowCom'
var = 'MSZoningGroup'
segm_target(var)


# In[ ]:


var = "Street"
segm_target(var)


# In[ ]:


var = "Alley"
segm_target(var)


# In[ ]:


df_train.loc[(df_train.Alley == 'Grvl') | (df_train.Alley == 'Pave'), 'AlleyGroup'] = 'Alley'
df_train.loc[df_train.Alley == 'NoAlley', 'AlleyGroup'] = 'NoAlley'

var = 'AlleyGroup'
segm_target(var)


# In[ ]:


var = "LotShape"
segm_target(var)


# In[ ]:


irr = ['IR1', 'IR2', 'IR3']
df_train.loc[(df_train.LotShape.isin(irr)), 'LotShapeGroup'] = 'Irreg'
df_train.loc[df_train.LotShape == 'Reg', 'LotShapeGroup'] = 'Reg'

var = 'LotShapeGroup'
segm_target(var)


# In[ ]:


var = "LandContour"
segm_target(var)


# In[ ]:


irr = ['Bnk', 'Low', 'HLS']
df_train.loc[(df_train.LandContour.isin(irr)), 'LandContourGroup'] = 'NotLvl'
df_train.loc[df_train.LandContour == 'Lvl', 'LandContourGroup'] = 'Lvl'

var = 'LandContourGroup'
segm_target(var)


# In[ ]:


var = "LotConfig"
segm_target(var)


# In[ ]:


df_train.loc[(df_train.LotConfig == 'FR2') | (df_train.LotConfig == 'FR3'), 'LotConfigGroup'] = 'FR'
df_train.loc[df_train.LotConfig == 'Corner', 'LotConfigGroup'] = 'Corner'
df_train.loc[df_train.LotConfig == 'CulDSac', 'LotConfigGroup'] = 'CulDSac'
df_train.loc[df_train.LotConfig == 'Inside', 'LotConfigGroup'] = 'Inside'

var = 'LotConfigGroup'
segm_target(var)


# In[ ]:


var = "LandSlope"
segm_target(var)


# In[ ]:


df_train.loc[(df_train.LandSlope == 'Mod') | (df_train.LandSlope == 'Sev'), 'LandSlopeGroup'] = 'NonGlt'
df_train.loc[df_train.LandSlope == 'Gtl', 'LandSlopeGroup'] = 'Gtl'

var = 'LandSlopeGroup'
segm_target(var)


# In[ ]:


var = "Neighborhood"
segm_target(var)


# In[ ]:


var = "Condition1"
segm_target(var)


# In[ ]:


ArtFee = ['Artery', 'Feedr']
stat = ['RRAe', 'RRAn', 'RRNe', 'RRNn']
pos = ['PosA', 'PosN']
df_train.loc[(df_train.Condition1.isin(ArtFee)), 'Condition1Group'] = 'ArtFee'
df_train.loc[(df_train.Condition1.isin(stat)), 'Condition1Group'] = 'Station'
df_train.loc[(df_train.Condition1.isin(pos)), 'Condition1Group'] = 'Station'
df_train.loc[df_train.Condition1 == 'Norm', 'Condition1Group'] = 'Norm'

var = 'Condition1Group'
segm_target(var)


# In[ ]:


var = "Condition2"
segm_target(var)


# ### Correlation between features
# 
# Here I will just check that those features that gave me a good impression are not too correlated and perform some bivariate analysis.

# In[ ]:


pd.crosstab(df_train['LotShapeGroup'], df_train['LotConfigGroup'])


# In[ ]:


pd.crosstab(df_train['Condition1Group'], df_train['LotShapeGroup'])


# In[ ]:


g = sns.FacetGrid(df_train, col='LotConfigGroup', hue='LotShapeGroup')
g.map(plt.hist, 'LogPrice', alpha= 0.3, bins=20)
g.add_legend()


# In[ ]:


g = sns.FacetGrid(df_train, col='Condition1Group', hue='LotShapeGroup', size = 5)
g.map(plt.hist, 'LogPrice', alpha= 0.3, bins=20)
g.add_legend()


# * Small correlations between Price and MSZoning, Alley, LotShape, and Condition1
# * No correlation between the features
# * LotShape seems to have more importance than Condition1. I can try to combine them.
# 
# 
# ## Type, Quality, and Condition
# 
# * **Numerical features**: OverallQual, OverallCond, YearBuilt, YearRemodAdd
# * **Categorical features**: MSSubClass, BldgType, HouseStyle
# 
# ### Correlation with target

# In[ ]:


corr_target('OverallQual', 'OverallCond','SalePrice')


# In[ ]:


corr_target('OverallQual', 'OverallCond','LogPrice')


# In[ ]:


corr_target('YearBuilt', 'YearRemodAdd', 'SalePrice')


# In[ ]:


x = df_train['YearBuilt']
x1 = df_train['YearRemodAdd']
y = df_train['LogPrice']

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0], x_estimator = np.mean)
sns.regplot(x=x1, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()


# In[ ]:


# let's take the most recent year between build and remod
df_train['YearMostRec'] = df_train[['YearBuilt', 'YearRemodAdd']].apply(np.max, axis=1)

x = df_train['YearMostRec']
y = df_train['LogPrice']

print(df_train[['YearMostRec', 'LogPrice']].corr())
fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()


# In[ ]:


var = "MSSubClass"
segm_target(var)


# In[ ]:


var = "BldgType"
segm_target(var)


# In[ ]:


df_train.loc[(df_train.BldgType == '2fmCon') | (df_train.BldgType == 'Duplex'), 'BldTypeGroup'] = '2FamDup'
df_train.loc[(df_train.BldgType == 'Twnhs') | (df_train.BldgType == 'TwnhsE'), 'BldTypeGroup'] = 'Twnhs+E'
df_train.loc[(df_train.BldgType == '1Fam'), 'BldTypeGroup'] = '1Fam'
var = 'BldTypeGroup'
segm_target(var)


# In[ ]:


var = "HouseStyle"
segm_target(var)


# In[ ]:


onepl = ['1.5Fin', '1.5Unf']
twopl = ['2.5Fin', '2.5Unf', '2Story']
spl = ['SFoyer', 'SLvl']
df_train.loc[df_train.HouseStyle.isin(onepl), 'HStyleGroup'] = '1.5'
df_train.loc[df_train.HouseStyle.isin(twopl), 'HStyleGroup'] = '2plus'
df_train.loc[df_train.HouseStyle.isin(spl), 'HStyleGroup'] = 'Split'
df_train.loc[df_train.HouseStyle == '1Story', 'HStyleGroup'] = "1Story"
var = 'HStyleGroup'
segm_target(var)


# ### Correlations between features

# In[ ]:


pd.crosstab(df_train["HStyleGroup"],df_train['OverallQual'])


# In[ ]:


g = sns.FacetGrid(df_train, hue="HStyleGroup", size = 5)
g.map(plt.hist, 'OverallQual', alpha= 0.5, bins=10)
g.add_legend()


# In[ ]:


pd.crosstab(df_train["BldTypeGroup"],df_train['OverallQual'])


# In[ ]:


g = sns.FacetGrid(df_train, hue="BldTypeGroup", size = 5)
g.map(plt.hist, 'OverallQual', alpha= 0.5, bins=10)
g.add_legend()


# In[ ]:


g = sns.FacetGrid(df_train, col="HStyleGroup", hue="BldTypeGroup")
g.map(plt.hist, 'OverallQual', alpha= 0.3, bins=10)
g.add_legend()


# In[ ]:


g = sns.FacetGrid(df_train, hue="BldTypeGroup", size= 7)
g.map(plt.hist, 'YearBuilt', alpha= 0.3, bins=50)
g.add_legend()


# In[ ]:


g = sns.FacetGrid(df_train, hue="BldTypeGroup", size = 7)
g.map(plt.scatter, 'YearBuilt', 'LogPrice', edgecolor="w")
g.add_legend()


# In[ ]:


g = sns.FacetGrid(df_train, hue="HStyleGroup", size=7)
g.map(plt.hist, 'YearBuilt', alpha= 0.3, bins=50)
g.add_legend()


# * Still don't know what to do with MSSubClass
# * Price and OverallQual are very well correlated, the log of the price gives an even better reading
# * Most of the results in OverallCond are 5, general discending trend that is surpising. Looking at the mean, excluding 2 and 5, we get an ascending trend.
# * For YearBlt and Remod, in both cases an ascending trend, very clear for YearRemodAdd
# * No data of YearRemodAdd prior 1950, probably missing data filled by the owner, this makes the feature untrustable to me since I can't build a model on something which is collected in an unclear way.
# * 1Fam is very populated and seems to have a very large distribution and outliers
# * 2Fam and Duplex have a few outliers but their price is very consistent
# * Town houses are in between these two categories
# * Slightly better distributed in the lower part of the prices, probably a log transformation will show different behaviour
# 
# **HouseStyle and OverallQual**: all with same distribution, but houses with two or more stories have generally a better quality
# 
# ** BldType and YearBuilt**: Town houses are more recent, 2FamDup are built till the 80's
# 
# ** HouseStyle and YearBuilt**: 1.5 are old news, replaced by split levels. 2Plus is fairly recent.
# 
# 
# ## Exterior and materials
# 
# * **Numerical Features**: MasVnrArea 
# * **Categorical Features**: Foundation, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, ExterQual, ExterCond

# In[ ]:


x = df_train['MasVnrArea']
y = df_train['SalePrice']

sns.regplot(x=x, y=y)


# In[ ]:


x = df_train['MasVnrArea']
y = df_train['LogPrice']

sns.regplot(x=x, y=y)


# In[ ]:


var = "Foundation"
segm_target(var)


# In[ ]:


fancy = ['BrkTil', 'Stone', 'Wood']
cement = ['PConc', 'Slab']
df_train.loc[df_train.Foundation.isin(fancy), 'FoundGroup'] = 'Fancy'
df_train.loc[df_train.Foundation.isin(cement), 'FoundGroup'] = 'Cement'
df_train.loc[df_train.Foundation == 'CBlock', 'FoundGroup'] = 'Cider'

var = "FoundGroup"
segm_target(var)


# In[ ]:


var = "RoofStyle"
segm_target(var)


# In[ ]:


nogable = ['Flat', 'Gambrel', 'Hip', 'Mansard', 'Shed']
df_train.loc[df_train.RoofStyle.isin(nogable), 'RoofStyleGroup'] = 'NoGable'
df_train.loc[df_train.RoofStyle == 'Gable', 'RoofStyleGroup'] = 'Gable'

var = "RoofStyleGroup"
segm_target(var)


# In[ ]:


var = "RoofMatl"
segm_target(var)


# In[ ]:


var = "Exterior1st"
segm_target(var)


# In[ ]:


other = ['Stucco', 'ImStucc', 'CemntBd', 'AsbShng', 'AsphShn', 'CBlock', 'Stone']
wood = ['Wd Sdng', 'WdShing', 'Plywood']

df_train.loc[df_train.Exterior1st.isin(other), 'Ext1Group'] = 'Other'
df_train.loc[df_train.Exterior1st.isin(wood), 'Ext1Group'] = 'Wood'
df_train.loc[df_train.Exterior1st == 'MetalSd', 'Ext1Group'] = 'MetalSd'
df_train.loc[df_train.Exterior1st == 'HdBoard', 'Ext1Group'] = 'HdBoard'

var = "Ext1Group"
segm_target(var)


# In[ ]:


var = "Exterior2nd"
segm_target(var)


# In[ ]:


other = ['Stucco', 'ImStucc', 'BrkFace', 'Brk Cmn', 'CmentBd', 'AsbShng', 
        'AsphShn', 'CBlock', 'Stone', 'Other']
wood = ['Wd Sdng', 'Wd Shng']

df_train.loc[df_train.Exterior2nd.isin(other), 'Ext2Group'] = 'Other'
df_train.loc[df_train.Exterior2nd.isin(wood), 'Ext2Group'] = 'Wood'
df_train.loc[df_train.Exterior2nd == 'MetalSd', 'Ext2Group'] = 'MetalSd'
df_train.loc[df_train.Exterior2nd == 'HdBoard', 'Ext2Group'] = 'HdBoard'

var = "Ext2Group"
segm_target(var)


# In[ ]:


var = "MasVnrType"
segm_target(var)


# In[ ]:


df_train.loc[df_train.MasVnrType == 'None', 'MasTypeGroup'] = 'None'
df_train.loc[df_train.MasVnrType == 'Stone', 'MasTypeGroup'] = 'Stone'
df_train.loc[(df_train.MasVnrType == 'BrkCmn') | (df_train.MasVnrType == 'BrkFace'), 'MasTypeGroup'] = 'Bricks'

var = "MasTypeGroup"
segm_target(var)


# In[ ]:


var = "ExterQual"
segm_target(var)


# In[ ]:


df_train.loc[df_train.ExterQual == 'Fa', 'ExtQuGroup'] = 1
df_train.loc[df_train.ExterQual == 'TA', 'ExtQuGroup'] = 1
df_train.loc[df_train.ExterQual == 'Gd', 'ExtQuGroup'] = 2
df_train.loc[df_train.ExterQual == 'Ex', 'ExtQuGroup'] = 3

x = df_train['ExtQuGroup']
y = df_train['LogPrice']
print(df_train[['ExtQuGroup','SalePrice' ]].corr())

sns.regplot(x = x, y = y, x_estimator = np.mean)


# In[ ]:


var = "ExterCond"
segm_target(var)


# In[ ]:


df_train.loc[df_train.ExterCond == 'Po', 'ExtCoGroup'] = 1
df_train.loc[df_train.ExterCond == 'Fa', 'ExtCoGroup'] = 1
df_train.loc[df_train.ExterCond == 'TA', 'ExtCoGroup'] = 1
df_train.loc[df_train.ExterCond == 'Gd', 'ExtCoGroup'] = 2
df_train.loc[df_train.ExterCond == 'Ex', 'ExtCoGroup'] = 2

x = df_train['ExtCoGroup']
y = df_train['LogPrice']
print(df_train[['ExtCoGroup','SalePrice']].corr())

sns.regplot(x = x, y = y, x_estimator = np.mean)


# ### Correlation between features

# In[ ]:


x = df_train['ExtQuGroup']
y = df_train['MasVnrArea']

print(df_train[['ExtQuGroup', 'MasVnrArea']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
fig.show()


# In[ ]:


g = sns.FacetGrid(df_train, hue='MasTypeGroup', size = 5)
g.map(plt.hist, 'MasVnrArea', alpha= 0.5, bins=10)
g.add_legend()


# In[ ]:


pd.crosstab(df_train['MasTypeGroup'], df_train['ExtQuGroup'])


# In[ ]:


pd.crosstab(df_train['MasTypeGroup'], df_train['FoundGroup'])


# In[ ]:


g = sns.FacetGrid(df_train, col='MasTypeGroup', hue='FoundGroup')
g.map(plt.hist, 'LogPrice', alpha= 0.3, bins=50)
g.add_legend()


# * There is a small correlation between Price and MasVnrArea, Foundation, MasVnrType, ExterQual
# * No particular correlation between the features
# 
# ## Basement
# 
# * **Numerical Features**: BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath
# * **Categorical Features**: BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, MisBsm

# In[ ]:


corr_target('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'LogPrice')


# In[ ]:


# I am interested in the total of finished SF
df_train['BsmtFinTotSF'] = df_train['BsmtFinSF1'] + df_train['BsmtFinSF2']

print(df_train[['BsmtFinTotSF', 'LogPrice']].corr())

x = df_train['BsmtFinTotSF']
y = df_train['LogPrice']

sns.regplot(x = x, y = y)


# In[ ]:


# What about the percentage of unfinished SF
df_train['BsmtPercUnf'] = df_train['BsmtUnfSF'] / df_train['TotalBsmtSF']
df_train['BsmtPercUnf'].fillna(1) #37 missing basements are actually complete

print(df_train[['BsmtPercUnf', 'LogPrice']].corr())

x = df_train['BsmtPercUnf']
y = df_train['LogPrice']

sns.regplot(x = x, y = y)


# In[ ]:


corr_target('BsmtFullBath', 'BsmtHalfBath', 'LogPrice')


# In[ ]:


x = df_train['BsmtFullBath']
x1 = df_train['BsmtHalfBath']
y = df_train['LogPrice']

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0], x_estimator = np.mean)
sns.regplot(x=x1, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()


# In[ ]:


df_train['BsmtBath'] = 0
df_train.loc[(df_train['BsmtFullBath'] > 0) | (df_train['BsmtHalfBath'] > 0), 'BsmtBath'] = 1

var = 'BsmtBath'
segm_target(var)


# In[ ]:


var = "BsmtQual"
segm_target(var)


# In[ ]:


df_train.loc[df_train.BsmtQual == 'NoBsmt', 'BsmtQuGroup'] = 0
df_train.loc[df_train.BsmtQual == 'Fa', 'BsmtQuGroup'] = 1
df_train.loc[df_train.BsmtQual == 'TA', 'BsmtQuGroup'] = 4
df_train.loc[df_train.BsmtQual == 'Gd', 'BsmtQuGroup'] = 10
df_train.loc[df_train.BsmtQual == 'Ex', 'BsmtQuGroup'] = 21

x = df_train['BsmtQuGroup']
y = df_train['LogPrice']
print(df_train[['BsmtQuGroup','SalePrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)


# In[ ]:


var = "BsmtCond"
segm_target(var)


# In[ ]:


var = "BsmtExposure"
segm_target(var)


# In[ ]:


df_train.loc[df_train.BsmtExposure == 'NoBsmt', 'BsmtExGroup'] = 0
df_train.loc[df_train.BsmtExposure == 'No', 'BsmtExGroup'] = 6
df_train.loc[df_train.BsmtExposure == 'Mn', 'BsmtExGroup'] = 7
df_train.loc[df_train.BsmtExposure == 'Av', 'BsmtExGroup'] = 8
df_train.loc[df_train.BsmtExposure == 'Gd', 'BsmtExGroup'] = 12


x = df_train['BsmtExGroup']
y = df_train['LogPrice']
print(df_train[['BsmtExGroup','SalePrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)


# In[ ]:


var = "BsmtFinType1"
segm_target(var)


# In[ ]:


df_train.loc[df_train.BsmtFinType1 == 'NoBsmt', 'BsmtF1Group'] = 0
df_train.loc[df_train.BsmtFinType1 == 'Unf', 'BsmtF1Group'] = 6
df_train.loc[df_train.BsmtFinType1 == 'LwQ', 'BsmtF1Group'] = 4
df_train.loc[df_train.BsmtFinType1 == 'Rec', 'BsmtF1Group'] = 4
df_train.loc[df_train.BsmtFinType1 == 'BLQ', 'BsmtF1Group'] = 4
df_train.loc[df_train.BsmtFinType1 == 'ALQ', 'BsmtF1Group'] = 5
df_train.loc[df_train.BsmtFinType1 == 'GLQ', 'BsmtF1Group'] = 11


x = df_train['BsmtF1Group']
y = df_train['LogPrice']
print(df_train[['BsmtF1Group','SalePrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,8))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)


# In[ ]:


var = "BsmtFinType2"
segm_target(var)


# In[ ]:


var = "MisBsm"
segm_target(var)


# ### Correlations between features

# In[ ]:


df_train[['TotalBsmtSF','BsmtFinTotSF', 'BsmtBath', 'BsmtQuGroup']].corr()


# In[ ]:


g = sns.FacetGrid(df_train, hue='BsmtBath', size = 5)
g.map(plt.hist, 'TotalBsmtSF', alpha= 0.5, bins=50)
g.add_legend()


# In[ ]:


g = sns.FacetGrid(df_train, hue='BsmtBath', size = 7)
g.map(plt.scatter, 'TotalBsmtSF', 'LogPrice', edgecolor="w")
g.add_legend()


# In[ ]:


pd.crosstab(df_train['BsmtQuGroup'], df_train['BsmtBath'])


# * Some kind of correlations between LogPrice and TotalBsmtSF, BsmtTotFinSF, BsmtBath, BsmtQual, MisBsmt
# * The unfinished SF should play a role in the cost (out of logic)
# * TotalBsmtSF and BsmtTotFinSF are mildly correlated, the first gives a stronger correlation with the target and the second is redundant if we include the unfinished SF (the three features are consistent). I will drop the TotFin because it is the more redundant
# * It seems that better basements have also a bathroom, which makes sense
# 
# ## Heating, Electricity, and Air conditioning
# 
# * **Categorical Features**: Utilities, Heating, HeatingQC, CentralAir, Electrical
# 
# ### Correlation with the target

# In[ ]:


var = "Utilities"
segm_target(var)


# In[ ]:


var = "Heating"
segm_target(var)


# In[ ]:


var = "HeatingQC"
segm_target(var)


# In[ ]:


df_train.loc[df_train.HeatingQC == 'Po', 'HeatQGroup'] = 1
df_train.loc[df_train.HeatingQC == 'Fa', 'HeatQGroup'] = 1
df_train.loc[df_train.HeatingQC == 'TA', 'HeatQGroup'] = 3
df_train.loc[df_train.HeatingQC == 'Gd', 'HeatQGroup'] = 4
df_train.loc[df_train.HeatingQC == 'Ex', 'HeatQGroup'] = 7

x = df_train['HeatQGroup']
y = df_train['LogPrice']

print(df_train[['HeatQGroup', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()


# In[ ]:


var = "CentralAir"
segm_target(var)


# In[ ]:


var = "Electrical"
segm_target(var)


# In[ ]:


df_train.loc[df_train.Electrical == "SBrkr", "ElecGroup"] = "SBrkr"
fuse = ['FuseA', 'FuseF', 'FuseP', 'Mix']
df_train.loc[df_train.Electrical.isin(fuse), "ElecGroup"] = "Fuse"

var = "ElecGroup"
segm_target(var)


# In[ ]:


g = sns.FacetGrid(df_train, hue='CentralAir', size = 5)
g.map(plt.hist, 'HeatQGroup', alpha= 0.5, bins=10)
g.add_legend()


# * Some correlation between LogPrice and HeatingQu, CentralAir, Electrical
# 
# ## Spaces and Rooms
# 
# * **Numerical Features**: 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd 
# * **Categorical Features**:KitchenQual, Functional

# In[ ]:


corr_target('1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'LogPrice')


# In[ ]:


# checking the sum of 1st and 2nd floor, not always it consistent with GrLivArea, thus I assume it is a 3rd floor
df_train['HasThirdFl'] = 0
df_train.loc[df_train.GrLivArea - df_train['1stFlrSF'] - df_train['2ndFlrSF'] > 0, 'HasThirdFl'] = 1

var = 'HasThirdFl'
segm_target(var)


# In[ ]:


x = df_train['FullBath']
x1 = df_train['HalfBath']
y = df_train['LogPrice']

print(df_train[['FullBath', 'HalfBath', 'LogPrice']].corr())

fig, ax =plt.subplots(2,2, figsize=(12,10))
sns.regplot(x=x, y=y, ax=ax[0][0])
sns.regplot(x=x, y=y, ax=ax[0][1], x_estimator = np.mean)
sns.regplot(x=x1, y=y, ax=ax[1][0])
sns.regplot(x=x1, y=y, ax=ax[1][1], x_estimator = np.mean)

fig.show()


# In[ ]:


# I bet the total number of bathrooms is more important than the segmentation per type
df_train['TotBath'] = df_train.FullBath + df_train.HalfBath

x = df_train['TotBath']
y = df_train['LogPrice']

print(df_train[['FullBath', 'HalfBath', 'TotBath', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)


# In[ ]:


# No house is actually without a bathroom
df_train[df_train.TotBath == 0]['BsmtBath']


# In[ ]:


x = df_train['BedroomAbvGr']
x1 = df_train['KitchenAbvGr']
x2 = df_train['TotRmsAbvGrd']
y = df_train['LogPrice']

print(df_train[['BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'LogPrice']].corr())

fig, ax =plt.subplots(3,2, figsize=(12,15))
sns.regplot(x=x, y=y, ax=ax[0][0])
sns.regplot(x=x, y=y, ax=ax[0][1], x_estimator = np.mean)
sns.regplot(x=x1, y=y, ax=ax[1][0])
sns.regplot(x=x1, y=y, ax=ax[1][1], x_estimator = np.mean)
sns.regplot(x=x2, y=y, ax=ax[2][0])
sns.regplot(x=x2, y=y, ax=ax[2][1], x_estimator = np.mean)

fig.show()


# In[ ]:


df_train['TotRooms+Bath'] = df_train.TotRmsAbvGrd + df_train.TotBath

x = df_train['TotRooms+Bath']
y = df_train['LogPrice']

print(df_train[['TotRooms+Bath', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)


# In[ ]:


var = "KitchenQual"
segm_target(var)


# In[ ]:


df_train.loc[df_train.KitchenQual == 'Fa', 'KitchQuGroup'] = 1
df_train.loc[df_train.KitchenQual == 'TA', 'KitchQuGroup'] = 4
df_train.loc[df_train.KitchenQual == 'Gd', 'KitchQuGroup'] = 10
df_train.loc[df_train.KitchenQual == 'Ex', 'KitchQuGroup'] = 21

x = df_train['KitchQuGroup']
y = df_train['LogPrice']

print(df_train[['KitchQuGroup','SalePrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)


# In[ ]:


var = "Functional"
segm_target(var)


# ### Correlation between features

# In[ ]:


x = df_train['TotRmsAbvGrd']
y = df_train['GrLivArea']

print(df_train[['TotRmsAbvGrd', 'GrLivArea']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
fig.show()


# In[ ]:


g = sns.FacetGrid(df_train, hue='TotRmsAbvGrd', size = 8)
g.map(plt.scatter, 'GrLivArea', 'LogPrice', edgecolor="w")
g.add_legend()


# In[ ]:


x = df_train['TotBath']
y = df_train['GrLivArea']

print(df_train[['TotBath', 'GrLivArea']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
fig.show()


# In[ ]:


g = sns.FacetGrid(df_train, hue='TotBath', size = 8)
g.map(plt.scatter, 'GrLivArea', 'LogPrice', edgecolor="w")
g.add_legend()


# In[ ]:


x = df_train['KitchQuGroup']
y = df_train['GrLivArea']

print(df_train[['KitchQuGroup', 'GrLivArea']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
fig.show()


# In[ ]:


g = sns.FacetGrid(df_train, hue='KitchQuGroup', size = 8)
g.map(plt.scatter, 'GrLivArea', 'LogPrice', edgecolor="w")
g.add_legend()


# * The features correlated with the target are GrLivArea, TotBath, TotRmsAbvGrd, and KitchenQual
# * GrLivArea and TotRmsAbvGrd are very correlated, one can be the "class" of the other
# 
# 
# ## Fireplaces and Garage
# 
# * **Numerical Features**: Fireplaces, GarageYrBlt, GarageCars, GarageArea
# * **Categorical Features**: FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PavedDrive
# 
# ### Correlation with target

# In[ ]:


corr_target('GarageCars', 'GarageArea', 'SalePrice')


# In[ ]:


x = df_train['GarageArea']
y = df_train['LogPrice']

sns.regplot(x=x, y=y)


# In[ ]:


x = df_train[df_train.GarageYrBlt > 1000]['GarageYrBlt'] #filter to avoid the one I filled in with neg values
y = df_train[df_train.GarageYrBlt > 1000]['LogPrice']

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()


# In[ ]:


x = df_train['Fireplaces']
y = df_train['LogPrice']

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()


# In[ ]:


df_train.loc[df_train.Fireplaces > 0, 'FireGroup'] = 1
df_train.loc[df_train.Fireplaces == 0, 'FireGroup'] = 0

var = "FireGroup"
segm_target(var)


# In[ ]:


var = "FireplaceQu"
segm_target(var)


# In[ ]:


df_train.loc[df_train.FireplaceQu == 'NoFire', 'FrpQuGroup'] = 0
df_train.loc[df_train.FireplaceQu == 'Po', 'FrpQuGroup'] = 0
df_train.loc[df_train.FireplaceQu == 'Fa', 'FrpQuGroup'] = 2
df_train.loc[df_train.FireplaceQu == 'TA', 'FrpQuGroup'] = 3
df_train.loc[df_train.FireplaceQu == 'Gd', 'FrpQuGroup'] = 4
df_train.loc[df_train.FireplaceQu == 'Ex', 'FrpQuGroup'] = 8

x = df_train['FrpQuGroup']
y = df_train['LogPrice']

print(df_train[['FrpQuGroup', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()


# In[ ]:


var = "GarageType"
segm_target(var)


# In[ ]:


incl = ['Attchd', 'Basment', 'BuiltIn']
escl = ['2Types', 'CarPort', 'Detchd']
df_train.loc[df_train.GarageType.isin(incl), 'GrgTypeGroup'] = 'Connected'
df_train.loc[df_train.GarageType.isin(escl), 'GrgTypeGroup'] = 'NonConnected'
df_train.loc[df_train.GarageType == 'NoGrg', 'GrgTypeGroup'] = 'NoGrg'

var = 'GrgTypeGroup'
segm_target(var)


# In[ ]:


var = "GarageFinish"
segm_target(var)


# In[ ]:


var = "GarageQual"
segm_target(var)


# In[ ]:


var = "GarageCond"
segm_target(var)


# In[ ]:


var = "PavedDrive"
segm_target(var)


# In[ ]:


df_train.loc[df_train.PavedDrive == 'N', 'PvdGroup'] = 0
df_train.loc[df_train.PavedDrive == 'P', 'PvdGroup'] = 1
df_train.loc[df_train.PavedDrive == 'Y', 'PvdGroup'] = 1

print(df_train[['PvdGroup', 'LogPrice']].corr())

var = "PvdGroup"
segm_target(var)


# In[ ]:


var = "MisGarage"
segm_target(var)


# ### Correlations between features

# In[ ]:


x = df_train['GarageCars']
y = df_train['GarageArea']

print(df_train[['GarageCars', 'GarageArea']].corr())

sns.regplot(x=x, y=y)


# In[ ]:


g = sns.FacetGrid(df_train, hue="GarageCars", size = 8)
g.map(plt.scatter, "GarageArea", "LogPrice", edgecolor="w")
g.add_legend()


# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(12, 5))

sns.boxplot('GarageType',"GarageArea",data=df_train, ax=ax[0])
sns.boxplot('GrgTypeGroup',"GarageArea",data=df_train, ax=ax[1])

fig.show()


# In[ ]:


pd.crosstab(df_train["GrgTypeGroup"],df_train['GarageFinish'])


# In[ ]:


g = sns.FacetGrid(df_train, hue="GarageFinish", size=7)
g.map(plt.hist, 'GarageArea', alpha= 0.3, bins=50)
g.add_legend()


# In[ ]:


g = sns.FacetGrid(df_train, hue="GarageFinish", size = 8)
g.map(plt.scatter, "GarageArea", "LogPrice", edgecolor="w")
g.add_legend()


# In[ ]:


pd.crosstab(df_train["PvdGroup"],df_train['GrgTypeGroup'])


# In[ ]:


pd.crosstab(df_train["PvdGroup"],df_train['GarageFinish'])


# * Fair correlation between GarageArea and Price
# * Correlation between presence of Fireplaces and Price
# * With the appropriate segmentation, it seems there is a linear relation between FrpQu and Price
# * Garage connected to the house tend to cost more than the nonconnected, both more than missing garage
# * It seems that a finished garage belongs to a more expensive house
# * Having a Paved drive leads to a higher price   
# * GarageCars and GarageArea are very correlated, although it is not one the category of the other, they seem very dependent from one another. The Price seems more dependent to the Area
# 
# 
# ## Decks, porch, fence
# 
# * **Numerical Features**: WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, 
# * **Categorical Features**: PoolQC, Fence
# 
# ### Correlation with target

# In[ ]:


corr_target('WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'LogPrice')


# In[ ]:


df_train['TotPorch'] = (df_train['WoodDeckSF'] + df_train['OpenPorchSF'] + df_train['EnclosedPorch'] + 
                       df_train['3SsnPorch'] + df_train['ScreenPorch'])

print(df_train[['TotPorch', 'LogPrice']].corr())

x = df_train['TotPorch']
y = df_train['LogPrice']

sns.regplot(x=x, y=y)


# In[ ]:


fil = ((df_train['TotPorch'] != df_train['WoodDeckSF']) &
      (df_train['TotPorch'] != df_train['OpenPorchSF']) &
      (df_train['TotPorch'] != df_train['EnclosedPorch']) &
      (df_train['TotPorch'] != df_train['3SsnPorch']) &
      (df_train['TotPorch'] != df_train['ScreenPorch']))

df_train[fil][['TotPorch', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].count()


# In[ ]:


var = "Fence"
segm_target(var)


# In[ ]:


df_train.loc[df_train.Fence == 'NoFence', 'FenceGroup'] = 'NoFence'
df_train.loc[(df_train.Fence == 'MnPrv') | (df_train.Fence == 'GdPrv'), 'FenceGroup'] = 'Prv'
df_train.loc[(df_train.Fence == 'MnWw') | (df_train.Fence == 'GdWo'), 'FenceGroup'] = 'Wo'

var = "FenceGroup"
segm_target(var)


# In[ ]:


var = "PoolQC"
segm_target(var)


# In[ ]:


g = sns.FacetGrid(df_train, hue="FenceGroup", size = 5)
g.map(plt.scatter, "TotPorch", "LogPrice", edgecolor="w")
g.add_legend()


# * Pool is not useful
# * The TotPorch feature looks the only promising
# * The type of fence seems to be a bit correlated with the price
# 
# There is not correlation between these features

# ## Sell and Miscellaneous
# 
# * **Numerical Features**: MiscVal, YrSold, MoSold
# * **Categorical Features**: MiscFeature, SaleType, SaleCondition

# In[ ]:


x = df_train['YrSold']
y = df_train['LogPrice']

print(df_train[['YrSold', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()


# In[ ]:


x = df_train['MoSold']
y = df_train['LogPrice']

print(df_train[['YrSold', 'MoSold', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()


# In[ ]:


# Checking if there is some seasonality in the air
df_train[['YrSold', 'MoSold', 'LogPrice']].groupby(['YrSold', 'MoSold']).mean().plot(figsize=(12,5))


# In[ ]:


df_train[['YrSold', 'MoSold', 'LogPrice']].groupby(['YrSold', 'MoSold']).median().plot(figsize=(12,5))


# In[ ]:


df_train['OldwhenSold'] = df_train.YrSold - df_train.YearBuilt

x = df_train['OldwhenSold']
y = df_train['LogPrice']

print(df_train[['OldwhenSold', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()


# In[ ]:


x = df_train['MiscVal']
y = df_train['LogPrice']

sns.regplot(x=x, y=y)


# In[ ]:


var = "MiscFeature"
segm_target(var)


# In[ ]:


var = "SaleType"
segm_target(var)


# In[ ]:


other = ['Oth', 'New']
cont = ['ConLw', 'ConLi', 'ConLD', 'Con', 'CWD', 'COD']
df_train.loc[df_train.SaleType.isin(other), 'SaleTyGroup'] = 'Other'
df_train.loc[df_train.SaleType.isin(cont), 'SaleTyGroup'] = 'Contract'
df_train.loc[df_train.SaleType == 'WD', 'SaleTyGroup'] = 'WD'

var = "SaleTyGroup"
segm_target(var)


# In[ ]:


var = "SaleCondition"
segm_target(var)


# * The only partially relevant one seems to be SaleType
# * Miscellaneous features seems to be a horrible feature to predict a price, but I can try to see it as something I omit from my machine learning model (because it would be distracting) and it get's added later
# 
# # Conclusion
# 
# Here is a list of feature I want to keep for my models. 

# In[ ]:


feat = ['MSZoningGroup', 'AlleyGroup', 'LotShapeGroup', 'Condition1Group',
       'YearBuilt', 'BldTypeGroup', 'HStyleGroup', 'OverallQual', 
       'MasVnrArea', 'MasTypeGroup', 'FoundGroup', 'ExtQuGroup', 
       'TotalBsmtSF',' BsmtUnfSF', 'BsmtBath', 'BsmtQuGroup', 'MisBsm',
       'HeatQGroup', 'CentralAir', 'ElecGroup',
       'GrLivArea', 'KitchQuGroup',
       'GarageArea', 'FireGroup', 'FrpQuGroup', 'GrgTypeGroup', 'GarageFinish', 'PvdGroup', 'MisGarage',
       'TotPorch', 'FenceGroup', 'SaleTyGroup']


# Done in this way, the exploration is very much time consuming but it gave me a few ideas and hints otherwise difficult to guess. 
# 
# Moreover, I enjoyed practicing in something that I usually don't do in detail here on Kaggle.
# 
# I hope to receive 

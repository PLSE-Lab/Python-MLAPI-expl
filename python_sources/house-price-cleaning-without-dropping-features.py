#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Having just started my adventure with this introductory competition, I tried to not look around to other kernels too much in order to keep my mind unbiased from other people's approaches and see how far I could get. Nevertheless, some popular kernels are so nice to read that it is impossible not to.
# 
# The one thing I have noticed so far is the general attitude of handling missing values like mistakes that can only distract your model and thus drop them. In this kernel, I want to do the heavy lifting and clean the dataset following these principles
# 
# * If a missing entry is documented, fill it
# * If a missing entry is obvious, fill it
# * If the remaining missing entries are not numerically relevant, fill them (not my favourite thing to do, I admit)
# * If a feature is clearly insignificant (for example, it has always the same value except for a limited amount of entries), drop it.
# * Create the features that turn potentially irrelevant information into relevant (this sounds cooler than it is, you will see)
# 
# The exploration and cleaning are not the most efficient time-wise but my ultimate goal is to verify that **better data beat better algorithms**. This is thus the first of 3 kernels that will hopefully help me understand better how to face such regression problems.
# 
# The other hope is to inspire some of you in giving me feedbacks or, even better, in stop dropping columns just because there is something weird in it.
# 
# The obvious problems of such approach are:
# 
# * I am still making choices about some features and there is always a way of being more scientific about it
# * It is very time-consuming (not computationally), thus it is probably less doable in a situation where the features are more than 80
# * It is done without looking at the final goal too much, which was done on purpose to be as unbiased as I can but there is always and healthy middle ground
# 
# Please let me know what else I am missing and enjoy.

# In[ ]:


# standard
import pandas as pd
import numpy as np

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# ---------- DF IMPORT -------------
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
combine = [df_train, df_test]
df_train.name = 'Train'
df_test.name = 'Test'


# # Exploratory Analysis
# 
# The goal of this part would be to just have a taste of what is in there.
# 
# ## Missing data

# In[ ]:


df_train.sample(10)


# In[ ]:


df_train.columns


# In[ ]:


for df in combine:
    if df.name == 'Train':
        mis_train = []
        cols = df.columns
        for col in cols:
            mis = df[col].isnull().sum()
            if mis > 0:
                print("{}: {} missing, {}%".format(col, mis, round(mis/df.shape[0] * 100, 3)))
                mis_train.append(col)
        print("_"*40)
        print("_"*40)
    if df.name == 'Test':
        mis_test = []
        cols = df.columns
        for col in cols:
            mis = df[col].isnull().sum()
            if mis > 0:
                print("{}: {} missing, {}%".format(col, mis, round(mis/df.shape[0] * 100, 3)))
                mis_test.append(col)

print("\n")
print(mis_train)
print("_"*40)
print(mis_test)


# * Alley is missing very often, but it just means that there is no alley, not that we have a missing entry
# * The basement features are missing 2% in test and 3% in train, but again it should mean that there is no basement (although the numbers do not add up completely, so I have to be more careful)
# * FireplaceQu is missing 47% in train and 50% in test, again documented
# * The garage features are missing 5% of the times, again documented (but numbers do not add up)
# * PoolQC is missing almost always, documented but probably irrelevant
# * Fence is missing always 80%, documented but probably irrelevant
# * MiscFeature is missing almost always
# 
# Given the variety of the missing values, I will have to pay extra attention during the data cleaning phase.
# 
# I now want to look at distributions. Thus, I will focus "room by room".
# 
# ## Zone, Street and access
# 
# * **Numeric Features**: LotFrontage, LotArea
# * **Categorical Features**: MSZoning, Street, Alley, LotShape, LandContour, LotConfig, LandSlope, Neighborhood, Condition1, Condition2

# In[ ]:


#helper functions to avoid repeating myself too much

def traintest_hist(feat, nbins):
    fig, axes = plt.subplots(1, 2)

    df_train[feat].hist(bins = nbins, ax=axes[0])
    df_test[feat].hist(bins = nbins, ax=axes[1])

    print("{}: {} missing, {}%".format('Train', df_train[feat].isnull().sum(), 
                                   round(df_train[feat].isnull().sum()/df_train.shape[0] * 100, 3)))
    print("{}: {} missing, {}%".format('Test', df_test[feat].isnull().sum(), 
                                   round(df_test[feat].isnull().sum()/df_test.shape[0] * 100, 3)))
    
def categ_dist_mis(cats):
    for cat in cats:
        print(cat)
        print("_"*40)
        print("In train: ")
        print(df_train[cat].value_counts(dropna = False))
        print('_'*40)
        print("In test: ")
        print(df_test[cat].value_counts(dropna = False))
        print("_"*40)
        print("_"*40)
        
def checkclean(feats):
    for feat in feats:
        print(feat)
        for df in combine:
            print("In {}".format(df.name))
            print(df[feat].value_counts(dropna = False))
            print("_"*40)
    print("_"*40)


# In[ ]:


traintest_hist('LotFrontage', 50)


# In[ ]:


traintest_hist('LotArea', 50)


# * Different distributions in train and test
# * More outliers in the train LotArea
# * Not sure if they are outliers yet
# * Missing values to verify
# 
# As of the categorical features

# In[ ]:


cats = ['MSZoning', 'Street', 'Alley', 'LotShape', 
       'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2']

categ_dist_mis(cats)


# * **MSZoning**: FV, RH and C are sparse classes, have to reassign them. Missing 4 times in test.
# * **Street**: ininfluential, I can drop it
# * **Alley**: a lot of missing, but it is documented, to evaluate if change it into a binary feature
# * **LotShape**: IR2 and IR3 can be grouped
# * **LandContour**: can be changed into binary
# * **LotConfig**: FR2 and FR3 can be grouped
# * **LandSlope**: Sev is a sparse class
# * **Neighborhood**: a lot of classes, I have to think of a better classification
# * **Condition1**: overwhelmingly normal, can be changed into binary
# * **Condition2**: ininfluencial, I can drop it
# 
# I will do the following:
# 
# * Drop **Street** and **Condition2** (later)
# * **MSZoning**: I will just change them into RL, RM, and Other
# * **Neighborhood**: don't know what to do with it, it should be important tho
# * **Condition1**: Will change in Normal/NotNormal
# * **LotFrontage**: Fill the missing with 0
# * **LotArea**: good for now.
# * **Alley**: Change to Alley/NotAlley
# * **LotShape**: Change into Regular/Irregular
# * **LotConfig**: group FR2 and FR3 into one category
# * **LanSlope**: change to Gentle/NotGentle
# * **LandContour**: change to Lvd/NotLvd

# In[ ]:


for df in combine:
    df.loc[(df.MSZoning == 'FV') | (df.MSZoning == 'RH') | (df.MSZoning == 'C (all)'), 'MSZoning'] = 3 #other
    df.loc[df.MSZoning == 'RL', 'MSZoning'] = 1
    df.loc[df.MSZoning == 'RM', 'MSZoning'] = 2
    #Condition1
    df.loc[(df.Condition1 != 'Norm'), 'Condition1'] = 0 # NotNormal
    df.loc[df.Condition1 == 'Norm', 'Condition1'] = 1
    #LotFrontage
    df.loc[df.LotFrontage.isnull(), 'LotFrontage'] = 0
    #Alley
    df.loc[df.Alley.notnull(), 'Alley'] = 1 # Alley
    df.loc[df.Alley.isnull(), 'Alley'] = 0 #NotAlley
    #LotShape
    df.loc[(df.LotShape != 'Reg'), 'LotShape'] = 0 # NotRegular
    df.loc[df.LotShape == 'Reg', 'LotShape'] = 1
    #LotConfig
    df.loc[df.LotConfig == 'Inside', 'LotConfig'] = 1
    df.loc[df.LotConfig == 'Corner', 'LotConfig'] = 2
    df.loc[df.LotConfig == 'CulDSac', 'LotConfig'] = 3
    df.loc[(df.LotConfig == 'FR2') | (df.LotConfig == 'FR3'), 'LotConfig'] = 4
    #LandSlope
    df.loc[(df.LandSlope != 'Gtl'), 'LandSlope'] = 0 # NotGentle
    df.loc[df.LandSlope == 'Gtl', 'LandSlope'] = 1
    #LandContour
    df.loc[(df.LandContour != 'Lvl'), 'LandContour'] = 0 # NotLeveled
    df.loc[df.LandContour == 'Lvl', 'LandContour'] = 1      
    
print("LotFrontage")    
print("In train: ")
print(df_train.LotFrontage.isnull().sum())
print('_'*40)
print("In test: ")
print(df_test.LotFrontage.isnull().sum())
print('_'*40)
print('_'*40)

feats = ["MSZoning", "Condition1", "Alley", "LotShape", "LotShape", "LotConfig", "LandSlope", "LandContour"]
    
checkclean(feats)


# ## Type, Quality, and Condition
# 
# * **Numerical features**: OverallQual, OverallCond, YearBuilt, YearRemodAdd
# * **Categorical features**: MSSubClass, BldgType, HouseStyle

# In[ ]:


traintest_hist('OverallQual', 10)


# In[ ]:


traintest_hist('OverallCond', 10)


# In[ ]:


traintest_hist('YearBuilt', 20)


# In[ ]:


traintest_hist('YearRemodAdd', 20)


# * Similar distributions for OverallCond in train and test
# * Small differences for OveralQual
# * The YearBuilt and YearRemodAdd features are distributed differently, but I can group by decade
# 
# Moving to the categorical variables

# In[ ]:


cats = ['MSSubClass', 'BldgType', 'HouseStyle']

categ_dist_mis(cats)


# * **MSSubClass**: a lot of classes that I don't know the meaning of
# * **BldgType**: some sparse classes, I have to think how to group it
# * **HouseStyle**: sparse classes to group
# 
# 
# I will do the following:
# 
# * **MSSubClass**: don't know what to do with it
# * **BldType**: change to 1Fam, 2Fam+Duplex, TownHouse
# * **YearBuilt**: good for now
# * **YearRemodAdd**: good for now
# * **HouseStyle**: 1Story, 2Story+all the half story, Split
# * **OverallQual** and **OverallCond**: good for now

# In[ ]:


for df in combine:
    df.loc[df.BldgType == '1Fam', 'BldgType'] = 1
    df.loc[(df.BldgType == '2fmCon') | (df.BldgType == 'Duplex'), 'BldgType'] = 2
    df.loc[df.BldgType== 'CulDSac', 'BldgType'] = 3
    df.loc[(df.BldgType == 'TwnhsE') | (df.BldgType == 'Twnhs'), 'BldgType'] = 4
    #HouseStyle
    df.loc[df.HouseStyle == '1Story', 'HouseStyle'] = 1
    df.loc[(df.HouseStyle == 'SFoyer') | (df.HouseStyle == 'SLvl'), 'HouseStyle'] = 2
    sto = ['1.5Fin', '1.5Unf', '2.5Fin', '2.5Unf', '2Story']
    df.loc[df.HouseStyle.isin(sto), 'HouseStyle'] = 3
    
feats = ["BldgType", "HouseStyle"]
    
checkclean(feats)


# ## Exterior and materials
# 
# * **Numerical Features**: MasVnrArea 
# * **Categorical Features**: Foundation, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, ExterQual, ExterCond

# In[ ]:


traintest_hist('MasVnrArea', 20)


# Simila distributions, but different ranges. Mostly 0.
# 
# Now the categories

# In[ ]:


cats = ['Foundation','RoofStyle', 'RoofMatl', 'Exterior1st', 
        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond']

categ_dist_mis(cats)


# * **Foundation**: Stone, Wood, and Slab are sparse
# * **RoofStyle**: sparse classes to group (but they all look the same to me)
# * **RoofMatl**: Ininfluencial, I can drop it
# * **Exterior1st**: sparse classes to group
# * **Exterior2nd**: same as 1st, 1 missing value in test
# * **MasVnrType**: 8 missing values in train, 16 in test, sparse classes. Train is consistent with numerical feature, test has an extra missing value.
# * **ExterQual**: the extremes are sparse classes
# * **ExterCond**: sparse classes
# 
# I will do the following
# 
# * Drop **RoofMatl** and **RoofStyle** (later)
# * **Foundation**: don't see how can it play a role, but it is my ignorance talking
# * **MasVnrArea**: I will change it into a binary variable since it is almost always 0
# * **MasVnrType**: Group the brick features, has to be consistent with the previous feature
# * **Exterior1st** and **Exterior2nd**: will group by material, considering of keeping a generic "Other" category
# * **ExterQual**: I will group Excellent and Good, TA and Fa
# * **ExterCond**: G+E, Ta+Fa+Po

# In[ ]:


for df in combine:
    #MasVnrArea
    df.loc[(df.MasVnrArea == 0), 'MasVnr'] = 0 
    df.loc[df.MasVnrArea > 0, 'MasVnr'] = 1
    #MasVnrType
    df.loc[(df.MasVnrType == 'None'), 'MasVnrType'] = 0 
    df.loc[(df.MasVnrType == 'BrkFace') |(df.MasVnrType == 'BrkCmn') , 'MasVnrType'] = 1 #bricks
    df.loc[(df.MasVnrType == 'Stone'), 'MasVnrType'] = 2
    #ExterQual
    df.loc[(df.ExterQual == 'TA') |(df.ExterQual == 'Fa') , 'ExterQual'] = 0 #Typical or Fair
    df.loc[(df.ExterQual == 'Gd') |(df.ExterQual == 'Ex') , 'ExterQual'] = 1 #Good or Excellent
    #ExterCond
    df.loc[(df.ExterCond == 'TA') |(df.ExterCond == 'Fa') | 
           (df.ExterCond == 'Po'), 'ExterCond'] = 0 #Typical or Fair or Poor
    df.loc[(df.ExterCond == 'Gd') |(df.ExterCond == 'Ex') , 'ExterCond'] = 1 #Good or Excellent
    ###
    feat = 'Exterior1st'
    ##
    df.loc[(df[feat] == 'VinylSd'), feat] = 0 
    df.loc[(df[feat] == 'Stucco') |(df[feat] == 'ImStucc') , feat] = 1 #Stucco or similar
    df.loc[(df[feat] == 'Wd Sdng') |(df[feat] == 'WdShing') | (df[feat] == 'Plywood'), feat] = 2 #woods
    df.loc[(df[feat] == 'MetalSd'), feat] = 3
    df.loc[(df[feat] == 'HdBoard'), feat] = 4
    df.loc[(df[feat] == 'BrkFace') |(df[feat] == 'BrkComm') , feat] = 5 #Bricks
    df.loc[(df[feat] == 'CemntBd') |(df[feat] == 'AsbShng') | 
           (df[feat] == 'AsphShn') | (df[feat] == 'CBlock') | (df[feat] == 'Stone'), feat] = 1 #Cement and similar
    # avoided sparse class
    feat = 'Exterior2nd'
    ##
    df.loc[(df[feat] == 'VinylSd'), feat] = 0 
    df.loc[(df[feat] == 'Stucco') |(df[feat] == 'ImStucc') , feat] = 1 #Stucco or similar (actually just Other)
    df.loc[(df[feat] == 'Wd Sdng') |(df[feat] == 'Wd Shng') | (df[feat] == 'Plywood'), feat] = 2 #woods
    df.loc[(df[feat] == 'MetalSd'), feat] = 3
    df.loc[(df[feat] == 'HdBoard'), feat] = 4
    df.loc[(df[feat] == 'BrkFace') |(df[feat] == 'Brk Cmn') , feat] = 1 #Bricks (actually just Other)
    df.loc[(df[feat] == 'CmentBd') |(df[feat] == 'AsbShng') | 
           (df[feat] == 'AsphShn') | (df[feat] == 'CBlock') | 
           (df[feat] == 'Stone') |(df[feat] == 'Other'), feat] = 1 #Cement and similar (actually just Other)
    # avoided sparse class
    
feats = ['MasVnr', 'MasVnrType', 'ExterQual', 'ExterCond', 'Exterior1st', 'Exterior2nd']

checkclean(feats)


# ## Basement
# 
# * **Numerical Features**: BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath
# * **Categorical Features**: BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2

# In[ ]:


traintest_hist('BsmtFinSF1', 30)


# In[ ]:


traintest_hist('BsmtFinSF2', 30)


# In[ ]:


traintest_hist('BsmtUnfSF', 30)


# In[ ]:


traintest_hist('TotalBsmtSF', 30)


# In[ ]:


traintest_hist('BsmtFullBath', 5)


# In[ ]:


traintest_hist('BsmtHalfBath', 5)


# * Not every house has a basement, investigate the 0's in all the features is essential
# * The BsmtFinSF2 looks useless, but I can add it to the FinSF1
# * The BsmtUnfSF might have importance to determine the price (just based on its definition tho)
# * The two features about the bathrooms might just turn into boolean
# 
# 1 missing value in the test (2 when we talk about its bathroom)
# 
# Now the categorical ones

# In[ ]:


cats = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

categ_dist_mis(cats)


# * **BsmtQual**: missing values for No Basement
# * **BsmtCond**: missing values, mismatch with the missing values of the previous category
# * **BsmtExposure**: missing values, again mismatch
# * **BsmtFinType1**: again mismatch, in this case it can be because there is nothing finished
# * **BsmtFinType2**: sparse classes and missing values mismatch
# 
# The numerical variables missing are really missing.
# 
# I will do this:
# 
# * **BsmFinSF1** and **BsmFinSF2**: I will sum them up
# * **BsmtFullBath** and **BsmtHalfBath**: I will combine them and transform to a boolean
# * **BsmtQual**, **BsmtCond**, **BsmtExposure**, **BsmtFinType1**, and **BsmtFinType2** have all missing values that in theory are documented. However, there is a mismatch across these features. I will create the class "missing" only if the value are missing in all of them.
# * **BsmtQual**: group Fa with TA to avoid sparse class
# * **BsmCond**: group TA, Fa, and Po
# * **BsmtExposure**: good for now
# * **BsmtFinType1**: good for now
# * **BsmtFinType2**: I will transform it into Finished/Unfinished

# In[ ]:


for df in combine:
    df['BsmFinSFTot'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
    #BsmtFullBath and BsmtHalfBath
    df['BsmBath'] = df['BsmtFullBath'] + df['BsmtHalfBath']
    df.loc[(df.BsmBath == 0), 'BsmBath'] = 0 #noBath
    df.loc[(df.BsmBath > 0), 'BsmBath'] = 1
    #MissingBasement
    fil = ((df.BsmtQual.isnull()) & (df.BsmtCond.isnull()) & (df.BsmtExposure.isnull()) &
          (df.BsmtFinType1.isnull()) & (df.BsmtFinType2.isnull()))
    df['MisBsm'] = 0
    df.loc[fil, 'MisBsm'] = 1
    #BsmtQual
    df.loc[fil, 'BsmtQual'] = -99 #missing basement
    df.loc[(df.BsmtQual == 'Fa') | (df.BsmtQual == 'TA'), 'BsmtQual'] = 1 #fair or typical
    df.loc[(df.BsmtQual == 'Gd'), 'BsmtQual'] = 2
    df.loc[(df.BsmtQual == 'Ex'), 'BsmtQual'] = 3
    #BsmtCond
    df.loc[fil, 'BsmtCond'] = -99 #missing basement
    df.loc[(df.BsmtCond == 'Fa') | (df.BsmtCond == 'TA') |
           (df.BsmtCond == 'Po'), 'BsmtCond'] = 1 #fair or typical or poor
    df.loc[(df.BsmtCond == 'Gd'), 'BsmtCond'] = 2
    df.loc[(df.BsmtCond == 'Ex'), 'BsmtCond'] = 3
    #BsmtExposure
    df.loc[fil, 'BsmtExposure'] = -99 #missing basement
    df.loc[df.BsmtExposure == 'Gd', 'BsmtExposure'] = 1
    df.loc[df.BsmtExposure == 'Av', 'BsmtExposure'] = 2
    df.loc[df.BsmtExposure == 'Mn', 'BsmtExposure'] = 3
    df.loc[df.BsmtExposure == 'No', 'BsmtExposure'] = 4
    #BsmtFinType1
    df.loc[fil, 'BsmtFinType1'] = -99 #missing basement
    df.loc[(df.BsmtFinType1 == 'Unf') , 'BsmtFinType1'] = 1
    df.loc[(df.BsmtFinType1 == 'GLQ') , 'BsmtFinType1'] = 2
    df.loc[(df.BsmtFinType1 == 'ALQ') , 'BsmtFinType1'] = 3
    df.loc[(df.BsmtFinType1 == 'BLQ') , 'BsmtFinType1'] = 4
    df.loc[(df.BsmtFinType1 == 'Rec') , 'BsmtFinType1'] = 5
    df.loc[(df.BsmtFinType1 == 'LwQ') , 'BsmtFinType1'] = 6
    #BsmtFinType2
    df.loc[fil, 'BsmtFinType2'] = -99 #missing basement
    df.loc[(df.BsmtFinType2 != 'Unf') & (df.BsmtFinType2 != -99), 'BsmtFinType2'] = 0 #notUnf
    df.loc[(df.BsmtFinType2 == 'Unf') , 'BsmtFinType2'] = 1
    
feats = ['BsmBath', 'MisBsm', 'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

checkclean(feats)

traintest_hist('BsmFinSFTot', 30)


# In[ ]:


# Checking consistency with numerical variable
df_train[df_train.BsmFinSFTot== 0]['MisBsm'].value_counts(dropna=False)


# In[ ]:


# Checking consistency between numerical variables
fil = (df_train.BsmFinSFTot + df_train.BsmtUnfSF != df_train.TotalBsmtSF)
df[fil].shape


# ## Heating, Electricity, and Air conditioning
# 
# * **Categorical Features**: Utilities, Heating, HeatingQC, CentralAir, Electrical

# In[ ]:


cats = ['Utilities', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical']

categ_dist_mis(cats)


# * **Utilities**: irrelevant, I can drop it
# * **Heating**: ininfluencial, I can drop it
# * **HeatingQC**: sparse classes to group
# * **CentralAir**: good
# * **Electrical**: 1 missing, sparse classes for Fuses and Mix
# 
# I will do the following
# 
# * Drop **Utilities** and **Heating** (later)
# * **HeatingQC**: aggregate Fa and Po
# * **CentralAir**: good for now
# * **Electrical**: group the fuses and mix

# In[ ]:


for df in combine:
    df.loc[df.HeatingQC == 'Ex', 'HeatingQC'] = 1
    df.loc[df.HeatingQC == 'Gd', 'HeatingQC'] = 2
    df.loc[df.HeatingQC == 'TA', 'HeatingQC'] = 3
    df.loc[(df.HeatingQC == 'Fa') | (df.HeatingQC == 'Po'), 'HeatingQC'] = 4 #Fair or poor
    #CentralAir
    df.loc[df.CentralAir == 'N', 'CentralAir'] = 0
    df.loc[df.CentralAir == 'Y', 'CentralAir'] = 1
    #Electrical
    fil = ((df.Electrical == 'FuseA') | (df.Electrical == 'FuseF') |
          (df.Electrical == 'FuseP') | (df.Electrical == 'Mix'))
    df.loc[fil, 'Electrical'] = 0
    df.loc[df.Electrical == 'SBrkr', 'Electrical'] = 1
    
feats = ['HeatingQC', 'CentralAir', 'Electrical']

checkclean(feats)


# ## Spaces and Rooms
# 
# * **Numerical Features**: 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd 
# * **Categorical Features**:KitchenQual, Functional

# In[ ]:


traintest_hist('1stFlrSF', 30)


# In[ ]:


traintest_hist('2ndFlrSF', 30)


# In[ ]:


traintest_hist('LowQualFinSF', 30)


# In[ ]:


traintest_hist('GrLivArea', 30)


# In[ ]:


traintest_hist('FullBath', 10)


# In[ ]:


traintest_hist('HalfBath', 10)


# In[ ]:


traintest_hist('BedroomAbvGr', 10)


# In[ ]:


traintest_hist('KitchenAbvGr', 10)


# In[ ]:


traintest_hist('TotRmsAbvGrd', 20)


# * **1stFlrSF**: well distributed, a couple of outliers
# * **2stFlrSF**: mostly 0, considering turning it into a binary feature
# * **LowQualFinSF**: ininfluential, I can drop it
# * **FullBath**: consider turning it into a binary feature
# * **HalfBath**: consider turning it into a binary feature
# * **BedroomAbvGr**: differently distributed
# * **KitchenAbvGr**: Moslty 1, consider turning it into a binary feature
# * **TotRmsAbvGrd**: similar distribution, some values seems too high
# 
# Now the categorical features

# In[ ]:


cats = ['KitchenQual', 'Functional']

categ_dist_mis(cats)


# * **KitchenQual**: 1 missing value in test
# * **Functional**: looks fairly ininfluential, if not: sparse classes and 2 missing values in test
# 
# I will do this
# 
# * **LowQualFinSF** will be dropped later
# * **1stFlrSF**: good for now
# * **2stFlrSF**: turn into with2floor/without2floor
# * **FullBath** and **HalfBath**: combine them and see how it goes
# * **KitchenAbvGr**: good for now
# * **BedroomAbvGr**: good for now
# * **KitchenQual**: group Fa and TA
# * **Functional**: turn into binary

# In[ ]:


for df in combine:
    df['2ndFlr'] = 0 #no2floor
    df.loc[df['2ndFlrSF'] > 0, '2ndFlr'] = 1 #with2floor
    #Baths
    df['BathsTot'] = df.FullBath + df.HalfBath 
    #KitchenQual
    df.loc[df.KitchenQual == 'Ex', 'KitchenQual'] = 1
    df.loc[df.KitchenQual == 'Gd', 'KitchenQual'] = 2
    df.loc[(df.KitchenQual == 'TA') | (df.KitchenQual == 'Fa'), 'KitchenQual'] = 3
    #Functional
    fun = ['Min2', 'Min1', 'Mod', 'Maj1', 'Sev', 'Maj2']
    fil = df.Functional.isin(fun)
    df.loc[fil, 'Functional'] = 0 #nontypical
    df.loc[df.Functional == 'Typ', 'Functional'] = 1
    
feats = ['2ndFlr', 'BathsTot', 'KitchenQual',  'Functional']

checkclean(feats)


# In[ ]:


# Checking consistency between numerical variables
fil = (df_train['1stFlrSF'] + df_train['2ndFlrSF'] != df_train.GrLivArea)
df[fil].shape


# In[ ]:


# Checking consistency between categorical variables
fil = (df_train['BedroomAbvGr'] + df_train['KitchenAbvGr'] != df_train.TotRmsAbvGrd)
df[fil].shape


# The first might indicate houses with 3 floors. The second tells me that living rooms exist.
# 
# ## Fireplaces and Garage
# 
# * **Numerical Features**: Fireplaces, GarageYrBlt, GarageCars, GarageArea
# * **Categorical Features**: FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PavedDrive

# In[ ]:


traintest_hist('Fireplaces', 10)


# In[ ]:


traintest_hist('GarageYrBlt', 30)


# In[ ]:


traintest_hist('GarageCars', 10)


# In[ ]:


traintest_hist('GarageArea', 30)


# * **Fireplaces**: consider turning it into binary
# * **GarageYrBlt**: missing values (maybe missing garage), one clear mistake in test
# * **GarageArea**: 1 missing value in test, similar distributions
# * **GarageCars**: 1 missing value in test, more a categorical version of the area
# 
# Now the categorical variables

# In[ ]:


cats = ['FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']

categ_dist_mis(cats)


# * **FireplaceQu**: missing values (missing fireplaces), sparse classes
# * **GarageType**: missing values in train are consistent, not in test (difference of 2), sparse classes
# * **GarageFinish**: missing values, but consistent with missing garage in numerical features
# * **GarageQual**: ok with the missing values, the rest looks pretty irrelevant
# * **GarageCond**: looks irrelevant
# * **PavedDrive**: probably better with binary values
# 
# I will do the following
# 
# * **GarageQual** and **GarageCond** will be dropped later
# * **Fireplaces**: turn into binary
# * **FireplaceQu**: Group Fa and Po, Gd and Ex, fill the missing (if consistent) with a Missing class
# * **GarageYrBlt**, **GarageType**, **GarageFinish**, **GarageQual**, and **GarageCond** has to have consistent missing values
# * **GarageYrBlt**: correct the mistake (2207) in test with 2007
# * **GarageArea**: fine for now
# * **GarageCars**: fine for now
# * **GarageType**: group Attchd, BuiltIn, Basment, 2Types  and Detchd, CarPort
# * **GarageFinish**: fine for now
# * **PavedDrive**: binary Y+P/N

# In[ ]:


for df in combine:
    df.loc[df['Fireplaces'] > 0, 'Fireplaces'] = 1 #Fireplaces
    df.loc[df['Fireplaces'] == 0, 'Fireplaces'] = 0
    #FireplaceQu
    df.loc[(df.Fireplaces == 0) & (df.FireplaceQu.isnull()), 'FireplaceQu'] = -99 #missing
    df.loc[(df.FireplaceQu == 'Ex') | (df.FireplaceQu== 'Gd'), 'FireplaceQu'] = 1
    df.loc[(df.FireplaceQu == 'TA'), 'FireplaceQu'] = 2
    df.loc[(df.FireplaceQu == 'Fa') | (df.FireplaceQu== 'Po'), 'FireplaceQu'] = 3
    #MisGarage
    fil = ((df.GarageYrBlt.isnull()) & (df.GarageType.isnull()) & (df.GarageFinish.isnull()) &
          (df.GarageQual.isnull()) & (df.GarageCond.isnull()))
    df['MisGarage'] = 0
    df.loc[fil, 'MisGarage'] = 1
    #GarageYrBlt
    df.loc[df.GarageYrBlt > 2200, 'GarageYrBlt'] = 2007 #correct mistake
    df.loc[fil, 'GarageYrBlt'] = -99
    #GarageType
    gty = ['Attchd', 'BuiltIn', 'Basment', '2Types']
    fil1 = df.GarageType.isin(gty)
    fil2 = ((df.GarageType == 'Detchd') | (df.GarageType == 'CarPort'))
    df.loc[fil, 'GarageType'] = -99 #missing garage
    df.loc[fil1, 'GarageType'] = 0 #attached and similar
    df.loc[fil2, 'GarageType'] = 1
    #GarageFinish
    df.loc[fil, 'GarageFinish'] = -99 #missing
    df.loc[df.GarageFinish == 'Unf', 'GarageFinish'] = 1
    df.loc[df.GarageFinish == 'RFn', 'GarageFinish'] = 2
    df.loc[df.GarageFinish == 'Fin', 'GarageFinish'] = 3
    #PavedDrive
    df.loc[df.PavedDrive == 'Y', 'PavedDrive'] = 1
    df.loc[df.PavedDrive == 'P', 'PavedDrive'] = 1
    df.loc[df.PavedDrive == 'N', 'PavedDrive'] = 0
    
    
feats = ['Fireplaces', 'FireplaceQu', 'MisGarage', 'GarageType', 'GarageFinish', 'PavedDrive']

checkclean(feats)

traintest_hist('GarageYrBlt', 30)


# ## Decks, porch, fence
# 
# * **Numerical Features**: WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, 
# * **Categorical Features**: PoolQC, Fence

# In[ ]:


traintest_hist('WoodDeckSF', 30)


# In[ ]:


traintest_hist('OpenPorchSF', 30)


# In[ ]:


traintest_hist('EnclosedPorch', 30)


# In[ ]:


traintest_hist('3SsnPorch', 30)


# In[ ]:


traintest_hist('ScreenPorch', 30)


# In[ ]:


traintest_hist('PoolArea', 30)


# * **WoodDeckSF**: mostly 0, maybe binary is better
# * **Porch features**: probably it is a good thing of combining them or turn them into binary
# * **PoolArea**: probably irrelevant
# 
# Now the categorical variables

# In[ ]:


cats = ['PoolQC', 'Fence']

categ_dist_mis(cats)


# * **PoolQC**: irrelevant, I can drop it
# * **Fence**: probably better as binary
# 
# I will do the following
# 
# * Drop **PoolQC** and **PoolArea** later
# * Combine **ScreenPorch**, **3SsnPorch**, **EnclosedPorch**, **OpenPorchSF**, and **WoodDeckSF** in a generic Porch area feature
# * **Fence**: a missing class, one for Prv's and one for W's

# In[ ]:


for df in combine:
    df['Porch'] = df.ScreenPorch + df['3SsnPorch'] + df.EnclosedPorch + df.OpenPorchSF + df.WoodDeckSF
    #Fence
    df.loc[df.Fence.isnull(), 'Fence'] = 0 #missing fence
    df.loc[(df.Fence == 'MnPrv') | (df.Fence == 'GdPrv'), 'Fence'] = 1 #privacy
    df.loc[(df.Fence == 'GdWo') | (df.Fence == 'MnWw'), 'Fence'] = 2 #woods
    
feats = ['Fence']

checkclean(feats) 

traintest_hist('Porch', 30)


# ## Sell and Miscellaneous
# 
# * **Numerical Features**: MiscVal, YrSold, MoSold
# * **Categorical Features**: MiscFeature, SaleType, SaleCondition

# In[ ]:


traintest_hist('MiscVal', 30)


# In[ ]:


traintest_hist('YrSold', 10)


# In[ ]:


traintest_hist('MoSold', 12)


# * **MiscVal**: I would say drop it
# * **YrSold**: looks ok
# * **MoSold**: looks ok, not sure if needed but I should check for seasonality
# 
# Now the categories

# In[ ]:


cats = ['MiscFeature', 'SaleType', 'SaleCondition']

categ_dist_mis(cats)


# * **MiscFeature**: ininfluencial, drop it
# * **SaleType**: sparse classes, probably ininfluential
# * **SaleCondition**: sparse classes, probably ininfluential
# 
# I will do the following
# 
# * Drop **MiscFeature** and **MiscVal** later
# * **SaleType**: group Warranties, Contracts, and New+other
# * **SaleCondition**: Turn into Normal/NotNormal

# In[ ]:


for df in combine:
    war = ['WD', 'CWD', 'VWD']
    con = ['Con', 'ConLw', 'ConLI', 'ConLD', 'COD']
    oth = ['New', 'Oth']
    df.loc[df.SaleType.isin(war), 'SaleType'] = 1
    df.loc[df.SaleType.isin(con), 'SaleType'] = 2
    df.loc[df.SaleType.isin(oth), 'SaleType'] = 3
    #SaleCondition
    nnor = ['Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']
    df.loc[df.SaleCondition.isin(nnor), 'SaleCondition'] = 0 #notNormal
    df.loc[df.SaleCondition == 'Normal', 'SaleCondition'] = 1
    
feats = ['SaleType', 'SaleCondition']

checkclean(feats)


# # Taking care of the truly missing data
# 
# Many missing values were just obvious or documented, how many are then left after this boring effort of fixing it?
# 
# First, I will eliminate the features I found to be irrelevant

# In[ ]:


irrs = ['Street', 'Condition2', 'Utilities', 'RoofMatl', 'RoofStyle', 'Heating',
      'LowQualFinSF', 'GarageQual', 'GarageCond', 'PoolArea', 'PoolQC', 'MiscVal', 'MiscFeature', 
       #the next features were just replaced
       'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2','BsmtFullBath','BsmtHalfBath', 'FullBath', 'HalfBath',
        'ScreenPorch', 'EnclosedPorch','3SsnPorch','OpenPorchSF', 'WoodDeckSF']

print('Before: {} features in train and {} in test'.format(len(df_train.columns), len(df_test.columns)))

for df in combine:
    for irr in irrs:
        del df[irr]
        
print('After: {} features in train and {} in test'.format(len(df_train.columns), len(df_test.columns)))


# In[ ]:


for df in combine:
    if df.name == 'Train':
        mis_train = []
        cols = df.columns
        for col in cols:
            mis = df[col].isnull().sum()
            if mis > 0:
                print("{}: {} missing, {}%".format(col, mis, round(mis/df.shape[0] * 100, 3)))
                mis_train.append(col)
        print("_"*40)
        print("_"*40)
    if df.name == 'Test':
        mis_test = []
        cols = df.columns
        for col in cols:
            mis = df[col].isnull().sum()
            if mis > 0:
                print("{}: {} missing, {}%".format(col, mis, round(mis/df.shape[0] * 100, 3)))
                mis_test.append(col)

print("\n")
print(mis_train)
print("_"*40)
print(mis_test)


# * **MasVnrType** and **MasVnr** are missing consistently in test and inconsistently in one case of the test. For each case, I will find the features that has only 1 value if those entries are missing, find the mode, impute with the mode. In the inconsistent case, I will use just an extra constraint to find the mode.
# 
# * **BsmExposure** is missing once in the train and twice in the test, I will impute with the mode of the values of those entries with the same basement characteristics. 
# 
# * **Electrical** is missing once in the train, I will impute with the mode of the values of those entries with the same energy characteristics
# 
# * **MSZoning** is missing 4 times in the test, will follow the same strategy of MasVnrType
# 
# * **Exterior1st** and **Exterior2nd	** missing once in test, I will impute with the mode of the values of those entries with the same exterior characteristics
# 
# * **BsmtQual** is missing twice in the test, same strategy as BsmExposure
# 
# * **BsmtCond** is missing 3 times in the test, same strategy
# 
# * **BsmtUnfSF** and **TotalBsmtSF** are missing once in the test, looking at it individually I can see that it is just a missing basement, so -99 for them
# 
# * **KitchenQual** missing once in the test, will be filled with the mode
# 
# * **Functional** missing twice in the test, same strategy as MasVnrType
# 
# * **GarageYrBlt** and **GarageFinish** missing consistently in test twice, same strategy as MasVnrArea
# 
# * **GarageCars** and **GarageArea** missing consistently once in test, I will do like in BsmExposure
# 
# * **SaleType** is missing once in test, same strategy of BsmExposure
# 
# * **BsmFinSFTot** and **BsmBath** are missing once in test but there the basement is missing too. BsmBath is missing one more time and it is missing the basement there too.
# 
# 
# I will need a couple of functions to work faster

# In[ ]:


def find_segment(df, feat): #returns the columns for which the missing entries have only 1 possible value
    mis = df[feat].isnull().sum()
    cols = df.columns
    seg = []
    for col in cols:
        vc = df[df[feat].isnull()][col].value_counts(dropna=False).iloc[0]
        if (vc == mis):
            seg.append(col)
    return seg

def find_mode(df, feat): #returns the mode to fill in the missing feat
    md = df[df[feat].isnull()][find_segment(df, feat)].dropna(axis = 1).mode()
    md = pd.merge(df, md, how='inner')[feat].mode().iloc[0]
    return md

def find_median(df, feat): #returns the median to fill in the missing feat
    md = df[df[feat].isnull()][find_segment(df, feat)].dropna(axis = 1).mode()
    md = pd.merge(df, md, how='inner')[feat].median()
    return md

def similar_mode(df, col, feats): #returns the mode in a segment made by similarity in feats
    sm = df[df[col].isnull()][feats]
    md = pd.merge(df, sm, how='inner')[col].mode().iloc[0]
    return md
    
def similar_median(df, col, feats): #returns the median in a segment made by similarity in feats
    sm = df[df[col].isnull()][feats]
    md = pd.merge(df, sm, how='inner')[col].median()
    return md


# In[ ]:


#Train
#MasVnrType
md = find_mode(df_train, 'MasVnrType')
print("MasVnrType {}".format(md))
df_train[['MasVnrType']] = df_train[['MasVnrType']].fillna(md)
#MasVnrArea
md = find_mode(df_train, 'MasVnr')
print("MasVnr {}".format(md))
df_train[['MasVnr']] = df_train[['MasVnr']].fillna(md)
#BsmtExposure
simi = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2']
md = similar_mode(df_train, 'BsmtExposure', simi)
print("BsmtExposure {}".format(md))
df_train[['BsmtExposure']] = df_train[['BsmtExposure']].fillna(md)
#Electrical
simi = ['HeatingQC', 'CentralAir']
md = similar_mode(df_train, 'Electrical', simi)
print("Electrical {}".format(md))
df_train[['Electrical']] = df_train[['Electrical']].fillna(md)

cols = df_train.columns
print("Start printing the missing values...")
for col in cols:
    mis = df_train[col].isnull().sum()
    if mis > 0:
        print("{}: {} missing, {}%".format(col, mis, round(mis/df_train.shape[0] * 100, 3)))
print("...done printing the missing values")


# In[ ]:


#Test
#MSZoning
md = find_mode(df_test, 'MSZoning')
print("MSZoning {}".format(md))
df_test[['MSZoning']] = df_test[['MSZoning']].fillna(md)
#Exterior1st
simi = ['ExterQual', 'ExterCond']
md = similar_mode(df_test, 'Exterior1st', simi)
print("Exterior1st {}".format(md))
df_test[['Exterior1st']] = df_test[['Exterior1st']].fillna(md)
#Exterior2nd
simi = ['ExterQual', 'ExterCond']
md = similar_mode(df_test, 'Exterior2nd', simi)
print("Exterior2nd {}".format(md))
df_test[['Exterior2nd']] = df_test[['Exterior2nd']].fillna(md)
#MasVnrType
md = find_mode(df_test, 'MasVnrType')
print("MasVnrType {}".format(md))
df_test[['MasVnrType']] = df_test[['MasVnrType']].fillna(md)
#MasVnrArea
md = find_mode(df_test, 'MasVnr')
print("MasVnr {}".format(md))
df_test[['MasVnr']] = df_test[['MasVnr']].fillna(md)
#BsmtQual
simi = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
md = similar_mode(df_test, 'BsmtQual', simi)
print("BsmtQual {}".format(md))
df_test[['BsmtQual']] = df_test[['BsmtQual']].fillna(md)
#BsmtCond
simi = ['BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
md = similar_mode(df_test, 'BsmtCond', simi)
print("BsmtCond {}".format(md))
df_test[['BsmtCond']] = df_test[['BsmtCond']].fillna(md)
#BsmtExposure
simi = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2']
md = similar_mode(df_test, 'BsmtExposure', simi)
print("BsmtExposure {}".format(md))
df_test[['BsmtExposure']] = df_test[['BsmtExposure']].fillna(md)
#BsmtUnfSF and TotalBsmtSF
df_test[['BsmtUnfSF', 'TotalBsmtSF']] = df_test[['BsmtUnfSF', 'TotalBsmtSF']].fillna(-99)
#KitchenQual
md = df_test.KitchenQual.mode().iloc[0]
print("KitchenQual {}".format(md))
df_test[['KitchenQual']] = df_test[['KitchenQual']].fillna(md)
#Functional
df_test[['Functional']] = df_test[['Functional']].fillna(1) #manually inserting the mode
#GarageYrBlt
simi = ['YearBuilt']
md = similar_median(df_test, 'GarageYrBlt', simi)
print("GarageYrBlt {}".format(md))
df_test[['GarageYrBlt']] = df_test[['GarageYrBlt']].fillna(md)
#GarageFinish
simi = ['YearBuilt', 'GarageYrBlt']
md = similar_median(df_test, 'GarageFinish', simi)
print("GarageFinish {}".format(md))
df_test[['GarageFinish']] = df_test[['GarageFinish']].fillna(md)
#GarageCars
simi = ['GarageType', 'MisGarage']
md = similar_mode(df_test, 'GarageCars', simi)
print("GarageCars {}".format(md))
df_test[['GarageCars']] = df_test[['GarageCars']].fillna(md)
#GarageArea
simi = ['GarageType', 'MisGarage', 'GarageCars']
md = similar_median(df_test, 'GarageArea', simi)
print("GarageArea {}".format(md))
df_test[['GarageArea']] = df_test[['GarageArea']].fillna(md)
#SaleType
simi = ['SaleCondition']
md = similar_mode(df_test, 'SaleType', simi)
print("SaleType {}".format(md))
df_test[['SaleType']] = df_test[['SaleType']].fillna(md)
#BsmFinSFTot and BsmBath
df_test[['BsmFinSFTot', 'BsmBath']] = df_test[['BsmFinSFTot', 'BsmBath']].fillna(-99)

cols = df_test.columns
print("Start printing the missing values...")
for col in cols:
    mis = df_test[col].isnull().sum()
    if mis > 0:
        print("{}: {} missing, {}%".format(col, mis, round(mis/df_test.shape[0] * 100, 3)))
print("...done printing the missing values")


# In[ ]:


#fixing the datatype
strings = ['MSSubClass']
ints = ['Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope',
       'Condition1', 'BldgType', 'HouseStyle', 'Exterior1st', 'Exterior2nd',
       'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2',
       'Electrical', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'FireplaceQu',
       'GarageType', 'GarageFinish', 'PavedDrive', 'Fence', 'SaleType', 'SaleCondition']

for df in combine:
    df[strings] = df[strings].astype(str)
    df[ints] = df[ints].astype(int)


# # Conclusions
# 
# This extremely boring kernel produced a train and test dataset with no obvious mistakes and fairly ready to be used for data analysis, feature engineering, and model training.  There will be 2 more kernels with those sections soon.
# 
# Thank you for reading this far, I hope I gave you some useful ideas for your own models. Thank you in advance for any comment and help.

# In[ ]:


df_train.to_csv("trainclean.csv", index=False)
df_test.to_csv("testclean.csv", index=False)


# In[ ]:





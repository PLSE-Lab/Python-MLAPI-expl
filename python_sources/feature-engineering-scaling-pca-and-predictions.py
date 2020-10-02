#!/usr/bin/env python
# coding: utf-8

# This is not the final version, optmisation still needed for some of the features and grid search to select better hyperparameters for XGBoost, Lasso and ElasticNet.
# 
# **Any feedback greatly appreciated!**

# In[ ]:


# data analysis
import pandas as pd
import numpy as np

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# metrics and algorithm validation
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

# encoding and modeling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import skew


# In[ ]:


import os
mingw_path = 'C:\Program Files\mingw-w64\x86_64-7.1.0-posix-seh-rt_v5-rev0\mingw64\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb


# In[ ]:


# import the dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


df = train.copy()


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


# viewing the correlation between columns
plt.figure(figsize=(18,15))
sns.heatmap(df.corr())


# # Analyse columns individually

# In[ ]:


df.columns.tolist()


# In[ ]:


df.info()


# # SalePrice
# 
# Eliminating outliers.

# In[ ]:


df.SalePrice.describe()


# In[ ]:


# eliminating all columns with a price over 350,000
to_el = df[df.SalePrice > 350000].index.tolist()


# In[ ]:


df.drop(to_el,inplace=True)


# # MSSubClass 
# 
# Identifies the type of dwelling involved in the sale.
# 
# As this is a column which aggregates the types of dwelling, it will be dropped.
# 
# House styles, build year and building style are all contained in other columns.
# 
# Columns to be created
# - Story
# - PUD

# In[ ]:


# creating PUD column

# indexes where PUD = 1
pud_i = df[df.MSSubClass.isin([120,150,160,180])].index.tolist()

# create PUD
df['PUD'] = 0

# add ones in PUD
for i in pud_i:
    df.set_value(i,'PUD',1)
    
# test set
# indexes where PUD = 1
pud_i = test[test.MSSubClass.isin([120,150,160,180])].index.tolist()

# create PUD
test['PUD'] = 0

# add ones in PUD
for i in pud_i:
    test.set_value(i,'PUD',1)


# In[ ]:


# dropping the MSSubClass
df.drop('MSSubClass',axis=1,inplace=True)
test.drop('MSSubClass',axis=1,inplace=True)


# # Items related to MSSubClass
# 
# Will work on adjusting the items related to the deleted column.

# # HouseStyle: Style of dwelling
# 
# Decided to clasify them into 4 categories:
# - 0 - 1 Story
# - 1 - 1.5Fin, 1.5Unf
# - 1 - 2 Stories, SLvl, SFoyer
# - 2 - 2.5Fin, 2.5Unf

# In[ ]:


df.HouseStyle.value_counts()


# In[ ]:


for i in df.index:
    if df.HouseStyle[i] == '1Story':
        df.set_value(i,'HouseStyle',0)
    elif df.HouseStyle[i] in ['1.5Fin','1.5Unf']:
        df.set_value(i,'HouseStyle',1)
    elif df.HouseStyle[i] in ['2Story','SLvl','SFoyer']:
        df.set_value(i,'HouseStyle',2)
    elif df.HouseStyle[i] in ['2.5Unf','2.5Fin']:
        df.set_value(i,'HouseStyle',3)
        
for i in test.index:
    if test.HouseStyle[i] == '1Story':
        test.set_value(i,'HouseStyle',0)
    elif test.HouseStyle[i] in ['1.5Fin','1.5Unf']:
        test.set_value(i,'HouseStyle',1)
    elif test.HouseStyle[i] in ['2Story','SLvl','SFoyer']:
        test.set_value(i,'HouseStyle',2)
    elif test.HouseStyle[i] in ['2.5Unf','2.5Fin']:
        test.set_value(i,'HouseStyle',3)


# In[ ]:


# add for OneHot
hot = ['HouseStyle']


# In[ ]:


df.HouseStyle = pd.to_numeric(df.HouseStyle)
test.HouseStyle = pd.to_numeric(test.HouseStyle)


# # MSZoning
# 
# Clasify in 5 categories:
# - 0 - Agriculture, Commercial, Industrial
# - 1 - RL, RP
# - 2 - RM
# - 3 - RH
# - 4 - FV

# In[ ]:


# some empty values
ind = test[test.MSZoning.isnull()].index.tolist()


# In[ ]:


for i in ind:
    test.set_value(i,'MSZoning',1)


# In[ ]:


df.MSZoning.value_counts()


# In[ ]:


test.MSZoning.value_counts()


# In[ ]:


for i in df.index:
    if df.MSZoning[i] == 'C (all)':
        df.set_value(i,'MSZoning',0)
    elif df.MSZoning[i] == 'RL':
        df.set_value(i,'MSZoning',1)
    elif df.MSZoning[i] == 'RM':
        df.set_value(i,'MSZoning',2)
    elif df.MSZoning[i] == 'RH':
        df.set_value(i,'MSZoning',3)
    elif df.MSZoning[i] == 'FV':
        df.set_value(i,'MSZoning',4)


# In[ ]:


for i in test.index:
    if test.MSZoning[i] == 'C (all)':
        test.set_value(i,'MSZoning',0)
    elif test.MSZoning[i] == 'RL':
        test.set_value(i,'MSZoning',1)
    elif test.MSZoning[i] == 'RM':
        test.set_value(i,'MSZoning',2)
    elif test.MSZoning[i] == 'RH':
        test.set_value(i,'MSZoning',3)
    elif test.MSZoning[i] == 'FV':
        test.set_value(i,'MSZoning',4)


# In[ ]:


# add for HotEncoder
hot.append('MSZoning')


# # LotArea
# 
# Dropping the outlier columns, greater than 60,000. Not goint lower because the test set contains values in those regions.

# In[ ]:


df.LotArea.describe()


# In[ ]:


df.LotArea.hist()


# In[ ]:


to_drop = df.LotArea[df.LotArea > 60000].keys().tolist()
df.drop(to_drop, inplace=True)


# In[ ]:


skw = ['LotArea']


# # Street
# 
# Heavily skewed, will drop it.

# In[ ]:


df.drop('Street',axis=1,inplace=True)
test.drop('Street',axis=1,inplace=True)


# # Alley
# 
# A lot of empty rows, will do more harm than good to the model. Eliminating.

# In[ ]:


df.drop('Alley',axis=1,inplace=True)
test.drop('Alley',axis=1,inplace=True)


# # LotShape
# 
# Clasify as 0 - Irregular, 1 - Regular

# In[ ]:


for i in df.index:
    if df.LotShape[i] == 'Reg':
        df.set_value(i,'LotShape',1)
    else:
        df.set_value(i,'LotShape',0)

for i in test.index:
    if test.LotShape[i] == 'Reg':
        test.set_value(i,'LotShape',1)
    else:
        test.set_value(i,'LotShape',0)


# # LandContour
# 
# Split in 4 categories. Add to OneHot.

# In[ ]:


df.LandContour.value_counts()


# In[ ]:


for i in df.index:
    if df.LandContour[i] == 'Lvl':
        df.set_value(i,'LandContour', 0)
    elif df.LandContour[i] == 'Bnk':
        df.set_value(i,'LandContour', 1)
    elif df.LandContour[i] == 'HLS':
        df.set_value(i,'LandContour', 2)
    elif df.LandContour[i] == 'Low':
        df.set_value(i,'LandContour', 3)


# In[ ]:


for i in test.index:
    if test.LandContour[i] == 'Lvl':
        test.set_value(i,'LandContour', 0)
    elif test.LandContour[i] == 'Bnk':
        test.set_value(i,'LandContour', 1)
    elif test.LandContour[i] == 'HLS':
        test.set_value(i,'LandContour', 2)
    elif test.LandContour[i] == 'Low':
        test.set_value(i,'LandContour', 3)


# In[ ]:


hot.append('LandContour')


# # Utilities
# 
# Pointless column. Eliminating it

# In[ ]:


df.Utilities.value_counts()


# In[ ]:


test.Utilities.value_counts()


# In[ ]:


df.drop('Utilities',axis=1,inplace=True)
test.drop('Utilities',axis=1,inplace=True)


# # LotConfig
# 
# 4 Categories, add to OneHot

# In[ ]:


df.LotConfig.value_counts()


# In[ ]:


test.LotConfig.value_counts()


# In[ ]:


for i in df.index:
    if df.LotConfig[i] == 'Inside':
        df.set_value(i,'LotConfig', 0)
    elif df.LotConfig[i] == 'Corner':
        df.set_value(i,'LotConfig', 1)
    elif df.LotConfig[i] == 'CulDSac':
        df.set_value(i,'LotConfig', 2)
    elif df.LotConfig[i] in ['FR2','FR3']:
        df.set_value(i,'LotConfig', 3)


# In[ ]:


for i in test.index:
    if test.LotConfig[i] == 'Inside':
        test.set_value(i,'LotConfig', 0)
    elif test.LotConfig[i] == 'Corner':
        test.set_value(i,'LotConfig', 1)
    elif test.LotConfig[i] == 'CulDSac':
        test.set_value(i,'LotConfig', 2)
    elif test.LotConfig[i] in ['FR2','FR3']:
        test.set_value(i,'LotConfig', 3)


# In[ ]:


hot.append('LotConfig')


# # LandSlope
# 
# Not helpful, dropping it.

# In[ ]:


df.LandSlope.value_counts()


# In[ ]:


test.LandSlope.value_counts()


# In[ ]:


df.drop('LandSlope',axis=1,inplace=True)
test.drop('LandSlope',axis=1,inplace=True)


# # Neighborhood
# 
# LabelEncoder adding for OneHotEncoder at the end.

# In[ ]:


encoder = LabelEncoder()


# In[ ]:


# first fitting and transforming the df column
df.Neighborhood = encoder.fit_transform(df.Neighborhood)


# In[ ]:


# then fitting the test column
test.Neighborhood = encoder.transform(test.Neighborhood)


# In[ ]:


hot.append('Neighborhood')


# # Condition1, Condition 2
# 
# Identified the categories as having the following impact, based on the mean price where they are conditions:
# - Positive: PosN, RRNn, RRNe, PosA - 2
# - Neutral: Norm, RRAn - 1
# - Negative: Feedr, Artery, RRAe - 0
# 
# Adding the final result to a new column, Conditions.

# In[ ]:


df.Condition1.value_counts()


# In[ ]:


df.Condition2.value_counts()


# In[ ]:


# classifying ones that appear in both columns and are not Norm
df[df.Condition2 != 'Norm'][['Condition1', 'Condition2']]

# 9 - Neg
# 29 - Neg
# 63 - Neg
# 88 - Neg
# 184 - Neg
# 523 - Pos
# 531 - Normal
# 548 - Neg
# 583 - Pos
# 589 - Neg
# 974 - Neg
# 1003 - Neg
# 1186 - Neg
# 1230 - Neg


# In[ ]:


# positive and negative categories
pos_c = ['PosN', 'RRNn', 'RRNe', 'PosA']
neg_c = ['Feedr', 'Artery', 'RRAe']


# In[ ]:


# finding the indexes of those in Condition1 column
negs = []
pos = []
for i in df.index:
    if df['Condition1'][i] in pos_c:
        pos.append(i)
    elif df['Condition1'][i] in neg_c:
        negs.append(i)


# In[ ]:


# adding the extra ones in Condition2 in their corresponding categories
negs.append([9,29,63,88,184,548,589,974,1003,1186,1230])
pos.append([523,583])


# In[ ]:


# creating the new column with Normal as standard
df['Conditions'] = 1


# In[ ]:


# replacing the corresponding rows with the categories
for i in negs:
    df.set_value(i,'Conditions',0)
    
for i in pos:
    df.set_value(i,'Conditions',2)


# In[ ]:


# same for test set
test.Condition1.value_counts()


# In[ ]:


test.Condition2.value_counts()


# In[ ]:


# classifying ones that appear in both columns and are not Norm
test[test.Condition2 != 'Norm'][['Condition1', 'Condition2']]

# 81 - Neg
# 203 - Pos
# 245 - Pos
# 486 - Pos
# 593 - Neg
# 650 - Neg
# 778 - Neg
# 807 - Pos
# 940 - Neg
# 995 - Neg
# 1138 - Pos
# 1258 - Neg
# 1336 - Neg
# 1342 - Neg


# In[ ]:


negs = []
pos = []
for i in test.index:
    if test['Condition1'][i] in pos_c:
        pos.append(i)
    elif test['Condition1'][i] in neg_c:
        negs.append(i)


# In[ ]:


negs.append([81,593,650,778,940,995,1258,1336,1342])
pos.append([203,245,486,807,1138])


# In[ ]:


test['Conditions'] = 1


# In[ ]:


for i in negs:
    test.set_value(i,'Conditions',0)
    
for i in pos:
    test.set_value(i,'Conditions',2)


# In[ ]:


# dropping them once done
df.drop('Condition1',axis=1,inplace=True)
test.drop('Condition1',axis=1,inplace=True)
df.drop('Condition2',axis=1,inplace=True)
test.drop('Condition2',axis=1,inplace=True)


# # BldgType
# 
# Convert to family - 1, not family - 0

# In[ ]:


df.BldgType.value_counts()


# In[ ]:


test.BldgType.value_counts()


# In[ ]:


for i in df.index:
    if df.BldgType[i] in ['1Fam', '2fmCon']:
        df.set_value(i,'BldgType',1)
    else:
        df.set_value(i,'BldgType',0)
        
for i in test.index:
    if test.BldgType[i] in ['1Fam', '2fmCon']:
        test.set_value(i,'BldgType',1)
    else:
        test.set_value(i,'BldgType',0)


# # OverallQual
# 
# Split in 3: 0: Poor, 1 - Average, 2 - Good

# In[ ]:


df.OverallQual.value_counts()


# In[ ]:


good = [10,9,8,7]
average = [6,5,4]
bad = [3,2,1]


# In[ ]:


# train
for i in df.index:
    if df['OverallQual'][i] in good:
        df.set_value(i,'OverallQual',2)
    elif df['OverallQual'][i] in average:
        df.set_value(i,'OverallQual',1)
    elif df['OverallQual'][i] in bad:
        df.set_value(i,'OverallQual',0)


# In[ ]:


# test
for i in test.index:
    if test['OverallQual'][i] in good:
        test.set_value(i,'OverallQual',2)
    elif test['OverallQual'][i] in average:
        test.set_value(i,'OverallQual',1)
    elif test['OverallQual'][i] in bad:
        test.set_value(i,'OverallQual',0)


# # OverallCond
# 
# Split in 3: 0: Poor, 1 - Average, 2 - Good

# In[ ]:


# train
for i in df.index:
    if df['OverallCond'][i] in good:
        df.set_value(i,'OverallCond',2)
    elif df['OverallCond'][i] in average:
        df.set_value(i,'OverallCond',1)
    elif df['OverallCond'][i] in bad:
        df.set_value(i,'OverallCond',0)
        
# test
for i in test.index:
    if test['OverallCond'][i] in good:
        test.set_value(i,'OverallCond',2)
    elif test['OverallCond'][i] in average:
        test.set_value(i,'OverallCond',1)
    elif test['OverallCond'][i] in bad:
        test.set_value(i,'OverallCond',0)


# # YearRemodAdd
# 
# Replacing with 1 - Yes and 0 - No.

# In[ ]:


for i in df.index:
    if df['YearBuilt'][i] == df['YearRemodAdd'][i]:
        df.set_value(i,'YearRemodAdd',1)
    else:
        df.set_value(i,'YearRemodAdd',0)
        
for i in test.index:
    if test['YearBuilt'][i] == test['YearRemodAdd'][i]:
        test.set_value(i,'YearRemodAdd',1)
    else:
        test.set_value(i,'YearRemodAdd',0)


# # YearBuilt
# 
# Replacing continous with 6 categories.
# 
# Add for OneHot Encoder.

# In[ ]:


df.YearBuilt.describe()


# In[ ]:


df.YearBuilt.hist(bins=6)


# In[ ]:


test.YearBuilt.describe()


# In[ ]:


test.YearBuilt.hist()


# In[ ]:


# train
for i in df.index:
    if df['YearBuilt'][i] > 0 and df['YearBuilt'][i] <= 1895:
        df.set_value(i,'YearBuilt', 0)
    elif df['YearBuilt'][i] > 1895 and df['YearBuilt'][i] <= 1918:
        df.set_value(i,'YearBuilt', 1)
    elif df['YearBuilt'][i] > 1918 and df['YearBuilt'][i] <= 1941:
        df.set_value(i,'YearBuilt', 2)
    elif df['YearBuilt'][i] > 1941 and df['YearBuilt'][i] <= 1964:
        df.set_value(i,'YearBuilt', 3)
    elif df['YearBuilt'][i] > 1964 and df['YearBuilt'][i] <= 1987:
        df.set_value(i,'YearBuilt', 4)
    elif df['YearBuilt'][i] > 1987:
        df.set_value(i,'YearBuilt', 5)


# In[ ]:


# test
for i in test.index:
    if test['YearBuilt'][i] > 0 and test['YearBuilt'][i] <= 1895:
        test.set_value(i,'YearBuilt', 0)
    elif test['YearBuilt'][i] > 1895 and test['YearBuilt'][i] <= 1918:
        test.set_value(i,'YearBuilt', 1)
    elif test['YearBuilt'][i] > 1918 and test['YearBuilt'][i] <= 1941:
        test.set_value(i,'YearBuilt', 2)
    elif test['YearBuilt'][i] > 1941 and test['YearBuilt'][i] <= 1964:
        test.set_value(i,'YearBuilt', 3)
    elif test['YearBuilt'][i] > 1964 and test['YearBuilt'][i] <= 1987:
        test.set_value(i,'YearBuilt', 4)
    elif test['YearBuilt'][i] > 1987:
        test.set_value(i,'YearBuilt', 5)


# In[ ]:


hot.append('YearBuilt')


# # RoofStyle
# 
# 3 Categories:
# - 0 - Gable
# - 1 - Hip
# - 2 - Rest

# In[ ]:


df.RoofStyle.value_counts()


# In[ ]:


test.RoofStyle.value_counts()


# In[ ]:


for i in df.index:
    if df.RoofStyle[i] == 'Gable':
        df.set_value(i,'RoofStyle',0)
    elif df.RoofStyle[i] == 'Hip':
        df.set_value(i,'RoofStyle',1)
    else:
        df.set_value(i,'RoofStyle',2)


# In[ ]:


for i in test.index:
    if test.RoofStyle[i] == 'Gable':
        test.set_value(i,'RoofStyle',0)
    elif test.RoofStyle[i] == 'Hip':
        test.set_value(i,'RoofStyle',1)
    else:
        test.set_value(i,'RoofStyle',2)


# In[ ]:


hot.append('RoofStyle')


# # RoofMatl
# 
# Very skewed, dropping it.

# In[ ]:


df.RoofMatl.value_counts()


# In[ ]:


test.RoofMatl.value_counts()


# In[ ]:


df.drop('RoofMatl',axis=1,inplace=True)
test.drop('RoofMatl',axis=1,inplace=True)


# # Exterior1st, Exterior2nd
# 
# Creating and encoding a column combination of the two.

# In[ ]:


df['SalePrice'].groupby(df['Exterior1st']).mean()


# In[ ]:


df['SalePrice'].groupby(df['Exterior2nd']).mean()


# In[ ]:


# getting all the combinations
combs = []
for i in df.index:
    combs.append((df.Exterior1st[i], df.Exterior2nd[i]))


# In[ ]:


# eliminating duplicates
combs = set(combs)


# In[ ]:


# different combos from the training set
test_diff = []
count = 0
for i in test.index:
    found = 0
    for j in combs:
        if (test.Exterior1st[i], test.Exterior2nd[i]) == j or (test.Exterior2nd[i], test.Exterior1st[i]) == j:
            found = 1
    if found == 0:
        test_diff.append((test.Exterior1st[i], test.Exterior2nd[i]))
        count += 1


# In[ ]:


# eliminating duplicates
test_diff = set(test_diff)


# In[ ]:


# rows with different fields in test set
count


# In[ ]:


# creating a new column
df['Exterior'] = ''


# In[ ]:


# adding combinations
for i in df.index:
    for j in combs:
        if (df.Exterior1st[i], df.Exterior2nd[i]) == j or (df.Exterior2nd[i], df.Exterior1st[i]) == j:
            df.set_value(i,'Exterior',j)


# In[ ]:


# mode values grouped by combination
# chosen mode as it brigns the mean closest to the overall one
grp = pd.DataFrame(df.SalePrice.groupby(df.Exterior).agg(lambda x:x.value_counts().index[0]).sort_values(ascending=False))


# In[ ]:


# number of occurances
grp2 = pd.DataFrame(df.Exterior.value_counts())


# In[ ]:


# joining them
new = grp.join(grp2)
new.head()


# In[ ]:


new.describe()


# In[ ]:


new.SalePrice.hist(bins=5)


# In[ ]:


# value categories split by combinations
zeros = new[new.SalePrice <= 134600]['Exterior'].keys().tolist()
ones = new[(new.SalePrice > 134600) & (new.SalePrice <= 187200)]['Exterior'].keys().tolist()
twos = new[(new.SalePrice > 187200) & (new.SalePrice <= 239800)]['Exterior'].keys().tolist()
threes = new[new.SalePrice > 239800]['Exterior'].keys().tolist()


# In[ ]:


# replace on training set
for i in df.index:
    if df.Exterior[i] in zeros:
        df.set_value(i,'Exterior',0)
    elif df.Exterior[i] in ones:
        df.set_value(i,'Exterior',1)
    elif df.Exterior[i] in twos:
        df.set_value(i,'Exterior',2)
    elif df.Exterior[i] in threes:
        df.set_value(i,'Exterior',3)


# In[ ]:


# same for test set
test['Exterior'] = ''

# adding combinations
for i in test.index:
    for j in combs:
        if (test.Exterior1st[i], test.Exterior2nd[i]) == j or (test.Exterior2nd[i], test.Exterior1st[i]) == j:
            test.set_value(i,'Exterior',j)


# In[ ]:


# replace on test set
for i in test.index:
    if test.Exterior[i] in zeros:
        test.set_value(i,'Exterior',0)
    elif test.Exterior[i] in ones:
        test.set_value(i,'Exterior',1)
    elif test.Exterior[i] in twos:
        test.set_value(i,'Exterior',2)
    elif test.Exterior[i] in threes:
        test.set_value(i,'Exterior',3)
    else:
        test.set_value(i,'Exterior',1)


# In[ ]:


# dropping Exterior1st and Exterior2nd
df.drop('Exterior1st',axis=1,inplace=True)
df.drop('Exterior2nd',axis=1,inplace=True)
test.drop('Exterior1st',axis=1,inplace=True)
test.drop('Exterior2nd',axis=1,inplace=True)


# # MasVnrType
# 
# Classifying BrkCmn, None - 0, BrkFace - 1, Stone - 2

# In[ ]:


df.MasVnrType.value_counts()


# In[ ]:


ind = df.MasVnrType[df.MasVnrType.isnull()].index.tolist()


# In[ ]:


for i in ind:
    df.set_value(i,'MasVnrType',0)


# In[ ]:


ind = test.MasVnrType[test.MasVnrType.isnull()].index.tolist()

for i in ind:
    test.set_value(i,'MasVnrType',0)


# In[ ]:


for i in df.index:
    if df.MasVnrType[i] in ['BrkCmn', 'None']:
        df.set_value(i,'MasVnrType', 0)
    elif df.MasVnrType[i] == 'BrkFace':
        df.set_value(i,'MasVnrType', 1)
    elif df.MasVnrType[i] == 'Stone':
        df.set_value(i,'MasVnrType', 2)


# In[ ]:


for i in test.index:
    if test.MasVnrType[i] in ['BrkCmn', 'None']:
        test.set_value(i,'MasVnrType', 0)
    elif test.MasVnrType[i] == 'BrkFace':
        test.set_value(i,'MasVnrType', 1)
    elif test.MasVnrType[i] == 'Stone':
        test.set_value(i,'MasVnrType', 2)


# # MasVnrArea
# 
# Appply log to limit skewness.

# In[ ]:


df.MasVnrArea.describe()


# In[ ]:


ind = df.MasVnrArea[df.MasVnrArea.isnull()].index.tolist()
# same empty columns as above
for i in ind:
    df.set_value(i,'MasVnrArea',0)


# In[ ]:


ind = test.MasVnrArea[test.MasVnrArea.isnull()].index.tolist()

for i in ind:
    test.set_value(i,'MasVnrArea',0)


# In[ ]:


df[df.MasVnrArea != 0]['MasVnrArea'].hist()


# In[ ]:


df[df.MasVnrArea != 0]['MasVnrArea'].sort_values(ascending=False).head()


# In[ ]:


# drop outlier
df.drop(297,inplace=True)


# In[ ]:


skw.append('MasVnrArea')


# # ExterQual
# 
# Classify as 0 - Poor, 1 - Average, 2 - Good 

# In[ ]:


df.ExterQual.value_counts()


# In[ ]:


# train
for i in df.index:
    if df.ExterQual[i] in ['Gd', 'Ex']:
        df.set_value(i,'ExterQual', 2)
    elif df.ExterQual[i] == 'TA':
        df.set_value(i,'ExterQual', 1)
    elif df.ExterQual[i] in ['Fa', 'Po']:
        df.set_value(i,'ExterQual', 0)


# In[ ]:


# test
for i in test.index:
    if test.ExterQual[i] in ['Gd', 'Ex']:
        test.set_value(i,'ExterQual', 2)
    elif test.ExterQual[i] == 'TA':
        test.set_value(i,'ExterQual', 1)
    elif test.ExterQual[i] in ['Fa', 'Po']:
        test.set_value(i,'ExterQual', 0)


# # ExterCond
# 
# Same as above

# In[ ]:


# train
for i in df.index:
    if df.ExterCond[i] in ['Gd', 'Ex']:
        df.set_value(i,'ExterCond', 2)
    elif df.ExterCond[i] == 'TA':
        df.set_value(i,'ExterCond', 1)
    elif df.ExterCond[i] in ['Fa', 'Po']:
        df.set_value(i,'ExterCond', 0)
        
# test
for i in test.index:
    if test.ExterCond[i] in ['Gd', 'Ex']:
        test.set_value(i,'ExterCond', 2)
    elif test.ExterCond[i] == 'TA':
        test.set_value(i,'ExterCond', 1)
    elif test.ExterCond[i] in ['Fa', 'Po']:
        test.set_value(i,'ExterCond', 0)


# # Foundation
# 
# 4 categories
# - 0 - BrkTil
# - 1 - CBlock
# - 2 - PConc
# - 3 - Slab, Stone, Wood
# 
# Add to hot.

# In[ ]:


df.Foundation.value_counts()


# In[ ]:


for i in df.index:
    if df.Foundation[i] == 'BrkTil':
        df.set_value(i,'Foundation',0)
    elif df.Foundation[i] == 'CBlock':
        df.set_value(i,'Foundation',1)
    elif df.Foundation[i] == 'PConc':
        df.set_value(i,'Foundation',2)
    else:
        df.set_value(i,'Foundation',3)


# In[ ]:


for i in test.index:
    if test.Foundation[i] == 'BrkTil':
        test.set_value(i,'Foundation',0)
    elif test.Foundation[i] == 'CBlock':
        test.set_value(i,'Foundation',1)
    elif test.Foundation[i] == 'PConc':
        test.set_value(i,'Foundation',2)
    else:
        test.set_value(i,'Foundation',3)


# In[ ]:


hot.append('Foundation')


# # BsmtQual
# 
# Create new column, HasBasement.
# 
# Split this column in 4 categories:
# 0 - NA
# 1 - Po, Fa
# 2 - TA
# 3 - Gd, Ex

# In[ ]:


df.BsmtQual.value_counts()


# In[ ]:


# creating new column
df['HasBsmt'] = 1


# In[ ]:


# empty indexes
idx = df.BsmtQual[df.BsmtQual.isnull()].index.tolist()


# In[ ]:


# adding the empty ones
for i in idx:
    df.set_value(i,'HasBsmt',0)


# In[ ]:


test['HasBsmt'] = 1
# empty indexes
idx = test.BsmtQual[test.BsmtQual.isnull()].index.tolist()
# adding the empty ones
for i in idx:
    test.set_value(i,'HasBsmt',0)


# In[ ]:


for i in df.index:
    if df.BsmtQual[i] in ['Po', 'Fa']:
        df.set_value(i,'BsmtQual',1)
    elif df.BsmtQual[i] == 'TA':
        df.set_value(i,'BsmtQual',2)
    elif df.BsmtQual[i] in ['Gd', 'Ex']:
        df.set_value(i,'BsmtQual',3)
    else:
        df.set_value(i,'BsmtQual',0)


# In[ ]:


for i in test.index:
    if test.BsmtQual[i] in ['Po', 'Fa']:
        test.set_value(i,'BsmtQual',1)
    elif test.BsmtQual[i] == 'TA':
        test.set_value(i,'BsmtQual',2)
    elif test.BsmtQual[i] in ['Gd', 'Ex']:
        test.set_value(i,'BsmtQual',3)
    else:
        test.set_value(i,'BsmtQual',0)


# # BsmtCond
# 
# Same as above.

# In[ ]:


df.BsmtCond.value_counts()


# In[ ]:


for i in df.index:
    if df.BsmtCond[i] in ['Po', 'Fa']:
        df.set_value(i,'BsmtCond',1)
    elif df.BsmtCond[i] == 'TA':
        df.set_value(i,'BsmtCond',2)
    elif df.BsmtCond[i] in ['Gd', 'Ex']:
        df.set_value(i,'BsmtCond',3)
    else:
        df.set_value(i,'BsmtCond',0)
        
for i in test.index:
    if test.BsmtCond[i] in ['Po', 'Fa']:
        test.set_value(i,'BsmtCond',1)
    elif test.BsmtCond[i] == 'TA':
        test.set_value(i,'BsmtCond',2)
    elif test.BsmtCond[i] in ['Gd', 'Ex']:
        test.set_value(i,'BsmtCond',3)
    else:
        test.set_value(i,'BsmtCond',0)


# # BsmtExposure
# 
# Categorise:
# - 0 - Gd, Av
# - 1 - Mn
# - 2 - No
# - 3 -NA

# In[ ]:


df.BsmtExposure.value_counts()


# In[ ]:


for i in df.index:
    if df.BsmtExposure[i] in ['Gd', 'Av']:
        df.set_value(i,'BsmtExposure',0)
    elif df.BsmtExposure[i] == 'Mn':
        df.set_value(i,'BsmtExposure',1)
    elif df.BsmtExposure[i] == 'No':
        df.set_value(i,'BsmtExposure',2)
    else:
        df.set_value(i,'BsmtExposure',3)
        
for i in test.index:
    if test.BsmtExposure[i] in ['Gd', 'Av']:
        test.set_value(i,'BsmtExposure',0)
    elif test.BsmtExposure[i] == 'Mn':
        test.set_value(i,'BsmtExposure',1)
    elif test.BsmtExposure[i] == 'No':
        test.set_value(i,'BsmtExposure',2)
    else:
        test.set_value(i,'BsmtExposure',3)


# In[ ]:


hot.append('BsmtExposure')


# # BsmtFinType1, BsmtFinType2
# 
# 4 Categories:
# - 0 - NA
# - 1 - Unf
# - 2 - BLQ, LwQ 
# - 3 - GLQ, ALQ, Rec

# In[ ]:


df.BsmtFinType1.value_counts()


# In[ ]:


#Type1
for i in df.index:
    if df.BsmtFinType1[i] == 'Unf':
        df.set_value(i,'BsmtFinType1',1)
    elif df.BsmtFinType1[i] in ['BLQ', 'LwQ']:
        df.set_value(i,'BsmtFinType1',2)
    elif df.BsmtFinType1[i] in ['GLQ', 'ALQ', 'Rec']:
        df.set_value(i,'BsmtFinType1',3)
    else:
        df.set_value(i,'BsmtFinType1',0)
        
for i in test.index:
    if test.BsmtFinType1[i] == 'Unf':
        test.set_value(i,'BsmtFinType1',1)
    elif test.BsmtFinType1[i] in ['BLQ', 'LwQ']:
        test.set_value(i,'BsmtFinType1',2)
    elif test.BsmtFinType1[i] in ['GLQ', 'ALQ', 'Rec']:
        test.set_value(i,'BsmtFinType1',3)
    else:
        test.set_value(i,'BsmtFinType1',0)


# In[ ]:


# Type2
for i in df.index:
    if df.BsmtFinType2[i] == 'Unf':
        df.set_value(i,'BsmtFinType2',1)
    elif df.BsmtFinType2[i] in ['BLQ', 'LwQ']:
        df.set_value(i,'BsmtFinType2',2)
    elif df.BsmtFinType2[i] in ['GLQ', 'ALQ', 'Rec']:
        df.set_value(i,'BsmtFinType2',3)
    else:
        df.set_value(i,'BsmtFinType2',0)
        
for i in test.index:
    if test.BsmtFinType2[i] == 'Unf':
        test.set_value(i,'BsmtFinType2',1)
    elif test.BsmtFinType2[i] in ['BLQ', 'LwQ']:
        test.set_value(i,'BsmtFinType2',2)
    elif test.BsmtFinType2[i] in ['GLQ', 'ALQ', 'Rec']:
        test.set_value(i,'BsmtFinType2',3)
    else:
        test.set_value(i,'BsmtFinType2',0)


# # BsmtFinSF1, BsmtFinSF2
# 
# Add for skew.

# In[ ]:


df.BsmtFinSF1.hist()


# In[ ]:


df.BsmtFinSF2.hist()


# In[ ]:


# empty value
test.set_value(test.BsmtFinSF1[test.BsmtFinSF1.isnull()].index[0],'BsmtFinSF1',0)
print('---')


# In[ ]:


# empty value
test.set_value(test.BsmtFinSF2[test.BsmtFinSF2.isnull()].index[0],'BsmtFinSF2',0)
print('---')


# In[ ]:


skw.append(['BsmtFinSF1', 'BsmtFinSF2'])


# # BsmtUnfSF
# 
# This column will be added for skew processing.
# 
# New column BsmtFinished finished - 1, unfinished - 0

# In[ ]:


no_b = df.HasBsmt[df.HasBsmt == 0].index.tolist()


# In[ ]:


unf = df.BsmtUnfSF[df.BsmtUnfSF == 0].index.tolist()


# In[ ]:


df['BsmtFinished'] = 0
for i in df.index:
    if df.BsmtUnfSF[i] == 0 and i in unf and i not in no_b:
        df.set_value(i,'BsmtFinished',1)


# In[ ]:


df.BsmtUnfSF.hist()


# In[ ]:


test.set_value(test.BsmtUnfSF[test.BsmtUnfSF.isnull()].index[0],'BsmtUnfSF',0)

no_b = test.HasBsmt[test.HasBsmt == 0].index.tolist()
unf = test.BsmtUnfSF[test.BsmtUnfSF == 0].index.tolist()

test['BsmtFinished'] = 0
for i in test.index:
    if test.BsmtUnfSF[i] == 0 and i in unf and i not in no_b:
        test.set_value(i,'BsmtFinished',1)


# In[ ]:


skw.append('BsmtUnfSF')


# # TotalBsmtSF
# 
# Dropping outliers.
# 
# Nothing else to change here, all good.

# In[ ]:


df.TotalBsmtSF.hist()


# In[ ]:


# dropping outliers
elim = df[df.TotalBsmtSF > 2300].index.tolist()


# In[ ]:


df.drop(elim,inplace=True)


# In[ ]:


test.set_value(test.TotalBsmtSF[test.TotalBsmtSF.isnull()].index[0],'TotalBsmtSF',0)
print('---')


# In[ ]:


test.TotalBsmtSF.hist()


# # Heating
# 
# Dropping this one.

# In[ ]:


df.Heating.value_counts()


# In[ ]:


test.Heating.value_counts()


# In[ ]:


df.drop('Heating', axis=1, inplace=True)
test.drop('Heating', axis=1, inplace=True)


# # HeatingQC
# 
# Categorise:
# - 0 - Po, Fa
# - 1 - TA
# - 2 - Gd
# - 3 - Ex

# In[ ]:


df.HeatingQC.value_counts()


# In[ ]:


for i in df.index:
    if df.HeatingQC[i] in ['Po', 'Fa']:
        df.set_value(i,'HeatingQC',0)
    elif df.HeatingQC[i] == 'TA':
        df.set_value(i,'HeatingQC',1)
    elif df.HeatingQC[i] == 'Gd':
        df.set_value(i,'HeatingQC',2)
    elif df.HeatingQC[i] == 'Ex':
        df.set_value(i,'HeatingQC',3)


# In[ ]:


for i in test.index:
    if test.HeatingQC[i] in ['Po', 'Fa']:
        test.set_value(i,'HeatingQC',0)
    elif test.HeatingQC[i] == 'TA':
        test.set_value(i,'HeatingQC',1)
    elif test.HeatingQC[i] == 'Gd':
        test.set_value(i,'HeatingQC',2)
    elif test.HeatingQC[i] == 'Ex':
        test.set_value(i,'HeatingQC',3)


# # CentralAir
# 
# Encode.

# In[ ]:


encoder = LabelEncoder()

# train
df.CentralAir = encoder.fit_transform(df.CentralAir)
# test
test.CentralAir = encoder.transform(test.CentralAir)


# # Electrical
# 
# Categorise:
# - 0 - Mix, FuseP, FuseF
# - 1 - FuseA
# - 2 - SBrkr

# In[ ]:


df.Electrical.value_counts()


# In[ ]:


test.Electrical.value_counts()


# In[ ]:


for i in df.index:
    if df.Electrical[i] in ['Mix','FuseP','FuseF']:
        df.set_value(i,'Electrical',0)
    elif df.Electrical[i] == 'FuseA':
        df.set_value(i,'Electrical',1)
    elif df.Electrical[i] == 'SBrkr':
        df.set_value(i,'Electrical',2)


# In[ ]:


for i in test.index:
    if test.Electrical[i] in ['Mix','FuseP','FuseF']:
        test.set_value(i,'Electrical',0)
    elif test.Electrical[i] == 'FuseA':
        test.set_value(i,'Electrical',1)
    elif test.Electrical[i] == 'SBrkr':
        test.set_value(i,'Electrical',2)


# In[ ]:


df.set_value(df.Electrical[df.Electrical.isnull()].index[0],'Electrical',2)
print('---')


# In[ ]:


hot.append('Electrical')


# # 1stFlrSF
# 
# Add for skew.
# 
# Removed outliers.

# In[ ]:


df['1stFlrSF'].hist()


# In[ ]:


outs = df[df['1stFlrSF'] > 2300].index.tolist()


# In[ ]:


df.drop(outs,inplace=True)


# In[ ]:


test['1stFlrSF'].hist()


# In[ ]:


skw.append('1stFlrSF')


# # 2ndFlrSF
# 
# This should be fine as is.

# In[ ]:


df['2ndFlrSF'].hist()


# In[ ]:


test['2ndFlrSF'].hist()


# # LowQualFinSF
# 
# Dropping this one.

# In[ ]:


df.LowQualFinSF.value_counts()


# In[ ]:


df.drop('LowQualFinSF', axis=1,inplace=True)
test.drop('LowQualFinSF', axis=1,inplace=True)


# # GrLivArea

# In[ ]:


df.GrLivArea.hist()


# In[ ]:


test.GrLivArea.hist()


# In[ ]:


skw.append('GrLivArea')


# # BsmtFullBath
# 
# 3 as 2, outlier may influence model 
# 
# Add to OneHot

# In[ ]:


df.BsmtFullBath.value_counts()


# In[ ]:


test.BsmtFullBath.value_counts()


# In[ ]:


# 3 to 2
df.set_value(df.BsmtFullBath[df.BsmtFullBath == 3].index[0],'BsmtFullBath',2)
test.set_value(test.BsmtFullBath[test.BsmtFullBath == 3].index[0],'BsmtFullBath',2)
print('---')


# In[ ]:


# filling in some empty values
ind = test.BsmtFullBath[test.BsmtFullBath.isnull()].index.tolist()

for i in ind:
    test.set_value(i,'BsmtFullBath',0)


# # BsmtHalfBath
# 
# Moving 2s to 1s.
# 
# Merging half bath/2 with full bath.
# 
# Dropping BsmtHalfBath and BsmtFullBath

# In[ ]:


df.BsmtHalfBath.value_counts()


# In[ ]:


test.BsmtHalfBath.value_counts()


# In[ ]:


# 2 to 1
df.set_value(df.BsmtHalfBath[df.BsmtHalfBath == 2].index[0],'BsmtHalfBath',1)
test.set_value(test.BsmtHalfBath[test.BsmtHalfBath == 2].index[0],'BsmtHalfBath',1)
df.set_value(df.BsmtHalfBath[df.BsmtHalfBath == 2].index[0],'BsmtHalfBath',1)
test.set_value(test.BsmtHalfBath[test.BsmtHalfBath == 2].index[0],'BsmtHalfBath',1)
print('---')


# In[ ]:


# filling in some nulls
ind = test.BsmtHalfBath[test.BsmtHalfBath.isnull()].index.tolist()

for i in ind:
    test.set_value(i,'BsmtHalfBath',0)


# In[ ]:


df['BsmtBath'] = df['BsmtHalfBath']/2 + df['BsmtFullBath']
test['BsmtBath'] = test['BsmtHalfBath']/2 + test['BsmtFullBath']


# In[ ]:


df.drop('BsmtHalfBath',axis=1,inplace=True)
df.drop('BsmtFullBath',axis=1,inplace=True)
test.drop('BsmtHalfBath',axis=1,inplace=True)
test.drop('BsmtFullBath',axis=1,inplace=True)


# # FullBath & HalfBath
# 
# Creating new column Bath.

# In[ ]:


df.FullBath.value_counts()


# In[ ]:


test.FullBath.value_counts()


# In[ ]:


for i in test.index:
    if test.FullBath[i] == 4:
        test.set_value(i,'FullBath',3)    


# In[ ]:


df.HalfBath.value_counts()


# In[ ]:


test.HalfBath.value_counts()


# In[ ]:


# correlations before
df.corr()['FullBath']['SalePrice']


# In[ ]:


df.corr()['HalfBath']['SalePrice']


# In[ ]:


df['Bath'] = df['HalfBath']/2 + df['FullBath']
test['Bath'] = test['HalfBath']/2 + test['FullBath']


# In[ ]:


# correlation after
df.corr()['Bath']['SalePrice']


# # BedroomAbvGr
# 
# 8 beds to 6 beds

# In[ ]:


df.BedroomAbvGr.value_counts()


# In[ ]:


test.BedroomAbvGr.value_counts()


# In[ ]:


df.set_value(df.BedroomAbvGr[df.BedroomAbvGr == 8].index[0],'BedroomAbvGr',6)
print('---')


# # Kitchen
# 
# 0 to 1

# In[ ]:


df.KitchenAbvGr.value_counts()


# In[ ]:


test.KitchenAbvGr.value_counts()


# In[ ]:


df.set_value(df[df.KitchenAbvGr == 0].index[0],'KitchenAbvGr',1)
print('---')


# # KitchenQual

# In[ ]:


df.KitchenQual.value_counts()


# In[ ]:


test.KitchenQual.value_counts()


# In[ ]:


for i in df.index:
    if df.KitchenQual[i] in ['Ex','Gd']:
        df.set_value(i,'KitchenQual',2)
    elif df.KitchenQual[i] == 'TA':
        df.set_value(i,'KitchenQual',1)
    else:
        df.set_value(i,'KitchenQual',0)


# In[ ]:


for i in test.index:
    if test.KitchenQual[i] in ['Ex','Gd']:
        test.set_value(i,'KitchenQual',2)
    elif test.KitchenQual[i] == 'TA':
        test.set_value(i,'KitchenQual',1)
    else:
        test.set_value(i,'KitchenQual',0)


# # TotRmsAbvGrd
# 
# Leaving it as is for now.

# In[ ]:


df.TotRmsAbvGrd.value_counts()


# In[ ]:


test.TotRmsAbvGrd.value_counts()


# In[ ]:


hot.append('TotRmsAbvGrd')


# # Functional
# 
# Categorise:
# - 2 - Maj1, Maj2, Sev, Sal
# - 1 - Min1, Min2, Mod
# - 0 - Typ

# In[ ]:


df.Functional.value_counts()


# In[ ]:


test.Functional.value_counts()


# In[ ]:


for i in df.index:
    if df.Functional[i] == 'Typ':
        df.set_value(i,'Functional',0)
    elif df.Functional[i] in ['Min1','Min2','Mod']:
        df.set_value(i,'Functional',1)
    elif df.Functional[i] in ['Maj1','Maj2','Sal','Sev']:
        df.set_value(i,'Functional',2)


# In[ ]:


for i in test.index:
    if test.Functional[i] == 'Typ':
        test.set_value(i,'Functional',0)
    elif test.Functional[i] in ['Min1','Min2','Mod']:
        test.set_value(i,'Functional',1)
    elif test.Functional[i] in ['Maj1','Maj2','Sal','Sev']:
        test.set_value(i,'Functional',2)


# In[ ]:


# empty values
ind = test.Functional[test.Functional.isnull()].index.tolist()

for i in ind:
    test.set_value(i,'Functional',0)


# In[ ]:


hot.append('Functional')


# # Fireplaces
# 
# Moving 3 and 4 to 2 category
# 
# Add to OneHot

# In[ ]:


df.Fireplaces.value_counts()


# In[ ]:


test.Fireplaces.value_counts()


# In[ ]:


for i in df.index:
    if df.Fireplaces[i] == 3:
        df.set_value(i,'Fireplaces',2)

for i in test.index:
    if test.Fireplaces[i] in [3,4]:
        test.set_value(i,'Fireplaces',2)


# In[ ]:


hot.append('Fireplaces')


# # FireplaceQu

# In[ ]:


df.FireplaceQu.value_counts()


# In[ ]:


test.FireplaceQu.value_counts()


# In[ ]:


for i in df.index:
    if df.FireplaceQu[i] in ['Ex','Gd']:
        df.set_value(i,'FireplaceQu',3)
    elif df.FireplaceQu[i] == 'TA':
        df.set_value(i,'FireplaceQu',2)
    elif df.FireplaceQu[i] in ['Po','Fa']:
        df.set_value(i,'FireplaceQu',1)
    else:
        df.set_value(i,'FireplaceQu',0)


# In[ ]:


for i in test.index:
    if test.FireplaceQu[i] in ['Ex','Gd']:
        test.set_value(i,'FireplaceQu',3)
    elif test.FireplaceQu[i] == 'TA':
        test.set_value(i,'FireplaceQu',2)
    elif test.FireplaceQu[i] in ['Po','Fa']:
        test.set_value(i,'FireplaceQu',1)
    else:
        test.set_value(i,'FireplaceQu',0)


# # GarageType
# 
# Will split into Attached - 1, Detached - 2, no garage - 0

# In[ ]:


df.GarageType.value_counts()


# In[ ]:


test.GarageType.value_counts()


# In[ ]:


for i in df.index:
    if df.GarageType[i] in ['Attchd', 'BuiltIn', 'Basment']:
        df.set_value(i,'GarageType',1)
    elif df.GarageType[i] in ['Detchd', 'CarPort', '2Types']:
        df.set_value(i,'GarageType',2)
    else:
        df.set_value(i,'GarageType',0)


# In[ ]:


for i in test.index:
    if test.GarageType[i] in ['Attchd', 'BuiltIn', 'Basment']:
        test.set_value(i,'GarageType',1)
    elif test.GarageType[i] in ['Detchd', 'CarPort', '2Types']:
        test.set_value(i,'GarageType',2)
    else:
        test.set_value(i,'GarageType',0)


# # GarageYrBlt
# 
# Add for skew, one year in test set is off.

# In[ ]:


df.GarageYrBlt.describe()


# In[ ]:


df.GarageYrBlt.hist(bins=5)


# In[ ]:


test.GarageYrBlt.describe()


# In[ ]:


# assume they meant 2007 instead of 2207
test.set_value(test.GarageYrBlt[test.GarageYrBlt > 2010].index[0],'GarageYrBlt', 2007)
print('---')


# In[ ]:


# empty values as 0
df.GarageYrBlt = df.GarageYrBlt.apply(lambda x: np.nan_to_num(x))
test.GarageYrBlt = test.GarageYrBlt.apply(lambda x: np.nan_to_num(x))


# In[ ]:


# train
for i in df.index:
    if df['GarageYrBlt'][i] <= 1918:
        df.set_value(i,'GarageYrBlt', 0)
    elif df['GarageYrBlt'][i] > 1918 and df['GarageYrBlt'][i] <= 1941:
        df.set_value(i,'GarageYrBlt', 1)
    elif df['GarageYrBlt'][i] > 1941 and df['GarageYrBlt'][i] <= 1964:
        df.set_value(i,'GarageYrBlt', 2)
    elif df['GarageYrBlt'][i] > 1964 and df['GarageYrBlt'][i] <= 1987:
        df.set_value(i,'GarageYrBlt', 3)
    elif df['GarageYrBlt'][i] > 1987:
        df.set_value(i,'GarageYrBlt', 4)


# In[ ]:


# test
for i in test.index:
    if test['GarageYrBlt'][i] <= 1918:
        test.set_value(i,'GarageYrBlt', 0)
    elif test['GarageYrBlt'][i] > 1918 and test['GarageYrBlt'][i] <= 1941:
        test.set_value(i,'GarageYrBlt', 1)
    elif test['GarageYrBlt'][i] > 1941 and test['GarageYrBlt'][i] <= 1964:
        test.set_value(i,'GarageYrBlt', 2)
    elif test['GarageYrBlt'][i] > 1964 and test['GarageYrBlt'][i] <= 1987:
        test.set_value(i,'GarageYrBlt', 3)
    elif test['GarageYrBlt'][i] > 1987:
        test.set_value(i,'GarageYrBlt', 4)


# In[ ]:


hot.append('GarageYrBlt')


# # GarageFinish
# 
# Categorise

# In[ ]:


df.GarageFinish.value_counts()


# In[ ]:


test.GarageFinish.value_counts()


# In[ ]:


for i in df.index:
    if df.GarageFinish[i] == 'Unf':
        df.set_value(i,'GarageFinish',0)
    elif df.GarageFinish[i] == 'RFn':
        df.set_value(i,'GarageFinish',1)
    elif df.GarageFinish[i] == 'Fin':
        df.set_value(i,'GarageFinish',2)
    else:
        df.set_value(i,'GarageFinish',3)


# In[ ]:


for i in test.index:
    if test.GarageFinish[i] == 'Unf':
        test.set_value(i,'GarageFinish',0)
    elif test.GarageFinish[i] == 'RFn':
        test.set_value(i,'GarageFinish',1)
    elif test.GarageFinish[i] == 'Fin':
        test.set_value(i,'GarageFinish',2)
    else:
        test.set_value(i,'GarageFinish',3)


# In[ ]:


hot.append('GarageFinish')


# # GarageCars
# 
# A bit of adjusting.

# In[ ]:


df.GarageCars.value_counts()


# In[ ]:


test.GarageCars.value_counts()


# In[ ]:


# moving from 5 to 4
test.set_value(test.GarageCars[test.GarageCars == 5].index[0],'GarageCars',4)
print('---')


# In[ ]:


# filling empty column
test.set_value(test.GarageCars[test.GarageCars.isnull()].index[0],'GarageCars',0)
print('---')


# In[ ]:


hot.append('GarageCars')


# # GarageArea

# In[ ]:


df.GarageArea.hist()


# In[ ]:


test.GarageArea.hist()


# In[ ]:


test.set_value(test.GarageArea[test.GarageArea.isnull()].index[0],'GarageArea',0)
print('---')


# # GarageQual
# 
# Categorising:
# - 0 - No garage
# - 1 - Fa, Po
# - 2 - TA
# - 3 - Gd, Ex

# In[ ]:


df.GarageQual.value_counts()


# In[ ]:


test.GarageQual.value_counts()


# In[ ]:


for i in df.index:
    if df.GarageQual[i] in ['Gd', 'Ex']:
        df.set_value(i,'GarageQual',3)
    elif df.GarageQual[i] == 'TA':
        df.set_value(i,'GarageQual',2)
    elif df.GarageQual[i] in ['Fa', 'Po']:
        df.set_value(i,'GarageQual',1)
    else:
        df.set_value(i,'GarageQual',0)


# In[ ]:


for i in test.index:
    if test.GarageQual[i] in ['Gd', 'Ex']:
        test.set_value(i,'GarageQual',3)
    elif test.GarageQual[i] == 'TA':
        test.set_value(i,'GarageQual',2)
    elif test.GarageQual[i] in ['Fa', 'Po']:
        test.set_value(i,'GarageQual',1)
    else:
        test.set_value(i,'GarageQual',0)


# In[ ]:


hot.append('GarageQual')


# # GarageCond
# 
# Same as above

# In[ ]:


df.GarageCond.value_counts()


# In[ ]:


test.GarageCond.value_counts()


# In[ ]:


# train
for i in df.index:
    if df.GarageCond[i] in ['Gd', 'Ex']:
        df.set_value(i,'GarageCond',3)
    elif df.GarageCond[i] == 'TA':
        df.set_value(i,'GarageCond',2)
    elif df.GarageCond[i] in ['Fa', 'Po']:
        df.set_value(i,'GarageCond',1)
    else:
        df.set_value(i,'GarageCond',0)
        
# test
for i in test.index:
    if test.GarageCond[i] in ['Gd', 'Ex']:
        test.set_value(i,'GarageCond',3)
    elif test.GarageCond[i] == 'TA':
        test.set_value(i,'GarageCond',2)
    elif test.GarageCond[i] in ['Fa', 'Po']:
        test.set_value(i,'GarageCond',1)
    else:
        test.set_value(i,'GarageCond',0)


# In[ ]:


hot.append('GarageCond')


# # PavedDrive
# 1 - Yes, 0 - No

# In[ ]:


df.PavedDrive.value_counts()


# In[ ]:


test.PavedDrive.value_counts()


# In[ ]:


# train
for i in df.index:
    if df.PavedDrive[i] == 'Y':
        df.set_value(i,'PavedDrive',1)
    else:
        df.set_value(i,'PavedDrive',0)
        
# test
for i in test.index:
    if test.PavedDrive[i] == 'Y':
        test.set_value(i,'PavedDrive',1)
    else:
        test.set_value(i,'PavedDrive',0)


# # OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, WoodDeckSF
# 
# Join those together as follows:
# - For each row, calculate the total sum of the above and put it in a new column, PorchSF
# - For each row, calculate the maxium value out of the columns selected, except for WoodDeckSF (correclation is much better without it), and put in under a new column, PorchType
# 
# Binarise OpenPorchSF, EnclosedPorch

# In[ ]:


# WoodDeckSF
# train
wood_tr = {'id':[]}
for i in df.index:
    if df.WoodDeckSF[i] > 0:
        wood_tr['id'].append(i)
        
wood_tr['class'] = 5
wood_tr['name'] = 'WoodDeckSF'

# test
wood_ts = {'id':[]}
for i in df.index:
    if df.WoodDeckSF[i] > 0:
        wood_ts['id'].append(i)
        
wood_ts['class'] = 5
wood_ts['name'] = 'WoodDeckSF'


# In[ ]:


# OpenPorchSF
# train
open_tr = {'id':[]}
for i in df.index:
    if df.OpenPorchSF[i] > 0:
        open_tr['id'].append(i)

open_tr['class'] = 0
open_tr['name'] = 'OpenPorchSF'

# test
open_ts = {'id':[]}
for i in test.index:
    if test.OpenPorchSF[i] > 0:
        open_ts['id'].append(i)

open_ts['class'] = 0
open_ts['name'] = 'OpenPorchSF'


# In[ ]:


# EnclosedPorch
# train
encl_tr = {'id':[]}
for i in df.index:
    if df.EnclosedPorch[i] > 0:
        encl_tr['id'].append(i)
        
encl_tr['class'] = 1
encl_tr['name'] = 'EnclosedPorch'
        
# test
encl_ts = {'id':[]}
for i in test.index:
    if test.EnclosedPorch[i] > 0:
        encl_ts['id'].append(i)
        
encl_ts['class'] = 1
encl_ts['name'] = 'EnclosedPorch'


# In[ ]:


# 3SsnPorch
# train
sn3_tr = {'id':[]}
for i in df.index:
    if df['3SsnPorch'][i] > 0:
        sn3_tr['id'].append(i)
        
sn3_tr['class'] = 2
sn3_tr['name'] = '3SsnPorch'
        
# test
sn3_ts = {'id':[]}
for i in test.index:
    if test['3SsnPorch'][i] > 0:
        sn3_ts['id'].append(i)
        
sn3_ts['class'] = 2
sn3_ts['name'] = '3SsnPorch'


# In[ ]:


# ScreenPorch
# train
scp_tr = {'id':[]}
for i in df.index:
    if df['ScreenPorch'][i] > 0:
        scp_tr['id'].append(i)
        
scp_tr['class'] = 3
scp_tr['name'] = 'ScreenPorch'
        
# test
scp_ts = {'id':[]}
for i in test.index:
    if test['ScreenPorch'][i] > 0:
        scp_ts['id'].append(i)

scp_ts['class'] = 3
scp_ts['name'] = 'ScreenPorch'


# In[ ]:


df['PorchSF'] = 0
df['PorchType'] = 4


# In[ ]:


k = [open_tr,encl_tr,sn3_tr,scp_tr,wood_tr]
for i in df.index:
    f = [0] * 5
    v_max = 0
    name = ''
    s = 0
    if i in open_tr['id']:
        f[0] = 1
    if i in encl_tr['id']:
        f[1] = 1
    if i in sn3_tr['id']:
        f[2] = 1
    if i in scp_tr['id']:
        f[3] = 1
    if i in wood_tr['id']:
        f[4] = 1
    if sum(f) > 1:
        for j in range(0,4):
            if f[j] != 0 and df[k[j]['name']][i] > v_max:
                v_max = df[k[j]['name']][i]
                name = k[j]['class']
        for j in range(0,5):      
            s += df[k[j]['name']][i]
        df.set_value(i,'PorchSF',s)
        df.set_value(i,'PorchType',name)
    else:
        for w,z in enumerate(f):
            if f[w] == 1:
                df.set_value(i,'PorchSF',df[k[w]['name']][i])
                df.set_value(i,'PorchType',k[w]['class'])


# In[ ]:


test['PorchSF'] = 0
test['PorchType'] = 4


# In[ ]:


k = [open_ts,encl_ts,sn3_ts,scp_ts,wood_ts]
for i in test.index:
    f = [0] * 5
    v_max = 0
    name = ''
    s = 0
    if i in open_ts['id']:
        f[0] = 1
    if i in encl_ts['id']:
        f[1] = 1
    if i in sn3_ts['id']:
        f[2] = 1
    if i in scp_ts['id']:
        f[3] = 1
    if i in wood_ts['id']:
        f[4] = 1
    if sum(f) > 1:
        for j in range(0,4):
            if f[j] != 0 and test[k[j]['name']][i] > v_max:
                v_max = test[k[j]['name']][i]
                name = k[j]['class']
        for j in range(0,5):      
            s += test[k[j]['name']][i]
        test.set_value(i,'PorchSF',s)
        test.set_value(i,'PorchType',name)
    else:
        for w,z in enumerate(f):
            if f[w] == 1:
                test.set_value(i,'PorchSF',test[k[w]['name']][i])
                test.set_value(i,'PorchType',k[w]['class'])


# In[ ]:


df.corr()['PorchSF']['SalePrice']


# In[ ]:


df.corr()['PorchType']['SalePrice']


# In[ ]:


df.OpenPorchSF = df.OpenPorchSF.apply(lambda x: 0 if x == 0 else 1)
df.EnclosedPorch = df.EnclosedPorch.apply(lambda x: 0 if x == 0 else 1)
test.OpenPorchSF = df.OpenPorchSF.apply(lambda x: 0 if x == 0 else 1)
test.EnclosedPorch = df.EnclosedPorch.apply(lambda x: 0 if x == 0 else 1)


# In[ ]:


# some empty ones I missed
ind = test.OpenPorchSF[test.OpenPorchSF.isnull()].index.tolist()
for i in ind:
    test.set_value(i,'OpenPorchSF',0)


# In[ ]:


ind = test.EnclosedPorch[test.EnclosedPorch.isnull()].index.tolist()
for i in ind:
    test.set_value(i,'EnclosedPorch',0)


# In[ ]:


hot.append('PorchType')


# # PoolArea, PoolQC
# 
# Dropping those two.

# In[ ]:


df.PoolArea.value_counts()


# In[ ]:


df.PoolQC.value_counts()


# In[ ]:


df.drop(['PoolArea','PoolQC'],axis=1,inplace=True)
test.drop(['PoolArea','PoolQC'],axis=1,inplace=True)


# # Fence
# 1 - has fence, 0 - no fence

# In[ ]:


df.Fence.value_counts()


# In[ ]:


test.Fence.value_counts()


# In[ ]:


df.Fence = df.Fence.apply(lambda x: 1 if x in ['GdPrv','MnPrv','GdWo','MnWw'] else 0)
test.Fence = test.Fence.apply(lambda x: 1 if x in ['GdPrv','MnPrv','GdWo','MnWw'] else 0)


# In[ ]:


test.Fence.value_counts()


# # MiscFeature
# 
# Dropping it.

# In[ ]:


df.MiscFeature.value_counts()


# In[ ]:


test.MiscFeature.value_counts()


# In[ ]:


df.drop('MiscFeature',axis=1,inplace=True)
test.drop('MiscFeature',axis=1,inplace=True)


# # MiscVal
# Dropping it.

# In[ ]:


df.MiscVal.hist()


# In[ ]:


df.drop('MiscVal',axis=1,inplace=True)
test.drop('MiscVal',axis=1,inplace=True)


# # MoSold
# 
# Leave as is, apply OneHot.

# In[ ]:


df.MoSold.value_counts()


# In[ ]:


df.MoSold.hist()


# In[ ]:


hot.append('MoSold')


# # YrSold
# 
# Encode.

# In[ ]:


df.YrSold.value_counts()


# In[ ]:


test.YrSold.value_counts()


# In[ ]:


encoder = LabelEncoder()


# In[ ]:


df.YrSold = encoder.fit_transform(df.YrSold)


# In[ ]:


test.YrSold = encoder.transform(test.YrSold)


# # SaleType
# 
# Categorising:
# - 0 - WD, CWD, VWD
# - 1 - New
# - 2 - COD
# - 3 - Con, ConLw, ConLI, ConLD, Oth

# In[ ]:


df.SaleType.value_counts()


# In[ ]:


for i in df.index:
    if df.SaleType[i] in ['WD','CWD','VWD']:
        df.set_value(i,'SaleType',0)
    elif df.SaleType[i] in ['Con','ConLw','ConLI','ConLD','Oth']:
        df.set_value(i,'SaleType',1)
    elif df.SaleType[i] == 'New':
        df.set_value(i,'SaleType',2)
    elif df.SaleType[i] == 'COD':
        df.set_value(i,'SaleType',3)


# In[ ]:


for i in test.index:
    if test.SaleType[i] in ['WD','CWD','VWD']:
        test.set_value(i,'SaleType',0)
    elif test.SaleType[i] in ['Con','ConLw','ConLI','ConLD','Oth']:
        test.set_value(i,'SaleType',1)
    elif test.SaleType[i] == 'New':
        test.set_value(i,'SaleType',2)
    elif test.SaleType[i] == 'COD':
        test.set_value(i,'SaleType',3)


# In[ ]:


# missed one row
test.set_value(test.SaleType[test.SaleType.isnull()].index[0],'SaleType',0)
print('---')


# In[ ]:


hot.append('SaleType')


# # SaleCondition
# 
# Encode

# In[ ]:


df.SaleCondition.value_counts()


# In[ ]:


test.SaleCondition.value_counts()


# In[ ]:


encoder = LabelEncoder()


# In[ ]:


df.SaleCondition = encoder.fit_transform(df.SaleCondition)


# In[ ]:


test.SaleCondition = encoder.transform(test.SaleCondition)


# # All to numerical

# In[ ]:


df = df.apply(pd.to_numeric)
test = test.apply(pd.to_numeric)


# # LotFrontage
# 
# Use Linear Regression to predict values

# In[ ]:


# use values mostly correlated with LotFrontage for X
X = df[~df.LotFrontage.isnull()][['LotArea','BldgType','1stFlrSF','GrLivArea','TotRmsAbvGrd']].values
y = df[~df.LotFrontage.isnull()]['LotFrontage'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train,y_train)
prd = regr.predict(X_test)
print('Roor Mean squared error: ',mean_squared_error(y_test,prd)**0.5)

to_pred = df[df.LotFrontage.isnull()][['LotArea','BldgType','1stFlrSF','GrLivArea','TotRmsAbvGrd']].values
predictions = regr.predict(to_pred)
idx = df[df.LotFrontage.isnull()].index.tolist()

x = 0
for i in idx:
    df.set_value(i,'LotFrontage',round(predictions[x]))
    x += 1


# In[ ]:


# same for test
X = test[~test.LotFrontage.isnull()][['LotArea','BldgType','1stFlrSF','GrLivArea','TotRmsAbvGrd']].values
y = test[~test.LotFrontage.isnull()]['LotFrontage'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train,y_train)
prd = regr.predict(X_test)
print('Root Mean squared error: ',mean_squared_error(y_test,prd)**0.5)

to_pred = test[test.LotFrontage.isnull()][['LotArea','BldgType','1stFlrSF','GrLivArea','TotRmsAbvGrd']].values
predictions = regr.predict(to_pred)
idx = test[test.LotFrontage.isnull()].index.tolist()

x = 0
for i in idx:
    test.set_value(i,'LotFrontage',round(predictions[x]))
    x += 1


# # Id
# Dropping it.

# In[ ]:


df.drop('Id',axis=1,inplace=True)
test.drop('Id',axis=1,inplace=True)


# In[ ]:


test_c = test.copy()
df_c = df.copy()


# # Normalising data

# In[ ]:


from scipy.stats import skew


# In[ ]:


skw


# # LotArea

# In[ ]:


print('Train: ',skew(df.LotArea))
print('Test: ',skew(test.LotArea))


# In[ ]:


(np.log1p(df.LotArea)).hist()
print('New skew: ',skew(np.log1p(df.LotArea)))


# In[ ]:


(np.log1p(test.LotArea)).hist()
print('New skew: ',skew(np.log1p(test.LotArea)))


# In[ ]:


df.LotArea = np.log1p(df.LotArea)
test.LotArea = np.log1p(test.LotArea)


# # MasVnrArea

# In[ ]:


print('Train: ',skew(df.MasVnrArea))
print('Test: ',skew(test.MasVnrArea))


# In[ ]:


(np.log1p(df.MasVnrArea)).hist()
print('New skew: ',skew(np.log1p(df.MasVnrArea)))


# In[ ]:


(np.log1p(test.MasVnrArea)).hist()
print('New skew: ',skew(np.log1p(test.MasVnrArea)))


# In[ ]:


df.MasVnrArea = np.log1p(df.MasVnrArea)
test.MasVnrArea = np.log1p(test.MasVnrArea)


# # BsmtFinSF1
# Correlation would go down, not pursuing it.

# In[ ]:


print('Train: ',skew(df.BsmtFinSF1))
print('Test: ',skew(test.BsmtFinSF1))


# In[ ]:


(np.sqrt(df.BsmtFinSF1)).hist()
print('New skew: ',skew(np.sqrt(df.BsmtFinSF1)))


# In[ ]:


(np.sqrt(test.BsmtFinSF1)).hist()
print('New skew: ',skew(np.sqrt(test.BsmtFinSF1)))


# # BsmtFinSF2

# In[ ]:


print('Train: ',skew(df.BsmtFinSF2))
print('Test: ',skew(test.BsmtFinSF2))


# In[ ]:


(np.log1p(df.BsmtFinSF2)).hist()
print('New skew: ',skew(np.log1p(df.BsmtFinSF2)))


# In[ ]:


(np.log1p(test.BsmtFinSF2)).hist()
print('New skew: ',skew(np.log1p(test.BsmtFinSF2)))


# In[ ]:


df.BsmtFinSF2 = np.log1p(df.BsmtFinSF2)
test.BmstFinSF2 = np.log1p(test.BsmtFinSF2)


# # BsmtUnfSF

# In[ ]:


print('Train: ',skew(df.BsmtUnfSF))
print('Test: ',skew(test.BsmtUnfSF))


# In[ ]:


(np.sqrt(df.BsmtUnfSF)).hist()
print('New skew: ',skew(np.sqrt(df.BsmtUnfSF)))


# In[ ]:


(np.sqrt(test.BsmtUnfSF)).hist()
print('New skew: ',skew(np.sqrt(test.BsmtUnfSF)))


# In[ ]:


df.BsmtUnfSF = np.log1p(df.BsmtUnfSF)
test.BsmtUnfSF = np.log1p(test.BsmtUnfSF)


# # 1stFlrSF

# In[ ]:


print('Train: ',skew(df['1stFlrSF']))
print('Test: ',skew(test['1stFlrSF']))


# In[ ]:


(np.sqrt(df['1stFlrSF'])).hist()
print('New skew: ',skew(np.sqrt(df['1stFlrSF'])))


# In[ ]:


(np.sqrt(test['1stFlrSF'])).hist()
print('New skew: ',skew(np.sqrt(test['1stFlrSF'])))


# In[ ]:


df['1stFlrSF'] = np.sqrt(df['1stFlrSF'])
test['1stFlrSF'] = np.sqrt(test['1stFlrSF'])


# # GrLivArea

# In[ ]:


print('Train: ',skew(df.GrLivArea))
print('Test: ',skew(test.GrLivArea))


# In[ ]:


(np.sqrt(df.GrLivArea)).hist()
print('New skew: ',skew(np.sqrt(df.GrLivArea)))


# In[ ]:


(np.sqrt(test.GrLivArea)).hist()
print('New skew: ',skew(np.sqrt(test.GrLivArea)))


# In[ ]:


df.GrLivArea = np.sqrt(df.GrLivArea)
test.GrLivArea = np.sqrt(test.GrLivArea)


# # Adding additional highly correlated columns.

# In[ ]:


add_f = df.corr()['SalePrice'].sort_values(ascending=False).head(11).keys().tolist()[1:]


# In[ ]:


# for j,i in enumerate(add_f):
#     df[i+'_2'] = df[i] ** 2
#     df[i+'_3'] = df[i] ** 3
#     df[i+'_sqrt'] = df[i] ** 0.5


# In[ ]:


# for j,i in enumerate(add_f):
#     test[i+'_2'] = test[i] ** 2
#     test[i+'_3'] = test[i] ** 3
#     test[i+'_sqrt'] = test[i] ** 0.5


# # OneHotEncoder
# Will try first without and then with this applied. Writting the code so it's here when needed.

# In[ ]:


df_c = df.copy()
df_c.drop('SalePrice',axis=1,inplace=True)


# In[ ]:


cols = []
for i,j in enumerate(df_c.columns.tolist()):
    cols.append((i,j))


# In[ ]:


for i,w in enumerate(hot):
    for j,k in cols:
        if w == k:
            hot[i] = (j,w)           


# In[ ]:


# arranging them in order
gh = hot[0]
for i in range(1,5):
    hot[i-1] = hot[i] 
hot[4] = gh


# In[ ]:


ar = hot[-3]
for i in range(19,21):
    hot[i-1] = hot[i]
hot[-1] = ar


# In[ ]:


X = df_c.values


# In[ ]:


shpr = X.shape[1]


# In[ ]:


# training set
for i,j in hot:
    shp = X.shape[1] - shpr
    hot_encoder = OneHotEncoder(categorical_features=[i+shp])
    X = hot_encoder.fit_transform(X).toarray()
    X = X[:,1:]


# In[ ]:


tst = test.copy()


# In[ ]:


tst = test.values


# In[ ]:


# test prediction set
shpr = tst.shape[1]
for i,j in hot:
    shp = tst.shape[1] - shpr
    hot_encoder = OneHotEncoder(categorical_features=[i+shp])
    tst = hot_encoder.fit_transform(tst).toarray()
    tst = tst[:,1:]


# # Feature scalling

# In[ ]:


y = df.SalePrice.values


# In[ ]:


# splitting the train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[ ]:


# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
tst = sc_X.transform(tst)


# In[ ]:


from sklearn.decomposition import PCA
# leave it as None initially to explore the variance first, then change to the choosen number from explained_variance
pca = PCA(n_components=125)

# fitting and transforming the training set and transforming the test set
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
tst = pca.transform(tst)

# cumulated explained vairance of the principal components
explained_variance = pca.explained_variance_ratio_
explained_variance


# In[ ]:


sum(explained_variance.tolist()[0:125])


# In[ ]:


len(explained_variance)


# In[ ]:


# importing libraries

from sklearn.linear_model import ElasticNet as EN
from sklearn.linear_model import Lasso as LS

# cross validation

algorithms = []

algorithms.append(('XGB', xgb.XGBRegressor()))
algorithms.append(('ElasticNet', EN()))
algorithms.append(('Lasso', LS()))

results = []
names = []
scoring = 'r2'

for name, model in algorithms:
    cv_results = cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f (%f)' %(name, cv_results.mean(), cv_results.std())
    print(msg)
    
fig = plt.figure(figsize=(22,5))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)


# In[ ]:


regr = xgb.XGBRegressor()


# In[ ]:


regr.fit(X_train,y_train)


# In[ ]:


pred = regr.predict(X_test)


# In[ ]:


mean_squared_error(y_test,pred)**0.5


# In[ ]:


pred = regr.predict(tst)


# In[ ]:


regressor = LS(alpha=0.001, max_iter=50000)


# In[ ]:


regressor.fit(X_train,y_train)


# In[ ]:


prediction = regressor.predict(X_test)


# In[ ]:


mean_squared_error(y_test,prediction)**0.5


# In[ ]:


predi = regressor.predict(tst)


# In[ ]:


pp = (pred + predi)/2


# In[ ]:


# # exporting the final result
# import csv
# with open('output.csv','w') as resultFile:
#      wr = csv.writer(resultFile, dialect='excel')
#      wr.writerow(pp)


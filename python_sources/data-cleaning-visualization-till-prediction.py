#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<h1>BASIC PROJECT DATA CLEANING, VISUALIZATION TILL PREDICTION</h1>\n<h3>Created by : Yonela Nuba using several kennels</h3>')


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '\n<h3>Links to associated Kernels </h3>')


# **Link One:**
# [https://www.kaggle.com/stephaniestallworth/housing-feature-engineering-regression](https://www.kaggle.com/stephaniestallworth/housing-feature-engineering-regression) <br />
# **Link Two:** 

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<h1> Kernel still under Construction </h1>')


# In[ ]:





# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import missingno as msno
sns.set_style('darkgrid')
color = sns.color_palette()
import matplotlib.mlab as mlab
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train = train_data.copy()
test = test_data.copy()


# In[ ]:


train.select_dtypes(include = [np.number]).columns


# In[ ]:


train.select_dtypes(include = [np.object]).columns


# In[ ]:


numerical_features = train.dtypes[train.dtypes != 'object'].index
print('Total Numerical Features: {}'.format(len(numerical_features)))

categorical_features = train.dtypes[train.dtypes == 'object'].index
print('Total Categorical features: {}'.format(len(categorical_features)))

print('Total number of features: {}'.format(len(numerical_features) + len(categorical_features)))


# In[ ]:


train.describe()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<h1>Lets see how are the features correlated to the target value </h1>')


# In[ ]:


correlation = train.select_dtypes(include = [np.number]).corr()
print(correlation['SalePrice'].sort_values(ascending = False))


# In[ ]:


f , ax = plt.subplots(figsize = (14,12))
plt.title('Correlation of numeric features',size=15)
sns.heatmap(correlation,square = True,  vmax=0.8, cmap = 'viridis')


# In[ ]:


highCorrelation = correlation.loc[['SalePrice','GrLivArea','TotalBsmtSF','OverallQual',
                                     'FullBath','TotRmsAbvGrd','YearBuilt','1stFlrSF','GarageYrBlt',
                                     'GarageCars','GarageArea'],
                                    ['SalePrice','GrLivArea','TotalBsmtSF','OverallQual','FullBath',
                                     'TotRmsAbvGrd','YearBuilt','1stFlrSF','GarageYrBlt','GarageCars','GarageArea']]
f , ax = plt.subplots(figsize = (14,12))
plt.title('Correlation of numeric features',size=15)
sns.heatmap(highCorrelation, square = True, linewidths=0.01, vmax=0.8, annot=True,cmap='viridis', linecolor="black", annot_kws = {'size':12})


# In[ ]:


sns.set()
cols = ['SalePrice','GrLivArea','TotalBsmtSF','OverallQual','FullBath',
        'TotRmsAbvGrd','YearBuilt','1stFlrSF','GarageYrBlt','GarageCars','GarageArea']
sns.pairplot(train[cols], kind = 'scatter', size = 2, diag_kind='kde')
plt.show()


# In[ ]:





# In[ ]:


plt.figure(figsize = (7,5))
plt.scatter(x = train.TotalBsmtSF, y = train.SalePrice)
plt.title('TotalBsmtSF', size = 15)

plt.figure(figsize = (7,5))
plt.scatter(x = train['1stFlrSF'], y = train.SalePrice)
plt.title('1stFlrSF', size = 15)

plt.figure(figsize = (7,5))
plt.scatter(x = train.GrLivArea, y = train.SalePrice)
plt.title('GrLivArea', size = 15)


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<h1> Removing outliers </h1>')


# In[ ]:


train.drop(train[train['TotalBsmtSF'] > 5000].index, inplace = True)
train.drop(train[train['1stFlrSF'] > 4000].index, inplace = True)
train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 3000)].index, inplace = True)


# In[ ]:


train.drop('Id', axis = 1, inplace = True)


# In[ ]:


y_train = train.SalePrice


# In[ ]:


train.shape, y_train.shape, test.shape


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<h1> Visualising missing values </h1>')


# In[ ]:


def missing_values(df):
    total = df.isnull().sum() # Total Missing values
    percent = 100 * total/len(df) # The percentage of missing values
    missingValue_table = pd.concat([total, percent], axis = 1)
    ren_table = missingValue_table.rename(columns = {0: 'Total Missing values', 1: '% of missing values'})
    ren_table = ren_table[ren_table.iloc[:,1]!=0].sort_values('% of missing values', ascending = False).round(2)
    
    print('Your dataset contains: {}'.format(df.shape[1]) + ' columns and there are: {}'.format(ren_table.shape[0]) + ' Columns that contains missing values')
    
    return ren_table


# In[ ]:


#Visualising numerical missing values in all numerical columns
msno.matrix(train.select_dtypes(include = [np.number]).sample(200))


# In[ ]:


#Visualising categorical missing values in categorical features
msno.matrix(train.select_dtypes(include = [np.object]).sample(200))


# In[ ]:


missing_values(train.select_dtypes(include = [np.number]))


# In[ ]:


missing_values(train.select_dtypes(include = [np.object]))


# In[ ]:


#Lets see a bar graph of missing values in the entire dataframe
msno.bar((train.select_dtypes(include = [np.number]).sample(1000)))


# In[ ]:


msno.bar((train.select_dtypes(include = [np.object]).sample(1000)))


# In[ ]:


msno.bar(train.sample(1000))


# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<h1> Lets Deal with Null values </h1>\n\n<p> Some Columns in the training data with null values actually has no null values but the null value means that, that feature is not present in that house </p>\n<p> Columns such as: <h3>'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 'Electrical',<br />\n                    'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2',<br />\n                     'MSZoning', 'Utilities'</h3> </p>\n            \n<p>These columns actually have a meeaning and we will have to fill the nulls with 'none' </p>")


# In[ ]:


cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish',
        'GarageType', 'Electrical','KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st', 'BsmtExposure','BsmtCond',
        'BsmtQual', 'BsmtFinType1', 'BsmtFinType2','MSZoning', 'Utilities']

for col in cols:
    train[col].fillna('None', inplace = True)
    test[col].fillna('None', inplace = True)


# In[ ]:


missing_values(train.select_dtypes(include = [np.object]))


# In[ ]:


missing_values(train.select_dtypes(include = [np.number]))


# In[ ]:


train.fillna(train.mean(), inplace = True)
test.fillna(test.mean(), inplace = True)


# In[ ]:


missing_values(train)


# In[ ]:


train.head()


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<h1>LETS GENERATE OUR OWN FEATURES FROM THE EXISTING FEATURES </h1>')


# In[ ]:


train['FullHouseSF'] = train['1stFlrSF'] + train['2ndFlrSF'] + train['TotalBsmtSF']
test['FullHouseSF'] = test['1stFlrSF'] + test['2ndFlrSF'] + test['TotalBsmtSF']
train['PorchSF'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']
test['PorchSF'] = test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch']
train['TotalSF'] = train['FullHouseSF'] + train['PorchSF'] + train['GarageArea']
test['TotalSF'] = test['FullHouseSF'] + test['PorchSF'] + test['GarageArea']

test.shape, train.shape


# In[ ]:


train.head()


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<h2> Now lets deal with categorical features </h2>')


# In[ ]:


catcols = train.select_dtypes(['object'])
for cat in catcols:
    print(cat)
    print(train[cat].value_counts())
    print('--'*20)


# In[ ]:


train['MSZoning'] = train['MSZoning'].map({'RL':0, 'RM':1, 'FV':2, 'RH':3, 'C (all)':4})
train['Street'] = train['Street'].map({'Pave':0, 'Grvl':1})
train['Alley'] = train['Alley'].map({'None':0, 'Grvl':1, 'Pave':2})
train['LotShape'] = train['LotShape'].map({'Reg':0, 'IR1':1, 'IR2':2, 'IR3':3})
train['LandContour'] = train['LandContour'].map({'Lvl':0, 'Bnk':1, 'HLS':2, 'Low':3})
train['Utilities'] = train['Utilities'].map({'AllPub':0, 'NoSeWa':1})
train['LotConfig'] = train['LotConfig'].map({'Inside':0, 'Corner':1, 'CulDSac':2, 'FR2':3, 'FR3':4})
train['LandSlope'] = train['LandSlope'].map({'Gtl':0, 'Mod':1, 'Sev':2})
train['Neighborhood'] = train['Neighborhood'].map({'NAmes':0, 'CollgCr':1, 'OldTown':2, 'Edwards':3, 'Somerst':4, 'Gilbert':5,
                                                  'NridgHt':6, 'Sawyer':7, 'NWAmes':8, 'SawyerW':9, 'BrkSide':10, 'Crawfor':11,
                                                  'Mitchel':12, 'NoRidge':13, 'Timber':14, 'IDOTRR':15, 'ClearCr':16, 'SWISU':17,
                                                  'StoneBr':18, 'Blmngtn':19, 'MeadowV':20, 'BrDale':21, 'Veenker':22, 'NPkVill':23, 'Blueste':24})
train['Condition1'] = train['Condition1'].map({'Norm':0, 'Feedr':1, 'Artery':2, 'RRAn':3, 'PosN':4, 'RRAe':5, 'PosA':6,
                                              'RRNn':7, 'RRNe':8})
train['Condition2'] = train['Condition2'].map({'Norm':0, 'Feedr':1, 'Artery':2, 'RRNn':3, 'PosN':4, 'RRAe':5, 'PosA':6, 'RRAn':7})
train['BldgType'] = train['BldgType'].map({'1Fam':0, 'TwnhsE':1, 'Duplex':2, 'Twnhs':3, '2fmCon':4})
train['HouseStyle'] = train['HouseStyle'].map({'1Story':0, '2Story':1, '1.5Fin':2, 'SLvl':3, 'SFoyer':4, '1.5Unf':5,
                                              '2.5Unf':6, '2.5Fin':7})
train['RoofStyle'] = train['RoofStyle'].map({'Gable':0, 'Hip':1, 'Flat':2, 'Gambrel':3, 'Mansard':4, 'Shed':5})
train['RoofMatl'] = train['RoofMatl'].map({'CompShg':0, 'Tar&Grv':1, 'WdShngl':2, 'WdShake':3, 'Roll':4, 'Membran':5, 'Metal':6})
train['Exterior1st'] = train['Exterior1st'].map({'VinylSd':0, 'HdBoard':1, 'MetalSd':2, 'Wd Sdng':3, 'Plywood':4, 'CemntBd':5, 'BrkFace':6, 'WdShing':7,
                                                'Stucco':8, 'AsbShng':9, 'Stone':10, 'BrkComm':11, 'CBlock':12, 'ImStucc':13, 'AsphShn':14})
train['Exterior2nd'] = train['Exterior2nd'].map({'VinylSd':0, 'MetalSd':1, 'HdBoard':2, 'Wd Sdng':3, 'Plwood':4, 'CmentBd':5, 'Wd Shng':6, 'Stucco':7,
                                                'BrkFace':8, 'AsbShng':9, 'ImStucc':10, 'Brk Cmn':11, 'Stone':12, 'AsphShn':13, 'Other':14, 'CBlock':15})
train['MasVnrType'] = train['MasVnrType'].map({'None':0, 'BrkFace':1, 'Stone':2, 'BrkCmn':3})
train['ExterQual'] = train['ExterQual'].map({'TA':0, 'Gd':1, 'Ex':2, 'Fa':3, 'Po':4})
train['ExterCond'] = train['ExterCond'].map({'TA':0, 'Gd':1, 'Fa':3, 'Ex':2, 'Po':4})
train['Foundation'] = train['Foundation'].map({'PConc':0, 'CBlock':1, 'BrkTil':2, 'Slab':3, 'Stone':4, 'Wood':5})
train['BsmtQual'] = train['BsmtQual'].map({'TA':0, 'Gd':1, 'Ex':2, 'None':3, 'Fa':4})
train['BsmtCond'] = train['BsmtCond'].map({'TA':0, 'Gd':1, 'Fa':2, 'None':3, 'Po':4})
train['BsmtExposure'] = train['BsmtExposure'].map({'No':0, 'Av':1, 'Gd':2, 'Mn':3, 'None':4})
train['BsmtFinType1'] = train['BsmtFinType1'].map({'Unf':0, 'GLQ':1, 'ALQ':2, 'BLQ':3, 'Rec':4, 'LwQ':5, 'None':6})
train['BsmtFinType2'] = train['BsmtFinType2'].map({'Unf':0, 'Rec':1, 'LwQ':2, 'None':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
train['Heating'] = train['Heating'].map({'GasA':0, 'GasW':1, 'Grav':2, 'Wall':3, 'OthW':4, 'Floor':5})
train['HeatingQC'] = train['HeatingQC'].map({'Ex':0, 'TA':1, 'Gd':2, 'Fa':3, 'Po':4})
train['CentralAir'] = train['CentralAir'].map({'Y':1, 'N':0})
train['Electrical'] = train['Electrical'].map({'SBrkr':0, 'FuseA':1, 'FuseF':2, 'FuseP':3, 'Mix':4, 'None':5})
train['KitchenQual'] = train['KitchenQual'].map({'TA':0, 'Gd':1, 'Ex':2, 'Fa':3})
train['Functional'] = train['Functional'].map({'Typ':0, 'Min2':1, 'Min1':2, 'Mod':3, 'Maj1':4, 'Maj2':5, 'Sev':6})
train['FireplaceQu'] = train['FireplaceQu'].map({'None':0, 'Gd':1, 'TA':2, 'Fa':3, 'Ex':4, 'Po':5})
train['GarageType'] = train['GarageType'].map({'Attchd':0, 'Detchd':1, 'BuiltIn':2, 'None':3, 'Basement':4, 'CarPort':5, '2Types':6})
train['GarageFinish'] = train['GarageFinish'].map({'Unf':0, 'RFn':1, 'Fin':2, 'None':3})
train['GarageQual'] = train['GarageQual'].map({'TA':0, 'None':1, 'Fa':2, 'Gd':3, 'Ex':4, 'Po':5})
train['GarageCond'] = train['GarageCond'].map({'TA':0, 'None':1, 'Fa':2, 'Gd':3, 'Po':4, 'Ex':5})
train['PavedDrive'] = train['PavedDrive'].map({'Y':0, 'N':1, 'P':2})
train['PoolQC'] = train['PoolQC'].map({'None':0, 'Ex':1, 'Gd':2, 'Fa':3})
train['Fence'] = train['Fence'].map({'None':0, 'MnPrv':1, 'GdPrv':2, 'GdWo':3, 'MnWw':4})
train['MiscFeature'] = train['MiscFeature'].map({'None':0, 'Shed':1, 'Othr':2, 'Gar2':3, 'TenC':4})
train['SaleType'] = train['SaleType'].map({'New':1, 'COD':2, 'ConLD':3, 'ConLI':4, 'CWD':5, 'Oth':6, 'Con':7, 'WD':0})
train['SaleCondition'] = train['SaleCondition'].map({'Normal':0, 'Partial':1, 'Abnorml':2, 'Family':3, 'Alloca':4, 'AdjLand':5})


# In[ ]:


train.head()


# In[ ]:


train_app = train.copy()
train_app.shape


# In[ ]:


test.head()


# In[ ]:


len(test.select_dtypes(['object']).columns)


# In[ ]:


catcols = test.select_dtypes(['object'])
for cat in catcols:
    print(cat)
    print(test[cat].value_counts())
    print('--'*20)


# In[ ]:


test.select_dtypes(['object']).columns


# In[ ]:


test['MSZoning'] = test['MSZoning'].map({'RL':0, 'RM':1, 'FV':2, 'RH':3, 'C (all)':4})
test['Street'] = test['Street'].map({'Pave':0, 'Grvl':1})
test['Alley'] = test['Alley'].map({'None':0, 'Grvl':1, 'Pave':2})
test['LotShape'] = test['LotShape'].map({'Reg':0, 'IR1':1, 'IR2':2, 'IR3':3})
test['LandContour'] = test['LandContour'].map({'Lvl':0, 'Bnk':1, 'HLS':2, 'Low':3})
test['Utilities'] = test['Utilities'].map({'AllPub':0, 'NoSeWa':1})
test['LotConfig'] = test['LotConfig'].map({'Inside':0, 'Corner':1, 'CulDSac':2, 'FR2':3, 'FR3':4})
test['LandSlope'] = test['LandSlope'].map({'Gtl':0, 'Mod':1, 'Sev':2})
test['Neighborhood'] = test['Neighborhood'].map({'NAmes':0, 'CollgCr':1, 'OldTown':2, 'Edwards':3, 'Somerst':4, 'Gilbert':5,
                                                  'NridgHt':6, 'Sawyer':7, 'NWAmes':8, 'SawyerW':9, 'BrkSide':10, 'Crawfor':11,
                                                  'Mitchel':12, 'NoRidge':13, 'Timber':14, 'IDOTRR':15, 'ClearCr':16, 'SWISU':17,
                                                  'StoneBr':18, 'Blmngtn':19, 'MeadowV':20, 'BrDale':21, 'Veenker':22, 'NPkVill':23, 'Blueste':24})
test['Condition1'] = test['Condition1'].map({'Norm':0, 'Feedr':1, 'Artery':2, 'RRAn':3, 'PosN':4, 'RRAe':5, 'PosA':6,
                                              'RRNn':7, 'RRNe':8})
test['Condition2'] = test['Condition2'].map({'Norm':0, 'Feedr':1, 'Artery':2, 'RRNn':3, 'PosN':4, 'RRAe':5, 'PosA':6, 'RRAn':7})
test['BldgType'] = test['BldgType'].map({'1Fam':0, 'TwnhsE':1, 'Duplex':2, 'Twnhs':3, '2fmCon':4})
test['HouseStyle'] = test['HouseStyle'].map({'1Story':0, '2Story':1, '1.5Fin':2, 'SLvl':3, 'SFoyer':4, '1.5Unf':5,
                                              '2.5Unf':6, '2.5Fin':7})
test['RoofStyle'] = test['RoofStyle'].map({'Gable':0, 'Hip':1, 'Flat':2, 'Gambrel':3, 'Mansard':4, 'Shed':5})
test['RoofMatl'] = test['RoofMatl'].map({'CompShg':0, 'Tar&Grv':1, 'WdShngl':2, 'WdShake':3, 'Roll':4, 'Membran':5, 'Metal':6})
test['Exterior1st'] = test['Exterior1st'].map({'VinylSd':0, 'HdBoard':1, 'MetalSd':2, 'Wd Sdng':3, 'Plywood':4, 'CemntBd':5, 'BrkFace':6, 'WdShing':7,
                                                'Stucco':8, 'AsbShng':9, 'Stone':10, 'BrkComm':11, 'CBlock':12, 'ImStucc':13, 'AsphShn':14})
test['Exterior2nd'] = test['Exterior2nd'].map({'VinylSd':0, 'MetalSd':1, 'HdBoard':2, 'Wd Sdng':3, 'Plwood':4, 'CmentBd':5, 'Wd Shng':6, 'Stucco':7,
                                                'BrkFace':8, 'AsbShng':9, 'ImStucc':10, 'Brk Cmn':11, 'Stone':12, 'AsphShn':13, 'Other':14, 'CBlock':15})
test['MasVnrType'] = test['MasVnrType'].map({'None':0, 'BrkFace':1, 'Stone':2, 'BrkCmn':3})
test['ExterQual'] = test['ExterQual'].map({'TA':0, 'Gd':1, 'Ex':2, 'Fa':3, 'Po':4})
test['ExterCond'] = test['ExterCond'].map({'TA':0, 'Gd':1, 'Fa':3, 'Ex':2, 'Po':4})
test['Foundation'] = test['Foundation'].map({'PConc':0, 'CBlock':1, 'BrkTil':2, 'Slab':3, 'Stone':4, 'Wood':5})
test['BsmtQual'] = test['BsmtQual'].map({'TA':0, 'Gd':1, 'Ex':2, 'None':3, 'Fa':4})
test['BsmtCond'] = test['BsmtCond'].map({'TA':0, 'Gd':1, 'Fa':2, 'None':3, 'Po':4})
test['BsmtExposure'] = test['BsmtExposure'].map({'No':0, 'Av':1, 'Gd':2, 'Mn':3, 'None':4})
test['BsmtFinType1'] = test['BsmtFinType1'].map({'Unf':0, 'GLQ':1, 'ALQ':2, 'BLQ':3, 'Rec':4, 'LwQ':5, 'None':6})
test['BsmtFinType2'] = test['BsmtFinType2'].map({'Unf':0, 'Rec':1, 'LwQ':2, 'None':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
test['Heating'] = test['Heating'].map({'GasA':0, 'GasW':1, 'Grav':2, 'Wall':3, 'OthW':4, 'Floor':5})
test['HeatingQC'] = test['HeatingQC'].map({'Ex':0, 'TA':1, 'Gd':2, 'Fa':3, 'Po':4})
test['CentralAir'] = test['CentralAir'].map({'Y':1, 'N':0})
test['Electrical'] = test['Electrical'].map({'SBrkr':0, 'FuseA':1, 'FuseF':2, 'FuseP':3, 'Mix':4, 'None':5})
test['KitchenQual'] = test['KitchenQual'].map({'TA':0, 'Gd':1, 'Ex':2, 'Fa':3})
test['Functional'] = test['Functional'].map({'Typ':0, 'Min2':1, 'Min1':2, 'Mod':3, 'Maj1':4, 'Maj2':5, 'Sev':6})
test['FireplaceQu'] = test['FireplaceQu'].map({'None':0, 'Gd':1, 'TA':2, 'Fa':3, 'Ex':4, 'Po':5})
test['GarageType'] = test['GarageType'].map({'Attchd':0, 'Detchd':1, 'BuiltIn':2, 'None':3, 'Basement':4, 'CarPort':5, '2Types':6})
test['GarageFinish'] = test['GarageFinish'].map({'Unf':0, 'RFn':1, 'Fin':2, 'None':3})
test['GarageQual'] = test['GarageQual'].map({'TA':0, 'None':1, 'Fa':2, 'Gd':3, 'Ex':4, 'Po':5})
test['GarageCond'] = test['GarageCond'].map({'TA':0, 'None':1, 'Fa':2, 'Gd':3, 'Po':4, 'Ex':5})
test['PavedDrive'] = test['PavedDrive'].map({'Y':0, 'N':1, 'P':2})
test['PoolQC'] = test['PoolQC'].map({'None':0, 'Ex':1, 'Gd':2, 'Fa':3})
test['Fence'] = test['Fence'].map({'None':0, 'MnPrv':1, 'GdPrv':2, 'GdWo':3, 'MnWw':4})
test['MiscFeature'] = test['MiscFeature'].map({'None':0, 'Shed':1, 'Othr':2, 'Gar2':3, 'TenC':4})
test['SaleType'] = test['SaleType'].map({'New':1, 'COD':2, 'ConLD':3, 'ConLI':4, 'CWD':5, 'Oth':6, 'Con':7, 'WD':0})
test['SaleCondition'] = test['SaleCondition'].map({'Normal':0, 'Partial':1, 'Abnorml':2, 'Family':3, 'Alloca':4, 'AdjLand':5})

test.head()


# In[ ]:


missing_values(test.select_dtypes([np.number]))


# In[ ]:


train.fillna(train.mean(), inplace = True)
test.fillna(test.mean(), inplace = True)
train.fillna(train.mode(), inplace = True)
test.fillna(test.mode(), inplace = True)


# In[ ]:


missing_values(test)


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<h2> Now lets create our model </h2>')


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


y = train['SalePrice'].copy()
X = train.drop(columns = 'SalePrice', inplace = True)


# In[ ]:


tree = DecisionTreeClassifier(random_state=101)
train = train.astype(float)


# In[ ]:


X = train


# In[ ]:


X.head()


# In[ ]:


treeReg = DecisionTreeRegressor(random_state=0, max_depth=5)
modelRegTree = treeReg.fit(X,y)
print(f'Decision tree has {treeReg.tree_.node_count} nodes with maximum depth {treeReg.tree_.max_depth}.')
print('*'*40)
print(f'Model Accuracy: {treeReg.score(X, y)}')


# In[ ]:


tree.fit(X,y)


# In[ ]:


print(f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')


# In[ ]:


print(f'Model Accuracy: {tree.score(X, y)}')


# In[ ]:


test = test.astype(float)


# In[ ]:


test_app = test['Id'].copy().astype(int).round()
test.drop(columns = 'Id', inplace = True)


# In[ ]:


y_pred = tree.predict(test)


# In[ ]:


y_pred_prob = tree.predict_proba(test)


# In[ ]:


features = list(train.columns)
fi = pd.DataFrame({'Feature': features, 'importance':tree.feature_importances_}).sort_values('importance', ascending = False)


# In[ ]:


fi.head()


# In[ ]:


print('SalePrice skewness: {}'.format(y_train.skew()))


# In[ ]:


mu = y_train.mean()
sigma = y_train.std()

num_bins = 40

plt.figure(figsize = (14,7))

n,bins,patches = plt.hist(y_train, num_bins, normed = 1, edgecolor = 'black', lw = 1, alpha = .40)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth = 2)
plt.xlabel('SalePrice')
plt.ylabel('Probability density')
plt.title(r'$\mathrm {Histogram\ of\ SalePrice:}\ \mu=%.3f, \ \sigma = %.3f$'%(mu, sigma))
plt.grid(True)
plt.show()


# In[ ]:


salePrice_normal = np.log1p(y_train)
mu = salePrice_normal.mean()
sigma = salePrice_normal.std()

num_bins = 40

plt.figure(figsize = (14,7))

n,bins,patches = plt.hist(salePrice_normal, num_bins, normed = 1, edgecolor = 'black', lw = 1, alpha = .40)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth = 2)
plt.xlabel('SalePrice')
plt.ylabel('Probability density')
plt.title(r'$\mathrm {Histogram\ of\ SalePrice:}\ \mu=%.3f, \ \sigma = %.3f$'%(mu, sigma))
plt.grid(True)
plt.show()


# In[ ]:


target = salePrice_normal.astype(float)


# In[ ]:


from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils


# In[ ]:


label = preprocessing.LabelEncoder()
target_encoded = label.fit_transform(y_train)
print(target_encoded)
print(utils.multiclass.type_of_target(y_train))
print(utils.multiclass.type_of_target(y_train.astype('int')))
print(utils.multiclass.type_of_target(target_encoded))


# In[ ]:


model = tree.fit(X, target_encoded)


# In[ ]:


pred = model.predict(test)


# In[ ]:


submit = pd.DataFrame(data = {'Id':test_app, 'SalePrice':pred})


# In[ ]:


pred


# In[ ]:


submit.head()


# In[ ]:


submit.to_csv('FirstSubmission.csv', index = False)


# In[ ]:


submit2 = pd.DataFrame(data = {'ID':test_app, 'SalePrice':y_pred})


# In[ ]:


submit2.to_csv('SecondSubmission.csv', index = False)


# In[ ]:


model13 = modelRegTree.predict(test)
model13


# In[ ]:


submit13 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':model13})
submit13.to_csv('ThirteenSub.csv', index = False)
submit13.head()


# In[ ]:





# In[ ]:





# In[ ]:


logic = LogisticRegression(random_state = 101, C=0.5, solver='lbfgs', verbose=1)


# In[ ]:


mod = logic.fit(X, y_train)


# In[ ]:


predic = mod.predict(test)
predic


# In[ ]:


submit3 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':predic})
submit3.to_csv('Submission3.csv', index = False)


# In[ ]:


submit3.head()


# In[ ]:


logics = LogisticRegression(random_state = 0, C = 1.0, solver='lbfgs', verbose=1)
mods = logics.fit(X, y_train)
prediss = mods.predict(test)
submit5 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':prediss})
submit5.to_csv('Submission5.csv', index = False)


# In[ ]:


submit5.head()


# In[ ]:


log = LogisticRegression()
logis = log.fit(X, y_train)
predi = logis.predict(test)
predi


# In[ ]:


submit4 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':predi})
submit4.to_csv('Submission4.csv', index = False)


# In[ ]:


from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score


# In[ ]:


model1 = XGBClassifier()
model2 = XGBRegressor()


# In[ ]:


classifier = model1.fit(X, y_train)
regressor = model2.fit(X, y_train)


# In[ ]:


y_prediction = classifier.predict(test)
y_prediction2 = regressor.predict(test)


# In[ ]:


y_prediction


# In[ ]:


y_prediction2


# In[ ]:


submit6 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':y_prediction})
submit6.to_csv('sixthSubmision.csv', index = False)


# In[ ]:


submit7 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':y_prediction2})
submit7.to_csv('SeventhSubmision.csv', index = False)


# In[ ]:


submit6.head()


# In[ ]:


submit7.head()


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<h2> Since XGBRegressor gives us a good score we need to work on our parameters </h2>')


# In[ ]:


params = {
    'booster':'gblinear',
    'objective':'reg:linear',
    'silent':1,
    'eta':1,
    'learning_rate':0.05,
    'n_estimators':50,
    'max_depth':5,
    'min_child_weight':1,
    'gamma':0,
    'subsample':0.8,
    'nthread':1
}
num_rounds = 15


# In[ ]:


from xgboost.sklearn import XGBRegressor
import xgboost as xgb
# from sklearn import cross_validation, metrics
# from sklearn.grid_search import GridSearchCV


# In[ ]:


regre = XGBRegressor(booster='gblinear', objective='reg:linear', silent = 1, eta = 1, learning_rate = 0.05,
    n_estimators = 50, max_depth = 5, min_child_weight = 1, gamma = 0, subsample = 0.8)


# In[ ]:


last = regre.fit(X, y_train)
less = last.predict(test)
less


# In[ ]:


submit8 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':less})
submit8.to_csv('EigthSubmission.csv', index = False)
submit8.head()


# In[ ]:


regres = XGBRegressor(booster='gblinear', objective='reg:linear', silent = 1, eta = 0.3, learning_rate = 0.05,
    n_estimators = 50, max_depth = 5, min_child_weight = 1, gamma = 0, subsample = 0.8, colsample_bytree = 0.7)


# In[ ]:


model8= regres.fit(X, y_train)
model = model8.predict(test)
model


# In[ ]:


submit9 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':model})
submit9.to_csv('Nineth.csv', index = False)
submit9.head()


# In[ ]:


regres10 = XGBRegressor(booster='gblinear', objective='reg:linear', silent = 1, eta = 1, learning_rate = 0.1,
    n_estimators = 500, max_depth = 6, min_child_weight = 1, gamma = 0.1, subsample = 0.8, colsample_bytree = 0.8, scale_pos_weight = 1, nthread=4, seed = 27, alpha = 10)


# In[ ]:


model10= regres10.fit(X, y_train)
model10 = model10.predict(test)
model10


# In[ ]:


submit10 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':model10})
submit10.to_csv('Ten.csv', index = False)
submit10.head()


# In[ ]:


regres11 = XGBRegressor(booster='gblinear', objective='reg:linear', silent = 1, eta = 1, learning_rate = 0.3,
                        n_estimators = 100, max_depth = 8, min_child_weight = 1, gamma = 0.1, subsample = 0.8,
                        colsample_bytree = 0.3, scale_pos_weight = 1,nthread=4, seed = 50, early_stopping_rounds=50
                       )


# In[ ]:


model11= regres11.fit(X, y_train)
model11 = model11.predict(test)
model11


# In[ ]:


submit11 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':model11})
submit11.to_csv('Eleven.csv', index = False)
submit11.head()


# In[ ]:


regres12 = XGBRegressor(booster='gblinear', objective='reg:linear', silent = 1, eta = 1, learning_rate = 0.3,
                        n_estimators = 1500, max_depth = 8, min_child_weight = 1, gamma = 0.1, subsample = 0.8,
                        colsample_bytree = 0.8, scale_pos_weight = 1,nthread=4, seed = 50, early_stopping_rounds=50
                       )
model12= regres12.fit(X, y_train)
model12 = model12.predict(test)
model12


# In[ ]:


submit12 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':model12})
submit12.to_csv('Twelve.csv', index = False)
submit12.head()


# In[ ]:


from xgboost.sklearn import XGBRegressor  
import scipy.stats as st
from scipy.stats import randint

one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}

xgbreg = XGBRegressor(nthreads=-1)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(xgbreg, params, n_jobs=1,random_state=100)  
gs.fit(X, y_train)  
prediction14 = gs.predict(test)


# In[ ]:


submission14 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':prediction14})
submission14.head()


# In[ ]:


gs1 = RandomizedSearchCV(xgbreg, params, n_jobs=1)  
gs1.fit(X, y_train)  
prediction15 = gs1.predict(test)
submission15 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':prediction15})
submission15.head()


# In[ ]:


gs2 = RandomizedSearchCV(xgbreg, params, n_jobs=1)  
gs2.fit(X, y_train)  
prediction16 = gs2.predict(test)
submission16 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':prediction16})
submission16.head()


# In[ ]:


submission15.to_csv('FifteenSub.csv', index = False)
submission16.to_csv('Sixteen.csv', index = False)


# In[ ]:





# In[ ]:


submission14.to_csv('FouteenSub.csv', index = False)


# In[ ]:


gs3 = RandomizedSearchCV(model2, params, n_jobs=1)  
gs3.fit(X, y_train)  
prediction17 = gs3.predict(test)
submission17 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':prediction17})
submission17.head()


# In[ ]:


gs4 = RandomizedSearchCV(xgbreg, params, n_jobs=1)  
gs4.fit(X, y_train)  
prediction18 = gs4.predict(test)
submission18 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':prediction18})
submission18.head()


# In[ ]:


gs5 = RandomizedSearchCV(model2, params, n_jobs=1)  
gs5.fit(X, y_train)  
prediction19 = gs5.predict(test)
submission19 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':prediction19})
submission19.head()


# In[ ]:


gs6 = RandomizedSearchCV(model2, params, n_jobs=1)  
gs6.fit(X, y_train)  
prediction20 = gs6.predict(test)
submission20 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':prediction20})
submission20.head()


# In[ ]:


submission17.to_csv('SeventeenSub.csv', index = False)
submission18.to_csv('EighteenSub.csv', index = False)
submission19.to_csv('NineteenSub.csv', index = False)
submission20.to_csv('TwentySub.csv', index = False)


# In[ ]:


gs7 = RandomizedSearchCV(model1, params, n_jobs=1)  
gs7.fit(X, y_train)  
prediction21 = gs7.predict(test)
submission21 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':prediction21})
submission21.head()


# In[ ]:


gs8 = RandomizedSearchCV(xgbreg, params, n_jobs=1, random_state= 0)  
gs8.fit(X, y_train)  
prediction22 = gs8.predict(test)
submission22 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':prediction22})
submission22.head()


# In[ ]:


gs9 = RandomizedSearchCV(xgbreg, params, n_jobs=1, random_state = 101)  
gs9.fit(X, y_train)  
prediction23 = gs9.predict(test)
submission23 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':prediction23})
submission23.head()


# In[ ]:


submission21.to_csv('TwentyOneSub.csv', index = False)
submission22.to_csv('TwentyTwoSub.csv', index = False)
submission23.to_csv('TentyThreeSub.csv', index = False)


# In[ ]:


gs10 = RandomizedSearchCV(model2, params, n_jobs=1, random_state = 50)  
gs10.fit(X, y_train)  
prediction24 = gs10.predict(test)
submission24 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':prediction24})
submission24.head()


# In[ ]:


submission24.to_csv('TwentyFourSubmission.csv', index = False)


# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<h2>SO THE SCORE OF OUR PREDICTION DOES NOT IMPROVE WHEN TRYING TO MANIPULATE THE PARAMETORS (PARAMETOR TUNING)</h2> <br />\n\nSometimes its best to use lets features than all of our features.Lets look at our data again.')


# In[ ]:


train.head()


# In[ ]:





# In[ ]:


import lightgbm as lgb


# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',
                              num_leaves=4,
                              learning_rate=0.05, 
                              n_estimators=1050,
                              max_bin=75, 
                              bagging_fraction=0.8,
                              bagging_freq=5, 
                              feature_fraction=0.2319,
                              feature_fraction_seed=9, 
                              bagging_seed=9,
                              min_data_in_leaf=6, 
                              min_sum_hessian_in_leaf=11)


# In[ ]:


model_lgb.fit(X, y_train)
lgb_pred = np.expm1(model_lgb.predict(test))


# In[ ]:


submission25 = pd.DataFrame(data = {'Id':test_app, 'SalePrice':lgb_pred})
submission25.head()


# In[ ]:





# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('HTML', '', '\n<h1> To be Continued</h1>\n\n<p>Now that we have tried to manipulate our data and even worked with parametors but still our model does not improve </p>\n<p> Lets try to work with our data again start cleaning it and use few features than we were using before. We will remove some features that are least corellated to the target feature and also remove some features that are some sort of duplicates. </p>')


# ** This is what we will do on this section**
# 
# * Load Data
# * Pickout Features (ID) that I will use when I am done with the prediction
# * Filling in missing values (both objects and numeric values)
# * Change data to correct data type 
# * Remove outliers
# * LabelEncode categorical features
# * Generate features
# * Skew numeric features

# In[ ]:


train_app = train_data.copy()
test_app = test_data.copy()


# In[ ]:


train_ID = train_app['Id']
test_ID = test_app['Id']
train_app = train_app.drop('Id', axis = 1)
test_app = test_app.drop('Id', axis = 1)


# In[ ]:


train_app.shape, test_app.shape


# In[ ]:


train_app.head()


# In[ ]:


test_app.head()


# In[ ]:


missing_values(train_app)


# In[ ]:


missing_values(test_app)


# In[ ]:


train_app.select_dtypes(include = [np.object]).columns


# In[ ]:


missing_values(train_app.select_dtypes(include = [np.object]))


# In[ ]:


# train_app.select_dtypes('object').apply(pd.Series.unique, axis = 0)
from scipy.stats import norm


# In[ ]:


sns.distplot(train_app['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_app['SalePrice'], plot=plt)


# In[ ]:


train_app['SalePrice'] = np.log(train_app['SalePrice'])
sns.distplot(train_app['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train_app['SalePrice'], plot=plt)


# In[ ]:


plt.figure(figsize=(7,5))
plt.scatter(x = train_app.TotalBsmtSF,y = train_app.SalePrice)
plt.title('TotalBsmtSF', size = 15)
plt.figure(figsize=(7,5))
plt.scatter(x = train_app['1stFlrSF'],y = train_app.SalePrice)
plt.title('1stFlrSF', size = 15)
plt.figure(figsize=(7,5))
plt.scatter(x = train_app.GrLivArea,y = train_app.SalePrice)
plt.title('GrLivArea', size = 15)


# In[ ]:


train_app.drop(train_app[train_app['TotalBsmtSF'] > 5000].index,inplace = True)
train_app.drop(train_app[train_app['1stFlrSF'] > 4000].index,inplace = True)
train_app.drop(train_app[(train_app['GrLivArea'] > 4000) & (train_app['SalePrice']<300000)].index,inplace = True)
train_app.shape


# In[ ]:


train_numerical_features = train_app.dtypes[train_app.dtypes != 'object'].index
train_cat_features = train_app.dtypes[train_app.dtypes == 'object'].index
print('Train Numerical Features: {}'.format(len(train_numerical_features)))
print('Train Categorical Features: {}'.format(len(train_cat_features)))


# In[ ]:


cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']
for col in cols_fillna:
    train_app[col].fillna('None',inplace=True)
    test_app[col].fillna('None',inplace=True)


# In[ ]:


missing_values(train_app)


# In[ ]:


train_app.fillna(train_app.mean(), inplace = True)
test_app.fillna(test_app.mean(), inplace = True)
train_app.fillna(train_app.mode(), inplace = True)
test_app.fillna(test_app.mode(), inplace = True)


# In[ ]:


missing_values(test_app)


# In[ ]:


for cols in train_numerical_features:
    print(cols)
    print('Skewness: {}'.format(train_app[cols].skew()))
    print('Kurtosis: {}'.format(train_app[cols].kurt()))
    print('**'*30)


# In[ ]:


for data in [train_app, test_app]:
    data['GrLivArea_Log'] = np.log(data['GrLivArea'])
    data.drop('GrLivArea', axis = 1, inplace = True)


# In[ ]:


for data in [train_app, test_app]:
    data['LotArea_log'] = np.log(data['LotArea'])
    data.drop('LotArea', axis = 1, inplace = True)


# In[ ]:


train_numerical_features = train_app.dtypes[train_app.dtypes != 'object'].index


# In[ ]:


train_app['SalePrice_Log'] = np.log1p(train_app['SalePrice'])


# In[ ]:


train_app.drop('SalePrice_Log', axis = 1, inplace = True)


# In[ ]:


# target = train_app.SalePrice
target = 'SalePrice'


# In[ ]:


nr_Cv = 5
use_lgvals = 1
min_val_corr = 0.4
drop_similar = 1


# In[ ]:


corr = train_app.corr()
corr_abs = corr.abs()

nr_num_cols = len(train_numerical_features)
ser_corr = corr_abs.nlargest(nr_num_cols, target)[target]

cols_abv_limit = list(ser_corr[ser_corr.values > min_val_corr].index)
cols_bel_limit = list(ser_corr[ser_corr.values <= min_val_corr].index)


# In[ ]:


print(ser_corr)
print('**'*50)
print('Columns With correlation above limit:')
print('=='*25)
print(cols_abv_limit)
print('**'*50)
print('Columns with correlation below limit:')
print('=='*25)
print(cols_bel_limit)


# In[ ]:


for col in list(train_cat_features):
    print(col)
    print(train_app[col].value_counts())
    print('==='*15)


# In[ ]:


catg_strong_corr = [ 'MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 
                     'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']

catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                  'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 
                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 
                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 
                  'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                  'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                  'SaleCondition' ]


# In[ ]:


def plot_corr_matrix(df, nr_c, targ) :
    
    corr = df.corr()
    corr_abs = corr.abs()
    cols = corr_abs.nlargest(nr_c, targ)[targ].index
    cm = np.corrcoef(df[cols].values.T)

    plt.figure(figsize=(nr_c/1.5, nr_c/1.5))
    sns.set(font_scale=1.25)
    sns.heatmap(cm, linewidths=1.5, annot=True, square=True, 
                fmt='.2f', annot_kws={'size': 10}, 
                yticklabels=cols.values, xticklabels=cols.values
               )
    plt.show()


# In[ ]:


nr_feats = len(cols_abv_limit)
plot_corr_matrix(train_app, nr_feats, target)


# In[ ]:





# In[ ]:


to_drop_num = cols_bel_limit
to_drop_catg = catg_weak_corr
cols_to_drop = to_drop_num + to_drop_catg

for df in [train_app, test_app]:
    df.drop(cols_to_drop, axis = 1, inplace = True)


# In[ ]:


catg_list = catg_strong_corr.copy()
catg_list.remove('Neighborhood')

for catg in catg_list:
    sns.violinplot(x = catg, y = target, data = train_app)
    plt.show()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(16,5)
sns.violinplot(x = 'Neighborhood', y = target, data = train_app, ax = ax)
plt.xticks(rotation = 45)
plt.show()


# In[ ]:


for catg in catg_list:
    categ = train_app.groupby(catg)[target].mean()
    print(categ)
    print('=='*15)


# In[ ]:


# 'MSZoning'
msz_catg2 = ['RM', 'RH']
msz_catg3 = ['RL', 'FV'] 


# Neighborhood
nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']
nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']

# Condition2
cond2_catg2 = ['Norm', 'RRAe']
cond2_catg3 = ['PosA', 'PosN'] 

# SaleType
SlTy_catg1 = ['Oth']
SlTy_catg3 = ['CWD']
SlTy_catg4 = ['New', 'Con']


# In[ ]:


for df in [train_app, test_app]:
    
    df['MSZ_num'] = 1  
    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2    
    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3        
    
    df['NbHd_num'] = 1       
    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2    
    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3    

    df['Cond2_num'] = 1       
    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2    
    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3    
    
    df['Mas_num'] = 1       
    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2 
    
    df['ExtQ_num'] = 1       
    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2     
    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3     
    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4     
   
    df['BsQ_num'] = 1          
    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2     
    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3     
 
    df['CA_num'] = 0          
    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1    

    df['Elc_num'] = 1       
    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 


    df['KiQ_num'] = 1       
    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2     
    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3     
    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4      
    
    df['SlTy_num'] = 2       
    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1  
    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3  
    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4  


# In[ ]:


catg_cols_to_drop = ['Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual',
                     'CentralAir', 'Electrical', 'KitchenQual', 'SaleType']


# In[ ]:


corr1 = train_app.corr()
corr_abs = corr1.abs()

nr_all_cols = len(train_app)
ser_all_1 = corr_abs.nlargest(nr_all_cols, target)[target]
print(ser_all_1)
cols_bel_corr_limit_1 = list(ser_all_1[ser_all_1.values <= min_val_corr].index)

for df in [train_app, test_app]:
    df.drop(catg_cols_to_drop, axis = 1, inplace = True)
    df.drop(cols_bel_corr_limit_1, axis = 1, inplace = True)


# In[ ]:


corr2 = train_app.corr()
corr_abs_2 = corr2.abs()
nr_all_cols = len(train_app)
ser_corr_2 = corr_abs_2.nlargest(nr_all_cols, target)[target]
print(ser_corr_2)


# In[ ]:


train_app.head()


# In[ ]:


corr3 = train_app.corr()
corr_abs_3 = corr3.abs()
nr_all_cols_3 = len(train_app)
print(corr_abs_3.nlargest(nr_all_cols, target)[target])


# In[ ]:


nr_features = len(train_app.columns)
plot_corr_matrix(train_app, nr_features, target)


# In[ ]:


cols = corr_abs_3.nlargest(nr_all_cols_3, target)[target].index

cols = list(cols)
if drop_similar == 1:
    for col in ['GarageArea','1stFlrSF','TotRmsAbvGrd','GarageYrBlt']:
        if col in cols:
            cols.remove(col)
            
            
cols = list(cols)
print(cols)


# In[ ]:


feats = cols.copy()
feats.remove('SalePrice')
print(feats)


# In[ ]:


train_ml = train_app[cols].copy()
test_ml = test_app[feats].copy()


# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
train_ml_sc = sc.fit_transform(train_ml.drop(target, axis=1))
test_ml_sc = sc.transform(test_ml)


# In[ ]:


train_ml_sc = pd.DataFrame(train_ml_sc)
train_ml_sc.head()


# In[ ]:


X = train_ml.drop([target], axis = 1)
y = train_ml[target]
X_test = test_ml.copy()

X_sc = train_ml_sc.copy()
y_sc  =train_ml[target]
X_test_sc = test_ml_sc.copy()


# In[ ]:





# In[ ]:


def get_best_score(grid):
    
    best_score = np.sqrt(-grid.best_score_)
    print(best_score)    
    print(grid.best_params_)
    print(grid.best_estimator_)
    
    return best_score


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

score_calc = 'neg_mean_squared_error'
linreg = LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid_linear = GridSearchCV(linreg, parameters, cv = nr_Cv, verbose=1 , scoring = score_calc)
grid_linear.fit(X, y)

sc_linear = get_best_score(grid_linear)


# In[ ]:


linregr_all = LinearRegression()
#linregr_all.fit(X_train_all, y_train_all)
linregr_all.fit(X, y)
pred_linreg_all = linregr_all.predict(X_test)
pred_linreg_all[pred_linreg_all < 0] = pred_linreg_all.mean()


# In[ ]:


sub_linreg = pd.DataFrame()
sub_linreg['Id'] = test_ID
sub_linreg['SalePrice'] = pred_linreg_all
sub_linreg.tail()
sub_linreg.to_csv('linreg.csv',index=False)


# In[ ]:





# In[ ]:


from sklearn.linear_model import Ridge

ridge = Ridge()
parameters = {'alpha':[0.001,0.005,0.01,0.1,0.5,1], 'normalize':[True,False], 'tol':[1e-06,5e-06,1e-05,5e-05]}
grid_ridge = GridSearchCV(ridge, parameters, cv=nr_Cv, verbose=1, scoring = score_calc)
grid_ridge.fit(X, y)

sc_ridge = get_best_score(grid_ridge)


# In[ ]:


pred_ridge_all = grid_ridge.predict(X_test)


# In[ ]:


pred_ridge_all


# In[ ]:


grid_regressor = XGBRegressor()


# In[ ]:


y_train,y = y_train.align(y, join = 'inner', axis = 0)


# In[ ]:


grid_regressor.fit(X, y_train)


# In[ ]:


pred_regressor = grid_regressor.predict(X_test)


# In[ ]:


pred_regressor


# In[ ]:





# In[ ]:


y_train.shape, y.shape


# In[ ]:


submition_dont = pd.DataFrame()
submition_dont['Id'] = test_ID
submition_dont['SalePrice'] = pred_regressor

submition_dont.head()


# In[ ]:


submition_dont.to_csv('Submision_TwentyFour.csv', index = False)


# In[ ]:


submit30 = pd.DataFrame(data = {'Id':test_ID, 'SalePrice':pred_regressor})
# submit30.to_csv('TwentyFive.csv', index = False)
submit30.head()


# In[ ]:


submit30.to_csv('TwentyFive.csv', index = False)


# In[ ]:





# In[ ]:


from sklearn.linear_model import Lasso

lasso = Lasso()
parameters = {'alpha':[1e-03,0.01,0.1,0.5,0.8,1], 'normalize':[True,False], 'tol':[1e-06,1e-05,5e-05,1e-04,5e-04,1e-03]}
grid_lasso = GridSearchCV(lasso, parameters, cv=nr_Cv, verbose=1, scoring = score_calc)
grid_lasso.fit(X, y_train)

sc_lasso = get_best_score(grid_lasso)

pred_lasso = grid_lasso.predict(X_test)


# In[ ]:


submitLaso = pd.DataFrame(data = {'Id': test_ID, 'SalePrice': pred_lasso})
submitLaso.head()


# In[ ]:


submitLaso.to_csv('LassoSubmit.csv', index = False)


# In[ ]:


from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor()
parameters = {'max_iter' :[10000], 'alpha':[1e-05], 'epsilon':[1e-02], 'fit_intercept' : [True]  }
grid_sgd = GridSearchCV(sgd, parameters, cv=nr_Cv, verbose=1, scoring = score_calc)
grid_sgd.fit(X_sc, y_train)

sc_sgd = get_best_score(grid_sgd)

pred_sgd = grid_sgd.predict(X_test_sc)


# In[ ]:


submitsgd = pd.DataFrame(data = {'Id': test_ID, 'SalePrice': pred_sgd})
submitsgd.head()
submitsgd.to_csv('sgdSubmit.csv', index = False)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

param_grid = { 'max_depth' : [7,8,9,10] , 'max_features' : [11,12,13,14] ,
               'max_leaf_nodes' : [None, 12,15,18,20] ,'min_samples_split' : [20,25,30],
                'presort': [False,True] , 'random_state': [5] }
            
grid_dtree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=nr_Cv, refit=True, verbose=1, scoring = score_calc)
grid_dtree.fit(X, y_train)

sc_dtree = get_best_score(grid_dtree)

pred_dtree = grid_dtree.predict(X_test)


# In[ ]:


submitdtree = pd.DataFrame(data = {'Id': test_ID, 'SalePrice': pred_dtree})
submitdtree.head()
submitdtree.to_csv('dtreeSubmit.csv', index = False)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

param_grid = {'min_samples_split' : [3,4,6,10], 'n_estimators' : [70,100], 'random_state': [5] }
grid_rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=nr_Cv, refit=True, verbose=1, scoring = score_calc)
grid_rf.fit(X, y_train)

sc_rf = get_best_score(grid_rf)
pred_rf = grid_rf.predict(X_test)


# In[ ]:


submitrf = pd.DataFrame(data = {'Id': test_ID, 'SalePrice': pred_rf})
submitrf.to_csv('RfSubmit.csv', index = False)


# In[ ]:


tree.fit(X, y_train)
tree_pred = tree.predict(X_test)


# In[ ]:


submittree = pd.DataFrame(data = {'Id': test_ID, 'SalePrice':tree_pred})
submittree.to_csv('Tree_submision.csv', index = False)
submittree.head()


# In[ ]:





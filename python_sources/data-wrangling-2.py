# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
import pylab as plt


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
                 
                 
#train.describe()
''' 
Fill in NA values in LotFrontage with mean and in MasvnrArea with zeros
'''

train.LotFrontage=train.LotFrontage.fillna(train.LotFrontage.mean())
train.MasVnrArea=train.MasVnrArea.fillna(0)

'''
checking numeric columns for outliers: looking at numeric values
'''

#numeric_columns=list(train.ix[1:2,train.dtypes!=object].columns)

#for i in numeric_columns:
#    print(i)
#    print(train[i].describe())
 
    
'''
checking numeric columns for outliers: looking at boxplots
'''
    
#for i in numeric_columns:
#    plt.figure()
#    plt.clf()
#    plt.boxplot(list(train[i]))
#    plt.title(i)
#    plt.show()


    
'''
candidates for outliers check and removal:
LotFrontage-Linear feet of street connected to property
LotArea-Lot size in square feet
MasVnrArea-Masonry veneer area in square feet
BsmtFinSF1: Type 1 finished square feet
BsmtFinSF2: Type 2 finished square feet
TotalBsmtSF: Total square feet of basement area
1stFlrSF: First Floor square feet
EnclosedPorch: Enclosed porch area in square feet
MiscVal: $Value of miscellaneous feature
    

'''
'''
LotFrontage
'''
# check how many observations in outliers area in train and test sets
#for i in [test,train]:
#   print(sum(i.LotFrontage>200))
#   
#check outliers observatios      
#train.ix[train.LotFrontage>200,:]

#remove observations with lotfrontage above 200
train=train.ix[train.LotFrontage<=200,:]

'''
LotArea
'''
#for i in [test,train]:
#   print(sum(i.LotArea>60000))
   
#check outliers observatios      
#train.ix[train.LotArea>60000,:]

#remove observations with lotArea above 60000
train=train.ix[train.LotArea<60000,:]

'''
MasVnrArea
'''
#for i in [test,train]:
#   print(sum(i.MasVnrArea>=1300))

#check outliers observatios   
#train.ix[train.MasVnrArea>=1300,:]

#remove observations with MasVnrArea above 1300
train=train.ix[train.MasVnrArea<1300,:]



'''
BsmtFinSF1
'''
#for i in [test,train]:
#   print(sum(i.BsmtFinSF1>=2000))



'''
BsmtFinSF2
'''
#for i in [test,train]:
#   print(sum(i.BsmtFinSF2>=1400))



'''
TotalBsmtSF
'''
#for i in [test,train]:
#   print(sum(i.TotalBsmtSF>=3200))

 
'''   
1stFlrSF   
'''
#for i in [test,train]:
#   print(sum(i['1stFlrSF']>=3200))
   
 
'''   
EnclosedPorch   
'''
#for i in [test,train]:
#   print(sum(i['EnclosedPorch']>=500))
   
'''   
MiscVal
'''
#for i in [test,train]:
#   print(sum(i['MiscVal']>=15000))

#train.ix[train['MiscVal']>15000,:]

'''
look again at numeric data
'''
train.describe()   
   

# <codecell>
'''
1.extracting label value
2.merging test and train files
3.removing null values
4.representing categorical features
'''

data_label=train['SalePrice'].copy()
train=train.drop(labels=['SalePrice'],axis=1)

#append test data to train data
data=train.append(test)

# check for columns with many null values 

data.isnull().sum()
'''
Alley-2713
PoolQC           2901
Fence            2339
MiscFeature      2807

out of 2910 observations

'''
'''
 Removing columns with many missing values
 
 '''
data=data.drop(labels=['Alley','PoolQC', 'Fence', 'MiscFeature' ], axis=1)

'''
Filling in columns with missing values
''' 
data.FireplaceQu=data.FireplaceQu.fillna('NA')   
        
     
data.MasVnrType=data.MasVnrType.fillna('None')


# fill in all the rest null values with 0

data=data.fillna(0)

#print(data.shape)

##check for nulls
#print(data.isnull().sum())
 
'''
Feature represntation: changing features into categorical values or dummmies
'''  
## Feature representation 
#print(data.dtypes)

'''
categorical nominal features
'''
#Turn categorical nominal features into dummies
data=pd.get_dummies(data,columns=['MSZoning', 'Street',\
'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',\
'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle','CentralAir','Electrical','Foundation','RoofStyle','RoofMatl', 'Exterior1st','Exterior2nd',\
'MasVnrType','SaleType','SaleCondition'])

'''
categorical ordinal features
'''
    
#train.ExterQual.unique()
order1=['Fa', 'TA','Gd', 'Ex' ]
data.ExterQual=data.ExterQual.astype('category', ordered=True,categories=order1).cat.codes
       

#train.ExterCond.unique()
order1=['Fa', 'TA','Gd', 'Ex' ]
data.ExterCond=data.ExterCond.astype('category', ordered=True,categories=order1).cat.codes
   

#train.BsmtQual.unique()
order1=['Fa', 'TA','Gd', 'Ex' ]
data.BsmtQual=data.BsmtQual.astype('category', ordered=True,categories=order1).cat.codes
       
    
#train.BsmtCond.unique()
order2=['Po', 'Fa', 'TA','Gd' ]
data.BsmtCond=data.BsmtCond.astype('category', ordered=True,categories=order2).cat.codes

#train.BsmtExposure.unique()
order3=['No','Mn', 'Av', 'Gd']
data.BsmtExposure=data.BsmtExposure.astype('category', ordered=True,categories=order3).cat.codes
    
#train.BsmtFinType1.unique()    
order4=['Unf', 'LwQ','Rec','BLQ','ALQ','GLQ']
data.BsmtFinType1=data.BsmtFinType1.astype('category', ordered=True,categories=order4).cat.codes
 
#train.BsmtFinType2.unique()    
order4=['Unf', 'LwQ','Rec','BLQ','ALQ','GLQ']
data.BsmtFinType2=data.BsmtFinType2.astype('category', ordered=True,categories=order4).cat.codes
 
data.Heating =data.Heating.astype('category').cat.codes

#train.HeatingQC.unique()
order5=[ 'Po', 'Fa','TA', 'Gd','Ex' ]
data.HeatingQC=data.HeatingQC.astype('category', ordered=True,categories=order5).cat.codes


#train.KitchenQual.unique()
order6=[ 'Fa','TA', 'Gd','Ex' ]
data.KitchenQual=data.KitchenQual.astype("category", ordered=True, categories=order6).cat.codes



#train.Functional.unique()
order7=['Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']
data.Functional=data.Functional.astype('category', ordered=True, categories=order7).cat.codes

#train.FireplaceQu.unique()
order8=['NA','Po', 'Fa','TA', 'Gd','Ex' ]
data.FireplaceQu=data.FireplaceQu.astype('category', ordered=True, categories=order8).cat.codes

#train.GarageType.unique()
order9=['Detchd' , 'CarPort', 'BuiltIn', 'Basment','Attchd',   '2Types']
data.GarageType=data.GarageType.astype('category', ordered=True, categories=order9).cat.codes


#train.GarageFinish.unique()
order10=['Unf','RFn', 'Fin']
data.GarageFinish=data.GarageFinish.astype('category', ordered=True, categories=order10).cat.codes


#train.GarageQual.unique()
order11=['Po', 'Fa','TA', 'Gd','Ex' ]
data.GarageQual=data.GarageQual.astype('category', ordered=True, categories=order11).cat.codes


#train.GarageCond.unique()
order12=['Po', 'Fa','TA', 'Gd','Ex' ]
data.GarageCond=data.GarageCond.astype('category', ordered=True, categories=order12).cat.codes



#train.PavedDrive.unique()
order13=['N','P','Y']
data.PavedDrive=data.PavedDrive.astype('category', ordered=True, categories=order13).cat.codes

# <codecell>
#reset indexes
data.reset_index(inplace=True)

# <codecell>
#reset indexes
data=data.drop(labels=['index'],axis=1)
# <codecell>
''' splitting and merging back data to original sets
'''
data_train=data.ix[:1450,:]
data_test=data.ix[1451:,:]
print(data_train.shape)
print(data_test.shape)
print(data_label.shape)

# <codecell>
''' merge back train data and train label'''

data_label_frame=pd.DataFrame(data_label)
data_label_frame.reset_index(inplace=True)
data_label_frame=data_label_frame.drop(labels=['index'],axis=1)
data_train['SalePrice']=data_label_frame['SalePrice']

# <codecell>
'''
write processed data into file
''' 

data_train.to_csv('House_train_processed.csv')
data_test.to_csv('House_test_processed.csv')
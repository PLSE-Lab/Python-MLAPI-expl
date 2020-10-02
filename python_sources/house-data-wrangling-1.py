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

train=pd.read_csv('../input/train.csv',index_col=0)
test=pd.read_csv('../input/test.csv',index_col=0)
data_label=train['SalePrice'].copy()
train=train.drop(labels=['SalePrice'],axis=1)

#append test data to train data
data=train.append(test)

## Removing missing values
data=data.drop(labels=['Alley','PoolQC', 'Fence', 'MiscFeature' ], axis=1) 
data.FireplaceQu=train.FireplaceQu.fillna('NA')   
    
#train.FireplaceQu.unique()    
    
#train.ix[train.MasVnrArea.isnull(),:]           

data.MasVnrArea=data.MasVnrArea.fillna(0)      
     
data.MasVnrType=data.MasVnrType.fillna('None')

#train.LotFrontage.describe()
data.LotFrontage=data.LotFrontage.fillna(data.LotFrontage.mean())


#train.isnull().sum()

# fill in all the rest null values with 0

data=data.fillna(0)

print(data.shape)
   
## Feature representation 
print(data.dtypes)


#Turn categorical nominal features into dummies
data=pd.get_dummies(data,columns=['MSZoning', 'Street',\
'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',\
'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle','CentralAir','Electrical','Foundation','RoofStyle','RoofMatl', 'Exterior1st','Exterior2nd',\
'MasVnrType','SaleType','SaleCondition'])

    
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

print(data.dtypes)

# standarize data
from sklearn import preprocessing
scaled_data=preprocessing.StandardScaler().fit_transform(data)
scaled_data=pd.DataFrame(scaled_data, columns=data.columns)


# split data int data_train and data_test
data_train=scaled_data.ix[0:1459,:]

data_test=scaled_data.ix[1460:,:]

# print shapes of data_sets
print(data_train.shape)
print(data_test.shape)
print(data_label.shape)

#build a random forest regressor

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100, min_samples_split =5)
rf.fit(data_train,data_label)
predicted=rf.predict(data_test)
test['SalePrice']=predicted

submission=pd.DataFrame(test['SalePrice'])


submission.to_csv('House_sub1.csv')  















data_test=data.ix[1461:,:]
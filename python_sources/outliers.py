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


train=pd.read_csv('../input/train.csv',index_col=0)
test=pd.read_csv('../input/test.csv',index_col=0)
                 
                 
train.describe()
''' 
Fill in NA values in LotFrontage with mean and in MasvnrArea with zeros
'''

train.LotFrontage=train.LotFrontage.fillna(train.LotFrontage.mean())
train.MasVnrArea=train.MasVnrArea.fillna(0)

'''
checking numeric columns for outliers: looking at numeric values
'''

numeric_columns=list(train.ix[1:2,train.dtypes!=object].columns)

for i in numeric_columns:
    print(i)
    print(train[i].describe())
 
    
'''
checking numeric columns for outliers: looking at boxplots
'''
    
for i in numeric_columns:
    plt.figure()
    plt.clf()
    plt.boxplot(list(train[i]))
    plt.title(i)
    plt.show()


    
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
for i in [test,train]:
   print(sum(i.LotFrontage>200))
   
#check outliers observatios      
train.ix[train.LotFrontage>200,:]

#remove observations with lotfrontage above 200
train=train.ix[train.LotFrontage<=200,:]

'''
LotArea
'''
for i in [test,train]:
   print(sum(i.LotArea>60000))
   
#check outliers observatios      
train.ix[train.LotArea>60000,:]

#remove observations with lotArea above 60000
train=train.ix[train.LotArea<60000,:]

'''
MasVnrArea
'''
for i in [test,train]:
   print(sum(i.MasVnrArea>=1300))

#check outliers observatios   
train.ix[train.MasVnrArea>=1300,:]

#remove observations with MasVnrArea above 1300
train=train.ix[train.MasVnrArea<1300,:]



'''
BsmtFinSF1
'''
for i in [test,train]:
   print(sum(i.BsmtFinSF1>=2000))



'''
BsmtFinSF2
'''
for i in [test,train]:
   print(sum(i.BsmtFinSF2>=1400))



'''
TotalBsmtSF
'''
for i in [test,train]:
   print(sum(i.TotalBsmtSF>=3200))

 
'''   
1stFlrSF   
'''
for i in [test,train]:
   print(sum(i['1stFlrSF']>=3200))
   
 
'''   
EnclosedPorch   
'''
for i in [test,train]:
   print(sum(i['EnclosedPorch']>=500))
   
'''   
MiscVal
'''
for i in [test,train]:
   print(sum(i['MiscVal']>=15000))

train.ix[train['MiscVal']>15000,:]

'''
look again at numeric data

'''

plt.close('all')




train.describe()   
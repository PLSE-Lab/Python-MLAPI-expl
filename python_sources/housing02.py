# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math as math
from scipy.stats.stats import pearsonr 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#Import and process
train = pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
trainY=train.loc[:,'SalePrice']
testIn=test.loc[:,'Id']
allAttr = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))





#Alley
allAttr["Alley"]=allAttr["Alley"].fillna("none")

#ExterQual
allAttr["ExterQual"]=allAttr["ExterQual"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})

#ExterCond
allAttr["ExterCond"]=allAttr["ExterCond"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})

#BsmtQual
allAttr["BsmtQual"]=allAttr["BsmtQual"].map({"Ex":100, "Gd":95, "TA":85, "Fa":75, "Po":50})
allAttr["BsmtQual"]=allAttr["BsmtQual"].fillna(0)

#BsmtCond
allAttr["BsmtCond"]=allAttr["BsmtCond"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
allAttr["BsmtCond"]=allAttr["BsmtCond"].fillna(0)

#BsmtExposure
allAttr["BsmtExposure"]=allAttr["BsmtExposure"].fillna("none")

#BsmtFinType1
allAttr["BsmtFinType1"]=allAttr["BsmtFinType1"].map({"GLQ":6, "ALQ":5, "BLQ":4, "Rec":3, "LwQ":2, "Unf":1})
allAttr["BsmtFinType1"]=allAttr["BsmtFinType1"].fillna(0)

#BsmtFinType2
allAttr["BsmtFinType2"]=allAttr["BsmtFinType2"].map({"GLQ":6, "ALQ":5, "BLQ":4, "Rec":3, "LwQ":2, "Unf":1})
allAttr["BsmtFinType2"]=allAttr["BsmtFinType2"].fillna(0)

#HeatingQC
allAttr["HeatingQC"]=allAttr["HeatingQC"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})

#KitchenQual
allAttr["KitchenQual"]=allAttr["KitchenQual"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})

#FireplaceQu
allAttr["FireplaceQu"]=allAttr["FireplaceQu"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
allAttr["FireplaceQu"]=allAttr["FireplaceQu"].fillna(0)

#GarageType
allAttr["GarageType"]=allAttr["GarageType"].fillna("none")

#GarageFinish
allAttr["GarageFinish"]=allAttr["GarageFinish"].fillna("none")

#GarageQual
allAttr["GarageQual"]=allAttr["GarageQual"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
allAttr["GarageQual"]=allAttr["GarageQual"].fillna(0)

#GarageCond
allAttr["GarageCond"]=allAttr["GarageCond"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
allAttr["GarageCond"]=allAttr["GarageCond"].fillna(0)

#PoolQC
allAttr["PoolQC"]=allAttr["PoolQC"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
allAttr["PoolQC"]=allAttr["PoolQC"].fillna(0)

#Fence
allAttr["Fence"]=allAttr["Fence"].fillna("none")

#MiscFeature
allAttr["MiscFeature"]=allAttr["MiscFeature"].fillna("none")

#MoSold
allAttr["MoSold"]=allAttr["MoSold"].map({1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"})



allAttr=pd.get_dummies(allAttr)
allAttr=allAttr.fillna(allAttr.mean())



    



def logFunc(x):
    return math.log10(x+1)

allAttr=allAttr.applymap(logFunc)

#corr=np.corrcoef(pd.DataFrame.transpose(allAttr))
#corr=pd.DataFrame(corr)

#print(allAttr["LotArea"])

trainX=allAttr.iloc[0:1460,:]
testX=allAttr.iloc[1460:2919,:]

correlation=pd.DataFrame

for column in trainX:
    print(pearsonr(trainX[column],trainY))
    


#corr.to_csv("CorrMatrix.csv")
#trainX.to_csv("trainX.csv")
#testX.to_csv("testX.csv")
#trainY.to_csv("trainY.csv")



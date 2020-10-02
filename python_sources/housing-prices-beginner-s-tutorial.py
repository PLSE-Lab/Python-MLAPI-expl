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

import scipy
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression



train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")




y= train_data.SalePrice
y_labs=test_data.Id

raw_train_data=train_data.drop(["Id","SalePrice"],axis=1)
raw_test_data=test_data.drop(["Id"],axis=1)

'''
raw_data = raw_train_data
raw_data = raw_data.concat(raw_test_data)
'''

raw_data = pd.concat([raw_train_data,raw_test_data])
#raw_test_data.head()



le=LabelEncoder()

def OneHotEncode(raw_data,enc):
    le.fit(np.unique(raw_data[enc]))
    le.transform(raw_data[enc])
    OneHotNeighborhood=pd.get_dummies(raw_data[enc],prefix=enc)
    raw_data=pd.concat([raw_data, OneHotNeighborhood], axis=1)
    raw_data.drop(enc, inplace=True, axis=1)
    return raw_data


'''
raw_data.head()

for _ in raw_data.columns:
    print(raw_data[_].head())



for _ in raw_data.columns:
    print(raw_data[_].describe())



for _ in raw_data.columns:
    print(_," : ",raw_data[_].unique())


LotFrontage True
Alley True
MasVnrType True
MasVnrArea True
BsmtQual True
BsmtCond True
BsmtExposure True
BsmtFinType1 True
BsmtFinType2 True
Electrical True
FireplaceQu True
GarageType True
GarageYrBlt True
GarageFinish True
GarageQual True
GarageCond True
PoolQC True
Fence True
MiscFeature True




for _ in raw_data.columns:
    if raw_data[_].isnull().values.any():
        #print(_,raw_data[_].isnull().values.any())
        raw_data[_][raw_data[_].isnull()]=np.mean(raw_data[_][raw_data[_].notnull()])
        
'''

#category_list=["MSZoning","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType","SaleCondition"]

#x=le.fit(raw_data.MSSubClass)


#x=x.transform(raw_data.MSSubClass)
def preprocessing(raw_data):
    '''
    'RL' 'RM' 'C (all)' 'FV' 'RH' nan
    '''

    raw_data.MSZoning[raw_data.MSZoning.isnull()] = 0
    raw_data.MSZoning[raw_data.MSZoning=="RL"] = 3
    raw_data.MSZoning[raw_data.MSZoning=="RM"] = 4
    raw_data.MSZoning[raw_data.MSZoning=="C (all)"] = 2
    raw_data.MSZoning[raw_data.MSZoning=="FV"] = 1
    raw_data.MSZoning[raw_data.MSZoning=="RH"] = 5

    '''
    'Pave' 'Grvl'
    '''

    raw_data.Street[raw_data.Street=="Pave"] = 1
    raw_data.Street[raw_data.Street=="Grvl"] = 0


    '''
    Alley : nan 'Grvl' 'Pave'
    '''

    raw_data.Alley[raw_data.Alley=="Pave"] = 1
    raw_data.Alley[raw_data.Alley=="Grvl"] = 0
    raw_data.Alley[raw_data.Alley.isnull()] = -1

    '''
    'Reg' 'IR1' 'IR2' 'IR3'
    '''

    raw_data.LotShape[raw_data.LotShape=="Reg"] = 4
    raw_data.LotShape[raw_data.LotShape=="IR1"] = 3
    raw_data.LotShape[raw_data.LotShape=="IR2"] = 2
    raw_data.LotShape[raw_data.LotShape=="IR3"] = 1


    '''
    LandContour  :  ['Lvl' 'Bnk' 'Low' 'HLS']
    '''

    raw_data.LandContour[raw_data.LandContour=="Lvl"] = 2
    raw_data.LandContour[raw_data.LandContour=="Bnk"] = 3
    raw_data.LandContour[raw_data.LandContour=="Low"] = 1
    raw_data.LandContour[raw_data.LandContour=="HLS"] = 4

    '''
    Utilities  :  ['AllPub' 'NoSeWa' nan]
    '''

    raw_data.Utilities[raw_data.Utilities=="AllPub"] = 2
    raw_data.Utilities[raw_data.Utilities=="NoSeWa"] = 1
    raw_data.Utilities[raw_data.Utilities.isnull()] = 0


    '''
    LotConfig  :  ['Inside' 'FR2' 'Corner' 'CulDSac' 'FR3']
    '''

    raw_data.LotConfig[raw_data.LotConfig=="Inside"] = 0
    raw_data.LotConfig[raw_data.LotConfig=="FR2"] = 2
    raw_data.LotConfig[raw_data.LotConfig=="Corner"] = 4
    raw_data.LotConfig[raw_data.LotConfig=="CulDSac"] = 1
    raw_data.LotConfig[raw_data.LotConfig=="FR3"] = 3

    '''
    LandSlope  :  ['Gtl' 'Mod' 'Sev']
    '''
    raw_data.LandSlope[raw_data.LandSlope=="Gtl"] = 2
    raw_data.LandSlope[raw_data.LandSlope=="Mod"] = 1
    raw_data.LandSlope[raw_data.LandSlope=="Sev"] = 0

    raw_data.LandSlope = np.exp(np.array(raw_data.LandSlope,dtype=np.float32))

    '''
    Neighborhood  :  ['CollgCr' 'Veenker' 'Crawfor' 'NoRidge' 'Mitchel' 'Somerst' 'NWAmes'
    'OldTown' 'BrkSide' 'Sawyer' 'NridgHt' 'NAmes' 'SawyerW' 'IDOTRR'
    'MeadowV' 'Edwards' 'Timber' 'Gilbert' 'StoneBr' 'ClearCr' 'NPkVill'
    'Blmngtn' 'BrDale' 'SWISU' 'Blueste']
    '''

    raw_data=OneHotEncode(raw_data,"Neighborhood")


    '''
    Condition1  :  ['Norm' 'Feedr' 'PosN' 'Artery' 'RRAe' 'RRNn' 'RRAn' 'PosA' 'RRNe']
    '''


    raw_data=OneHotEncode(raw_data,"Condition1")



    '''
    Condition2  :  ['Norm' 'Artery' 'RRNn' 'Feedr' 'PosN' 'PosA' 'RRAn' 'RRAe']
    '''


    raw_data=OneHotEncode(raw_data,"Condition2")



    '''
    BldgType  :  ['1Fam' '2fmCon' 'Duplex' 'TwnhsE' 'Twnhs']
    '''

    raw_data=OneHotEncode(raw_data,"BldgType")

    '''
    HouseStyle  :  ['2Story' '1Story' '1.5Fin' '1.5Unf' 'SFoyer' 'SLvl' '2.5Unf' '2.5Fin']
    '''


    raw_data=OneHotEncode(raw_data,"HouseStyle")


    '''
    RoofStyle  :  ['Gable' 'Hip' 'Gambrel' 'Mansard' 'Flat' 'Shed']
    '''
    raw_data=OneHotEncode(raw_data,"RoofStyle")
    '''
    RoofMatl  :  ['CompShg' 'WdShngl' 'Metal' 'WdShake' 'Membran' 'Tar&Grv' 'Roll'
    'ClyTile']
    '''
    raw_data=OneHotEncode(raw_data,"RoofMatl")
    '''
    Exterior1st  :  ['VinylSd' 'MetalSd' 'Wd Sdng' 'HdBoard' 'BrkFace' 'WdShing' 'CemntBd' 'Plywood' 'AsbShng' 'Stucco' 'BrkComm' 'AsphShn' 'Stone' 'ImStucc'
    'CBlock' nan]
    
    '''
    raw_data.Exterior1st[raw_data.Exterior1st.isnull()]="NULL"
    raw_data=OneHotEncode(raw_data,"Exterior1st")



    '''
    Exterior2nd  :  ['VinylSd' 'MetalSd' 'Wd Shng' 'HdBoard' 'Plywood' 'Wd Sdng' 'CmentBd'
    'BrkFace' 'Stucco' 'AsbShng' 'Brk Cmn' 'ImStucc' 'AsphShn' 'Stone'
    'Other' 'CBlock' nan]
    '''
    raw_data.Exterior2nd[raw_data.Exterior2nd.isnull()]="NULL"
    raw_data=OneHotEncode(raw_data,"Exterior2nd")
    '''
    MasVnrType  :  ['BrkFace' 'None' 'Stone' 'BrkCmn' nan]
    '''

    raw_data.MasVnrType[raw_data.MasVnrType.isnull()]="NULL"
    raw_data=OneHotEncode(raw_data,"MasVnrType")

    """
    MasVnrArea
    """

    raw_data.MasVnrArea[raw_data.MasVnrArea.isnull()] = np.mean(raw_data.MasVnrArea[raw_data.MasVnrArea.notnull()])

    '''
    ExterQual  :  ['Gd' 'TA' 'Ex' 'Fa']
    '''

    raw_data.ExterQual[raw_data.ExterQual=="Gd"] = 3
    raw_data.ExterQual[raw_data.ExterQual=="TA"] = 2
    raw_data.ExterQual[raw_data.ExterQual=="Ex"] = 4
    raw_data.ExterQual[raw_data.ExterQual=="Fa"] = 1

    '''
    ExterCond  :  ['TA' 'Gd' 'Fa' 'Po' 'Ex']
    '''

    raw_data.ExterCond[raw_data.ExterCond=="Gd"] = 3
    raw_data.ExterCond[raw_data.ExterCond=="TA"] = 2
    raw_data.ExterCond[raw_data.ExterCond=="Ex"] = 4
    raw_data.ExterCond[raw_data.ExterCond=="Fa"] = 1
    raw_data.ExterCond[raw_data.ExterCond=="Po"] = 0

    '''
    Foundation  :  ['PConc' 'CBlock' 'BrkTil' 'Wood' 'Slab' 'Stone']
    '''

    raw_data.Foundation[raw_data.Foundation=="PConc"] = 3
    raw_data.Foundation[raw_data.Foundation=="CBlock"] = 2
    raw_data.Foundation[raw_data.Foundation=="BrkTil"] = 1
    raw_data.Foundation[raw_data.Foundation=="Wood"] = 6
    raw_data.Foundation[raw_data.Foundation=="Slab"] = 5
    raw_data.Foundation[raw_data.Foundation=="Stone"] = 4

    '''
    BsmtQual  :  ['Gd' 'TA' 'Ex' nan 'Fa']
    '''

    raw_data.BsmtQual[raw_data.BsmtQual.isnull()] = 0
    raw_data.BsmtQual[raw_data.BsmtQual=="Gd"] = 4
    raw_data.BsmtQual[raw_data.BsmtQual=="TA"] = 3
    raw_data.BsmtQual[raw_data.BsmtQual=="Ex"] = 5
    raw_data.BsmtQual[raw_data.BsmtQual=="Fa"] = 2

    '''
    BsmtCond  :  ['TA' 'Gd' nan 'Fa' 'Po']
    '''

    raw_data.BsmtCond[raw_data.BsmtCond.isnull()] = 0
    raw_data.BsmtCond[raw_data.BsmtCond=="TA"] = 3
    raw_data.BsmtCond[raw_data.BsmtCond=="Gd"] = 4
    raw_data.BsmtCond[raw_data.BsmtCond=="Fa"] = 2
    raw_data.BsmtCond[raw_data.BsmtCond=="Po"] = 1

    '''
    BsmtExposure  :  ['No' 'Gd' 'Mn' 'Av' nan]
    '''

    raw_data.BsmtExposure[raw_data.BsmtExposure.isnull()] = 0
    raw_data.BsmtExposure[raw_data.BsmtExposure=="No"] = 1
    raw_data.BsmtExposure[raw_data.BsmtExposure=="Gd"] = 4
    raw_data.BsmtExposure[raw_data.BsmtExposure=="Mn"] = 2
    raw_data.BsmtExposure[raw_data.BsmtExposure=="Av"] = 3

    '''
    BsmtFinType1  :  ['GLQ' 'ALQ' 'Unf' 'Rec' 'BLQ' nan 'LwQ']
    '''

    raw_data.BsmtFinType1[raw_data.BsmtFinType1.isnull()] = 0
    raw_data.BsmtFinType1[raw_data.BsmtFinType1=="GLQ"] = 6
    raw_data.BsmtFinType1[raw_data.BsmtFinType1=="ALQ"] = 5
    raw_data.BsmtFinType1[raw_data.BsmtFinType1=="Unf"] = 1
    raw_data.BsmtFinType1[raw_data.BsmtFinType1=="Rec"] = 3
    raw_data.BsmtFinType1[raw_data.BsmtFinType1=="BLQ"] = 4
    raw_data.BsmtFinType1[raw_data.BsmtFinType1=="LwQ"] = 2

    '''
    BsmtFinSF1
    '''

    raw_data.BsmtFinSF1[raw_data.BsmtFinSF1.isnull()] = np.mean(raw_data.BsmtFinSF1[raw_data.BsmtFinSF1.notnull()])


    '''
    BsmtFinType2  :  ['Unf' 'BLQ' nan 'ALQ' 'Rec' 'LwQ' 'GLQ']
    '''

    raw_data.BsmtFinType2[raw_data.BsmtFinType2.isnull()] = 0
    raw_data.BsmtFinType2[raw_data.BsmtFinType2=="GLQ"] = 6
    raw_data.BsmtFinType2[raw_data.BsmtFinType2=="ALQ"] = 5
    raw_data.BsmtFinType2[raw_data.BsmtFinType2=="Unf"] = 1
    raw_data.BsmtFinType2[raw_data.BsmtFinType2=="Rec"] = 3
    raw_data.BsmtFinType2[raw_data.BsmtFinType2=="BLQ"] = 4
    raw_data.BsmtFinType2[raw_data.BsmtFinType2=="LwQ"] = 2

    '''
    BsmtFinSF2
    '''

    raw_data.BsmtFinSF2[raw_data.BsmtFinSF2.isnull()] = np.mean(raw_data.BsmtFinSF2[raw_data.BsmtFinSF2.notnull()])

    '''
    Heating  :  ['GasA' 'GasW' 'Grav' 'Wall' 'OthW' 'Floor']
    '''

    #raw_data.Heating[raw_data.Heating.isnull()]="NULL"
    raw_data=OneHotEncode(raw_data,"Heating")

    '''
    HeatingQC  :  ['Ex' 'Gd' 'TA' 'Fa' 'Po']
    '''

    raw_data.HeatingQC[raw_data.HeatingQC=="Ex"] = 5
    raw_data.HeatingQC[raw_data.HeatingQC=="Gd"] = 4
    raw_data.HeatingQC[raw_data.HeatingQC=="TA"] = 3
    raw_data.HeatingQC[raw_data.HeatingQC=="Fa"] = 2
    raw_data.HeatingQC[raw_data.HeatingQC=="Po"] = 1

    '''
    CentralAir  :  ['Y' 'N']
    '''

    raw_data.CentralAir[raw_data.CentralAir=="Y"] = 1
    raw_data.CentralAir[raw_data.CentralAir=="N"] = 1

    '''
    Electrical  :  ['SBrkr' 'FuseF' 'FuseA' 'FuseP' 'Mix' nan]
    '''

    raw_data.Electrical[raw_data.Electrical.isnull()] = 0
    raw_data.Electrical[raw_data.Electrical=="SBrkr"] = 5
    raw_data.Electrical[raw_data.Electrical=="FuseF"] = 3
    raw_data.Electrical[raw_data.Electrical=="FuseA"] = 4
    raw_data.Electrical[raw_data.Electrical=="FuseP"] = 1
    raw_data.Electrical[raw_data.Electrical=="Mix"] = 2

    '''
    BsmtFullBath
    '''

    raw_data.BsmtFullBath[raw_data.BsmtFullBath.isnull()] = np.mean(raw_data.BsmtFullBath[raw_data.BsmtFullBath.notnull()])

    '''
    BsmtHalfBath
    '''

    raw_data.BsmtHalfBath[raw_data.BsmtHalfBath.isnull()] = np.mean(raw_data.BsmtHalfBath[raw_data.BsmtHalfBath.notnull()])

    '''
    KitchenQual  :  ['Gd' 'TA' 'Ex' 'Fa' nan]
    '''

    raw_data.KitchenQual[raw_data.KitchenQual.isnull()] = 0
    raw_data.KitchenQual[raw_data.KitchenQual=="Gd"] = 3
    raw_data.KitchenQual[raw_data.KitchenQual=="TA"] = 2
    raw_data.KitchenQual[raw_data.KitchenQual=="Ex"] = 4
    raw_data.KitchenQual[raw_data.KitchenQual=="Fa"] = 1

    '''
    Functional  :  ['Typ' 'Min1' 'Maj1' 'Min2' 'Mod' 'Maj2' 'Sev' nan]
    '''

    raw_data.Functional[raw_data.Functional.isnull()] = 0
    raw_data.Functional[raw_data.Functional=="Typ"] = 4
    raw_data.Functional[raw_data.Functional=="Min1"] = 1
    raw_data.Functional[raw_data.Functional=="Maj1"] = 5
    raw_data.Functional[raw_data.Functional=="Min2"] = 3
    raw_data.Functional[raw_data.Functional=="Mod"] = 3
    raw_data.Functional[raw_data.Functional=="Maj2"] = 2
    raw_data.Functional[raw_data.Functional=="Sev"] = 7

    '''
    FireplaceQu  :  [nan 'TA' 'Gd' 'Fa' 'Ex' 'Po']
    '''

    raw_data.FireplaceQu[raw_data.FireplaceQu.isnull()] = 0
    raw_data.FireplaceQu[raw_data.FireplaceQu=="TA"] = 3
    raw_data.FireplaceQu[raw_data.FireplaceQu=="Gd"] = 4
    raw_data.FireplaceQu[raw_data.FireplaceQu=="Fa"] = 2
    raw_data.FireplaceQu[raw_data.FireplaceQu=="Ex"] = 5
    raw_data.FireplaceQu[raw_data.FireplaceQu=="Po"] = 1

    '''
    GarageType  :  ['Attchd' 'Detchd' 'BuiltIn' 'CarPort' nan 'Basment' '2Types']
    '''

    raw_data.GarageType[raw_data.GarageType.isnull()] = 0
    raw_data.GarageType[raw_data.GarageType=="Attchd"] = 2
    raw_data.GarageType[raw_data.GarageType=="Detchd"] = 1
    raw_data.GarageType[raw_data.GarageType=="BuiltIn"] = .75
    raw_data.GarageType[raw_data.GarageType=="CarPort"] = 3
    raw_data.GarageType[raw_data.GarageType=="Basment"] = 4
    raw_data.GarageType[raw_data.GarageType=="2Types"] = 5

    '''
    GarageFinish  :  ['RFn' 'Unf' 'Fin' nan]
    '''

    raw_data.GarageFinish[raw_data.GarageFinish.isnull()] = 0
    raw_data.GarageFinish[raw_data.GarageFinish=="RFn"] = 2
    raw_data.GarageFinish[raw_data.GarageFinish=="Unf"] = 1
    raw_data.GarageFinish[raw_data.GarageFinish=="Fin"] = 3

    '''
    GarageCars
    '''

    raw_data.GarageCars[raw_data.GarageCars.isnull()] = np.mean(raw_data.GarageCars[raw_data.GarageCars.notnull()])

    '''
    GarageQual  :  ['TA' 'Fa' 'Gd' nan 'Ex' 'Po']
    '''

    raw_data.GarageQual[raw_data.GarageQual.isnull()] = 0
    raw_data.GarageQual[raw_data.GarageQual=="TA"] = 3
    raw_data.GarageQual[raw_data.GarageQual=="Fa"] = 2
    raw_data.GarageQual[raw_data.GarageQual=="Gd"] = 4
    raw_data.GarageQual[raw_data.GarageQual=="Ex"] = 5
    raw_data.GarageQual[raw_data.GarageQual=="Po"] = 1


    '''
    GarageCond  :  ['TA' 'Fa' nan 'Gd' 'Po' 'Ex']
    '''

    raw_data.GarageCond[raw_data.GarageCond.isnull()] = 0
    raw_data.GarageCond[raw_data.GarageCond=="TA"] = 3
    raw_data.GarageCond[raw_data.GarageCond=="Fa"] = 2
    raw_data.GarageCond[raw_data.GarageCond=="Gd"] = 4
    raw_data.GarageCond[raw_data.GarageCond=="Po"] = 1
    raw_data.GarageCond[raw_data.GarageCond=="Ex"] = 5

    '''
    PavedDrive  :  ['Y' 'N' 'P']
    '''

    raw_data.PavedDrive[raw_data.PavedDrive=="Y"] = 2
    raw_data.PavedDrive[raw_data.PavedDrive=="N"] = 0
    raw_data.PavedDrive[raw_data.PavedDrive=="P"] = 1

    '''
    PoolQC  :  [nan 'Ex' 'Fa' 'Gd']
    '''

    raw_data.PoolQC[raw_data.PoolQC.isnull()] = 0
    raw_data.PoolQC[raw_data.PoolQC=="Ex"] = 3
    raw_data.PoolQC[raw_data.PoolQC=="Fa"] = 1
    raw_data.PoolQC[raw_data.PoolQC=="Gd"] = 2

    '''
    Fence  :  [nan 'MnPrv' 'GdWo' 'GdPrv' 'MnWw']
    '''

    raw_data.Fence[raw_data.Fence.isnull()] = "NULL"
    raw_data=OneHotEncode(raw_data,"Fence")


    '''
    MiscFeature  :  [nan 'Shed' 'Gar2' 'Othr' 'TenC']
    '''

    raw_data.MiscFeature[raw_data.MiscFeature.isnull()] = "NULL"
    raw_data=OneHotEncode(raw_data,"MiscFeature")

    '''
    SaleType  :  ['WD' 'New' 'COD' 'ConLD' 'ConLI' 'CWD' 'ConLw' 'Con' 'Oth' nan]
    '''

    raw_data.SaleType[raw_data.SaleType.isnull()] = "NULL"
    raw_data=OneHotEncode(raw_data,"SaleType")

    '''
    SaleCondition  :  ['Normal' 'Abnorml' 'Partial' 'AdjLand' 'Alloca' 'Family']
    '''

    raw_data.SaleCondition[raw_data.SaleCondition.isnull()] = "NULL"
    raw_data=OneHotEncode(raw_data,"SaleCondition")



    for _ in raw_data.columns:
        if raw_data[_].isnull().values.any():
            #print(_,raw_data[_].isnull().values.any())
            raw_data[_][raw_data[_].isnull()]=np.mean(raw_data[_][raw_data[_].notnull()])

    return raw_data

def Scale(raw_data):
    scaler = MinMaxScaler()
    for _ in raw_data.columns:
        scaler.fit(raw_data[_].reshape(-1, 1))
        scaler.transform(raw_data[_].reshape(-1, 1))
    return raw_data

raw_data=preprocessing(raw_data)

#raw_train_data = preprocessing(raw_train_data)

#raw_test_data = preprocessing(raw_test_data)

#raw_data_extra_cols=list(set(raw_data.columns) - set(raw_train_data.columns))
'''
for _ in raw_data_extra_cols:
    raw_train_data[_]=0
'''

raw_data = Scale(raw_data)


raw_train_data = raw_data.iloc[0:1460,]
raw_test_data = raw_data.iloc[1460:,]


clf = ExtraTreesClassifier()
clf = clf.fit(raw_train_data, y)
#print(clf.feature_importances_)
model = SelectFromModel(clf, prefit=True)
#model = model.fit(raw_train_data, y)
X_new = model.transform(raw_data)
print(X_new.shape)

raw_train_data = np.array(pd.DataFrame(X_new).iloc[0:1460,])
raw_test_data = np.array(pd.DataFrame(X_new).iloc[1460:,])

X_train, X_cv, y_train, y_cv = train_test_split(raw_train_data, y, test_size=0.30, random_state=42)

'''

#### XGB
print("Fitting The data")
model = XGBClassifier()

model.fit(X_train, y_train)
pickle.dump(model, open("/Users/Allu/Desktop/kaggle/housingPrices/pima.pickle.dat", "wb"))

loaded_model = pickle.load(open("/Users/Allu/Desktop/kaggle/housingPrices/pima.pickle.dat", "rb"))
print("predicting")
y_pred = loaded_model.predict(X_test)
print("calculating accuracy")
accuracy = mean_squared_error(y_test, y)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
'''

### Ensamble

######### SVC

# Training classifiers
'''
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=123)
clf3 = RandomForestClassifier()
clf4=AdaBoostClassifier()

eclf = VotingClassifier(estimators=[('dt', clf1), ('gb', clf2), ('rfc', clf3), ('ada', clf4)], voting='soft', weights=[2,2,2,2])
print("DecisionTreeClassifier")
clf1 = clf1.fit(X_train, y_train)
print("KNeighborsClassifier")
clf2 = clf2.fit(X_train, y_train)
print("SVC")
clf3 = clf3.fit(X_train, y_train)
print("AdaBoost")
clf4 = clf4.fit(X_train, y_train)
print("VotingClassifier")
eclf = eclf.fit(X_train, y_train)
'''
#reg = linear_model.Ridge (alpha = .5)

print("Riged Regression")

riged_alphas=[0.01,0.1, 1.0, 10.0,20,40,60,70,80,90,100]
riged_error=10000000000000.0
for _ in riged_alphas:
    reg = linear_model.Ridge (alpha = _)
    reg.fit(X_train, y_train)
    y_temp=reg.predict(X_cv)
    if mean_squared_error(y_temp, y_cv) < riged_error:
        y_t_riged = reg.predict(raw_test_data)



#y_t = reg.predict(raw_test_data)


## Lasso Regg

print("##########Lasso Regression###############")

lasso_alphas=[0.01,0.1, 1.0, 10.0,20,40,60,70,80,90,100]

#reg = linear_model.Lasso(alpha = 0.1)
lasso_error=10000000000000.0
for _ in lasso_alphas:
    reg = linear_model.Lasso (alpha = _)
    reg.fit(X_train, y_train)
    y_temp=reg.predict(X_cv)
    if mean_squared_error(y_temp, y_cv) < lasso_error:
        y_t_lasso = reg.predict(raw_test_data)

# LassoLARS


print("##########Lars Regression###############")

lars_alphas=[0.01,0.1, 1.0, 10.0,20,40,60,70,80,90,100]

#reg = linear_model.Lasso(alpha = 0.1)
lars_error=10000000000000.0
for _ in lars_alphas:
    reg = linear_model.LassoLars(alpha = _)
    reg.fit(X_train, y_train)
    y_temp=reg.predict(X_cv)
    if mean_squared_error(y_temp, y_cv) < lars_error:
        y_t_lars = reg.predict(raw_test_data)

## Baye's Regression

print("##########Bayes Regression###############")

reg = linear_model.BayesianRidge()
reg.fit(X_train, y_train)
y_bays=reg.predict(raw_test_data)

'''
## Linear Regression

print("##########Linear Regression###############")

model = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression(fit_intercept=False))])

model = model.fit(X_train, y_train)

y_lin_reg =model.predict(raw_test_data)
'''
## Mean of all Values

print("##########Mean Regression###############")

y_all = y_t_riged+y_t_lasso+y_t_lars+y_bays
y_mean = np.divide(y_all,4)


print("##########RSME Calculations###############")
'''
mse_lin_reg=mean_squared_error(y_lin_reg, y_cv)
mse_riged=mean_squared_error(y_t_riged, y_cv)
mse_lasso=mean_squared_error(y_t_lasso, y_cv)
mse_lars=mean_squared_error(y_t_lars, y_cv)
mse_bays=mean_squared_error(y_bays, y_cv)
mse_all=mean_squared_error(y_mean, y_cv)
'''

y_t=y_t_lars





'''
#clf.fit(X_train, y_train)
print("Scores")
scores = cross_val_score(eclf, X_test, y_test)

print("Score Means")
scores.mean()


print("Accuracy : ",scores)

y_pred=eclf.predict(X_test)
accuracy = mean_squared_error(y_pred, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


y_t = eclf.predict(raw_test_data)
'''

'''
y_t = np.round_(y_t,decimals=0)

output=pd.DataFrame({"Id":y_labs,"SalePrice":y_t})


output.to_csv("../input/solution.csv",index=False)

'''

#print("output",np.unique(y_test))




'''
print(raw_data.shape)
print(raw_train_data.shape)
print(raw_test_data.shape)
'''


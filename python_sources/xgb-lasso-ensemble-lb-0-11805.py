import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import copy
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso


np.random.seed(21)


data= pd.read_csv("../input/train.csv")
score= pd.read_csv("../input/test.csv")


#Missing Data Columns

#missing = pd.isnull(data).sum()
#missing = missing[missing>0]


def replaceMissingWithAnotherAverage(dataraw,missvar,groupvar):
    rv=dataraw.groupby([groupvar])[missvar].mean()
    for r in rv.index:
        idx = (dataraw[groupvar] == r) & (dataraw[missvar].isnull())
        dataraw.loc[idx, missvar] = rv.ix[r]
    return dataraw



def recodeMissing(dataraw):
    dataraw['Alley'].fillna('aNone',inplace=True)
    dataraw['MasVnrType'].fillna('None',inplace=True)
    dataraw['MasVnrArea'].fillna(0,inplace=True)
    dataraw['MiscFeature'].fillna('aNone',inplace=True)
    dataraw['Fence'].fillna('aNone',inplace=True)
    dataraw['PoolQC'].fillna('NA',inplace=True)
    dataraw['GarageCond'].fillna('aNone',inplace=True)
    dataraw['GarageQual'].fillna('NA',inplace=True)
    dataraw['GarageFinish'].fillna('aNone',inplace=True)
    dataraw['GarageType'].fillna('aNone',inplace=True)
    dataraw['FireplaceQu'].fillna('NA',inplace=True)
    dataraw['Electrical'].fillna('SBrkr',inplace=True)
    dataraw['BsmtFinType1'].fillna('Unf',inplace=True)
    dataraw['BsmtFinType2'].fillna('Unf',inplace=True)
    dataraw['BsmtExposure'].fillna('No',inplace=True)
    dataraw['BsmtCond'].fillna('TA',inplace=True)
    dataraw['BsmtQual'].fillna('TA',inplace=True)
    dataraw = replaceMissingWithAnotherAverage(dataraw,'LotFrontage','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'GarageYrBlt','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'LotFrontage','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'GarageYrBlt','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'BsmtHalfBath','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'BsmtFullBath','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'GarageCars','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'BsmtFinSF2','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'GarageArea','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'BsmtFinSF1','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'TotalBsmtSF','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'BsmtUnfSF','Neighborhood')
    return dataraw


data =recodeMissing(data)
score = recodeMissing(score)

data =data[data['GrLivArea']<4000]


#Missing Data Rows


cat = ['Street','Alley','Utilities','CentralAir','LandSlope','GarageFinish','PavedDrive',
       'PoolQC','LotShape','LandContour','MasVnrType','ExterQual','BsmtQual','BsmtCond',
       'BsmtExposure','KitchenQual','Fence','MiscFeature','MSZoning','LotConfig','BldgType',
       'ExterCond','HeatingQC','Electrical','FireplaceQu','GarageQual','GarageCond','RoofStyle',
       'Foundation','BsmtFinType1','BsmtFinType2','Heating','GarageType','SaleCondition',
       'Functional','Condition2','HouseStyle','RoofMatl','Condition1','SaleType','MoSold',
        'Exterior1st','MSSubClass','Exterior2nd','Neighborhood'
       ]

ratio = ['BsmtHalfBath','HalfBath','BsmtFullBath','KitchenAbvGr','FullBath','Fireplaces',
         'GarageCars','BedroomAbvGr','OverallCond','OverallQual','TotRmsAbvGrd','PoolArea',
         '3SsnPorch','LowQualFinSF','MiscVal','ScreenPorch','LotFrontage','EnclosedPorch',
         'OpenPorchSF','BsmtFinSF2','WoodDeckSF','MasVnrArea','GarageArea','2ndFlrSF','BsmtFinSF1',
         'TotalBsmtSF','1stFlrSF','BsmtUnfSF','GrLivArea','LotArea']


years = ['YrSold','YearRemodAdd','GarageYrBlt','YearBuilt']



translist=['BsmtQual','BsmtCond', 'ExterCond', 'ExterQual','GarageCond','GarageQual','KitchenQual',
           'FireplaceQu','PoolQC','HeatingQC']



def chunkYears(alldata,ratio):
    addtoratio=[]
    addtocat=[]
    for y in years:
        alldata[y+'decade']=(alldata[y]/10).round()
        alldata.loc[:,y+'decade']=alldata.loc[:,y+'decade'].fillna(999)
        alldata.loc[:,y+'decade']=alldata.loc[:,y+'decade'].astype('str')
        addtocat.append(y+"decade")
        alldata.loc[:,y]=2010-alldata[y]
        alldata.ix[alldata[y] < 0,y] = 0
        addtoratio.append(y)
    alldata['newBuild']=1*[alldata['YrSold']<=1][0]
    addtocat.append("newBuild")
    return alldata,addtoratio,addtocat



data,ratio1,cat1 = chunkYears(data,ratio)
score,ratio2,cat2 = chunkYears(score,ratio)

ratio = ratio+ratio1
cat=cat+cat1



def recodeQualRatings(alldata):
    chgdict = {}
    addtoratio=[]
    for t in translist:
        alldata.loc[:,t+'num']=alldata[t]
        thisval=5
        chgdict[t+'num']={}
        addtoratio.append(t+'num')
        for thisVal in ['Ex','Gd','TA','Fa','Po','NA']:
            chgdict[t+'num'][thisVal]=thisval
            thisval-=1
        chgdict[t+'num']['aNone']=0
        alldata.loc[:,t+'num']=alldata[t+'num'].fillna('NA')
    alldata.replace(to_replace=chgdict,inplace=True)
    return alldata,addtoratio



data,addtoratio1 = recodeQualRatings(data)
score,addtoratio2 = recodeQualRatings(score)

ratio = ratio+addtoratio1


def transformSF(alldata):
    #alldata['1stFlr_2ndFlr_Sf'] = np.log1p(alldata['1stFlrSF'] + alldata['2ndFlrSF'])
    #alldata['All_Liv_SF'] = np.log1p(alldata['1stFlr_2ndFlr_Sf'] + alldata['LowQualFinSF'] + alldata['GrLivArea'])
    #alldata['TotalSF']=alldata['GrLivArea']+alldata['TotalBsmtSF']
    alldata['allPorch']=alldata['WoodDeckSF']+alldata['OpenPorchSF']+alldata['EnclosedPorch']+alldata['3SsnPorch']+alldata['ScreenPorch']
    alldata['avgRoomSize']=alldata['GrLivArea']/alldata['TotRmsAbvGrd']
    alldata['lotDepth']=alldata['LotArea']/alldata['LotFrontage']
    alldata['netYard']=alldata['LotArea']-alldata['GarageArea']-alldata['1stFlrSF']-alldata['PoolArea']
    #alldata['smallHouse']=1*[alldata['GrLivArea']<=800][0]
    #alldata['bigHouse']=1*[alldata['GrLivArea']>=3500][0]
    #alldata['wideFront']=1*[alldata["LotFrontage"]>=150][0]
    #alldata['bigYard']=1*[alldata['LotArea']>35000][0]
    return alldata


data = transformSF(data)
score = transformSF(score)

ratio.append("netYard")
ratio.append('lotDepth')
ratio.append('avgRoomSize')
ratio.append('allPorch')




def neighborHoodScore(df,npdict=None):
    if npdict is None:
        nprice = df.groupby("Neighborhood")['SalePrice'].mean()
        npdict = {}
        for neigh in nprice.index:
            loadval=0
            if 100000<nprice[neigh]<=139000:
                loadval=1
            elif 139000<nprice[neigh]<=199000:
                loadval=2
            elif 199000<nprice[neigh]<=250000:
                loadval=3
            elif nprice[neigh]>250000:
                loadval=4
            npdict[neigh]=loadval
    df["Nval"] = df["Neighborhood"].map(npdict)
    return df,npdict


data,npdict=neighborHoodScore(data)
score,npdict = neighborHoodScore(score,npdict)

ratio.append('Nval')
cat.append('Nval')


def transformContinuous(alldata):
    tmpratio=[]
    for sfvar in ratio:
        if 'missing' not in sfvar:
            varname = 'trans_log_'+sfvar
            alldata[varname]=np.log(alldata[sfvar]+1.)
            tmpratio.append(varname)
    return alldata,tmpratio


data,ratio1 = transformContinuous(data)
score,ratio3 = transformContinuous(score)

ratio =ratio+ratio1


def dropZerosRecodeLow(alldata):
    drop=[]
    for c in alldata.columns:
        if len(alldata[c].value_counts(dropna=False))==1:
            drop.append(c)
            if c in cat:
                cat.remove(c)
            if c in ratio:
                ratio.remove(c)
    recodethreshold=1
    for c in cat:
        q=alldata[c].value_counts()
        q=q[q<recodethreshold]
        if len(q)>1:
            to_recode = list(q.index)
            alldata[c].replace(to_recode,value='LOW__',inplace=True)
    return alldata,drop


data,drop = dropZerosRecodeLow(data)
score,dropy = dropZerosRecodeLow(score)



def dummyfy(alldata,isTraining =True):
    d1 = alldata[['Id']+cat]
    d1=d1.set_index('Id')
    dummies = pd.get_dummies(d1,sparse=True,columns=cat,drop_first=True)
    if isTraining:
        vars =['Id','SalePrice']+ratio
    else:
        vars = ['Id']+ratio
    d2 = alldata[vars]
    d2=d2.set_index('Id')
    sparserat = d2.to_sparse()
    analyze = dummies.join(sparserat)
    return analyze


trainall = dummyfy(data)
score = dummyfy(score,isTraining=False)


for c in score.columns:
    if c not in trainall.columns:
        trainall[c]=0.


finalsetdrop=[]
trainalldrop=[]


for c in trainall.columns:
    if c not in score.columns:
        if c!='SalePrice':
            trainalldrop.append(c)


trainall = trainall.drop(trainalldrop,1)


train=trainall
score.fillna(0,inplace=True)


blocklist = list(train.columns[(train.sum()==2)])+list(train.columns[(train.sum()==0)])+list(train.columns[(train.sum()==1)])


trainvars=[x for x in train.columns if x!='SalePrice']
trainvars = [x for x in trainvars if x not in blocklist]


scaler = MinMaxScaler()
t2=copy.deepcopy(train)
s2 = copy.deepcopy(score)
t2[trainvars]=scaler.fit_transform(train[trainvars])
s2[trainvars]=scaler.transform(score[trainvars])
train = t2
score=s2


X_train = train[trainvars]
y_train = np.log(train['SalePrice'])

lassomodel = Lasso(alpha=.0005,selection = 'random',max_iter=100000,random_state=99).fit(X_train, y_train)


params = {
    'min_child_weight': 4.,
    'eta': 0.01,
    'base_score':7.8,
    'colsample_bylevel':.2,
    'colsample_bytree': .2,
    'max_depth': 5,
    'subsample': .2,
    'booster':'gbtree',
    'alpha': 0.4,
    'lambda':.6,
    'gamma': 0,
    'silent': 1,
    'verbose_eval': True,
    'seed': 2001
}


xgtrain=xgb.DMatrix(X_train,label=y_train)
xgmodel = xgb.train( params, xgtrain, num_boost_round=15000,verbose_eval=1, obj = None)


lasso_preds = np.exp(lassomodel.predict(score[trainvars]))
xgb_preds = np.exp(xgmodel.predict(xgb.DMatrix(score[trainvars])))

# LB 0.11744

ens = .35*xgb_preds+.65*lasso_preds
submission = pd.DataFrame()
submission['Id'] = score.index
submission["SalePrice"] = ens
submission.to_csv("submit_ensemble.csv", index=False)




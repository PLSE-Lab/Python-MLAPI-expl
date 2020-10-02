#!/usr/bin/env python
# coding: utf-8

# # Clean and more accurate results with converted features
# 
# In this project, I tried to differentiate from other examples as much as I can, especially in the data cleaning part. But as long as the dataset is the same and most effective approaches were already studied, there are of course similarities with other Kernels, in other words it is just another example as good as everyone's work, thank you for checking in advance!
# 
# 
# With this study I got 0.12135% Score, but I tried already trained ML models in this work with more complex Model Combination methods like adding a meta-model on averaged base models and it generated better results with around 10%-11%. But I decided not to put them, because I want to improve my expertise on these methods before tring to explain. 
# 

# In[ ]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ## Reading and cleaning data and Exploratory Data Analysis 
# 
# At first I imported both train and test dataset into different dataframes and merged them, which is going to help me in cleaning part. 
# 

# In[ ]:


train = pd.read_csv("../input/train.csv")
train.info()


# In[ ]:


test=pd.read_csv("../input/test.csv")
test.info()


# In[ ]:


trainnum = train._get_numeric_data()


# In[ ]:


traincat = train.drop(trainnum.columns, axis=1)


# Since only the train data has included SalePrice data, I separated them into categorical and numeric and then plot their relation with the target variable(SalePrice). For the categorical variables, for understanding the range more clearly, I took log of SalePrice. It allowed me to see the differences btw. different categories more clearly. 

# In[ ]:


for x in trainnum.drop('SalePrice', axis=1).columns:
    sns.regplot(x, "SalePrice", data=trainnum)
    plt.xlabel(x)
    plt.ylabel("SalePrice")
    plt.show()


# From the scatterplots of all numerical variables compared with the SalePrice, it can easily be said that, all variables have some affect on SalePrice. But it could be recognized from graph that there are some infulencer points, which would drop model prediction accuracy if I would not handle. So under the second topic called "Dealing with Missing Data" I will drop them. 

# In[ ]:


for x in traincat.columns:
    sns.boxplot(x, np.log(train.SalePrice), data=traincat)
    plt.xlabel(x)
    plt.ylabel("SalePrice")
    plt.show()


# Due to comparison graphs of categorical variables, most of them seems as expected. For example regarding to anyone of the Quality variables Excellent conditions corresponds to higher prices. However, I wolud like to mention that I saw among them as little unexpected results; it shows that "Utilities" dataset doesn't carry a useful information at all, because almost all of observations are in same category. Moreover BsmtFinType1 and 2 variables did not affect SalePrice that much. However I will keep them anyway because one from each category has little effects. But I am going to drop Utilities variable. 
# 
# After that, I will check the nr of missing values in each variable and correlations btw SalePrice and other numerical variables. 

# In[ ]:


total = pd.concat([train, test], sort=False).set_index('Id')


# In[ ]:


total.isnull().sum()


# In[ ]:


for col in trainnum.drop(['Id', 'SalePrice'],axis=1).columns :
    if (abs(train.SalePrice.corr(trainnum[col]))>0.2):
        print('Correlation btw SalePrice vs',trainnum[col].name, ':', round(train.SalePrice.corr(trainnum[col]),2))


# Since the correlation between to variable means one's effect to the compared one, the higher correlation resulted with a strong relation between variables regardless its sign(+/-). So I printed higher correlations than 0.2 to see which variable has stronger affect on SalePrice. 
# It seems according to the results, OverallQual got the highest score of 0.79, in other words the most important one in terms of its affect on SalePrice. However it may change after variable transformation. 

# ## Dealing with Missing Data

# At this stage, I coppied original dataset and tried to do some feature transformations on it. Since "data is expensive" argument is valid, I tried to find some solutions to fill Null cells in order to keep the dataset still unless it has not any affect or possibility to mislead my prediction models. And when dealing with this, I got help from data explanations in the information page of the competition. 
# 
# For example regarding some variables, as explained in the information page, the Null data means "not any available" rather than missing information and I will fill them regarding to this meaning. As an example to be clear, for example if the "Alley" column is Null, then there is not any Alley option at that specific houses. But when I was implementing that approach, I also changed some of the columns into dummy variables, in other words for example if there is an Alley access I converted it into 1, regardless its type of access, otherwise 0, because most of this variable is Null. I used same approach for MiscFeature variable as well because of same reason as before. And I also converted CentralAir variables into dummy. 

# In[ ]:


train.drop(train[(train.GrLivArea>4000) & (train.SalePrice<400000)].index, inplace=True)


# In[ ]:


train.drop(train[(train.TotRmsAbvGrd>12) & (train.SalePrice<250000)].index, inplace=True)


# In[ ]:


train.drop(train.loc[(train.TotalBsmtSF>6000) & (train.SalePrice<200000)].index, inplace=True)


# In[ ]:


train.drop(train.loc[(train.BsmtFinSF1>5000) & (train.SalePrice<200000)].index, inplace=True)


# In[ ]:


train.drop(train.loc[(train.PoolArea>500) & (train.SalePrice>700000)].index, inplace=True)


# In[ ]:


train.drop(train.loc[(train.MiscVal>3000)].index, inplace=True)


# In[ ]:


train.reset_index(drop=True, inplace=True)


# In[ ]:


total = pd.concat([train, test], sort=False).set_index('Id')


# In[ ]:


total.Alley.unique()


# In[ ]:


total.Alley.isnull().sum()


# In[ ]:


total.Alley[total.Alley=='Pave'].count()


# In[ ]:


total2 = total[:]


# In[ ]:


total2.Alley = total2.Alley.replace(total2.Alley.loc[(total2.Alley.isnull()==True)],0).replace(total2.Alley.loc[(total2.Alley!=0)],1)


# In[ ]:


total2.Alley.isnull().sum()


# In[ ]:


total3 = total2[:]


# In[ ]:


total3.groupby("MiscFeature").MiscFeature.count()


# In[ ]:


total3.MiscFeature = total3.MiscFeature.replace(total3.MiscFeature.loc[(total3.MiscFeature.isnull()==True)],0).replace(total3.MiscFeature.loc[(total3.MiscFeature!=0)],1)


# In[ ]:


total3.CentralAir = total3.CentralAir.replace(total3.CentralAir.loc[(total3.CentralAir=="Y")],1).replace(total3.CentralAir.loc[(total3.CentralAir=="N")],0)


# In[ ]:


total3.columns


# After that, I used different approaches to fill missing data. For numerical variables, I filled them with 0, and for categorical it was most common to fill. 

# In[ ]:


for x in ('MSZoning', 'Exterior1st','Exterior2nd','Electrical','BsmtFullBath','BsmtHalfBath','KitchenQual','SaleType') :
    total3[x] = total3[x].fillna(total3[x].mode()[0])


# In[ ]:


for x in ('MasVnrType', 'BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1','FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageQual', 'GarageCond','PoolQC','Fence') :
    total3[x] = total3[x].fillna("None")


# In[ ]:


total3[["BsmtFinSF1","BsmtUnfSF","TotalBsmtSF","BsmtFinSF2"]].loc[total3.BsmtFinSF2.isnull()==True]


# In[ ]:


for x in ('MasVnrArea', "BsmtFinSF1","BsmtUnfSF","TotalBsmtSF","BsmtFinSF2","BsmtFinType2",'GarageCars', 'GarageArea') :
    total3[x] = total3[x].fillna(0)


# In[ ]:


total3.groupby("Utilities").Utilities.count()


# In[ ]:


total3 = total3.drop(["Utilities"], axis=1)


# For the LotFrontage column, I firstly checked if it effects SalePrice differently according to its neighborhood location. And I saw it differantiates by Neighborhood. For example, 50sqft of LotFrontage area has lower prices in BrkSide than Blmngtn. So when I filling missing data in that column, I took into account Neigborhood data as well, and fill them according to their Neighborhood's average sqft. 

# In[ ]:


train.groupby(["Neighborhood","LotFrontage"]).SalePrice.mean().head(15)


# In[ ]:


total3.LotFrontage = total3.LotFrontage.fillna(total3.groupby(["Neighborhood"]).LotFrontage.mean()[0])


# In[ ]:


total3.Functional=total3.Functional.fillna(method='ffill')


# On the other hand, I thought YearBuilt variable is some kind of subcolumn of YearRemodAdd, because if a house is remodeled, then it may counted at that age, but if not, then it carries same information as YearBuilt. So I dropped YearBuilt and convert both YearRemodAdd, YrSold and GarageYrBuilt variables into numeric by substract them from 2011, I selected 2011 because the dataset's latest variables are from 2010, so I expect with this trading is it will give me the age rather than date as a continous variable. And when I was working on it I saw a GarageBuilt year of 2207 and I assumed it would be 2007 because of that specific house's built year and it would be a typo, so I changed it into 2007 first and then convert them. And also in GarageYrBuilt variable, there were some missing data, but since I aimed them to convert numeric, although NaN in that column means there is not any Garage, I replaced all missing data with 1880, which was 15 year lower than minimum of that column and would have less effect on SalePrice. 

# In[ ]:


total3 = total3.drop(["YearBuilt"], axis=1)


# In[ ]:


total3.loc[total3.GarageYrBlt=='None'].GarageType.unique()


# In[ ]:


total3.loc[(total3.GarageYrBlt=='None')&(total3.GarageType=='Detchd')].YearRemodAdd


# In[ ]:


total3.loc[total3.GarageYrBlt=='None', 'GarageYrBlt']=total3.YearRemodAdd


# In[ ]:


total3.loc[total3.GarageYrBlt==2207].YearRemodAdd


# In[ ]:


total3.loc[total3.GarageYrBlt==2207, 'GarageYrBlt']=total3.YearRemodAdd


# In[ ]:


min(total.GarageYrBlt)


# In[ ]:


total3.GarageYrBlt.replace("None",1800, inplace=True)


# In[ ]:


total3.GarageYrBlt = 2011 - total3.GarageYrBlt


# In[ ]:


total3=total3.rename(columns = {'GarageYrBlt':'GrAge'})


# In[ ]:


min(total3.GrAge)


# In[ ]:


max(total3.YearRemodAdd)


# In[ ]:


total3.YearRemodAdd = 2011 - total3.YearRemodAdd


# In[ ]:


total3=total3.rename(columns = {'YearRemodAdd':'HAge'})


# In[ ]:


min(total3.HAge)


# In[ ]:


max(total3.YrSold)


# In[ ]:


total3.YrSold = 2011 - total3.YrSold


# In[ ]:


total3=total3.rename(columns = {'YrSold':'SoldAge'})


# In[ ]:


min(total3.SoldAge)


# In[ ]:


total3.isnull().any().sum()


# From this check it approved that there is not any Null data remains, it resulted as 1, because it counts only True answers and there is only missing data in SalePrice now. 

# ## Feature Transformation
# 
# Since it requires to predict continous data, I first plan to use Linear Regression. So, at this part, firstly I will transform nearly all columns into numeric for better result. After that, I will jump to Prediction part and will create a valid model.
# 
# I can simply say that, I will scale all variables and convert them into values btw -1 to 1 and I also converted some of them with mean normalization in order to scale them with zero mean value, but scaling gave better results than this approach. So in my last study, I did not use mean normalization. 
# 
# Moreover, for the categorical variables, I change them into numeric manually if there is an order btw categories and then scale them, otherwise, I converted them into dummy variables. 
# 
# Additionally since I used Lasso, Ridge, ElasticNet as Regularized Linear Regression and also Gradient Boosting Regressor and XGBoost. 

# In[ ]:


plt.figure(figsize=[10,6])
sns.distplot(np.log(total3.loc[total3.SalePrice.isnull()!=True].SalePrice))


# In[ ]:


total3 = total3.replace({'Street': {'Pave': 1, 'Grvl': 0 },
                             'FireplaceQu': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'None': 0 
                                            },
                             'Fence':      {'GdPrv': 4, 
                                            'GdWo': 3, 
                                            'MnPrv': 2, 
                                            'MnWw': 1,
                                            'None': 0},
                             'ExterQual':  {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1
                                            },
                             'ExterCond':  {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1
                                            },
                             'BsmtQual':   {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'None': 0},
                            'BsmtExposure':{'Gd': 3, 
                                            'Av': 2, 
                                            'Mn': 1,
                                            'No': 0,
                                            'None': 0},
                             'BsmtCond':   {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'None': 0},
                             'GarageQual': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'None': 0},
                             'GarageCond': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'None': 0},
                            'KitchenQual': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1},
                             'Functional': {'Typ': 7,
                                            'Min1':6,
                                            'Min2':5,
                                            'Mod': 4,
                                            'Maj1':3,
                                            'Maj2':2,
                                            'Sev': 1,
                                            'Sal': 0},
                             'BsmtFinType1': {'GLQ' : 6,
                                              'ALQ' : 5,
                                              'BLQ' : 4,
                                              'Rec' : 3,
                                              'LwQ' : 2,
                                              'Unf' : 1,
                                              'None': 0},
                            'BsmtFinType2': {'GLQ' : 6,
                                              'ALQ' : 5,
                                              'BLQ' : 4,
                                              'Rec' : 3,
                                              'LwQ' : 2,
                                              'Unf' : 1,
                                              'None': 0},
                             'HeatingQC':   {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1
                                            },
                            'PoolQC' :    {'Ex': 4, 
                                            'Gd': 3, 
                                            'Fa': 2,
                                            'None': 1
                                            }
                            })


# In[ ]:


def NormVariable(x):
    x= (x - np.mean(x))/(max(x)-min(x))
    return(x)


# In[ ]:


def ScaleVariable(x):
    x = x/max(x)
    return(x)


# In[ ]:


total4 = total3[:]


# In[ ]:


total4.columns


# In[ ]:


for col in total4[['MSSubClass', 'MSZoning', 'Street','LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood',
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st','Exterior2nd', 'MiscFeature',
    'MasVnrType','Foundation',  'Heating', 'Electrical','GarageType','GarageFinish', 'PavedDrive','MoSold', 'SaleType',
                   'SaleCondition']]:
    col = pd.get_dummies(total4[col])
    total4 = total4.merge(col, left_index=True, right_index=True)


# In[ ]:


total4 = total4.drop(['MSSubClass', 'MSZoning', 'Street','LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood',
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st','Exterior2nd', 
    'MasVnrType','Foundation',  'Heating', 'Electrical','GarageType','GarageFinish', 'PavedDrive','MoSold', 'SaleType',
                      'SaleCondition'], axis=1)


# In[ ]:


for col in total4[[]]:
    total4[col] = NormVariable(pd.to_numeric(total4[col]))


# In[ ]:


for col in total4[['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                  'PoolQC','KitchenQual','GarageQual','GarageCond','Fence','OverallQual','OverallCond','Functional',
                   'FireplaceQu','HeatingQC', 'TotRmsAbvGrd','Fireplaces','GarageCars','Fireplaces', 'BedroomAbvGr']]:
    total4[col] = ScaleVariable(pd.to_numeric(total4[col]))


# In[ ]:


for col in total4[['SalePrice','SoldAge','GrAge','HAge','MiscVal','LotFrontage', 'LotArea','MasVnrArea','PoolArea',
                   'GrLivArea','GarageArea','WoodDeckSF', 'OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                   'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF','1stFlrSF','2ndFlrSF', 'LowQualFinSF']]:
    total4[col] = np.log(total4[col]+1)


# In[ ]:


test2 = total4.loc[total4.SalePrice.isnull()==True]


# In[ ]:


train2 = total4.loc[total4.SalePrice.isnull()==False]


# In[ ]:


from sklearn.linear_model import RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV,LinearRegression
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.model_selection import KFold,cross_val_score


# In[ ]:


import xgboost as xgb


# In[ ]:


ytrain=train2.SalePrice


# In[ ]:


train3 = train2[:]


# In[ ]:


train3 = train3.drop('SalePrice',1)


# In[ ]:


test3 = test2[:]


# In[ ]:


test3 = test3.drop('SalePrice',1)


# In[ ]:


kfolds = KFold(n_splits=10, shuffle=True, random_state=35)


# In[ ]:


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train3, ytrain, scoring="neg_mean_squared_error", cv = kfolds))
    return(rmse)


# ## Lasso
# 

# In[ ]:


model_lasso = LassoCV(alphas = [1, 0.1, 0.05, 0.001, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009], cv=kfolds).fit(train3, ytrain)


# In[ ]:


rmse_cv(model_lasso).mean()


# In[ ]:


coef = pd.Series(model_lasso.coef_, index = train3.columns)


# In[ ]:


print("Lasso picked " + str(sum(coef != 0)) + "variables at total")


# In[ ]:


imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])


# In[ ]:


plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients by Importance")


# In[ ]:


lasso_preds = np.expm1(model_lasso.predict(test3))-1


# In[ ]:


solution = pd.DataFrame({"id":test.Id, "SalePrice":lasso_preds})
solution.to_csv("submit_lasso.csv", index = False)


# ## Ridge

# In[ ]:


Ridge = RidgeCV(alphas = [0.01,0.1,0.5,1,3,5,10,20,30,50,100]).fit(train3, ytrain)


# In[ ]:


rmse_cv(Ridge).mean()


# In[ ]:


ridge_preds = np.expm1(Ridge.predict(test3))-1


# In[ ]:


solution = pd.DataFrame({"id":test.Id, "SalePrice":ridge_preds})
solution.to_csv("submit_ridge.csv", index = False)


# ## ElasticNet
# This time I am going to try Elastic-net, which is useful when there are multiple features which are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.

# In[ ]:


Enet = ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000, cv=kfolds).fit(train3, ytrain)


# In[ ]:


rmse_cv(Enet).mean()


# In[ ]:


enet_preds = np.expm1(Enet.predict(test3))-1


# In[ ]:


solution = pd.DataFrame({"id":test.Id, "SalePrice":enet_preds})
solution.to_csv("submit_enet.csv", index = False)


# ## Gradient Boosting Regressor

# In[ ]:


gbmodel = GradientBoostingRegressor(learning_rate=0.05, loss='huber', min_samples_split=10, n_estimators=3000,
                                       random_state=35).fit(train3, ytrain)


# In[ ]:


rmse_cv(gbmodel).mean()


# In[ ]:


gb_preds = np.expm1(gbmodel.predict(test3))-1


# ## XGBoost
# 
# Lastly I tried to use XGBoost in my dataset like other models, however it returned with duplicated column name error. So for trying this model, I had to find duplicated columns, so I used a function I got from [this link in Github](https://github.com/pandas-dev/pandas/issues/11250) and with this function I found which columns were duplicated. And then I changed them and created a new train and test dataset. With this I trained XGBoost, but it did not provide better result than regularized linear models. 

# In[ ]:


def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            iv = vs.iloc[:,i].tolist()
            for j in range(i+1, lcs):
                jv = vs.iloc[:,j].tolist()
                if iv == jv:
                    dups.append(cs[i])
                    break

    return dups


# In[ ]:


duplicate_columns(total4)


# In[ ]:


total4.columns.values


# In[ ]:


total5 = total4[:]


# In[ ]:


total5.columns = (['LotFrontage', 'LotArea', 'Alley', 'OverallQual', 'OverallCond',
       'HAge', 'MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       'HeatingQC', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GrAge',
       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
       'MiscVal', 'SoldAge', 'SalePrice', 20, 30, 40, 45, 50, 60, 70, 75,
       80, 85, 90, 120, 150, 160, 180, 190, 'C (all)', 'FV', 'RH', 'RL',
       'RM', '0_x', '1_x', 'IR1', 'IR2', 'IR3', 'Reg', 'Bnk', 'HLS',
       'Low', 'Lvl', 'Corner', 'CulDSac', 'FR2', 'FR3', 'Inside', 'Gtl',
       'Mod', 'Sev', 'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr',
       'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV',
       'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt',
       'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr',
       'Timber', 'Veenker', 'Artery_x', 'Feedr_x', 'Norm_x', 'PosA_x',
       'PosN_x', 'RRAe', 'RRAn_x', 'RRNe', 'RRNn_x', 'Artery_y',
       'Feedr_y', 'Norm_y', 'PosA_y', 'PosN_y', 'RRAn_y', 'RRNn_y',
       '1Fam', '2fmCon', 'Duplex', 'Twnhs', 'TwnhsE', '1.5Fin', '1.5Unf',
       '1Story', '2.5Fin', '2.5Unf', '2Story', 'SFoyer', 'SLvl', 'Flat',
       'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed', 'CompShg', 'Membran',
       'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl', 'AsbShng_x',
       'AsphShn_x', 'BrkComm', 'BrkFace_x', 'CBlock_x', 'CemntBd',
       'HdBoard_x', 'ImStucc_x', 'MetalSd_x', 'Plywood_x', 'Stone_x1',
       'Stucco_x', 'VinylSd_x', 'Wd Sdng_x', 'WdShing', 'AsbShng_y',
       'AsphShn_y', 'Brk Cmn', 'BrkFace_y', 'CBlock_y', 'CmentBd',
       'HdBoard_y', 'ImStucc_y', 'MetalSd_y', 'Other', 'Plywood_y',
       'Stone_y1', 'Stucco_y', 'VinylSd_y', 'Wd Sdng_y', 'Wd Shng', '0_y',
       '1_y', 'BrkCmn', 'BrkFace', 'None_x', 'Stone_x', 'BrkTil',
       'CBlock', 'PConc', 'Slab', 'Stone_y', 'Wood', 'Floor', 'GasA',
       'GasW', 'Grav', 'OthW', 'Wall', 'FuseA', 'FuseF', 'FuseP', 'Mix',
       'SBrkr', '2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort',
       'Detchd', 'None_y', 'Fin', 'None', 'RFn', 'Unf', 'N', 'P', 'Y', 1,
       2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'COD', 'CWD', 'Con', 'ConLD',
       'ConLI', 'ConLw', 'New', 'Oth', 'WD', 'Abnorml', 'AdjLand',
       'Alloca', 'Family', 'Normal', 'Partial'])


# In[ ]:


testxgb = total5.loc[total5.SalePrice.isnull()==True]


# In[ ]:


testxgb.drop('SalePrice',1, inplace=True)


# In[ ]:


trainxgb = total5.loc[total5.SalePrice.isnull()==False]


# In[ ]:


trainxgb.drop('SalePrice',1, inplace=True)


# In[ ]:


def rmse_cv1(model):
    rmse= np.sqrt(-cross_val_score(model, trainxgb, ytrain, scoring="neg_mean_squared_error", cv = kfolds))
    return(rmse)


# In[ ]:


model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1).fit(trainxgb, ytrain)


# In[ ]:


rmse_cv1(model_xgb).mean()


# In[ ]:


xgb_preds = np.expm1(model_xgb.predict(testxgb))-1


# It seems Lasso had the best results, it was also proven by Kaggle via submission scores. However, as my last submission, I calculate average of all predictions generated by models and submit it. After submission it scored as 0.12135. 

# In[ ]:


pred = (lasso_preds + ridge_preds + enet_preds + xgb_preds + gb_preds)/5


# In[ ]:


solution = pd.DataFrame({"id":test.Id, "SalePrice":pred})
solution.to_csv("submit_comb.csv", index = False)


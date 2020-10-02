# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew

le = LabelEncoder()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_one_hot(df, col_name, fill_val):
    if fill_val is not None:
        df[col_name].fillna(fill_val, inplace=True)

    dummies = pd.get_dummies(df[col_name], prefix="_" + col_name)
    df = df.join(dummies)
    df = df.drop([col_name], axis=1)
    return df
    
def factorize(df , column, fill_na=None):
    le = LabelEncoder()
    if fill_na is not None:
        df[column].fillna(fill_na, inplace=True)
    le.fit(df[column].unique())
    df[column] = le.transform(df[column])
    return df


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

def process(df):
    df['MSSubCls'] = df['MSSubClass']
    df = get_one_hot(df,'MSSubCls',None)
    df = factorize(df,'MSSubClass',None)

    # df = get_one_hot(df, 'MSZoning',None)

    neighborhood = df['Neighborhood']
    zone = df['MSZoning']

    corr = pd.crosstab(neighborhood,zone)
    print(corr)
    # zone = df.iloc[[1]].apply(lambda row: print(corr.loc[row['Neighborhood']].idxmax(axis=1)) ,axis=1)
    # print(df.iloc[[1]])
    zone = df.apply(lambda row: (corr.loc[row['Neighborhood']]).idxmax(axis=1) if pd.isnull(row['MSZoning']) else row['MSZoning'],axis=1)
    df = df.drop(['MSZoning'],axis=1)
    df['MSZoning'] = zone
    df = get_one_hot(df,'MSZoning',None)

    lot_frontage_by_neighborhood = df["LotFrontage"].groupby(df["Neighborhood"])
    for key, group in lot_frontage_by_neighborhood:
        idx = (df["Neighborhood"] == key) & (df["LotFrontage"].isnull())
        df.loc[idx, "LotFrontage"] = group.median() 
    
    df['Street'].fillna('Grvl', inplace=True)
    street_dict = {'Grvl':0,'Pave':1}
    df['Street'].replace(street_dict, inplace=True)

    # df['LotArea'] = df['LotArea'].map(lambda x:np.log1p(x))


 
    # print(df['Alley'].mode())


    df['Alley'].fillna('NA', inplace=True)  
    print(df.Alley.unique())
    #because no NA values there
    alley_dict = {'Grvl':1,'Pave':2,"NA":1} 
    df['Alley'].replace(alley_dict, inplace=True)


    # df['LotShape'].fillna('Grvl', inplace=True)
    #because no NA values there
    # print(df['LotShape'].mode())
    df['LotShape'].fillna('Reg', inplace=True)
    lot_dict = {'Reg':1,'IR1':2,"IR2":3,"IR3":4} 
    df['LotShape'].replace(lot_dict, inplace=True)

    df['LandContour'] = df['LandContour'].apply(lambda v: random.choice(['lvl','Bnk','HLS','Low']) if pd.isnull(v) else v)
    df['LandContour'].fillna('lvl', inplace=True)
    contour_dict = {'lvl':0,'Bnk':1,"HLS":1,"Low":-1} 
    df['LandContour'] = df['LandContour'].map(contour_dict)

    df = df.drop(['Utilities'], axis=1)

    df = get_one_hot(df,'LotConfig',None)

    df['LandSlope'].fillna('Gtl', inplace=True)
    slope_dict = {'Gtl':1,'Mod':2,"Sev":3} 
    df['LandSlope'] = df['LandSlope'].map(slope_dict)
    # print(df.LandSlope.unique())

    # Bin by neighborhood (a little arbitrarily). Values were computed by: 
    # train_df["SalePrice"].groupby(train_df["Neighborhood"]).median().sort_values()
    
        
    df.loc[df.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1   
    df.loc[df.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1
    df.loc[df.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1
    df.loc[df.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1
    df.loc[df.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1
    df["Neighborhood_Good"].fillna(0, inplace=True)

    neighborhood_map = {
        "MeadowV" : 0,  #  88000
        "IDOTRR" : 1,   # 103000
        "BrDale" : 1,   # 106000
        "OldTown" : 1,  # 119000
        "Edwards" : 1,  # 119500
        "BrkSide" : 1,  # 124300
        "Sawyer" : 1,   # 135000
        "Blueste" : 1,  # 137500
        "SWISU" : 2,    # 139500
        "NAmes" : 2,    # 140000
        "NPkVill" : 2,  # 146000
        "Mitchel" : 2,  # 153500
        "SawyerW" : 2,  # 179900
        "Gilbert" : 2,  # 181000
        "NWAmes" : 2,   # 182900
        "Blmngtn" : 2,  # 191000
        "CollgCr" : 2,  # 197200
        "ClearCr" : 3,  # 200250
        "Crawfor" : 3,  # 200624
        "Veenker" : 3,  # 218000
        "Somerst" : 3,  # 225500
        "Timber" : 3,   # 228475
        "StoneBr" : 4,  # 278000
        "NoRidge" : 4,  # 290000
        "NridgHt" : 4,  # 315000
    }

    df["NeighborhoodBin"] = df["Neighborhood"].map(neighborhood_map)
    df = get_one_hot(df,'Neighborhood',None)

    for condition in df.Condition1.unique():
        if not pd.isnull(condition):
            df[condition] = df['Condition1'].map(lambda x: 1 if x == condition else 0)
            df[condition] = df[condition] + df['Condition2'].map(lambda x: 1 if x == condition and condition!='Norm' else 0)
        
    df = df.drop(['Condition1','Condition2'], axis=1)        
    # print(df.Norm.unique())
    print(df['BldgType'].mode())
    
    df['hsStl'] = df['HouseStyle']
    df['bldtyp'] = df['BldgType']
    df = factorize(df,'bldtyp',None)
    df = factorize(df,'hsStl',None)

    df = get_one_hot(df,'BldgType',None)

    df = get_one_hot(df,'HouseStyle',None)



# drop_cols = [
#                 "_Exterior1st_ImStucc", "_Exterior1st_Stone",
#                 "_Exterior2nd_Other","_HouseStyle_2.5Fin", 
            
#                 "_RoofMatl_Membran", "_RoofMatl_Metal", "_RoofMatl_Roll",
#                 "_Condition2_RRAe", "_Condition2_RRAn", "_Condition2_RRNn",
#                 "_Heating_Floor", "_Heating_OthW",

#                 "_Electrical_Mix", 
#                 "_MiscFeature_TenC",
#                 "_GarageQual_Ex", "_PoolQC_Fa"
#             ]
            
# df.drop(drop_cols, axis=1, inplace=True)
    year_bin = [i+2000 for i in range(11,0,-1)]
    print(year_bin)
    year_bin = year_bin+[2000,1990,1980,1970,1960,1950,1940,1920,1900,-1]
    df['YearRemodAdd'].fillna(df.YearBuilt, inplace=True)
    for i in range(1,len(year_bin)):
        df['built_'+str(year_bin[i])] = 0
        df.loc[(df.YearBuilt >= year_bin[i]) & (df.YearBuilt < year_bin[i-1]),'built_'+ str(year_bin[i])] = 1
        df['remod_'+str(year_bin[i])] = 0
        df.loc[(df.YearRemodAdd >= year_bin[i]) & (df.YearRemodAdd < year_bin[i-1]),'remod_'+ str(year_bin[i])] = 1
    
    year_map = pd.concat(pd.Series("YearBin" + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))

    df["GarageYrBlt"].fillna(0.0, inplace=True)

    df["GarageYrBltBin"] = df.GarageYrBlt.map(year_map)
    df["GarageYrBltBin"].fillna("NoGarage", inplace=True)
    df = get_one_hot(df, "GarageYrBltBin", None)



    df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    df["Remodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]) * 1
    
    df["RecentRemodel"] = (df["YearRemodAdd"] == df["YrSold"]) * 1

    df["VeryNewHouse"] = (df["YearBuilt"] == df["YrSold"]) * 1

    df["Age"] = 2010 - df["YearBuilt"]
    df["TimeSinceSold"] = 2010 - df["YrSold"]

    df["SeasonSold"] = df["MoSold"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                                                  6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).astype(int)
    
    df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    # df = df.drop(['YearBuilt','YearRemodAdd'], axis=1)        

    df = get_one_hot(df,'RoofStyle',df['RoofStyle'].mode()[0])
    df = get_one_hot(df,'RoofMatl',df['RoofMatl'].mode()[0])

    df['Exterior1st'].fillna(df['Exterior1st'].mode()[0], inplace=True)
    df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0], inplace=True)
    for condition in df.Exterior1st.unique():
        if not pd.isnull(condition):
            df[condition] = df['Exterior1st'].map(lambda x: 1 if x == condition else 0)
            df[condition] = df[condition] + df['Exterior2nd'].map(lambda x: 1 if x == condition else 0)
            df.loc[df[condition]>1,condition] = 1
        
    df = df.drop(['Exterior1st','Exterior2nd','GarageYrBlt'], axis=1)

    df["MasVnrArea"].fillna(0, inplace=True)
    idx = (df["MasVnrArea"] != 0) & ((df["MasVnrType"] == "None") | (df["MasVnrType"].isnull()))
    df.loc[idx, "MasVnrType"] = "BrkFace"
    df = get_one_hot(df,'MasVnrType','None')

    df['ExterQual'].fillna('TA', inplace=True)
    qual_dict = {'Ex':4,'Gd':3,"TA":2,"Fa":1,"Po":0} 
    df['ExterQual'].replace(qual_dict, inplace=True)
    
    df = get_one_hot(df,'Foundation',df['Foundation'].mode()[0])

    df['BsmtQual'].fillna('TA', inplace=True)
    qual_dict = {'Ex':4,'Gd':3,"TA":2,"Fa":1,"Po":0,"NA":-2} 
    df['BsmtQual'].replace(qual_dict, inplace=True)
    df['BsmtCond'].replace(qual_dict, inplace=True)

    # df['BsmtExposure'].fillna('TA', inplace=True)
    qual_dict = {'Gd':3,'Av':2,"Mn":1,"No":0,None:-2} 
    df['BsmtExposure'].replace(qual_dict, inplace=True)
    # df['BsmtExposure'].replace(qual_dict, inplace=True)

# bsmt_fin_dict = {None: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
# df["BsmtFinType1"] = df["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
# df["BsmtFinType2"] = df["BsmtFinType2"].map(bsmt_fin_dict).astype(int)



    df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0], inplace=True)
    df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0], inplace=True)
    for condition in df.BsmtFinType1.unique():
        if not pd.isnull(condition):
            df[condition] = df['BsmtFinType1'].map(lambda x: 1 if x == condition else 0)
            df[condition] = df[condition] + df['BsmtFinType2'].map(lambda x: 1 if x == condition else 0)
            df.loc[df[condition]>1,condition] = 1
        
    df = df.drop(['BsmtFinType1','BsmtFinType2'], axis=1)

    qual_dict = {'None':-1,"NA":-1,None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    df["ExterCond"] = df["ExterCond"].map(qual_dict).astype(int)
    df["HeatingQC"] = df["HeatingQC"].map(qual_dict).astype(int)
    
        # description says NA = no pool, but there are entries with PoolArea >0 and PoolQC = NA. Fill the ones with values with average condition
    df.loc[(df['PoolQC'].isnull()) & (df['PoolArea']==0), 'PoolQC' ] = 'None'
    df.loc[(df['PoolQC'].isnull()) & (df['PoolArea']>0), 'PoolQC' ] = 'TA'
    df["PoolQC"] = df["PoolQC"].map(qual_dict).astype(int)
    df["KitchenQual"] = df["KitchenQual"].map(qual_dict).astype(int)
    df["FireplaceQu"] = df["FireplaceQu"].map(qual_dict).astype(int)
    df["GarageQual"] = df["GarageQual"].map(qual_dict).astype(int)
    df["GarageCond"] = df["GarageCond"].map(qual_dict).astype(int)



    df["Functional"] = df["Functional"].map(
        {None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, 
         "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)
         
    df["SimplOverallQual"] = df.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
    df["SimplOverallCond"] = df.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
    df["SimplPoolQC"] = df.PoolQC.replace({1 : 1, 2 : 1, # average
                                             3 : 2, 4 : 2 # good
                                            })
    df["SimplGarageCond"] = df.GarageCond.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
    df["SimplGarageQual"] = df.GarageQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
    df["SimplFireplaceQu"] = df.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
    df["SimplFireplaceQu"] = df.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
    df["SimplFunctional"] = df.Functional.replace({1 : 1, 2 : 1, # bad
                                                     3 : 2, 4 : 2, # major
                                                     5 : 3, 6 : 3, 7 : 3, # minor
                                                     8 : 4 # typical
                                                    })
    df["SimplKitchenQual"] = df.KitchenQual.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
    df["SimplHeatingQC"] = df.HeatingQC.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
                                                  


    electric_dict = {'SBrkr':5,'FuseA':0,'Mix':2,'FuseF':-1,'FuseP':-2}
    df['Electrical'].replace(electric_dict, inplace=True)

    air_dict = {'N':0,'Y':1}
    df['CentralAir'].replace(air_dict, inplace=True)



    garage_dict = {'Fin':2,'RFn':1,'Unf':-1,'NA':-5}
    df['GarageFinish'].replace(garage_dict, inplace=True)

    pave_dict = {'Y':1,'P':0,'N':-1}
    df['PavedDrive'].replace(pave_dict, inplace=True)

    df["Fence"] = df["Fence"].map(
            {None: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)
    df = get_one_hot(df,'Fence',0)


    #dealing with area related fields.
    df["BsmtFinSF1"].fillna(0, inplace=True)
    df["BsmtFinSF2"].fillna(0, inplace=True)
    df["BsmtUnfSF"].fillna(0, inplace=True)
    df["TotalBsmtSF"].fillna(0, inplace=True)
    df["GarageArea"].fillna(0, inplace=True)
    df["BsmtFullBath"].fillna(0, inplace=True)
    df["BsmtHalfBath"].fillna(0, inplace=True)
    df["GarageCars"].fillna(0, inplace=True)
    df["PoolArea"].fillna(0, inplace=True)


    df["IsRegularLotShape"] = (df["LotShape"] == 1) * 1

    # Most properties are level; bin the other possibilities together
    # as "not level".
    df["IsLandLevel"] = (df["LandContour"] == 0) * 1

    # Most land slopes are gentle; treat the others as "not gentle".
    df["IsLandSlopeGentle"] = (df["LandSlope"] == 1) * 1

    # Most properties use standard circuit breakers.
    df["IsElectricalSBrkr"] = (df["Electrical"] == 5) * 1

    # About 2/3rd have an attached garage.
    df["IsGarageDetached"] = (df["GarageType"] == "Detchd") * 1

    # Most have a paved drive. Treat dirt/gravel and partial pavement
    # as "not paved".
    df["IsPavedDrive"] = (df["PavedDrive"] == 1) * 1

    # The only interesting "misc. feature" is the presence of a shed.
    df["HasShed"] = (df["MiscFeature"] == "Shed") * 1.  



    df["Has2ndFloor"] = (df["2ndFlrSF"] == 0) * 1
    df["HasMasVnr"] = (df["MasVnrArea"] == 0) * 1
    df["HasWoodDeck"] = (df["WoodDeckSF"] == 0) * 1
    df["HasOpenPorch"] = (df["OpenPorchSF"] == 0) * 1
    df["HasEnclosedPorch"] = (df["EnclosedPorch"] == 0) * 1
    df["Has3SsnPorch"] = (df["3SsnPorch"] == 0) * 1
    df["HasScreenPorch"] = (df["ScreenPorch"] == 0) * 1


    # Months with the largest number of deals may be significant.
    df["HighSeason"] = df["MoSold"].replace( 
        {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

    df["NewerDwelling"] = df["MSSubClass"].replace(
        {20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
         90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})   


    df["SaleCondition_PriceDown"] = df.SaleCondition.replace(
        {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})

    # House completed before sale or not
    df["BoughtOffPlan"] = df.SaleCondition.replace(
        {"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})
    

    area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea' ]

    df["TotalArea"] = df[area_cols].sum(axis=1)

    df["TotalArea1st2nd"] = df["1stFlrSF"] + df["2ndFlrSF"]
    df['All_Liv_SF'] = np.log1p(df['TotalArea1st2nd'] + df['LowQualFinSF'] + df['GrLivArea'])
    
    for col in area_cols+['TotalArea','TotalArea1st2nd']:
        df[col].map(lambda x:np.log1p(x))
        
    area_cols = area_cols+['TotalArea','TotalArea1st2nd','All_Liv_SF']
    
    sub_df = df[area_cols]
    array_standard = StandardScaler().fit_transform(sub_df)
    df_standard = pd.DataFrame(array_standard, df.index, area_cols)
    df.drop(df[area_cols], axis=1, inplace=True)
    df = pd.concat([df, df_standard], axis=1)
    
    
    df = get_one_hot(df,'Functional','Typ')
    df = get_one_hot(df,'GarageType','NA')

    df = get_one_hot(df, "SaleType", "WD")
    df = get_one_hot(df, "SaleCondition", "Normal")                                                  
    df = get_one_hot(df,'Heating','none')
    df = get_one_hot(df,'MiscFeature','none')
    df['month'] = df['MoSold']
    df = get_one_hot(df,'MoSold','none')

    drop_cols = [
                "_Exterior1st_ImStucc", "_Exterior1st_Stone",
                "_Exterior2nd_Other","_HouseStyle_2.5Fin", 
            
                "_RoofMatl_Membran", "_RoofMatl_Metal", "_RoofMatl_Roll",
                "_Condition2_RRAe", "_Condition2_RRAn", "_Condition2_RRNn",
                "_Heating_Floor", "_Heating_OthW",

                "_Electrical_Mix", 
                "_MiscFeature_TenC",
                "_GarageQual_Ex", "_PoolQC_Fa","_MSSubCls_150","_Condition2_PosN",    # only two are not zero
    "_MSZoning_C (all)",
    "_MSSubCls_160", 'Stone', 'ImStucc','_HouseStyle_7','_RoofMatl_ClyTile','_Functional_0'
            ]
    for col in drop_cols:
        try:
            df.drop([col], axis=1, inplace=True)
        except:
            continue
    
    return df
    

label_df = pd.DataFrame(index = train_df.index, columns=["SalePrice"])
label_df["SalePrice"] = np.log(train_df["SalePrice"])
train_df.drop(['SalePrice'], axis=1, inplace=True)

# numeric_feats = train_df.dtypes[train_df.dtypes != "object"].index
# skewed_feats = train_df[numeric_feats].apply(lambda x: skew(x.dropna().astype(float))) #compute skewness
# skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]
# skewed_feats = skewed_feats.index

train_df = process(train_df)
# train_df[skewed_feats] = np.log1p(train_df[skewed_feats])

test_df.loc[666, "GarageQual"] = "TA"
test_df.loc[666, "GarageCond"] = "TA"
test_df.loc[666, "GarageFinish"] = "Unf"
test_df.loc[666, "GarageYrBlt"] = "1980"

# The test example 1116 only has GarageType but no other information. We'll 
# assume it does not have a garage.
test_df.loc[1116, "GarageType"] = np.nan

# numeric_feats = test_df.dtypes[test_df.dtypes != "object"].index
# skewed_feats = test_df[numeric_feats].apply(lambda x: skew(x.dropna().astype(float))) #compute skewness
# skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]
# skewed_feats = skewed_feats.index

test_df = process(test_df)
# test_df[skewed_feats] = np.log1p(test_df[skewed_feats])

train_df.sort_index(axis=1, inplace=True)
test_df.sort_index(axis=1,inplace=True)
# print (len(list(train_df)))
# print (len(list(test_df)))
print (set(list(test_df))-set(list(train_df)))


print("Training set size:", train_df.shape)
print("Test set size:", test_df.shape)

################################################################################

# XGBoost -- I did some "manual" cross-validation here but should really find
# these hyperparameters using CV. ;-)

import xgboost as xgb

regr = xgb.XGBRegressor(
                 colsample_bytree=0.8,
                 colsample_bylevel = 0.8,
                 gamma=0.01,
                 learning_rate=0.05,
                 max_depth=5,
                 min_child_weight=1.5,
                 n_estimators=6000,                                                                  
                 reg_alpha=0.5,
                 reg_lambda=0.5,
                 subsample=0.7,
                 seed=42,
                 silent=1)

regr.fit(train_df, label_df)

# Run prediction on training set to get a rough idea of how well it does.
y_pred = regr.predict(train_df)
y_test = label_df
print("XGBoost score on training set: ", rmse(y_test, y_pred))

# Run prediction on the Kaggle test set.
y_pred_xgb = regr.predict(test_df)

################################################################################

from sklearn.linear_model import LassoCV

best_alpha = 0.0001
train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(train_df.mean())

regr = LassoCV(eps=10**-6, n_alphas=75, max_iter=150000)
regr.fit(train_df, label_df)

# Run prediction on training set to get a rough idea of how well it does.
y_pred = regr.predict(train_df)
y_test = label_df
print("Lasso score on training set: ", rmse(y_test, y_pred))

# Run prediction on the Kaggle test set.
y_pred_lasso = regr.predict(test_df)

################################################################################

################################################################################

from sklearn.linear_model import ElasticNet

# I found this best alpha through cross-validation.
best_alpha = 0.0005
# train_df = train_df.fillna(train_df.mean())
# test_df = test_df.fillna(train_df.mean())

# regr = ElasticNet(alpha=best_alpha, max_iter=50000)
# regr.fit(train_df, label_df)

# # Run prediction on training set to get a rough idea of how well it does.
# y_pred = regr.predict(train_df)
# y_test = label_df
# print("Elastic score on training set: ", rmse(y_test, y_pred))

# # # Run prediction on the Kaggle test set.
# y_pred_elastic = regr.predict(test_df)

################################################################################

# Blend the results of the two regressors and save the prediction to a CSV file.

y_pred = (y_pred_xgb) / 1.0
y_pred = np.exp(y_pred)-1.0

pred_df = pd.DataFrame(y_pred, index=test_df["Id"], columns=["SalePrice"])
pred_df.to_csv('output.csv', header=True, index_label='Id')
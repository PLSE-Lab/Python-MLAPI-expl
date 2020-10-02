# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
import matplotlib.pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
data = pd.concat([train, test])

MSSubClass = data["MSSubClass"].replace([20, 30, 40, 45, 50, 120, 150], 1) \
.replace([60, 70, 75, 160], 2) \
.replace([80, 85, 90, 180, 190], 0)

new = pd.DataFrame({"MSSubClass": MSSubClass})

new["MSZoning"] = data["MSZoning"].map(lambda x: 1 if (x == "RL") or (x == "RM") else 0)

new["LotFrontage"] = data["LotFrontage"].interpolate()

new["LotArea"] = np.log(data["LotArea"])

# dropping Street due to lack of variation in data

new["Alley"] = data["Alley"].map(lambda x: 1 if (x == "Grvl") or (x == "Pave") else 0)

new["LotShape"] = data["LotShape"].map(lambda x: 1 if (x=="Reg") else 0)

new["LandContour"] = data["LandContour"].replace('Lvl', 1) \
.replace('Bnk', 2) \
.replace('HLS', 3) \
.replace('Low', 4)

# dropping Utilities due to lack of variation in data

new["LotConfig"] = data["LotConfig"].replace(['Inside', 'Corner'], 1) \
.replace('CulDSac', 2) \
.replace(['FR2', 'FR3'], 3)

new["LandSlope"] = data["LandSlope"].replace("Gtl", 1) \
.replace("Mod", 2) \
.replace("Sev", 3)

# Using k-Means to group houses in the most similar neighbourhoods together.
# This starts with giving each neighbourhood a number to be used as a distance metric for k-Means.
# Since we are unfamiliar with the actual geography of Ames, we will assume that the most
# densely populated localities are closer to each other than sparsely populated localities.
# These numbers have been assigned on that basis.


new["Neighborhood"] = data["Neighborhood"].replace(['NAmes', 'CollgCr'], 1) \
.replace(['OldTown', 'Edwards'], 2) \
.replace(['Somerst', 'Gilbert', 'NridgHt'], 3) \
.replace(['Sawyer', 'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor'], 4) \
.replace(['Mitchel', 'NoRidge', 'Timber', 'IDOTRR', 'ClearCr', 'StoneBr', 'SWISU'], 5) \
.replace(['Blmngtn', 'MeadowV', 'BrDale', 'Veenker'], 6) \
.replace(['NPkVill'], 7) \
.replace(['Blueste'], 8)

neighbor = new["Neighborhood"]
array = np.array(neighbor)
array_2d = np.reshape(array, (-1, 1))

model = KMeans(n_clusters=3)
model.fit(array_2d)

new["Neighborhood"] = data["Neighborhood"].replace(['Somerst', 'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor'], 0) \
.replace(['Mitchel', 'NoRidge', 'Timber', 'IDOTRR', 'ClearCr', 'StoneBr', 'SWISU', 'Blmngtn', 'MeadowV', 'BrDale', 'Veenker', 'NPkVill', 'Blueste'], 1) \
.replace(['NAmes', 'CollgCr', 'OldTown', 'Edwards'], 2)


new["Condition1"] = data["Condition1"].map(lambda x: 1 if (x == "Norm") else 0)

# Condition2 shows very little deviation, so dropping it.

new["BldgType"] = data["BldgType"].replace(['1Fam', 'TwnhsE'], 1) \
.replace('Duplex', 2) \
.replace(['Twnhs', '2fmCon'], 3)

new["HouseStyle"] = data["HouseStyle"].replace(['1Story', '1.5Fin', '1.5Unf'], 1) \
.replace(['2Story', '2.5Fin', '2.5Unf'], 2) \
.replace(['SLvl', 'SFoyer'], 3)

new["OverallQual"] = data["OverallQual"]

new["OverallCond"] = data["OverallCond"]

# The plot obtained between (from the lines below) the natural log of SalePrice and YearBuilt
# gave us the idea of diving YearBuilt into four categories.

# price = np.log(train["SalePrice"])
# matplotlib.pyplot.scatter(train["YearBuilt"], price)

# The houses built before 1900 are minimal and do not attribute for much of the sale price.
# The houses built roughly between 1900 and 1940 seem to effect price somewhat more.
# The houses built roughly between 1940 and 1985 show a significant amount of deviation.
# The houses built roughly between 1985 and now also show significant deviation.

new["YearBuilt"] = data["YearBuilt"]


for i in range(new["YearBuilt"].size):
    if new["YearBuilt"].iloc[i] < 1900:
        new["YearBuilt"].iloc[i] = 1
    elif new["YearBuilt"].iloc[i] > 1900 and new["YearBuilt"].iloc[i] < 1940:
        new["YearBuilt"].iloc[i] = 2
    elif new["YearBuilt"].iloc[i] > 1940 and new["YearBuilt"].iloc[i] < 1985:
        new["YearBuilt"].iloc[i] = 3
    else:
        new["YearBuilt"].iloc[i] = 4
        


# For YearRemodAdd, there is no easily observable pattern here through a lot.
# Also, there is significant variation in data for all years. It is better to simply classify these years into four categories:
# 1 - before 1900
# 2 - between 1900 and 1950
# 3 - between 1950 and 2000
# 4 - between 2000 and now

new["YearRemodAdd"] = data["YearRemodAdd"]


for i in range(new["YearRemodAdd"].size):
    if new["YearRemodAdd"].iloc[i] < 1900:
        new["YearRemodAdd"].iloc[i] = 1
    elif new["YearRemodAdd"].iloc[i] > 1900 and new["YearRemodAdd"].iloc[i] < 1950:
        new["YearRemodAdd"].iloc[i] = 2
    elif new["YearRemodAdd"].iloc[i] > 1950 and new["YearRemodAdd"].iloc[i] < 2000:
        new["YearRemodAdd"].iloc[i] = 3
    else:
        new["YearRemodAdd"].iloc[i] = 4
        


new["RoofStyle"] = np.where(data["RoofStyle"] == "Gable", 1, 0)

# Very little deviation in RoofMatl, hence dropping it.


# For Exterior1st, Labelling VinylSd as 1, HdBoard, MetalSd, Wd Sdng as 2, and others as 3.

new["Exterior1st"] = data["Exterior1st"].replace('VinylSd', 1) \
.replace(['HdBoard', 'MetalSd', 'Wd Sdng'], 2) \
.replace(['Plywood', 'CemntBd', 'BrkFace', 'WdShing', 'Stucco', 'AsbShng', 'Stone', 'BrkComm', 'AsphShn', 'ImStucc', 'CBlock'], 3)

# Exterior2nd is almost identical to Exterior1st, hence we drop it.

new["MasVnrType"] = data["MasVnrType"].fillna("None")
new["MasVnrType"] = new["MasVnrType"].map(lambda x: 0 if (x == "None") else 1)


# Ignoring MasVnrArea. We are only considering whether or not there is a masonry.


# ExterQual and ExterCond

original = data["ExterQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1})
current = data["ExterCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1})
original.value_counts()
cond = np.subtract(current, original)
cond.value_counts()
new["ExterCond"] = pd.Series(cond.map(lambda x: "depreciated" if x < 0 else "improved" if x > 0 else "no change"))
new["ExterCond"].value_counts()



new["Foundation"] = data["Foundation"].replace(["BrkTil", "Slab", "Stone", "Wood"], "Other")
new["Foundation"].value_counts()


new["BsmtQual"] = data["BsmtQual"].fillna("None").map({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 1, "None": 0})


new["BsmtCond"] = data["BsmtCond"].fillna("None").map({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 1, "None": 0})
new["BsmtCond"] = np.where(new["BsmtCond"] > 1, 1, 0)


new["BsmtExposure"] = data["BsmtExposure"].fillna("None").map({"Gd": 1, "Av": 1, "Mn": 1, "No": 0, "None": 0})


type1 = data["BsmtFinType1"].fillna("None").map({"GLQ": 3, "ALQ": 2, "Rec": 2, "BLQ": 1, "LwQ": 1, "Unf": 0, "None": 0})
type2 = data["BsmtFinType2"].fillna("None").map({"GLQ": 3, "ALQ": 2, "Rec": 2, "BLQ": 1, "LwQ": 1, "Unf": 0, "None": 0})
new["BsmtFinType"] = pd.Series(np.logical_or(type1 == 0.0, type2 == 0.0))
new["BsmtFinType"] = np.where(new["BsmtFinType"] == True, 1, 0)


data["TotalBsmtSF"] = data["TotalBsmtSF"].fillna(0)
new["LogTotalBsmtSF"] = np.log(data["TotalBsmtSF"] + 1)


Heating = pd.Series(np.where(data["Heating"] == "GasA", 1, 0))
#  dropping heating due to not enough variation


new["HeatingQC"] = data["HeatingQC"].map({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 1})


new["CentralAir"] = np.where(data["CentralAir"] == "Y", 1, 0)


new["Electrical"] = np.where(data["Electrical"] == "SBrkr", 1, 0)


new["Log1stFlrSF"] = np.log(data["1stFlrSF"])
new["SecondFlr"] = np.where(data["2ndFlrSF"] == 0, 0, 1)


new["LogGrLivArea"] = np.log(data["GrLivArea"])


data["BsmtFullBath"] = data["BsmtFullBath"].fillna(0)
full = pd.Series(np.sum([data["BsmtFullBath"], data["FullBath"]], axis = 0))
new["FullBath"] = full.replace([0], 1).replace([3, 4, 6], 3)


data["BsmtHalfBath"] = data["BsmtHalfBath"].fillna(0)
half = pd.Series(np.sum([data["BsmtHalfBath"], data["HalfBath"]], axis = 0))


new["HalfBath"] = half.replace([2, 3, 4], 1)


new["BedroomAbvGr"] = data["BedroomAbvGr"].replace([0], 1).replace([4, 5, 6, 8], 4)


new["KitchenQual"] = data["KitchenQual"].fillna("TA").map({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 1})
new["KitchenQual"] = np.where(new["KitchenQual"] > 2, 1, 0)


new["TotRmsAbvGrd"] = data["TotRmsAbvGrd"].replace([2, 3], 4).replace([10, 11, 12, 13, 14, 15], 9)


data["Functional"] = data["Functional"].fillna("Typ")
new["Functional"] = np.where(data["Functional"] == "Typ", 1, 0)


new["Fireplaces"] = np.where(data["Fireplaces"] > 0, 1, 0)


new["FireplaceQu"] = data["FireplaceQu"].fillna("None").map({"Ex": 3, "Gd": 3, "TA": 2, "Fa": 1, "Po": 1, "None": 0})


data["GarageType"] = data["GarageType"].fillna("None")
new["GarageType"] = np.where(data["GarageType"] == "Attchd", 1, 0)


data["GarageYrBlt"] = data["GarageYrBlt"].fillna(0).replace([data["GarageYrBlt"].max()], 0)
new["GarageYrBlt"] = pd.qcut(data["YearBuilt"], q = 4, labels = ["ancient", "older", "newer", "modern"])
new["GarageYrBlt"] = new["GarageYrBlt"].replace('ancient', 1) \
.replace('older', 2) \
.replace('newer', 3) \
.replace('modern', 4)


data["GarageFinish"] = data["GarageFinish"].fillna("None")
new["GarageFinish"] = data["GarageFinish"]


data["GarageCars"] = data["GarageCars"].fillna(2)
data["GarageArea"] = data["GarageArea"].fillna(data["GarageArea"].mean())
new["GarageCars"] = data["GarageCars"].replace([4, 5], 3)


data["GarageQual"] = data["GarageQual"].fillna("None")
data["GarageCond"] = data["GarageCond"].fillna("None")
original = data["GarageQual"].map({"Ex": 4, "Gd": 4, "TA": 2, "Fa": 1, "Po": 1, "None": 0})
current = data["GarageCond"].map({"Ex": 4, "Gd": 4, "TA": 2, "Fa": 1, "Po": 1, "None": 0})
cond = np.subtract(current, original)
new["GarageRemod"] = pd.Series(cond.map(lambda x: "depreciated" if x < 0 else "improved" if x > 0 else "no change"))
new["GarageRemod"] = np.where(new["GarageRemod"] == "no change", 1, 0)


new["PavedDrive"] = np.where(data["PavedDrive"] == "Y", 1, 0)


new["WoodDeck"] = np.where(data["WoodDeckSF"] == 0, 0, 1)


new["TotalPorchSF"] = np.sum([data["OpenPorchSF"], data["EnclosedPorch"], data["3SsnPorch"], data["ScreenPorch"]], axis = 0)


# dropping PoolArea and PoolQC due to lack of data


new["Fence"] = np.where(data["Fence"].isnull(), 0 , 1)


new["MoSold"] = data["MoSold"] \
.map({12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall"})
new["MoSold"] = new["MoSold"].replace('winter', 1) \
.replace('spring', 2) \
.replace('summer', 3) \
.replace('fall', 4)



new["YrSold"] = data["YrSold"]


new["SaleType"] = data["SaleType"] \
.replace(["CWD", "VWD"], "WD") \
.replace(["COD", "Con", "ConLw", "ConLI", "ConLD", "Oth"], "Other") \
.fillna("WD")


new["SaleCondition"] = np.where(data["SaleCondition"] == "Normal", 1, 0)


# creating a list of all categorical features to one hot encode them.

cat = ['MSZoning', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle',
'Exterior1st', 'MasVnrType', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'HeatingQC', 'CentralAir',
'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageRemod', 'PavedDrive', 'Fence', 'SaleType', 'SaleCondition']

# perfroming one hot encoding

for col in cat:
    new[col] = LabelEncoder().fit_transform(new[col])
    n = new[col].max()
    for i in range(n):
        col_name = col + '_' + str(i)
        new[col_name] = new[col].apply(lambda x: 1 if x == i else 0)

new = new.drop(col, axis = 1)
# print(new)
        
    
# creating train and test data from new

train_set = new[:1460]
test_set = new[1460:]


train_x = train_set.ix[:, 0:111]
train_pricelog = np.log(train["SalePrice"])

model_random = RandomForestRegressor()
model_random.fit(train_x, train_pricelog)
results_random = model_random.predict(test_set)
# results_random_final = 2.71 ** results_random
# results_random_df = pd.DataFrame(results_random_final)



model_ridge = Ridge(alpha=0.00099)
model_ridge.fit(train_x, train_pricelog)
results_ridge = model_ridge.predict(test_set)
# results_ridge_final = 2.71 ** results_ridge
# results_ridge_df = pd.DataFrame(results_ridge_final)



result_avg = (results_random + results_ridge)/2
result_avg = np.exp(result_avg)



result_df = pd.DataFrame(result_avg)
result_df.to_csv("sample_submission.csv", index=False)








#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import re

description = {}
values = {}
with open("/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt", 'r') as desc_file:
    for line in desc_file.readlines():
        if re.match(r'\w', line):
            name_desc = line.split(": ")
            name, desc = name_desc[0], name_desc[1].strip("\t\n")
            description[name] = desc
            values[name] = {}
        elif re.match(r' *\t*\n', line):
            pass
        elif re.match(r' +\w+', line):
            value_desc = line.split("\t")
            value, desc = value_desc[0].strip(), value_desc[1].strip("\n")
            values[name][value] = desc

print(values["KitchenQual"])


# In[ ]:


train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train.head()


# In[ ]:


test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test.head()


# In[ ]:


nan_counts = train.isna().sum()
nan_counts[nan_counts > 100]


# In[ ]:


nan_counts = test.isna().sum()
nan_counts[nan_counts > 100]


# In[ ]:


remove_columns = ["Id", "Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "MiscVal", "GarageYrBlt"]
df = train.drop(columns=remove_columns)
dft = test.drop(columns=remove_columns)
df.head()


# In[ ]:


nan_counts = df.isna().sum()
nan_counts[nan_counts > 0]


# In[ ]:


nan_counts = dft.isna().sum()
nan_counts[nan_counts > 0]


# In[ ]:


categorical_features = ["MSSubClass"]
ordinal_features_0 = []
ordinal_features_1 = []
numeric_features = []
for col in df:
    if isinstance(df[col][0], str):
        categorical_features.append(col)
    else:
        unique_values = df[col].unique()
        if len(unique_values) > 15:
            numeric_features.append(col)
        elif 0 in unique_values:
            ordinal_features_0.append(col)
        else:
            ordinal_features_1.append(col)
            
ordinal_features_1.remove("MSSubClass")
numeric_features.remove("SalePrice")


# In[ ]:


# from 1
to_ordinal_1 = ["ExterQual", "ExterCond", "HeatingQC", "KitchenQual", "Functional"]
# from 0
to_ordinal_0 = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "GarageFinish", "GarageQual", "GarageCond"]

to_ordinal = to_ordinal_1 + to_ordinal_0
for x in to_ordinal:
    categorical_features.remove(x)

ordinal_features_1 += to_ordinal_1
ordinal_features_0 += to_ordinal_0
ordinal_features = ordinal_features_1 + ordinal_features_0


# In[ ]:


print(categorical_features)
print(ordinal_features_0)
print(ordinal_features_1)
print(numeric_features)


# In[ ]:


nan_counts = df[ordinal_features_0].isna().sum()
nan_counts[nan_counts > 0]


# In[ ]:


variables = list(dft.columns)
basement = [x for x in variables if "Bsmt" in x]
garage = [x for x in variables if "Garage" in x]
basement_numeric = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath", "TotalBsmtSF"]
basement_non_numeric = [x for x in basement if x not in basement_numeric]


# In[ ]:


for var in garage:
    print(var, description[var])
    print(values[var])


# In[ ]:


df[basement][df.BsmtExposure.isna() & df.BsmtQual.notna()]


# In[ ]:


df[basement][df.BsmtFinType2.isna() & df.BsmtQual.notna()]


# In[ ]:


df[df.BsmtFinType1 == "GLQ"].BsmtFinType2.value_counts()


# In[ ]:


df.loc[948, "BsmtExposure"] = "No"
df.loc[332, "BsmtFinType2"] = "Unf"
df.loc[df.BsmtQual.isna(), basement_non_numeric] = "NA"


# In[ ]:


df.loc[df.GarageFinish.isna(), garage].head()


# In[ ]:


df.loc[df.GarageFinish.isna(), ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]] = "NA"


# In[ ]:


nan_counts = dft[ordinal_features_0].isna().sum()
nan_counts[nan_counts > 0]


# In[ ]:


dft[basement][dft.BsmtFullBath.isna() | dft.BsmtHalfBath.isna()]


# In[ ]:


dft.loc[660, basement_non_numeric] = "NA"
dft.loc[660, basement_numeric] = 0.0
dft.loc[728, basement_non_numeric] = "NA"
dft.loc[728, basement_numeric] = 0.0


# In[ ]:


dft[basement][dft.BsmtCond.isna() & dft.BsmtQual.notna()]


# In[ ]:


dft.loc[580, "BsmtCond"] = "Gd"
dft.loc[725, "BsmtCond"] = "TA"
dft.loc[1064, "BsmtCond"] = "TA"


# In[ ]:


dft[basement][dft.BsmtCond.notna() & dft.BsmtQual.isna()]


# In[ ]:


dft.loc[757, "BsmtQual"] = "Fa"
dft.loc[758, "BsmtQual"] = "TA"


# In[ ]:


dft[basement][dft.BsmtExposure.isna() & dft.BsmtQual.notna()]


# In[ ]:


dft.loc[27, "BsmtExposure"] = "No"
dft.loc[888, "BsmtExposure"] = "No"


# In[ ]:


dft.loc[dft.BsmtQual.isna(), basement_non_numeric] = "NA"


# In[ ]:


dft[garage][dft.GarageCars.isna()]


# In[ ]:


dft.loc[dft.GarageFinish.isna(), ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]] = "NA"
dft.loc[dft.GarageFinish.isna(), ["GarageCars", "GarageArea"]] = 0.0


# In[ ]:


nan_counts = df[ordinal_features_1].isna().sum()
nan_counts[nan_counts > 0]


# In[ ]:


nan_counts = dft[ordinal_features_1].isna().sum()
nan_counts[nan_counts > 0]


# In[ ]:


print(values["KitchenQual"])
print(dft[["OverallQual", "OverallCond"]][dft.KitchenQual.isna()])
print(dft[dft.OverallQual == 5].KitchenQual.mode())
print(dft[dft.OverallCond == 3].KitchenQual.mode())


# In[ ]:


dft.loc[95, "KitchenQual"] = "TA"


# In[ ]:


print(values["Functional"])
print(dft[["OverallQual", "OverallCond"]][dft.Functional.isna()])
print(dft[dft.OverallQual == 1].Functional.mode())
print(dft[dft.OverallCond == 5].Functional.mode())
print(dft[dft.OverallQual == 4].Functional.mode())
print(dft[dft.OverallCond == 1].Functional.mode())


# In[ ]:


dft.loc[756, "Functional"] = "Min2"
dft.loc[1013, "Functional"] = "Mod"


# In[ ]:


df2 = df.copy()
dft2 = dft.copy()

for label in to_ordinal_1:
    old_values = values[label].keys()
    new_values = reversed(range(1, len(old_values)+1))
    mapper = dict(zip(old_values, new_values))
    df2[label] = df2[label].replace(mapper)
    dft2[label] = dft2[label].replace(mapper)
    
for label in to_ordinal_0:
    old_values = values[label].keys()
    new_values = reversed(range(len(old_values)))
    mapper = dict(zip(old_values, new_values))
    df2[label] = df2[label].replace(mapper)
    dft2[label] = dft2[label].replace(mapper)


# In[ ]:


nan_counts = df2[categorical_features].isna().sum()
nan_counts[nan_counts > 0]


# In[ ]:


print(description["MasVnrType"])
print(values["MasVnrType"])


# In[ ]:


df2.loc[df2.MasVnrType.isna(),categorical_features[14:24]]


# In[ ]:


criteria = df2[(df2.RoofMatl == "CompShg") & (df2.Foundation == "PConc") & (df2.Heating == "GasA") & (df2.CentralAir == "Y") & (df2.Electrical == "SBrkr") & (df2.PavedDrive == "Y")]
criteria.MasVnrType.value_counts()


# In[ ]:


df2.loc[df2.MasVnrType.isna(),"MasVnrType"] = "None"


# In[ ]:


print(description["Electrical"])
print(values["Electrical"])


# In[ ]:


df2.loc[df2.Electrical.isna(), ["Utilities", "Heating", "CentralAir", "Electrical"]]


# In[ ]:


criteria = df2[(df2.Utilities == "AllPub") & (df2.Heating == "GasA") & (df2.CentralAir == "Y")]
criteria.Electrical.value_counts()


# In[ ]:


df2.loc[1379, "Electrical"] = "SBrkr"


# In[ ]:


nan_counts = dft2[categorical_features].isna().sum()
nan_counts[nan_counts > 0]


# In[ ]:


dft2.loc[dft2.MasVnrType.isna(),categorical_features[14:24]]


# In[ ]:


dft2.loc[dft2.MasVnrType.isna(),"MasVnrType"] = "None"


# In[ ]:


print(description["MSZoning"])
print(values["MSZoning"])


# In[ ]:


dft2.loc[dft2.MSZoning.isna(),["MSZoning", "LotConfig", "Condition2", "BldgType"]]


# In[ ]:


criteria = dft2[(dft2.LotConfig == "Inside") & (dft2.Condition2 == "Norm") & (dft2.BldgType == "1Fam")]
criteria.MSZoning.value_counts()


# In[ ]:


dft2.loc[dft2.MSZoning.isna(), "MSZoning"] = "RL"


# In[ ]:


print(description["Utilities"])
print(values["Utilities"])


# In[ ]:


dft2.loc[dft2.Utilities.isna(), ["Utilities", "Heating", "Electrical"]]


# In[ ]:


criteria = dft2[(dft2.Heating == "GasA") & (dft2.Electrical == "FuseA")]
criteria.Utilities.value_counts()


# In[ ]:


dft2.loc[[455, 485], "Utilities"] = "AllPub"


# In[ ]:


print(description["Exterior1st"])
print(values["Exterior1st"])
print(description["Exterior2nd"])
print(values["Exterior2nd"])


# In[ ]:


dft2.loc[dft2.Exterior1st.isna(), ["Exterior1st", "Exterior2nd", "HouseStyle", "RoofStyle", "RoofMatl"]]


# In[ ]:


criteria = dft2[(dft2.HouseStyle == "1Story") & (dft2.RoofStyle == "Flat") & (dft2.RoofMatl == "Tar&Grv")]
criteria.Exterior2nd.value_counts()


# In[ ]:


dft2.loc[691, ["Exterior1st", "Exterior2nd"]] = "Plywood"


# In[ ]:


print(description["SaleType"])
print(values["SaleType"])


# In[ ]:


dft2.loc[dft2.SaleType.isna(), ["SaleType", "SaleCondition"]]


# In[ ]:


dft2[dft2.SaleCondition == "Normal"].SaleType.value_counts()


# In[ ]:


dft2.loc[1029, "SaleType"] = "WD"


# In[ ]:


df2.loc[:,"MSSubClass"] = df2["MSSubClass"].astype(str)
dft2.loc[:,"MSSubClass"] = dft2["MSSubClass"].astype(str)


# In[ ]:


print(numeric_features)


# In[ ]:


nan_counts = df2.isna().sum()
nan_counts[nan_counts > 0]


# In[ ]:


print(description["MasVnrArea"])


# In[ ]:


df2.loc[df2.MasVnrArea.isna(), ["MasVnrType", "MasVnrArea"]]


# In[ ]:


df2.MasVnrArea[df2.MasVnrType == "None"].value_counts()


# In[ ]:


df2.loc[df2.MasVnrArea.isna(), "MasVnrArea"] = 0.0


# In[ ]:


print(description["LotFrontage"])
print(description["LotArea"])


# In[ ]:


df2.loc[df2.LotFrontage.isna(), ["LotFrontage", "LotArea"]].head(10)


# In[ ]:


from sklearn.linear_model import RANSACRegressor

data = df2[["LotFrontage", "LotArea"]].dropna()
x, y = data["LotArea"].to_numpy(), data["LotFrontage"].to_numpy()
x = x.reshape(-1, 1)
ransac = RANSACRegressor()
ransac.fit(x, y)
y_pred = ransac.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred, c='orange')
plt.axis([0, 40000, 0, 350])

plt.show()


# In[ ]:


x_new = df2.LotArea[df2.LotFrontage.isna()].to_numpy().reshape(-1, 1)
df2.loc[df2.LotFrontage.isna(), "LotFrontage"] = ransac.predict(x_new)


# In[ ]:


nan_counts = dft2.isna().sum()
nan_counts[nan_counts > 0]


# In[ ]:


dft2.loc[dft2.MasVnrArea.isna(), ["MasVnrType", "MasVnrArea"]]


# In[ ]:


dft2.loc[dft2.MasVnrArea.isna(), "MasVnrArea"] = 0.0


# In[ ]:


x_new = dft2.LotArea[dft2.LotFrontage.isna()].to_numpy().reshape(-1, 1)
dft2.loc[dft2.LotFrontage.isna(), "LotFrontage"] = ransac.predict(x_new)


# In[ ]:


dft2.loc[dft2.GarageCars.isna(), garage]


# In[ ]:


dft2.loc[1116, ["GarageCars", "GarageArea"]] = 0


# In[ ]:


print(df2.isna().sum().sum())
print(dft2.isna().sum().sum())


# In[ ]:


n = len(df2)
DF = pd.concat([df2.drop("SalePrice", axis=1), dft2])
DF = pd.get_dummies(DF)
df_train = DF.iloc[:n]
df_test = DF.iloc[n:]
y_train = df["SalePrice"]
df_test.iloc[:10, 46:]


# In[ ]:


import seaborn as sns

df_viz = pd.concat([y_train, df_train[numeric_features]], axis=1)
corr = df_viz.corr()
fig, ax = plt.subplots(figsize=(20, 20))
ax = sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True)
plt.show()


# In[ ]:


corr_target = corr.iloc[0].drop("SalePrice").sort_values()
numeric_selected = list(corr_target[corr_target > 0.2].index)

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(corr_target.index, corr_target)
ax.axvline(0.2, c="orange", ls="--")
ax.tick_params(labelbottom=True, labeltop=True)
plt.show()


# In[ ]:


df_viz = pd.concat([y_train, df_train[ordinal_features_0]], axis=1)
corr = df_viz.corr()
fig, ax = plt.subplots(figsize=(20, 20))
ax = sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True)
plt.show()


# In[ ]:


corr_target = corr.iloc[0].drop("SalePrice").sort_values()
ordinal_0_selected = list(corr_target[corr_target > 0.2].index)

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(corr_target.index, corr_target)
ax.axvline(0.2, c="orange", ls="--")
ax.tick_params(labelbottom=True, labeltop=True)
plt.show()


# In[ ]:


df_viz = pd.concat([y_train, df_train[ordinal_features_1]], axis=1)
corr = df_viz.corr()
fig, ax = plt.subplots(figsize=(12, 10))
ax = sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True)
plt.show()


# In[ ]:


corr_target = corr.iloc[0].drop("SalePrice").sort_values()
ordinal_1_selected = list(corr_target[corr_target > 0.2].index)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(corr_target.index, corr_target)
ax.axvline(0.2, c="orange", ls="--")
ax.tick_params(labelbottom=True, labeltop=True)
plt.show()


# In[ ]:


df_viz = pd.concat([y_train, df_train.iloc[:, 46:]], axis=1)
corr = df_viz.corr()
corr_target = corr.iloc[0].drop("SalePrice")
categorical_selected = list(corr_target[corr_target > 0.2].index)


# In[ ]:


ordinal_selected = ordinal_0_selected + ordinal_1_selected
print(numeric_selected)
print(ordinal_selected)
print(categorical_selected)


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = df_train[numeric_selected + ordinal_selected + categorical_selected]
X_test = df_test[numeric_selected + ordinal_selected + categorical_selected]
X_train.loc[:, numeric_selected + ordinal_selected] = scaler.fit_transform(df_train[numeric_selected + ordinal_selected])
X_test.loc[:, numeric_selected + ordinal_selected] = scaler.transform(df_test[numeric_selected + ordinal_selected])
X_train.head()


# In[ ]:


"""from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = df_train.copy()
X_test = df_test.copy()
X_train.loc[:, numeric_features + ordinal_features] = scaler.fit_transform(X_train[numeric_features + ordinal_features])
X_test.loc[:, numeric_features + ordinal_features] = scaler.transform(X_test[numeric_features + ordinal_features])
X_train.head()"""


# In[ ]:


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
scores = cross_val_score(lr, X_train, y_train, cv=10)

print(scores)
print(scores.mean())


# In[ ]:


from sklearn.linear_model import Ridge

rdg = Ridge()
param_grid = {'alpha': np.arange(1, 100)}
rdg_gs = GridSearchCV(rdg, param_grid, cv=5, n_jobs=-1)
rdg_gs.fit(X_train, y_train)

print(rdg_gs.best_params_)
print(rdg_gs.best_score_)


# In[ ]:


from sklearn.linear_model import Lasso

las = Lasso(max_iter=1000, tol=0.01)
param_grid = {'alpha': np.arange(1, 100)}
las_gs = GridSearchCV(las, param_grid, cv=5, n_jobs=-1)
las_gs.fit(X_train, y_train)

print(las_gs.best_params_)
print(las_gs.best_score_)


# In[ ]:


from sklearn.linear_model import LassoLars

lar = LassoLars(eps=0.001)
param_grid = {'alpha': np.arange(1, 100)}
lar_gs = GridSearchCV(lar, param_grid, cv=5, n_jobs=-1)
lar_gs.fit(X_train, y_train)

print(lar_gs.best_params_)
print(lar_gs.best_score_)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

"""rfr = RandomForestRegressor()
param_grid = {'n_estimators': [100, 500, 1000], 'max_depth': [10, 20, 30], 'min_impurity_decrease': [0.01, 0.001, 0.0001]}
rfr_gs = GridSearchCV(rfr, param_grid, cv=5, n_jobs=-1)
rfr_gs.fit(X_train, y_train)

print(rfr_gs.best_params_)
print(rfr_gs.best_score_)"""
best_params = {'max_depth': 20, 'min_impurity_decrease': 0.001, 'n_estimators': 100}
rfr = RandomForestRegressor(**best_params)
scores = cross_val_score(rfr, X_train, y_train, cv=10)
print(scores.mean())


# In[ ]:


from xgboost import XGBRegressor

"""xgb = XGBRegressor()
param_grid = {'n_estimators': [100, 500, 1000], 'max_depth': [2, 4, 6], 
              'gamma': [0.01, 0.001, 0.0001], 'learning_rate': np.arange(0.1, 1, 0.1)}
xgb_gs = GridSearchCV(xgb, param_grid, cv=5, n_jobs=-1)
xgb_gs.fit(X_train, y_train)

print(xgb_gs.best_params_)
print(xgb_gs.best_score_)"""
best_params = {'gamma': 0.01, 'learning_rate': 0.4, 'max_depth': 2, 'n_estimators': 500}
xgb = XGBRegressor(**best_params)
scores = cross_val_score(xgb, X_train, y_train, cv=10)
print(scores.mean())


# In[ ]:


xgb.fit(X_train, y_train)
predictions = xgb.predict(X_test)
output = pd.DataFrame({"Id": test.Id, "SalePrice": predictions})
output.to_csv("my_submission.csv", index=False)


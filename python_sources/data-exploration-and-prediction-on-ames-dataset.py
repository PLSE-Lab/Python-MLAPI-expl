#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings 
warnings.filterwarnings('ignore')
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


print(f" train.shape = {train.shape} \n test.shape = {test.shape}\n")


# In[ ]:


# for i in train.select_dtypes(include='object'):
#          print(f"{i} \n {train[i].value_counts().to_frame()} \n")


# In[ ]:


# Log transform the target for scoring
train.SalePrice = np.log1p(train.SalePrice)
y = train.SalePrice


# In[ ]:


len(set(train['Id'] + test['Id'])) == len(train['Id'] + test['Id'])


# So there are no rows with same 'Id'. Train & Test set have no comman 'Id'. 

# In[ ]:


data = pd.concat([train.loc[:,'SalePrice' != train.columns], test.loc[:,:]], axis=0)


# Get ready to rumble all columns:
# 
# As there are quite a lot of columns, we can to create priority for each column as whether high, medium or low in an[ excel sheet ](https://drive.google.com/file/d/1_4jVNVuoIN7m4fsPZqGHKTvbZ7U0kC_S/view?usp=sharing)with columns - Features, Priority. This is completely based on indivisual's intuition. So lets plot 'SalePrice' vrs some interesting features obtained from excel sheet to get a gist on out dataset. (Its always better to plot all columns and examine). Lets get going!!
# 

# In[ ]:


# cols = ['MSSubClass','MSZoning','LotArea','OverallCond','LotConfig','PoolArea','PoolQC','Fence','MiscFeature','MiscVal','SaleType','SaleCondition','SalePrice','Street','Alley','LotShape','LandContour',
#         'Utilities','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','YearBuilt','YearRemodAdd','Exterior1st','Foundation']
# cols_obj = train[cols].select_dtypes(include='object')
# cols_num = train[cols].select_dtypes(exclude='object')
# for col in cols_obj.columns:
#     sns.catplot(x=col, y="SalePrice", data=train, kind='box')
#     plt.show()

# for col in cols_num.columns:
#     sns.regplot(x=col, y="SalePrice", data=train)
#     plt.show()


# * So, lots of **outliers!!**
# * Some of the numeric variable considered are actually categorical variable such as 'OverallQual', 'OverallCond'. Just because they arent strings doesn't mean they are numeric( or continuous ), even nominal data here is considered as int object. So they have to be separated.
# Before moving on to fight outliers lets examine **missing values!!**

# In[ ]:


length = len(list(filter(lambda x: x>0, train.isnull().sum())))
print(length, 'variables have missing values in train set')
print(train.isnull().sum().sort_values(ascending=False)[:length])

length = len(list(filter(lambda x: x>0, test.isnull().sum())))
print(length, 'variables have missing values in test set')
print(test.isnull().sum().sort_values(ascending=False)[:length])


# As the columns [PoolQC, MiscFeature, Alley, Fence, FireplaceQu] have more than half of observations as missing its better we drop these columns.

# In[ ]:


data = data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1 )
# BsmtQual etc : data description says NA for basement features is "no basement"
data.loc[:, "BsmtQual"] = data.loc[:, "BsmtQual"].fillna("No")
data.loc[:, "BsmtCond"] = data.loc[:, "BsmtCond"].fillna("No")
data.loc[:, "BsmtExposure"] = data.loc[:, "BsmtExposure"].fillna("No")
data.loc[:, "BsmtFinType1"] = data.loc[:, "BsmtFinType1"].fillna("No")
data.loc[:, "BsmtFinType2"] = data.loc[:, "BsmtFinType2"].fillna("No")
data.loc[:, "BsmtFullBath"] = data.loc[:, "BsmtFullBath"].fillna(0)
data.loc[:, "BsmtHalfBath"] = data.loc[:, "BsmtHalfBath"].fillna(0)
data.loc[:, "BsmtUnfSF"] = data.loc[:, "BsmtUnfSF"].fillna(0)
# Electrical : NA can be most frequent value
most_frequent = data.loc[:,"Electrical"].value_counts().index[0]
data.loc[:,"Electrical"] = data.loc[:, "Electrical"].fillna(most_frequent)
# Functional : data description says NA means typical
data.loc[:, "Functional"] = data.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
data.loc[:, "GarageType"] = data.loc[:, "GarageType"].fillna("No")
data.loc[:, "GarageFinish"] = data.loc[:, "GarageFinish"].fillna("No")
data.loc[:, "GarageQual"] = data.loc[:, "GarageQual"].fillna("No")
data.loc[:, "GarageCond"] = data.loc[:, "GarageCond"].fillna("No")
data.loc[:, "GarageArea"] = data.loc[:, "GarageArea"].fillna(0)
data.loc[:, "GarageCars"] = data.loc[:, "GarageCars"].fillna(0)
# KitchenQual : NA most likely means typical
data.loc[:, "KitchenQual"] = data.loc[:, "KitchenQual"].fillna("TA")
# LotFrontage : NA most likely means no lot frontage
data.loc[:, "LotFrontage"] = data.loc[:, "LotFrontage"].fillna(0)
# MasVnrType : NA most likely means no veneer
data.loc[:, "MasVnrType"] = data.loc[:, "MasVnrType"].fillna("None")
data.loc[:, "MasVnrArea"] = data.loc[:, "MasVnrArea"].fillna(0)


# In[ ]:


data = data.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )


# In[ ]:


data["SimplOverallQual"] = data.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
data["SimplOverallCond"] = data.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
data["SimplGarageCond"] = data.GarageCond.replace({1 : 1, # bad
                                                      2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
data["SimplGarageQual"] = data.GarageQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
data["SimplFunctional"] = data.Functional.replace({1 : 1, 2 : 1, # bad
                                                     3 : 2, 4 : 2, # major
                                                     5 : 3, 6 : 3, 7 : 3, # minor
                                                     8 : 4 # typical
                                                    })
data["SimplKitchenQual"] = data.KitchenQual.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
data["SimplHeatingQC"] = data.HeatingQC.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                    4 : 2, 5 : 2 # good
                                                  })
data["SimplBsmtFinType1"] = data.BsmtFinType1.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
data["SimplBsmtFinType2"] = data.BsmtFinType2.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
data["SimplBsmtCond"] = data.BsmtCond.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
data["SimplBsmtQual"] = data.BsmtQual.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
data["SimplExterCond"] = data.ExterCond.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
data["SimplExterQual"] = data.ExterQual.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })


# In[ ]:


data["OverallGrade"] = data["OverallQual"] * data["OverallCond"]
# Overall quality of the garage
data["GarageGrade"] = data["GarageQual"] * data["GarageCond"]
# Overall quality of the exterior
data["ExterGrade"] = data["ExterQual"] * data["ExterCond"]
# Overall kitchen score
data["KitchenScore"] = data["KitchenAbvGr"] * data["KitchenQual"]
# Overall garage score
data["GarageScore"] = data["GarageArea"] * data["GarageQual"]
# Simplified overall quality of the house
data["SimplOverallGrade"] = data["SimplOverallQual"] * data["SimplOverallCond"]
# Simplified overall quality of the exterior
data["SimplExterGrade"] = data["SimplExterQual"] * data["SimplExterCond"]
# Simplified overall garage score
data["SimplGarageScore"] = data["GarageArea"] * data["SimplGarageQual"]
# Simplified overall kitchen score
data["SimplKitchenScore"] = data["KitchenAbvGr"] * data["SimplKitchenQual"]
# Total number of bathrooms
data["TotalBath"] = data["BsmtFullBath"] + (0.5 * data["BsmtHalfBath"]) + data["FullBath"] + (0.5 * data["HalfBath"])
# Total SF for house (incl. basement)
data["AllSF"] = data["GrLivArea"] + data["TotalBsmtSF"]
# Total SF for 1st + 2nd floors
data["AllFlrsSF"] = data["1stFlrSF"] + data["2ndFlrSF"]
# Total SF for porch
data["AllPorchSF"] = data["OpenPorchSF"] + data["EnclosedPorch"] + data["3SsnPorch"] + data["ScreenPorch"]
# Has masonry veneer or not
data["HasMasVnr"] = data.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
                                               "Stone" : 1, "None" : 0})
# House completed before sale or not
data["BoughtOffPlan"] = data.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})


# In[ ]:


train = pd.concat([data.iloc[:train.shape[0],:], train['SalePrice']], axis=1) 
test = data.iloc[train.shape[0]:, :]


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(train.corr())


# In[ ]:


categorical_features = data.select_dtypes(include = ["object"]).columns
numerical_features = data.select_dtypes(exclude = ["object"]).columns
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))


# In[ ]:


print("NAs for numerical features in train : ", train[numerical_features].isnull().values.sum(), test[numerical_features].isnull().values.sum())
train[numerical_features] = train[numerical_features].fillna(train.median())
test[numerical_features] = test[numerical_features].fillna(train.median())
print("Remaining NAs for numerical features in train & test: ",train[numerical_features].isnull().values.sum(),train[numerical_features].isnull().values.sum())


# In[ ]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train.loc[:,categorical_features] = imp.fit_transform(train.loc[:,categorical_features])
test.loc[:,categorical_features] = imp.transform(test.loc[:,categorical_features])


# In[ ]:


from scipy.stats import skew
skewness = train.loc[:,numerical_features].apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")


# In[ ]:


skewness = train.loc[:,numerical_features].apply(lambda x: skew(x))
pskewed = skewness[skewness > 0.5 ]
nskewed = skewness[skewness < -0.5 ]
print(pskewed.shape[0], nskewed.shape[0])


# In[ ]:


train.loc[:, pskewed.index] = np.log1p(train.loc[:, pskewed.index])
train.loc[:, nskewed.index] = np.log1p(train.loc[:, nskewed.index])
#train.loc[:, nskewed.index] = np.exp(train.loc[:, nskewed.index])
test.loc[:, pskewed.index] = np.log1p(test.loc[:, pskewed.index])
test.loc[:, nskewed.index] = np.log1p(test.loc[:, nskewed.index])
#test.loc[:, nskewed.index] = np.exp(test.loc[:, nskewed.index])


# In[ ]:


data = pd.concat([train.loc[:,'SalePrice' != train.columns], test.loc[:,:]], axis=0)


# In[ ]:


data_cat = pd.get_dummies(data.loc[:, categorical_features])
data = pd.concat([data.loc[:,numerical_features], data_cat[:][:]], axis=1)


# In[ ]:


train = pd.concat([data.iloc[:train.shape[0],:], train['SalePrice']], axis=1) 
test = data.iloc[train.shape[0]:, :]
train_X = train.loc[:, train.columns != 'SalePrice']
train_y = train.loc[:, 'SalePrice']


# So now i guess we have preprocessed data finally!!

# In[ ]:


import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

#make_pipeline
pipe = make_pipeline(StandardScaler(), XGBRegressor())
pipe.fit(train_X, train_y)
pred = pipe.predict(test)

#cross_val_score
scores = cross_val_score(pipe, train_X, train_y, scoring='neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * scores.mean()))
print(scores)

parameters = {'xgbregressor__objective':['reg:linear'],
              'xgbregressor__learning_rate': [.03, 0.05, .07], #so called `eta` value
              'xgbregressor__max_depth': [4, 5, 6, 7],
              'xgbregressor__min_child_weight': [4,3,5],
              'xgbregressor__n_estimators': [500]}

model = GridSearchCV(pipe, param_grid = parameters, cv=5)
model.fit(train_X, train_y)
print("best score:",model.best_score_)
print("best params:",model.best_params_)

# print("best features:",model.feature_importances_)


# In[ ]:


Id = test['Id']
fin_score = pd.DataFrame({'SalePrice': np.expm1(model.predict(test))})
fin_data = pd.concat([Id,fin_score],axis=1)
fin_data.to_csv('House_Prices_submit.csv', sep=',', index = False)


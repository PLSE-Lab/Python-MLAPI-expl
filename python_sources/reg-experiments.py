#Part 1 - Python
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 20:35:28 2017

@author: Denes
"""
#Libraries
import pandas as pd
import numpy as np
from scipy import stats
import sklearn.linear_model as lm
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb

#Functions
def write_to_file(pred_array):
    m_temp = np.zeros((len(pred_array), 2))
    
    for i in range(len(m_temp)):
        m_temp[i, 0] = i + 1461
        m_temp[i, 1] = pred_array[i]
        
    df_sample = pd.DataFrame(m_temp,columns = ['Id', 'SalePrice']).astype(int)
    
    del m_temp, i
    
    df_sample.to_csv("F:/Code/Python/4 House/submission_p.csv", index = False)
    return print("Submission saved to csv file.")

#Data import
np.random.seed(117)
df_train = pd.read_csv("F:/Code/Python/4 House/train.csv")
df_test = pd.read_csv("F:/Code/Python/4 House/test.csv")

end_train_x = df_train.iloc[:, 1:80]
end_train_y = df_train.iloc[:, -1]
end_test_x = df_test.iloc[:, 1:80]

del df_train, df_test

df_all = pd.concat((end_train_x, end_test_x), axis = 0, ignore_index = True)

del end_train_x, end_test_x

#Pre-processing
d_nan = {}
for column in enumerate(df_all.columns):
    d_nan[column] = np.sum(df_all.loc[:,column].isnull())[1]
del column

df_all.ix[df_all.iloc[:, 1].isnull() == True, 1] = np.argmax(pd.value_counts(df_all.iloc[:, 1]))
df_all.ix[df_all.iloc[:, 2].isnull() == True, 2] = 0
df_all.ix[df_all.iloc[:, 5].isnull() == True, 5] = 'None'
df_all.ix[df_all.iloc[:, 7].isnull() == True, 7] = 'None'
df_all.ix[df_all.iloc[:, 8].isnull() == True, 8] = np.argmax(pd.value_counts(df_all.iloc[:, 8]))
df_all.ix[df_all.iloc[:, 9].isnull() == True, 9] = 0
df_all.ix[df_all.iloc[:, 22].isnull() == True, 22] = np.argmax(pd.value_counts(df_all.iloc[:, 22]))
df_all.ix[df_all.iloc[:, 23].isnull() == True, 23] = np.argmax(pd.value_counts(df_all.iloc[:, 23]))
df_all.ix[(df_all.iloc[:, 24].isnull() == True) & (df_all.iloc[:, 25] != 0) & (df_all.iloc[:, 25].isnull() == False), 24] = np.argmax(pd.value_counts(df_all.ix[(df_all.iloc[:, 24] != 'None'), 24]))
df_all.ix[(df_all.iloc[:, 24] != 'None') & (df_all.iloc[:, 25] == 0), 24] = 'None'
df_all.ix[(df_all.iloc[:, 24] == 'None') & (df_all.iloc[:, 25] != 0) & (df_all.iloc[:, 25].isnull() == False), 24] = np.argmax(pd.value_counts(df_all.ix[(df_all.iloc[:, 24] != 'None'), 24]))
df_all.ix[df_all.iloc[:, 24].isnull() == True, 24] = np.argmax(pd.value_counts(df_all.iloc[:, 24]))
df_all.ix[df_all.iloc[:, 25].isnull() == True, 25] = np.argmax(pd.value_counts(df_all.iloc[:, 25]))
df_all.ix[df_all.iloc[:, 37].isnull() == True, 37] = 0
df_all.ix[(df_all.iloc[:, 37] == 0) & (df_all.iloc[:, 29].isnull() == True), 29] = 'None'
df_all.ix[(df_all.iloc[:, 37] == 0) & (df_all.iloc[:, 30].isnull() == True), 30] = 'None'
df_all.ix[(df_all.iloc[:, 37] == 0) & (df_all.iloc[:, 31].isnull() == True), 31] = 'None'
df_all.ix[(df_all.iloc[:, 37] == 0) & (df_all.iloc[:, 32].isnull() == True), 32] = 'None'
df_all.ix[(df_all.iloc[:, 37] == 0) & (df_all.iloc[:, 33].isnull() == True), 33] = 0
df_all.ix[(df_all.iloc[:, 37] == 0) & (df_all.iloc[:, 34].isnull() == True), 34] = 'None'
df_all.ix[(df_all.iloc[:, 37] == 0) & (df_all.iloc[:, 35].isnull() == True), 35] = 0
df_all.ix[(df_all.iloc[:, 37] == 0) & (df_all.iloc[:, 36].isnull() == True), 36] = 0
df_all.ix[df_all.iloc[:, 29].isnull() == True, 29] = np.argmax(pd.value_counts(df_all.iloc[:, 29]))
df_all.ix[df_all.iloc[:, 30].isnull() == True, 30] = np.argmax(pd.value_counts(df_all.iloc[:, 30]))
df_all.ix[df_all.iloc[:, 31].isnull() == True, 31] = np.argmax(pd.value_counts(df_all.iloc[:, 31]))
df_all.ix[df_all.iloc[:, 34].isnull() == True, 34] = np.argmax(pd.value_counts(df_all.iloc[:, 34]))
df_all.ix[df_all.iloc[:, 41].isnull() == True, 41] = np.argmax(pd.value_counts(df_all.iloc[:, 41]))
df_all.ix[df_all.iloc[:, 46].isnull() == True, 46] = np.argmax(pd.value_counts(df_all.iloc[:, 46]))
df_all.ix[df_all.iloc[:, 47].isnull() == True, 47] = np.argmax(pd.value_counts(df_all.iloc[:, 47]))
df_all.ix[df_all.iloc[:, 52].isnull() == True, 52] = np.argmax(pd.value_counts(df_all.iloc[:, 52]))
df_all.ix[df_all.iloc[:, 54].isnull() == True, 54] = np.argmax(pd.value_counts(df_all.iloc[:, 54]))
df_all.ix[df_all.iloc[:, 56].isnull() == True, 56] = 'None'
df_all.ix[(df_all.iloc[:, 57].isnull() == False) & (df_all.iloc[:, 58].isnull() == True), 58] = df_all.ix[(df_all.iloc[:, 57].isnull() == False) & (df_all.iloc[:, 58].isnull() == True), 18]
df_all.ix[(df_all.iloc[:, 60] == 0) & (df_all.iloc[:, 57].isnull() == True), 57] = 'None'
df_all.ix[(df_all.iloc[:, 60] == 0) & (df_all.iloc[:, 58].isnull() == True), 58] = 0
df_all.ix[(df_all.iloc[:, 60] == 0) & (df_all.iloc[:, 59].isnull() == True), 59] = 'None'
df_all.ix[df_all.iloc[:, 59].isnull() == True, 59] = np.argmax(pd.value_counts(df_all.iloc[:, 59]))
df_all.ix[df_all.iloc[:, 60].isnull() == True, 60] = np.argmax(pd.value_counts(df_all.iloc[:, 60]))
df_all.ix[df_all.iloc[:, 61].isnull() == True, 61] = np.argmax(pd.value_counts(df_all.ix[(df_all.iloc[:, 61] != 0) & (df_all.iloc[:, 60] == np.argmax(pd.value_counts(df_all.iloc[:, 60]))), 61]))
df_all.ix[(df_all.iloc[:, 61] == 0) & (df_all.iloc[:, 62].isnull() == True), 62] = 'None'
df_all.ix[(df_all.iloc[:, 61] == 0) & (df_all.iloc[:, 63].isnull() == True), 63] = 'None'
df_all.ix[df_all.iloc[:, 62].isnull() == True, 62] = np.argmax(pd.value_counts(df_all.iloc[:, 62]))
df_all.ix[df_all.iloc[:, 63].isnull() == True, 63] = np.argmax(pd.value_counts(df_all.iloc[:, 63]))
df_all.ix[df_all.iloc[:, 71].isnull() == True, 71] = 'None'
df_all.ix[df_all.iloc[:, 72].isnull() == True, 72] = 'None'
df_all.ix[df_all.iloc[:, 73].isnull() == True, 73] = 'None'
df_all.ix[df_all.iloc[:, 77].isnull() == True, 77] = np.argmax(pd.value_counts(df_all.iloc[:, 77]))

df_all.loc[(df_all['YrSold'] - df_all['YearRemodAdd'] < 0), 'YearRemodAdd'] = df_all['YrSold']
df_all.loc[np.max(df_all.loc[:,'YrSold']) < df_all.loc[:,'GarageYrBlt'], 'GarageYrBlt'] = 2007

df_all['Street'] = (df_all['Street'] == 'Pave').astype(int)

df_neighborhood = pd.concat([df_all.ix[0:len(end_train_y), 'Neighborhood'], end_train_y], axis = 1)
df_neighborhood = df_neighborhood.groupby('Neighborhood')['SalePrice'].mean().reset_index()
df_neighborhood = df_neighborhood.iloc[np.argsort(df_neighborhood.iloc[:, 1])]
df_neighborhood.describe()
df_all['Class'] = 'UltraRich'
df_all.loc[df_all['Neighborhood'].isin(df_neighborhood.ix[df_neighborhood.iloc[:, 1] < 300000, 0]) , 'Class'] = 'Rich'
df_all.loc[df_all['Neighborhood'].isin(df_neighborhood.ix[df_neighborhood.iloc[:, 1] < 200000, 0]) , 'Class'] = 'Wealthy'
df_all.loc[df_all['Neighborhood'].isin(df_neighborhood.ix[df_neighborhood.iloc[:, 1] < 180000, 0]) , 'Class'] = 'Middle'
df_all.loc[df_all['Neighborhood'].isin(df_neighborhood.ix[df_neighborhood.iloc[:, 1] < 130000, 0]) , 'Class'] = 'Lower'
del df_neighborhood

df_all = df_all.drop('Neighborhood', 1)

df_all = df_all.drop('MoSold', 1)

df_all['CentralAir'] = (df_all['CentralAir'] == 'Y').astype(bool).astype(int)

df_all.loc[(df_all['BldgType'] == 'TwnhsE') | (df_all['BldgType'] == 'TwnhsI'), 'BldgType'] = 'Twnhs'

df_all.loc[(df_all['Alley'] == 'Pave'), 'ExterQual'] = 2
df_all.loc[(df_all['Alley'] == 'Grvl'), 'ExterQual'] = 1
df_all.loc[(df_all['Alley'] == 'None'), 'ExterQual'] = 0

df_all.loc[(df_all['ExterQual'] == 'Ex'), 'ExterQual'] = 5
df_all.loc[(df_all['ExterQual'] == 'Gd'), 'ExterQual'] = 4
df_all.loc[(df_all['ExterQual'] == 'TA'), 'ExterQual'] = 3
df_all.loc[(df_all['ExterQual'] == 'Fa'), 'ExterQual'] = 2
df_all.loc[(df_all['ExterQual'] == 'Po'), 'ExterQual'] = 1

df_all.loc[(df_all['ExterCond'] == 'Ex'), 'ExterCond'] = 5
df_all.loc[(df_all['ExterCond'] == 'Gd'), 'ExterCond'] = 4
df_all.loc[(df_all['ExterCond'] == 'TA'), 'ExterCond'] = 3
df_all.loc[(df_all['ExterCond'] == 'Fa'), 'ExterCond'] = 2
df_all.loc[(df_all['ExterCond'] == 'Po'), 'ExterCond'] = 1

df_all.loc[(df_all['BsmtQual'] == 'Ex'), 'BsmtQual'] = 5
df_all.loc[(df_all['BsmtQual'] == 'Gd'), 'BsmtQual'] = 4
df_all.loc[(df_all['BsmtQual'] == 'TA'), 'BsmtQual'] = 3
df_all.loc[(df_all['BsmtQual'] == 'Fa'), 'BsmtQual'] = 2
df_all.loc[(df_all['BsmtQual'] == 'Po'), 'BsmtQual'] = 1
df_all.loc[(df_all['BsmtQual'] == 'None'), 'BsmtQual'] = 0
df_all['BsmtQual'] = df_all['BsmtQual'].astype(int)

df_all.loc[(df_all['BsmtCond'] == 'Ex'), 'BsmtCond'] = 5
df_all.loc[(df_all['BsmtCond'] == 'Gd'), 'BsmtCond'] = 4
df_all.loc[(df_all['BsmtCond'] == 'TA'), 'BsmtCond'] = 3
df_all.loc[(df_all['BsmtCond'] == 'Fa'), 'BsmtCond'] = 2
df_all.loc[(df_all['BsmtCond'] == 'Po'), 'BsmtCond'] = 1
df_all.loc[(df_all['BsmtCond'] == 'None'), 'BsmtCond'] = 0
df_all['BsmtCond'] = df_all['BsmtCond'].astype(int)

df_all.loc[(df_all['BsmtExposure'] == 'Gd'), 'BsmtExposure'] = 4
df_all.loc[(df_all['BsmtExposure'] == 'Av'), 'BsmtExposure'] = 3
df_all.loc[(df_all['BsmtExposure'] == 'Mn'), 'BsmtExposure'] = 2
df_all.loc[(df_all['BsmtExposure'] == 'No'), 'BsmtExposure'] = 1
df_all.loc[(df_all['BsmtExposure'] == 'None'), 'BsmtExposure'] = 0
df_all['BsmtExposure'] = df_all['BsmtExposure'].astype(int)

df_all.loc[(df_all['BsmtFinType1'] == 'GLQ'), 'BsmtFinType1'] = 6
df_all.loc[(df_all['BsmtFinType1'] == 'ALQ'), 'BsmtFinType1'] = 5
df_all.loc[(df_all['BsmtFinType1'] == 'BLQ'), 'BsmtFinType1'] = 4
df_all.loc[(df_all['BsmtFinType1'] == 'Rec'), 'BsmtFinType1'] = 3
df_all.loc[(df_all['BsmtFinType1'] == 'LwQ'), 'BsmtFinType1'] = 2
df_all.loc[(df_all['BsmtFinType1'] == 'Unf'), 'BsmtFinType1'] = 1
df_all.loc[(df_all['BsmtFinType1'] == 'None'), 'BsmtFinType1'] = 0
df_all['BsmtFinType1'] = df_all['BsmtFinType1'].astype(int)

df_all.loc[(df_all['BsmtFinType2'] == 'GLQ'), 'BsmtFinType2'] = 6
df_all.loc[(df_all['BsmtFinType2'] == 'ALQ'), 'BsmtFinType2'] = 5
df_all.loc[(df_all['BsmtFinType2'] == 'BLQ'), 'BsmtFinType2'] = 4
df_all.loc[(df_all['BsmtFinType2'] == 'Rec'), 'BsmtFinType2'] = 3
df_all.loc[(df_all['BsmtFinType2'] == 'LwQ'), 'BsmtFinType2'] = 2
df_all.loc[(df_all['BsmtFinType2'] == 'Unf'), 'BsmtFinType2'] = 1
df_all.loc[(df_all['BsmtFinType2'] == 'None'), 'BsmtFinType2'] = 0
df_all['BsmtFinType2'] = df_all['BsmtFinType2'].astype(int)

df_all.loc[(df_all['KitchenQual'] == 'Ex'), 'KitchenQual'] = 5
df_all.loc[(df_all['KitchenQual'] == 'Gd'), 'KitchenQual'] = 4
df_all.loc[(df_all['KitchenQual'] == 'TA'), 'KitchenQual'] = 3
df_all.loc[(df_all['KitchenQual'] == 'Fa'), 'KitchenQual'] = 2
df_all.loc[(df_all['KitchenQual'] == 'Po'), 'KitchenQual'] = 1
df_all['KitchenQual'] = df_all['KitchenQual'].astype(int)

#Skewedness
end_train_y = np.log1p(end_train_y)

integers = df_all.dtypes[df_all.dtypes != "object"].index

s_skewed = df_all.ix[:, integers].apply(lambda x: stats.skew(x.dropna()))
s_skewed = s_skewed[(s_skewed > 1)]
s_skewed = s_skewed.index

df_all[s_skewed] = np.log1p(df_all[s_skewed])
df_all = pd.get_dummies(df_all)

end_train_x = df_all.iloc[0:len(end_train_y), :]
end_test_x = df_all.iloc[len(end_train_y):, :]

#Models
#Lasso
alphas = [0.0001, 0.0005, 0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5]

rmse_lasso = []
for alpha in alphas:
    rmse_lasso.append(cross_val_score(lm.Lasso(alpha = alpha),
                                      X = end_train_x,
                                      y = end_train_y,
                                      cv = 5,
                                      scoring = "neg_mean_squared_error"
                                      ).mean() * -1
    )

rmse_lasso = pd.Series(rmse_lasso, index = alphas)
rmse_lasso.plot(title = "Validation")

model_lasso = lm.Lasso(alpha = rmse_lasso[rmse_lasso == rmse_lasso.min()].index[0])
model_lasso.fit(end_train_x, end_train_y)

end_train_x = end_train_x.drop(end_train_x.iloc[:, model_lasso.coef_ == 0].columns, 1)
end_test_x = end_test_x.drop(end_test_x.iloc[:, model_lasso.coef_ == 0].columns, 1)

model_lasso.fit(end_train_x, end_train_y)
end_test_pl = np.exp(model_lasso.predict(end_test_x))

#Ridge
rmse_ridge = []
for alpha in alphas:
    rmse_ridge.append(cross_val_score(lm.Ridge(alpha = alpha),
                                       X = end_train_x,
                                       y = end_train_y,
                                       cv = 5,
                                       scoring = "neg_mean_squared_error"
                                       ).mean() * -1
    )

rmse_ridge = pd.Series(rmse_ridge, index = alphas)
rmse_ridge.plot(title = "Validation")
rmse_ridge.min()

model_ridge = lm.Ridge(alpha = rmse_ridge[rmse_ridge == rmse_ridge.min()].index[0])
model_ridge.fit(end_train_x, end_train_y)
end_test_pr = np.exp(model_ridge.predict(end_test_x))

#Evaluate
rmse_history = np.zeros((0,2))
rmse_history = np.append(rmse_history, np.array(([[rmse_lasso.min(), rmse_ridge.min()]])), axis = 0)
print(rmse_lasso.min())
print(rmse_ridge.min())

#Write
end_test_p = (end_test_pr + end_test_pl) / 2
write_to_file(end_test_p)

#---------------------------------------------------------#

#Part 2 - R

###Libraries & Randomization
set.seed(117)
setwd("F:/Code/R/House")
library(functional)
library(e1071)
library(dummies)
library(gbm)
library(glmnet)
library(xgboost)
library(caret)
library(randomForest)
library(plyr)
library(parallel)
library(doParallel)
library(caret)

###Functions
dfnull <- function(df){
  df_null <- data.frame(apply(apply(df, 2, FUN = 'is.na'), 2, FUN = 'sum'))
  colnames(df_null)[1] <- "# of Null"
  return(df_null)
}

###Import
df_train_x <- read.csv("F:/Code/R/House/train.csv", stringsAsFactors = F)
df_test_x <- read.csv("F:/Code/R/House/test.csv", stringsAsFactors = F)
df_train_x[, 1] <- NULL
df_test_x[, 1] <- NULL

df_train_y <- data.frame(SalePrice = df_train_x[, ncol(df_train_x)])
df_train_x[, ncol(df_train_x)] <- NULL

df_all_x <- rbind(df_train_x, df_test_x)

###Pre-processing
##Imputation
#NAs to None
df_all_x[is.na(df_all_x[, 'Alley']), 'Alley'] <- 'None'
df_all_x[is.na(df_all_x[, 'PoolQC']), 'PoolQC'] <- 'None'
df_all_x[is.na(df_all_x[, 'Fence']), 'Fence'] <- 'None'
df_all_x[is.na(df_all_x[, 'MiscFeature']), 'MiscFeature'] <- 'None'
df_all_x[is.na(df_all_x[, 'FireplaceQu']), 'FireplaceQu'] <- 'None'
df_all_x[is.na(df_all_x[, 'BsmtFinType1']), 'BsmtFinType1'] <- 'None'
df_all_x[is.na(df_all_x[, 'GarageType']), 'GarageType'] <- 'None'

df_all_x[is.na(df_all_x[, 'MasVnrArea']), 'MasVnrArea'] <- 0
df_all_x[is.na(df_all_x[, 'BsmtFinSF1']), 'BsmtFinSF1'] <- 0
df_all_x[is.na(df_all_x[, 'BsmtFinSF2']), 'BsmtFinSF2'] <- 0
df_all_x[is.na(df_all_x[, 'BsmtUnfSF']), 'BsmtUnfSF'] <- 0
df_all_x[is.na(df_all_x[, 'TotalBsmtSF']), 'TotalBsmtSF'] <- 0

df_all_x[is.na(df_all_x[, 'MasVnrType']) & (is.na(df_all_x[, 'MasVnrArea']) == T), 'MasVnrType'] <- 'None'
df_all_x[is.na(df_all_x[, 'BsmtQual']) & (is.na(df_all_x[, 'BsmtCond']) == T), 'BsmtQual'] <- 'None'
df_all_x[is.na(df_all_x[, 'BsmtCond']) & df_all_x[, 'BsmtQual'] == 'None', 'BsmtCond'] <- 'None'
df_all_x[is.na(df_all_x[, 'BsmtExposure']) & df_all_x[, 'BsmtCond'] == 'None', 'BsmtExposure'] <- 'None'
df_all_x[is.na(df_all_x[, 'BsmtFinType2']) & df_all_x[, 'BsmtCond'] == 'None', 'BsmtFinType2'] <- 'None'
df_all_x[is.na(df_all_x[, 'GarageYrBlt']) & df_all_x[, 'GarageType'] == 'None', 'GarageYrBlt'] <- 0
df_all_x[is.na(df_all_x[, 'GarageFinish']) & df_all_x[, 'GarageType'] == 'None', 'GarageFinish'] <- 'None'
df_all_x[is.na(df_all_x[, 'GarageQual']) & df_all_x[, 'GarageType'] == 'None', 'GarageQual'] <- 'None'
df_all_x[is.na(df_all_x[, 'GarageCond']) & df_all_x[, 'GarageType'] == 'None', 'GarageCond'] <- 'None'

df_all_x[(max(df_all_x[,'YrSold']) < df_all_x[,'GarageYrBlt']) & (is.na(df_all_x[,'GarageYrBlt']) == F), 'GarageYrBlt'] <- 2007

df_all_x[(df_all_x['YrSold'] - df_all_x['YearRemodAdd'] < 0), 'YearRemodAdd'] <- df_all_x[(df_all_x['YrSold'] - df_all_x['YearRemodAdd'] < 0), 'YrSold']


#Mice
library(mice)
v_char <- names(df_all_x[, sapply(df_all_x, class) == 'character'])
df_all_x[v_char] <- lapply(df_all_x[v_char], as.factor)

df_all_x[,'BsmtFullBath'] <- as.factor(df_all_x[,'BsmtFullBath'])
df_all_x[,'BsmtHalfBath'] <- as.factor(df_all_x[,'BsmtHalfBath'])
df_all_x[,'GarageYrBlt'] <- as.factor(df_all_x[,'GarageYrBlt'])
df_all_x[,'GarageCars'] <- as.factor(df_all_x[,'GarageCars'])

imp <- mice(data = df_all_x,
            m = 5,
            method = 'rf'
)

df_all_x <- complete(imp)

df_all_x[,'BsmtFullBath'] <- as.integer(df_all_x[,'BsmtFullBath'])
df_all_x[,'BsmtHalfBath'] <- as.integer(df_all_x[,'BsmtHalfBath'])
df_all_x[,'GarageYrBlt'] <- as.integer(df_all_x[,'GarageYrBlt'])
df_all_x[,'GarageCars'] <- as.integer(df_all_x[,'GarageCars'])

write.csv(df_all_x, "F:/Code/R/House/imputed.csv", row.names = F)

##Engineerig
df_all_x <- read.csv("F:/Code/R/House/imputed.csv", stringsAsFactors = F)
df_null <- dfnull(df_all_x)

#Categorical to Ordinal
df_all_x[(df_all_x['ExterQual'] == 'Ex'), 'ExterQual'] = 5
df_all_x[(df_all_x['ExterQual'] == 'Gd'), 'ExterQual'] = 4
df_all_x[(df_all_x['ExterQual'] == 'TA'), 'ExterQual'] = 3
df_all_x[(df_all_x['ExterQual'] == 'Fa'), 'ExterQual'] = 2
df_all_x[(df_all_x['ExterQual'] == 'Po'), 'ExterQual'] = 1
df_all_x[, 'ExterQual'] <- as.integer(df_all_x[,'ExterQual'])

df_all_x[(df_all_x['ExterCond'] == 'Ex'), 'ExterCond'] = 5
df_all_x[(df_all_x['ExterCond'] == 'Gd'), 'ExterCond'] = 4
df_all_x[(df_all_x['ExterCond'] == 'TA'), 'ExterCond'] = 3
df_all_x[(df_all_x['ExterCond'] == 'Fa'), 'ExterCond'] = 2
df_all_x[(df_all_x['ExterCond'] == 'Po'), 'ExterCond'] = 1
df_all_x[, 'ExterCond'] <- as.integer(df_all_x[, 'ExterCond'])

df_all_x[(df_all_x['BsmtQual'] == 'Ex'), 'BsmtQual'] = 5
df_all_x[(df_all_x['BsmtQual'] == 'Gd'), 'BsmtQual'] = 4
df_all_x[(df_all_x['BsmtQual'] == 'TA'), 'BsmtQual'] = 3
df_all_x[(df_all_x['BsmtQual'] == 'Fa'), 'BsmtQual'] = 2
df_all_x[(df_all_x['BsmtQual'] == 'Po'), 'BsmtQual'] = 1
df_all_x[(df_all_x['BsmtQual'] == 'None'), 'BsmtQual'] = 0
df_all_x[, 'BsmtQual'] <- as.integer(df_all_x[, 'BsmtQual'])

df_all_x[(df_all_x['BsmtCond'] == 'Ex'), 'BsmtCond'] = 5
df_all_x[(df_all_x['BsmtCond'] == 'Gd'), 'BsmtCond'] = 4
df_all_x[(df_all_x['BsmtCond'] == 'TA'), 'BsmtCond'] = 3
df_all_x[(df_all_x['BsmtCond'] == 'Fa'), 'BsmtCond'] = 2
df_all_x[(df_all_x['BsmtCond'] == 'Po'), 'BsmtCond'] = 1
df_all_x[(df_all_x['BsmtCond'] == 'None'), 'BsmtCond'] = 0
df_all_x[, 'BsmtCond'] <- as.integer(df_all_x[, 'BsmtCond'])

df_all_x[(df_all_x['BsmtExposure'] == 'Gd'), 'BsmtExposure'] = 4
df_all_x[(df_all_x['BsmtExposure'] == 'Av'), 'BsmtExposure'] = 3
df_all_x[(df_all_x['BsmtExposure'] == 'Mn'), 'BsmtExposure'] = 2
df_all_x[(df_all_x['BsmtExposure'] == 'No'), 'BsmtExposure'] = 1
df_all_x[(df_all_x['BsmtExposure'] == 'None'), 'BsmtExposure'] = 0
df_all_x[, 'BsmtExposure'] <- as.integer(df_all_x[, 'BsmtExposure'])

df_all_x[(df_all_x['BsmtFinType1'] == 'GLQ'), 'BsmtFinType1'] = 6
df_all_x[(df_all_x['BsmtFinType1'] == 'ALQ'), 'BsmtFinType1'] = 5
df_all_x[(df_all_x['BsmtFinType1'] == 'BLQ'), 'BsmtFinType1'] = 4
df_all_x[(df_all_x['BsmtFinType1'] == 'Rec'), 'BsmtFinType1'] = 3
df_all_x[(df_all_x['BsmtFinType1'] == 'LwQ'), 'BsmtFinType1'] = 2
df_all_x[(df_all_x['BsmtFinType1'] == 'Unf'), 'BsmtFinType1'] = 1
df_all_x[(df_all_x['BsmtFinType1'] == 'None'), 'BsmtFinType1'] = 0
df_all_x[, 'BsmtFinType1'] <- as.integer(df_all_x[, 'BsmtFinType1'])

df_all_x[(df_all_x['BsmtFinType2'] == 'GLQ'), 'BsmtFinType2'] = 6
df_all_x[(df_all_x['BsmtFinType2'] == 'ALQ'), 'BsmtFinType2'] = 5
df_all_x[(df_all_x['BsmtFinType2'] == 'BLQ'), 'BsmtFinType2'] = 4
df_all_x[(df_all_x['BsmtFinType2'] == 'Rec'), 'BsmtFinType2'] = 3
df_all_x[(df_all_x['BsmtFinType2'] == 'LwQ'), 'BsmtFinType2'] = 2
df_all_x[(df_all_x['BsmtFinType2'] == 'Unf'), 'BsmtFinType2'] = 1
df_all_x[(df_all_x['BsmtFinType2'] == 'None'), 'BsmtFinType2'] = 0
df_all_x[, 'BsmtFinType2'] <- as.integer(df_all_x[, 'BsmtFinType2'])

df_all_x[(df_all_x['KitchenQual'] == 'Ex'), 'KitchenQual'] = 5
df_all_x[(df_all_x['KitchenQual'] == 'Gd'), 'KitchenQual'] = 4
df_all_x[(df_all_x['KitchenQual'] == 'TA'), 'KitchenQual'] = 3
df_all_x[(df_all_x['KitchenQual'] == 'Fa'), 'KitchenQual'] = 2
df_all_x[(df_all_x['KitchenQual'] == 'Po'), 'KitchenQual'] = 1
df_all_x[, 'KitchenQual'] = as.integer(df_all_x[, 'KitchenQual'])

df_all_x[(df_all_x['PoolQC'] == 'Ex'), 'PoolQC'] = 4
df_all_x[(df_all_x['PoolQC'] == 'Gd'), 'PoolQC'] = 3
df_all_x[(df_all_x['PoolQC'] == 'TA'), 'PoolQC'] = 2
df_all_x[(df_all_x['PoolQC'] == 'Fa'), 'PoolQC'] = 1
df_all_x[(df_all_x['PoolQC'] == 'None'), 'PoolQC'] = 0
df_all_x[, 'PoolQC'] = as.integer(df_all_x[, 'PoolQC'])

df_all_x['CentralAir'] <- as.integer(df_all_x['CentralAir'] == 'Y')

df_all_x[(df_all_x['Alley'] == 'Pave'), 'Alley'] = 2
df_all_x[(df_all_x['Alley'] == 'Grvl'), 'Alley'] = 1
df_all_x[(df_all_x['Alley'] == 'None'), 'Alley'] = 0
df_all_x[, 'Alley'] = as.integer(df_all_x[, 'Alley'])

df_all_x[(df_all_x['Street'] == 'Pave'), 'Street'] = 1
df_all_x[(df_all_x['Street'] == 'Grvl'), 'Street'] = 0
df_all_x[, 'Street'] = as.integer(df_all_x[, 'Street'])

df_all_x[(df_all_x['GarageFinish'] == 'Fin'), 'GarageFinish'] = 3
df_all_x[(df_all_x['GarageFinish'] == 'RFn'), 'GarageFinish'] = 2
df_all_x[(df_all_x['GarageFinish'] == 'Unf'), 'GarageFinish'] = 1
df_all_x[(df_all_x['GarageFinish'] == 'None'), 'GarageFinish'] = 0
df_all_x[, 'GarageFinish'] = as.integer(df_all_x[, 'GarageFinish'])

#Low Var or Cor
df_all_x['MoSold'] <- NULL

#Neighborhood
df_neighborhood <- data.frame(Neighborhood = df_all_x[(1:length(df_train_y$SalePrice)), 'Neighborhood'], SalePrice = df_train_y)
df_neighborhood <- aggregate(df_neighborhood[, 'SalePrice'] ~ df_neighborhood[, 'Neighborhood'], FUN = 'mean')
names(df_neighborhood) <- c('Neighborhood', 'SalePrice')
df_neighborhood <- df_neighborhood[order(df_neighborhood$SalePrice), ]
print(df_neighborhood)

df_all_x['Class'] = 5
df_all_x[df_all_x$Neighborhood %in% df_neighborhood[df_neighborhood[, 'SalePrice'] < 300000, 'Neighborhood'], 'Class'] = 4
df_all_x[df_all_x$Neighborhood %in% df_neighborhood[df_neighborhood[, 'SalePrice'] < 200000, 'Neighborhood'], 'Class'] = 3
df_all_x[df_all_x$Neighborhood %in% df_neighborhood[df_neighborhood[, 'SalePrice'] < 180000, 'Neighborhood'], 'Class'] = 2
df_all_x[df_all_x$Neighborhood %in% df_neighborhood[df_neighborhood[, 'SalePrice'] < 130000, 'Neighborhood'], 'Class'] = 1

df_all_x['Neighborhood'] <- NULL
rm(df_neighborhood)

df_all_x[(df_all_x['BldgType'] == 'TwnhsE') | (df_all_x['BldgType'] == 'TwnhsI'), 'BldgType'] <- 'Twnhs'

#Near Zero Variance
nzv <- nearZeroVar(dummy.data.frame(df_all_x), saveMetrics = T)
nzv[nzv$nzv,]

#Correlation
# v_num <- names(df_all_x[, sapply(df_all_x, class) == 'integer'])
# descrCor <- cor(df_all_x[v_num])
# summary(descrCor[upper.tri(descrCor)])
#
# highlyCorDescr <- findCorrelation(descrCor, cutoff = .7, verbose = T)
# df_all_x[v_num[highlyCorDescr]] <- NULL
#
# v_num <- names(df_all_x[, sapply(df_all_x, class) == 'integer'])
# descrCor_after <- cor(df_all_x[v_num])
# summary(descrCor_after[upper.tri(descrCor_after)])

# #LinearCombos
# v_num <- names(df_all_x[, sapply(df_all_x, class) == 'integer'])
# comboInfo <- findLinearCombos(df_all_x[v_num])
# print(comboInfo)
#
# df_all_x[,comboInfo$remove] <- NULL

#Box - Cox
df_train_y$SalePrice <- log1p(df_train_y$SalePrice)
hist(df_train_y$SalePrice)

v_num <- names(df_all_x[, sapply(df_all_x, class) == 'integer'])

#createDataPartition for Validation
v_char <- names(df_all_x[, sapply(df_all_x, class) == 'character'])
df_all_x[v_char] <- lapply(df_all_x[v_char], as.factor)

end_train_y <- df_train_y
end_train_x <- df_all_x[1 : nrow(end_train_y), ]
end_train_all <- cbind(end_train_x, end_train_y)

end_test_x <- df_all_x[(nrow(end_train_y) + 1) : nrow(df_all_x), ]

rm(df_train_x)
rm(df_test_x)
rm(df_null)

#Manual Validation/Test Sets
if (exists('id') != T){
  id <- createDataPartition(end_train_all$SalePrice,
                            p = 0.8,
                            list = F,
                            times = 1
  )
}

val_train_x <- end_train_all[id,]
val_valid_x <- end_train_all[-id,]

preProcValues <- preProcess(val_train_x, method = c("center", "scale", "BoxCox"))

val_train_x <- predict(preProcValues, val_train_x)
val_valid_x <- predict(preProcValues, val_valid_x)


#Feature Validation - RandomForest

fitControl <- trainControl(method = "repeatedcv",
                           number = 4,
                           repeats = 3,
                           search = 'grid',
                           allowParallel = T,
                           verbose = T
)

gridControl <- expand.grid(mtry = c(21, 25, 31, 35, 41))

no_cores <- makeCluster(detectCores() - 1)
registerDoParallel(no_cores)

tuner_rfo <- train(form = SalePrice ~ ., 
                   data = val_train_x,
                   method = "rf",
                   trControl = fitControl,
                   tuneGrid = gridControl
)

val_valid_y <- predict(object = tuner_rfo,
                       newdata = val_valid_x[, -which(names(val_valid_x) == 'SalePrice')],
                       n.trees = model_tuner_rfo$bestTune
                       )

stopCluster(no_cores)
registerDoSEQ()

print(tuner_rfo)
print(sum((val_valid_y - val_valid_x[, which(names(val_valid_x) == 'SalePrice')])^2))

#Model Optimization - Xgboost
no_cores <- makeCluster(detectCores() - 1)
registerDoParallel(no_cores)

fitControl <- trainControl(method = "repeatedcv",
                           number = 4,
                           repeats = 3,
                           search = 'grid',
                           allowParallel = T,
                           verbose = T
)

gridControl <- expand.grid(nrounds = c(1601, 1701),
                           max_depth = c(4, 5),
                           eta = c(115e-4, 120e-4),
                           gamma = c(85e-3, 80e-3),
                           colsample_bytree = c(0.51, 0.55),
                           min_child_weight = c(2, 3),
                           subsample = c(0.55, 0.575)
                           )

model_xgb_t <- train(form = SalePrice ~ .,
                   data = val_train_x,
                   method = "xgbTree",
                   trControl = fitControl,
                   tuneGrid = gridControl,
                   verbose = T
)

val_valid_y <-predict(object = model_xgb_t,
                      newdata = val_valid_x[, -which(names(val_valid_x) == 'SalePrice')],
                      n.trees = model_xgb$bestTune
)

stopCluster(no_cores)
registerDoSEQ()

print(model_xgb)
print(sum((val_valid_y - val_valid_x[, which(names(val_valid_x) == 'SalePrice')])^2))

#Model Training - Xgboost
no_cores <- makeCluster(detectCores() - 1)
registerDoParallel(no_cores)

model_xgb <- train(form = SalePrice ~ .,
                   data = end_train_all,
                   method = "xgbTree",
                   trControl = fitControl,
                   tuneGrid = gridControl,
                   verbose = T
)

end_test_y <- predict(object = model_xgb,
                      newdata = end_test_x,
                      n.trees = model_xgb_t$bestTune
)

stopCluster(no_cores)
registerDoSEQ()

print(model_xgb)

###Write
pyt_test_y <- read.csv("F:/Code/Python/4 House/submission.csv")
end_test_y <- exp(end_test_y) - 1
end_test_y <- data.frame((nrow(end_train_x) + 1) : (nrow(end_train_x) + nrow(end_test_x)), SalePrice = end_test_y)
colnames(end_test_y) <- c('Id', 'SalePrice')
end_test_y <- end_test_y
write.csv(end_test_y, "F:/Code/R/House/submission_r.csv", row.names = F)


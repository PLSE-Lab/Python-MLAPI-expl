import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

import csv
import matplotlib
import matplotlib.pyplot as plt
import itertools
import datetime
import seaborn as sns
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.metrics import make_scorer, mean_squared_error



# Import data sets
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# Data processing - remove outliers with areas > 4000 sqft as recommended
df_train = df_train[df_train.GrLivArea < 4000]

# Combine training and testing sets for data manipulation, remember the index so we can split later
df = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],
                df_test.loc[:,'MSSubClass':'SaleCondition']), keys=['train', 'test'])

###################################
# Preprocesing
###################################
df['Age'] = df['YrSold'] - df['YearBuilt']
df['RemodelAge'] = df['YrSold'] - df['YearRemodAdd']
df['FinishedBsmtSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
df['HasKitchen'] = np.where(df['KitchenQual'].isnull(), 0, 1)
df['NumBaths'] = df['BsmtFullBath'] + df['BsmtHalfBath']*0.5 + df['FullBath'] + df['HalfBath']*0.5
df['HasGarage'] = np.where(df['GarageType'].isnull(), 0, 1)
df['GarageCars'] = np.where(df['HasGarage'] == 0, 0, df['GarageCars'])

# FireplaceQu - prob can drop (Ex, Gd, TA, Fa, Po, NA)
# GarageYrBlt can prob drop
# GarageQual & GarageCond (Ex, Gd, TA, Fa, Po, NA) fixed earlier
# HasDeck = if WoodDeckSF > 0, otherwise make WoodDeckSF NAN
# HasPorch = if OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch > 0, if not keep all other variables NAN
# HasPool = PoolArea > 0, PoolQC (Ex, Gd, TA, Fa, NA) otherwise make NAN
# Can probably just drop the actual sizes...


new_columns = []
drop_columns = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtExposure', 'Alley', 'BsmtFinType1', 'BsmtFinType2', 'KitchenQual',
                'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'GarageYrBlt', 'FireplaceQu', 'GarageArea',
                'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
                'YrSold','YearBuilt', 'MoSold', 'MiscVal','1stFlrSF','2ndFlrSF', 'GarageType', 'GarageFinish',
                'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']

# Convert Quality to Numerical Scale
quality_scale = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po': 1, np.nan: 0}
quality_columns = ['ExterQual','ExterCond', 'HeatingQC', 'GarageQual', 'GarageCond', 'KitchenQual', 'BsmtQual', 'BsmtCond']

for col in quality_columns:
    df[col+"_numerical"] = np.nan # give a negative score to bias towards goodness
    for k, v in quality_scale.items():
        df.loc[:, col+"_numerical"].loc[:, df[col].isin([k])] = v
    new_columns.append(col+"_numerical")
    drop_columns.append(col)


quality_scale = {'Typ':0, 'Min1':-1, 'Min2':-2, 'Mod':-3, 'Maj1':-4, 'Maj2':-5, 'Sev':-6, 'Sal':-7}
col = 'Functional'
df[col+"_numerical"] = np.nan
for k, v in quality_scale.items():
    df.loc[:, col+"_numerical"].loc[:, df[col].isin([k])] = v
new_columns.append(col+"_numerical")
drop_columns.append(col)



# Drop the ones we don't want
df = df[:].drop(drop_columns, axis=1)


#Check how many missing values in general
def num_missing(x):
  return sum(x.isnull())
#print(df.loc[:,'Electrical':'SaleCondition'].apply(num_missing, axis=0))
#print(df.apply(num_missing, axis=0))


#matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
#prices = pd.DataFrame({"price":df["SalePrice"], "log(price + 1)":np.log1p(df["SalePrice"])})
#prices.hist()

#Fill NANs
df = df.fillna(df.median())

#Log transform skewed numeric features:
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
numeric_feats = df.dtypes[df.dtypes != "object"].index
skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
df[skewed_feats] = np.log1p(df[skewed_feats])
#print(skewed_feats)
#df_encoded.head()

categorical_columns = ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                    'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir',
                    'Electrical', 'PavedDrive', 'SaleType', 'SaleCondition']

df = pd.get_dummies(df, columns=categorical_columns)
#df = df_encoded.merge(dummies, left_index=True, right_index=True)
#dummies.describe()
#print(dummies)

#print(df.apply(num_missing, axis=0))

X_train = df[:df_train.shape[0]]
X_test = df[df_train.shape[0]:]
y = df_train.SalePrice




########################################
# Lasso
########################################

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
print('############## Lasso Model ##############')
model_ridge = Ridge()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
print('Lasso rmse: ' + str(rmse_cv(model_lasso).mean()))
lasso_coefficients = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(lasso_coefficients != 0)) + " variables and eliminated the other "
      +  str(sum(lasso_coefficients == 0)) + " variables")

print('Important Lasso Coefficients')
print(lasso_coefficients.sort_values(ascending=False).head(15))
#alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
#cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
#cv_ridge = pd.Series(cv_ridge, index = alphas)
#cv_ridge.plot(title = "Validation - Just Do It")
#plt.xlabel("alpha")
#plt.ylabel("rmse")
#plt.show()




########################################
# XGBOOST
########################################
print('############## XGB Model ##############')

# Model Fitting
n_estimators = 500
model = xgb.XGBClassifier(learning_rate = 0.1, n_estimators = n_estimators, max_depth=3, min_child_weight=1,
                          gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'reg:linear', nthread=4,
                          scale_pos_weight=1, seed=0)
print('Creating Matrix')
dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

print("Accuracy")
xgb_params = model.get_xgb_params()
cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=n_estimators, early_stopping_rounds=100)
print(cv_result[-1:].head())


print('Train Model')
# Fit the algorithm on the data
model_xgb = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.05) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)
#print (model_xgb)

print('Important Features (top 15)')
b = model_xgb.booster()
fs = b.get_fscore()
all_features = [fs.get(f, 0.) for f in b.feature_names]
all_features = np.array(all_features, dtype=np.float32)

#Create dataframe of importances
feature_importance = pd.DataFrame(data=(all_features / all_features.sum()*100), index=X_train.columns, columns=['values'])
feature_importance = feature_importance.sort_values(by='values', ascending=False)
print(feature_importance.head(15))

#b = sns.barplot(X_train.columns, all_features,color='blue')
#plt.show()



########################################
# RANDOM FOREST
########################################
print('############## Random Forest ##############')
rsme_scorer = make_scorer(mean_squared_error, False)

rfm = RandomForestRegressor(n_estimators=500, n_jobs=-1)
cv_score = np.sqrt(-cross_val_score(estimator=rfm, X=X_train, y=y, cv=15, scoring = rsme_scorer))
print('cv score' + str(cv_score.mean()))
rfm.fit(X_train, y)

# Output feature importance coefficients, map them to their feature name, and sort values
print('Important Features (top 15)')
feature_importance = pd.Series(rfm.feature_importances_, index = X_train.columns).sort_values(ascending=False)
print(feature_importance.head(15))



############### Output and Save #####################
xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))
random_forest_preds = np.expm1(rfm.predict(X_test))
preds = 0.5*lasso_preds + 0.3*xgb_preds + 0.2*random_forest_preds

#predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
#sns.jointplot(x="xgb", y="lasso", data=predictions, kind='reg')
#plt.show()

solution = pd.DataFrame({"id":df_test.Id, "SalePrice":preds})
solution.to_csv("solution.csv", index = False)

print('Done')
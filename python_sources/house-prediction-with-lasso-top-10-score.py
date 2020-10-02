import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from scipy.stats import skew

from scipy.special import boxcox1p

from sklearn.feature_selection import RFECV

from sklearn.linear_model import Lasso

from sklearn.model_selection import cross_val_score



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# remove outliers

train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000))]



all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))



# drop some features to avoid multicollinearity

all_data.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd'], axis=1, inplace=True)



train["SalePrice"] = np.log1p(train["SalePrice"])



numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.65]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = boxcox1p(all_data[skewed_feats], 0.15)



all_data = pd.get_dummies(all_data)



all_data = all_data.fillna(all_data.mean())



X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice





#### models selection

lasso = Lasso(alpha=0.0004)

model = lasso



### prediction

model.fit(X_train, y)



preds = np.expm1(model.predict(X_test))

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("full_features_lasso.csv", index = False)
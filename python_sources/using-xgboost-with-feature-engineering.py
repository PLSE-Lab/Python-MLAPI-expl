# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from scipy.stats import skew, skewtest

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('../input/train.csv', index_col=0)
test_df = pd.read_csv('../input/test.csv', index_col=0)

y_train = np.log(train_df.pop('SalePrice'))
all_df = pd.concat((train_df, test_df), axis=0)

# Variable transformations
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

feat_trial = (all_df['1stFlrSF'] + all_df['2ndFlrSF']).copy()
print("Skewness of the original intended feature:",skew(feat_trial))
print("Skewness of transformed feature", skew(np.log1p(feat_trial)))

# hence, we'll use the transformed feature
# lets create the feature then
all_df['1stFlr_2ndFlr_Sf'] = np.log1p(all_df['1stFlrSF'] + all_df['2ndFlrSF'])

feat_trial = (all_df['1stFlr_2ndFlr_Sf'] + all_df['LowQualFinSF'] + all_df['GrLivArea']).copy()
print("Skewness of the original intended feature:",skew(feat_trial))
print("Skewness of transformed feature", skew(np.log1p(feat_trial)))
all_df['All_Liv_SF'] = np.log1p(all_df['1stFlr_2ndFlr_Sf'] + all_df['LowQualFinSF'] + all_df['GrLivArea'])

numeric_feats = all_df.dtypes[all_df.dtypes != "object"].index
skewed_feats = all_df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
print(skewed_feats)
all_df[skewed_feats] = np.log1p(all_df[skewed_feats])

all_dummy_df = pd.get_dummies(all_df)

# Replacing missing values
mean_cols = all_dummy_df.mean()
all_dummy_df.isnull().sum().sum()

numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

# Model Building
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

# Not completely necessary, just converts dataframe to numpy array
X_train = dummy_train_df.values
X_test = dummy_test_df.values

# Prediction with XGBoost
import xgboost as xgb

clf = xgb.XGBRegressor(n_estimators = 1000, seed = 0, learning_rate = 0.03, gamma = 0.1, min_child_weight=1, max_depth = 10, subsample = 0.734, colsample_bytree = 0.4, colsample_bylevel = 0.8 )
clf.fit(X_train, y_train)
y_xgb = np.exp(clf.predict(X_test))
submission_xgb = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': y_xgb})
submission_xgb.to_csv('submisison_xgb.csv', index=False)




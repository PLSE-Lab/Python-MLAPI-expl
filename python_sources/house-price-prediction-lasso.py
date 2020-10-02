# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import skew
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))

train["SalePrice"]=np.log1p(train["SalePrice"])

# Transform the skew data 
numeric_feats = all_data.dtypes[all_data.dtypes!='object'].index
skew_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
#print(skew_feats>0.75)
skew_feats = skew_feats[skew_feats>0.75].index
#print(skew_feats)
all_data[skew_feats] = np.log1p(all_data[skew_feats])

all_data = pd.get_dummies(all_data)
# Fill the missing data with column mean
all_data = all_data.fillna(all_data.mean())

# Create training and testing data set
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice
#print(X_train.describe(), y.describe())

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring = "neg_mean_squared_error", cv = 5))
    return rmse.mean()
    
model_ridge = Ridge()

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
ridge_rmse = [rmse_cv(Ridge(alpha = alpha)) for alpha in alphas]

ridge_rmse = pd.Series(ridge_rmse, index=alphas)
#print(ridge_rmse)

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
#print(rmse_cv(model_lasso))

preds = model_lasso.predict(X_test)
submission = pd.DataFrame({"Id":test["Id"],"SalePrice":np.exp(preds)-1})
submission.to_csv('submission.csv', index=False)
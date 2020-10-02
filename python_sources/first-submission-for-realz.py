#all the imports
import numpy as np
import pandas as pd
import xgboost as xgb

from scipy.stats import skew

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn import linear_model
import xgboost as xgb


#load data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
id = test.Id
y = np.log1p(train.SalePrice)
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))


#measuring function
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train, y, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)

#massage data
#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())


#inputs
train = all_data[:train.shape[0]]
test = all_data[train.shape[0]:]

#X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=.33, random_state=0)


#Simple linear regression first
model_lr = linear_model.LinearRegression()
print("Linear regression")
print(rmse_cv(model_lr))


#XGB
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)
print("XGB")
print(rmse_cv(model_xgb))

#submit the answer
model_xgb.fit(train,y)
predictions = np.expm1(model_xgb.predict(test))

solution = pd.DataFrame({"id":id, "SalePrice":predictions})
solution.to_csv("linear_predictions.csv", index = False)

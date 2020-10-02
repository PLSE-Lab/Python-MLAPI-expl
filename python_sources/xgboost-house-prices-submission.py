# DataCamp XGBoost Practice Kernel

# import packages
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import xgboost as xgb

# import train and test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# dividing data into target and predictor variables
train_X = train.drop('SalePrice', axis=1)
test_X = test
train_y = train.SalePrice

# one-hot encoding categorical variables
onehot_train_X = pd.get_dummies(train_X)
onehot_test_X = pd.get_dummies(test_X)
train_X, test_X = onehot_train_X.align(onehot_test_X, join='left', axis=1)

# imputing missing values with the mean value
my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

# Creating a DMatrix and parameters to use while fine-tuning NUM_ROUNDS parameter
housing_dmatrix = xgb.DMatrix(data=train_X, label=train_y)
params = {"objective":"reg:linear", "max_depth":3}

# Fine-tuned the number of boosting rounds below, found 230 to be the best value
num_rounds = [225, 230, 235]
final_rmse_per_round = []

for x in num_rounds:
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3,
                        num_boost_round=x, metrics="rmse",
                        as_pandas=True, seed=123)
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

num_round_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_round_rmses, columns=["num_boosting_rounds","rmse"]))

# creating a DMatrix and parameters to use while fine-tuning MAX_DEPTH parameter
dmatrix2 = xgb.DMatrix(data=train_X, label=train_y)
params2 = {"objective":"reg:linear", "num_boost_round":230}

# Fine-tuned the max_depth parameter below, found 4 to be the best value
depth_vals = [3, 4, 5, 6, 7, 8]
best_rmse = []

for x in depth_vals:
    params2["max_depth"] = x
    cv_results2 = xgb.cv(dtrain=dmatrix2, params=params2, nfold=3, metrics="rmse", as_pandas=True, seed=123)
    best_rmse.append(cv_results2["test-rmse-mean"].tail().values[-1])

max_depth_rmses = list(zip(depth_vals, best_rmse))
print(pd.DataFrame(max_depth_rmses, columns=["max_depth","rmse"]))



# creating a DMatrix and parameters to use while fine-tuning COLSAMPLE_BYTREE parameter
dmatrix3 = xgb.DMatrix(data=train_X, label=train_y)
params3 = {"objective":"reg:linear", "num_boost_round":230, "max_depth":4}

# Fine-tuned the eta (learning rate) parameter below, found x to be the best value
colsample_vals = [0.01, 0.1, 0.5, 0.9]
best_colsample_rmse = []

for x in colsample_vals:
    colsample_bytree = x
    cv_results3 = xgb.cv(dtrain=dmatrix3, params=params3, nfold=3, metrics="rmse", as_pandas=True, seed=123)
    best_colsample_rmse.append(cv_results3["test-rmse-mean"].tail().values[-1])

colsample_rmses = list(zip(colsample_vals, best_colsample_rmse))
print(pd.DataFrame(colsample_rmses, columns=["colsample","rmse"]))


# using this section below as the testing and final submission
#--------------------------------------------------------------------------------------
# linear regression base learner
model = xgb.XGBRegressor(objective = "reg:linear", n_estimators = 230, colsample_bytree = 0.5, eta = 0.1, max_depth = 4, seed=123)
model.fit(train_X, train_y)
preds = model.predict(test_X)

# creating the submission file
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice':preds})
my_submission.to_csv('submission.csv', index=False)






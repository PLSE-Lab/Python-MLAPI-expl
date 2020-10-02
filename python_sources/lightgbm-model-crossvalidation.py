#· Light GBM is is a gradient boosting framework: https://lightgbm.readthedocs.io/en/latest/Python-Intro.html
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
#import lightgbm as lgb
from lightgbm import LGBMRegressor
#Load data from csv
train_csv = pd.read_csv('../input/train.csv')
test_csv = pd.read_csv('../input/test.csv')
#Identify Ids
train_id = train_csv['Id']
test_id = test_csv['Id']
# 1. Spot missing values by counting number of entries per column: appears less entries compared to number of rows on the rest.  
# 2. Spot outliers by looking into the min and max and standard deviation
train_csv.describe()
#Dividing the training set into two: one for actual training and the other for cross-validation:
df_X =train_csv.drop('SalePrice', axis=1, inplace = False)
df_y = train_csv['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)
y_test
#take id
y_train_id = X_test['Id']
# Return datatype and count the number of type of value
X_train.dtypes.value_counts()
# Combine data of train and test sets
data_all = pd.concat((X_train, X_test), sort = True)
data_all = pd.get_dummies(data_all)
#Replace NA values 
data_all = data_all.fillna(data_all.mean())
data_all.describe()
# Skewness
numeric_feats = data_all.dtypes[data_all.dtypes != "object"].index
data_skewed = data_all[numeric_feats].apply(lambda x: skew(x.dropna()))
data_skewed = data_skewed[abs(data_skewed) > 0.6]
data_skewed = data_skewed.index
data_all[data_skewed] = np.log1p(data_all[data_skewed])
#Divide data back to slots
X_train_slots = data_all[:X_train.shape[0]]
X_test_slots = data_all[X_train.shape[0]:]
y_train_unlog = y_train
#Saleprice Skewness
y_train_slots = np.log1p(y_train_unlog)
y_train_slots
# LightGBM Cross-validation Dataset
lgb = LGBMRegressor(num_iterations=50000, num_leaves=7, learning_rate=0.001, min_data_in_leaf=2).fit(X_train_slots, y_train_slots)
y_pred_log = lgb.predict(X_test_slots)
#convert back from logarithmic values to SalePrice
y_pred = np.expm1(y_pred_log) 
# Model Evaluation
variance = pd.DataFrame({"y_predicted":y_pred, "y_real":y_test})
variance.plot(x = "y_predicted", y = "y_real", kind = "scatter")

#As a simple model, we can see the cross-validation plot looks nicely to gather predictions with the real data.
#So next, we will implement the LightGBM with the full training set, so to take our test set results, which we will submit.

#remove left-side Id-column
train_csv.drop('Id', axis = 1, inplace=True)
test_csv.drop('Id', axis = 1, inplace=True)
# 1. Spot missing values by counting number of entries per column: appears less entries compared to number of rows on the rest.  
# 2. Spot outliers by looking into the min and max and standard deviation
train_csv.describe()
# Return datatype and count the number of type of value
train_csv.dtypes.value_counts()
# Combine data of train and test sets
data_all = pd.concat((train_csv, test_csv), sort = True)
data_all.drop('SalePrice', axis = 1, inplace = True) 
data_all = pd.get_dummies(data_all)
#Replace NA values 
data_all = data_all.fillna(data_all.mean())
data_all.describe()
# Skewness
numeric_feats = data_all.dtypes[data_all.dtypes != "object"].index
data_skewed = data_all[numeric_feats].apply(lambda x: skew(x.dropna()))
data_skewed = data_skewed[abs(data_skewed) > 0.6]
data_skewed = data_skewed.index
data_all[data_skewed] = np.log1p(data_all[data_skewed])

#Divide data back to slots
X_train = data_all[:train_csv.shape[0]]
X_test = data_all[train_csv.shape[0]:]
y_train_unlog = train_csv.SalePrice
#Saleprice Skewness
y_train = np.log1p(y_train_unlog)

# LightGBM Dataset
lgb = LGBMRegressor(num_iterations=50000, num_leaves=7, learning_rate=0.001, min_data_in_leaf=2).fit(X_train, y_train)
y_pred_log = lgb.predict(X_test)
#convert back from logarithmic values to SalePrice
y_pred =np.expm1(y_pred_log) 
#Return output
results = pd.DataFrame({'Id': test_id, 'SalePrice': y_pred})
results.to_csv('results.csv', index = False)
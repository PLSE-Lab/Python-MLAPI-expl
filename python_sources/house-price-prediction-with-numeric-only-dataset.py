# Visit http://arcanumysl.wordpress.com for recent works.
# I am new to Machine Learning. This is my first solution without referring or looking any other kernels.
# Comment me for suggestions.

# What to do next
# 1. Include categorical(string) data whose corr is high enough
# 2. Perform Feature Engineering to make better representative feature data by removing/combing the current dataset.
# 3. Fine-tune the model: Grid Search, HyperParameters
# 4. More Feature Engineering using grid_search.best_estimator_.feature_importances_
# 5. Do these analysis with other models such as SVM
# 6. Pick the best combination and submit.

# ------------------------------------------- #

# Here I am only dealing with numeric data as one-step at a time approach.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/train.csv")


# [Selecting numeric values] - whose corr > 0.4
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)
# print(df_num.corr()['SalePrice'].sort_values(ascending=False))

important = abs(np.array(df_num.corr()['SalePrice']))>0.4
important_index = df_num.keys()[important]
df_num_important = df_num.loc[:,important_index]

# [Data preparation to feed the model]
X = df_num_important.drop('SalePrice', axis=1)
y = df_num_important['SalePrice']

X.isnull().values.any()
X = X.fillna(X.mean())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)

# ------------------------------------------- #

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor:
lin_reg = LinearRegression()
forest_reg = RandomForestRegressor()

# Fit the regressor to the training data
lin_reg.fit(X_train, y_train)
forest_reg.fit(X_train, y_train)


# Predict on the test data: y_pred
y_pred_lin = lin_reg.predict(X_test)
y_pred_forest = forest_reg.predict(X_test)

# [Linear Regression] - Compute and print R^2 and RMSE
print("R^2: {}".format(lin_reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lin))
print("Root Mean Squared Error: {}".format(rmse))

# [Random Forest] - Compute and print R^2 and RMSE
print("R^2: {}".format(forest_reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred_forest))
print("Root Mean Squared Error: {}".format(rmse))

# ------------------------------------------- #

# Import the necessary modules
from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(lin_reg, X, y, cv=5)
print(cv_scores)
print("Average Lin_Reg 5-Fold CV Score: {}".format(np.mean(cv_scores)))


cv_scores = cross_val_score(forest_reg, X, y, cv=5)
print(cv_scores)
print("Average Forest_reg 5-Fold CV Score: {}".format(np.mean(cv_scores)))
# Look like forest model is more powerful as expected

# ------------------------------------------- #
# [Data preparation for test data]
df_test = pd.read_csv("../input/test.csv")
important_index_wo_price = important_index.drop(['SalePrice'])

df_test_num = df_test.loc[:,important_index_wo_price]
df_test_num = df_test_num.fillna(df_test_num.mean())
# print(df_test_num.isnull().values.any())

scaler.fit(df_test_num) # scaling made slight difference # score from 0.17423 -> 0.17194

# I don't want to fit for 'Id' features.
ids = df_test['Id']
predictions = forest_reg.predict(df_test_num)

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions })
output.to_csv('house-predictions_youngseok.csv', index = False)
output.head()


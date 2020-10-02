from xgboost import XGBModel
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error

import pandas as pd

data = pd.read_csv('../input/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
est_X = my_imputer.transform(test_X)

my_model = XGBRegressor(n_estimators=1000, learing_rate=0.05, n_jobs=4)
my_model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X,test_y)], verbose=False)

predictions = my_model.predict(test_X)
print(mean_absolute_error(predictions, test_y))
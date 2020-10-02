# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import time
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from catboost import CatBoostRegressor
import os
print(os.listdir("../input"))

data_train = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')


features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']


#create X and y for training
X = data_train[features]
y = data_train.SalePrice
test_X = test_data[features]

# split data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
#model = XGBRegressor()

print("Train CatBoost Decision Tree")
model_start = time.time()
model = CatBoostRegressor(iterations= 1000,
                          learning_rate=0.05,
                          depth = 5,
                          eval_metric='MAE',
                          random_seed=1,
                          bagging_temperature=22,
                          od_type='Iter',
                          metric_period=100,
                          od_wait=100)

model.fit(X, y,
          use_best_model=True,
          plot=True
          )

test_preds = model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
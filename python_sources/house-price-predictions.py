# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Path of the file to be read
file_path = '/kaggle/input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(file_path)
home_data.columns

features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "TotRmsAbvGrd"]
X = home_data[features]
y = home_data.SalePrice

# Split into validation and test set
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 1)

# Specifying model
iowa_model = DecisionTreeRegressor(random_state = 1)

# Fit Model 
iowa_model.fit(X_train, y_train)
val_predictions = iowa_model.predict(X_val)
val_mae = mean_absolute_error(val_predictions, y_val)
print("Validation MAE when not specifying the leaf nodes:{}".format(val_mae))

# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X, y)

test_file = '/kaggle/input/home-data-for-ml-course/test.csv'
test_data = pd.read_csv(test_file)
testing_X = test_data[features]
test_predict = rf_model.predict(testing_X)

output = pd.DataFrame({'Id': test_data.Id,
                      'SalePrice': test_predict})
output.to_csv('submission.csv', index=False)
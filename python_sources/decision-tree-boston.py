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
        
train_path = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'

unfil = pd.read_csv(train_path)
unfil.head()

feature_names = ['LotFrontage','YearBuilt','OverallQual','1stFlrSF','GarageArea']

train_x = unfil[feature_names]
train_y = unfil['SalePrice']

train_x.fillna(value=70.049958, inplace=True)

from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(max_features = 5, criterion = 'mae')

history = tree_model.fit(train_x, train_y)


test_path = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'
test_data = pd.read_csv(test_path)

x_test = test_data[feature_names]

x_test['LotFrontage'].fillna(value=68.580357, inplace=True)
x_test['GarageArea'].fillna(value = 472.768861, inplace= True)

x_test.isnull().sum() #beauty

prediction = history.predict(x_test)

print(prediction)

submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': prediction})

submission.to_csv('submission.csv', index=False)




# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
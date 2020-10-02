# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Import data set as DataFrame
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Save SalePrice column individually, and drop the column from train set
price = train['SalePrice']
train = train.drop(['SalePrice'], axis=1)
# Combine train and test dataset
data = pd.concat([train, test], axis=0, ignore_index=True)
# Set columns' dtype as object for dummy creation
data['MSSubClass'] = data['MSSubClass'].astype('object')
data['MoSold'] = data['MoSold'].astype('object')
# Create dummies
data_dummies = pd.get_dummies(data)
# Check for entries with NA values
na_counts = data_dummies.isnull().sum()
na_items = list(na_counts[na_counts>0].index)
na_rows = {}
for item in na_items:
	na_rows[item]=list(data[data[item].isnull()].index)
# Fill NA values with problematic rows
# Some NA values can be substituted with 0.0
na_index1 = na_rows['BsmtFinSF1']
na_index2 = na_rows['GarageCars']
na_index3 = na_rows['MasVnrArea']
na_index4 = na_rows['BsmtFullBath']
na_index = na_index1 + na_index2 + na_index3 + na_index4
for index in na_index:
	data_dummies.iloc[index] = data_dummies.iloc[index].fillna(0.0)
# Others need to be filled with random numbers
rnd1 = np.random.normal(
	data_dummies['LotFrontage'].mean(),
	data_dummies['LotFrontage'].std(),
	data_dummies['LotFrontage'].isnull().sum())
rnd1[rnd1<0]=0
rnd2 = np.random.normal(
	data_dummies['GarageYrBlt'].mean(),
	data_dummies['GarageYrBlt'].std(),
	data_dummies['GarageYrBlt'].isnull().sum())
rnd2[rnd2<0]=0
data_dummies['LotFrontage'][data_dummies['LotFrontage'].isnull()] = rnd1
data_dummies['GarageYrBlt'][data_dummies['GarageYrBlt'].isnull()] = rnd2
# Split processed dataset into train and test
train = data_dummies.iloc[:1460]
test = data_dummies.iloc[1460:]
# Train model
# After trying a few diferent regressors I Choose Lasso
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=100).fit(train.as_matrix(), price.as_matrix())
result = lasso.predict(test.as_matrix())
# Organize into submission format
submission = pd.DataFrame(result)
submission.to_csv('submission.csv')
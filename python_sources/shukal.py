# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from xgboost import XGBRegressor as xgbr
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer

x_full=pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv',index_col='Id')
x_test_full=pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv',)

x_full.dropna(axis=0,subset=['SalePrice'],inplace=True)
y=x_full.SalePrice
x_full.drop(['SalePrice'],axis=1,inplace=True)

x_train_full,x_valid_full,y_train,y_valid=train_test_split(x_full,y,train_size=0.8,random_state=0)

categorical_cols = [cname for cname in x_train_full.columns if
                    x_train_full[cname].nunique() < 10 and 
                    x_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in x_train_full.columns if 
                x_train_full[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
x_train = x_train_full[my_cols].copy()
x_valid = x_valid_full[my_cols].copy()
x_test = x_test_full[my_cols].copy()

numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

my_model_2 = xgbr(random_state=42,n_estimators=2000,learning_rate=0.055)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model2', my_model_2)])
clf.fit(x_train,y_train)
preds=clf.predict(x_valid)

print('MAE:', mean_absolute_error(y_valid, preds))





preds_test = clf.predict(x_test) 

# Save test predictions to file
output = pd.DataFrame({'Id': x_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission1.csv', index=False)


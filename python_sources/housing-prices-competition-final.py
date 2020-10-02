# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from scipy.stats import skew
import matplotlib.pyplot as plt

def get_score(n_esti, preprocessor, X_train, y):
    xgboost_model = XGBRegressor(n_estimators = n_esti, learning_rate = 0.01, n_jobs = 5)    

    train_pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('model', xgboost_model)
        ])
        
    train_score = -1 * cross_val_score(train_pipeline, X_train, y, cv = 5, scoring = 'neg_mean_absolute_error')
    
    return train_score.mean()
    
    
    
def get_plots(column, salesprice, column_name):
    plt.scatter(column, salesprice, c = "blue", marker = "s")
    plt.title("Looking for outliers")
    plt.xlabel(column_name)
    plt.ylabel('SalePrice')
    plt.show()

    

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
home_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

print(home_data.shape)

#home_data = home_data[home_data.GrLivArea < 4500]
#print(home_data.shape)
#home_data.reset_index(drop = True, inplace = True)
#print(home_data.shape)

# Removing Outliers
home_data = home_data[home_data.GrLivArea < 4000]
#home_data = home_data[home_data.LotFrontage < 300]
home_data = home_data[home_data.LotArea < 100000]
home_data = home_data[home_data.MasVnrArea < 1400]
home_data = home_data[home_data.BsmtFinSF2 < 1400]
home_data = home_data[home_data.EnclosedPorch < 400]
home_data.reset_index(drop = True, inplace = True)

y = home_data.SalePrice # Predicting Column

X = home_data.drop(['Id', 'SalePrice'], axis = 1) # Training Data
print(len(X.columns))

test_id = test_data.Id
test_data = test_data.drop(['Id'], axis = 1) # testing data

all_data = pd.concat((X, test_data))

categorical_col = all_data.dtypes[all_data.dtypes == 'object'].index#[col for col in X.columns if X[col].nunique() < 10 and X[col].dtype == 'object']
print(categorical_col)

numerical_col = all_data.dtypes[all_data.dtypes != 'object'].index#[col for col in X.columns if X[col].dtype != 'object']
print(numerical_col)
'''
#my_col = categorical_col + numerical_col
#print(len(my_col))

#numeric_feats = X.dtypes[X.dtypes != "object"]#.index
#print('Numeric Features: ', numeric_feats)

# compute skewness
skew_feats = all_data[numerical_col].apply(lambda i: skew(i.dropna()))
skew_feats = skew_feats[skew_feats > 0.75]
skew_feats = skew_feats.index
#print('skew_feats : ', skew_feats)

#all_data[skew_feats] = np.log1p(all_data[skew_feats])
#print('ALL_DATA: ', all_data.columns)
#all_data = pd.get_dummies(all_data)
#print('ALL_DATA: ', all_data.columns)
#print('Length: ', len(all_data.columns))
#print('X: ', X)

#y = np.log1p(y)
#print('y: ', y)

#X_train = X[my_col].copy()
#X_test = test_data[my_col].copy()'''

X_train = all_data[:X.shape[0]]
X_test = all_data[X.shape[0]:]

#print(len(X_train.columns))
#print((X_test.columns))

numeric_transformer = SimpleImputer(strategy = 'constant')

cat_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
    ])
    
preprocessing = ColumnTransformer(transformers = [
    ('num', numeric_transformer, numerical_col),
    ('cat', cat_transformer, categorical_col)
    ])    
    
'''number = 2000    
scores = {}
    
for i in range(1, 9):
    scores[number] = get_score((number), preprocessing, X_train, y)
    number += 250
    
print(scores)'''
    
xgboost_model = XGBRegressor(n_estimators = 2750, learning_rate = 0.01, n_jobs = 4)    

train_pipeline = Pipeline(steps = [
    ('preprocesser', preprocessing),
    ('model', xgboost_model)])
    
train_score = -1 * cross_val_score(train_pipeline, X_train, y, cv = 5, scoring = 'neg_mean_absolute_error')
 
print(train_score.mean())

train_pipeline.fit(X_train, y)

test_preds = train_pipeline.predict(X_test)

output = pd.DataFrame({'Id': test_id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)

'''for column in numerical_col:
    get_plots(X_train[column], y, column)'''
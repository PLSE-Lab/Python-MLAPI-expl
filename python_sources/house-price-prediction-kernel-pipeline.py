# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

home=pd.read_csv('../input/train.csv')
print(home.columns)
y=home['SalePrice']
features=['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X=home[features]


randforest_model=RandomForestRegressor(random_state=1)
randforest_model.fit(X,y)

testdata=pd.read_csv('../input/test.csv')
testx=testdata[features]
predictions=randforest_model.predict(testx)
output=pd.DataFrame({'Id':testdata.Id, 'SalePrice':predictions})
output.to_csv('submission.csv',index=False)



#Now including categorical Data as well

train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')

train_data.dropna(axis=0, subset=['SalePrice'], how='all', inplace=True)
#train_data.shape
#test_data.dropna(axis=0,how='all',inplace=True)
#test_data.shape

target = train_data.SalePrice

cols_with_missing=[col for col in train_data.columns if train_data[col].isnull().any()]
candidate_train_predictors=train_data.drop(['Id','SalePrice']+cols_with_missing, axis=1)
candidate_test_predictors=test_data.drop(['Id']+cols_with_missing,axis=1)

low_cardinality_cols=[cname for cname in candidate_train_predictors.columns if candidate_train_predictors[cname].nunique()<10 and candidate_train_predictors[cname].dtype=='object']
numeric_cols=[cname for cname in candidate_train_predictors.columns if candidate_train_predictors[cname].dtype in ['int64','float64']]

my_columns=low_cardinality_cols+numeric_cols
train_predictors=candidate_train_predictors[my_columns]
test_predictors=candidate_test_predictors[my_columns]

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors, join='left', axis=1)

#my_new_model=RandomForestRegressor()
#my_new_model.fit(final_train,target)
#from sklearn.model_selection import cross_val_score
#def get_mae(X, y):
# return -1 * cross_val_score(RandomForestRegressor(50), X, y, scoring = 'neg_mean_absolute_error').mean()

#predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

#mae_without_categoricals = get_mae(predictors_without_categoricals, target)

#mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)
#np.where(final_test.values >= np.finfo(np.float64).max)
from sklearn.impute import SimpleImputer
my_pipeline=make_pipeline(SimpleImputer(), RandomForestRegressor())
my_pipeline.fit(final_train,target)
#my_imputer = SimpleImputer()
#final_train = my_imputer.fit_transform(final_train)
#final_test = my_imputer.transform(final_test)
#final_test.shape
#final_train.shape
#print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
#print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))

prediction=my_pipeline.predict(final_test)
output=pd.DataFrame({'Id':test_data.Id, 'SalePrice':prediction})
output.to_csv('submission_categorical_included.csv',index=False)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
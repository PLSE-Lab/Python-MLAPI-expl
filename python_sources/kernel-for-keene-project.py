# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


Data_train_raw = pd.read_csv('../input/trainFeatures.csv', delimiter=',')
Data_test_raw = pd.read_csv('../input/testFeatures.csv', delimiter=',')
Label_train_raw = pd.read_csv('../input/trainLabels.csv', delimiter=',')
Data_header = list(Data_train_raw.columns.values)
nullheader = ['ids', 'erkey', 'RatingID', 'AccountabilityID']
Data_train_raw = Data_train_raw.drop(5448)
Label_train_raw = Label_train_raw.drop(5448)


def checknan(data, header):
    for item in header:
        if pd.notnull(data.iloc[0][item]) is False:
            print(item)
            count = 0
            for i in range(0, 9999, 1):
                if pd.notnull(data.iloc[i][item]) is False:
                    count = count + 1
                if count == 9999:
                    print(count)
                    nullheader.append(item)


checknan(Data_train_raw, Data_header)
Data_train_raw = Data_train_raw.drop(columns=nullheader, axis=1)
Data_test_raw = Data_test_raw.drop(columns=nullheader, axis=1)

missing_header = Data_train_raw.columns[Data_train_raw.isna().any()].tolist()
l1_header = ['RatingYear', 'BaseYear', 'Creation_Date', 'date_calculated', 'exclude',
             'ExcludeFromLists', 'Rpt_Comp_Date', 'EmployeeID', 'Rpt_Ap_Date',
             'Rpt_Ap_Emp', 'BoardList_Status', 'StaffList_Status',
             'AuditedFinancial_status', 'Form990_status', 'Privacy_Status',
             'RatingTableID', 'CNVersion', 'ProgSvcRev', 'MemDues', 'PrimaryRev',
             'Other_Revenue', 'Program_Expenses', 'Administration_Expenses',
             'Fundraising_Expenses', 'Total_Liabilities', 'Total_Net_Assets']

for item in list(Data_train_raw.columns.values):
    if item not in missing_header:
        if item not in l1_header:
            Data_train_raw.drop(columns=item, axis=1)
            Data_test_raw.drop(columns=item, axis=1)

'''
pca doest not make model better and result in weird number
pca = PCA(n_components=10)
pca.fit(Data_train_raw.loc[:, l1_header])
pca_result = pca.transform(Data_train_raw.loc[:, l1_header])
pca_data = pd.DataFrame(pca_result)

X_train = pd.concat([pca_data, Data_train_raw.loc[:, missing_header]], axis=1)
X_test = Data_test_raw
Y_train = Label_train_raw.drop(columns=['ids', 'ATScore'], axis=1)
'''

X_train = Data_train_raw
X_test = Data_test_raw
Y_train = Label_train_raw.drop(columns=['ids', 'ATScore'], axis=1)

X_train_train, X_train_test, Y_train_train, Y_train_test = train_test_split(X_train, Y_train, test_size=0.2)

dtrain = xgb.DMatrix(X_train_train, label=Y_train_train, missing=np.nan)
dtest = xgb.DMatrix(X_train_test, label=Y_train_test, missing=np.nan)
evallist = [(dtest, 'eval'), (dtrain, 'train')]

'''
# this part is for CV grid search for optimal hyperparameter

param = {'max_deoth': 10, 'eta': 0.1, 'silent': 0, 'objective': 'reg:linear',
         'colsample_bylevel': 0.8, 'min_child_weight': 5, 'subsample': 0.7, 'colsample_bytree': 0.9,
         'gamma': 0.1, 'nthread': 4, 'eval_metric': 'rmse', 'alpha': 0.05, 'lambda': 0.05}
cv_params = {'eta': [0.001, 0.003, 0.005, 0.007, 0.009]}
num_round = 3000
model = xgb.XGBRegressor(**param)
optimized_model = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=0, n_jobs=4)
optimized_model.fit(X_train, Y_train)
print(optimized_model.best_params_)
print(optimized_model.best_score_)
'''


param_final = {'max_deoth': 10, 'eta': 0.01, 'silent': 0, 'objective': 'reg:linear',
               'colsample_bylevel': 0.8, 'min_child_weight': 5, 'subsample': 0.7, 'colsample_bytree': 0.9,
               'gamma': 0.1, 'nthread': 4, 'eval_metric': 'rmse', 'alpha': 0.05, 'lambda': 0.05}
#num_train = 20000
num_train = 2000
bst = xgb.train(param_final, dtrain, num_train, evallist, early_stopping_rounds=10)

dpred = xgb.DMatrix(X_test)
Y_test = bst.predict(dpred)

result = pd.DataFrame()
result['Id'] = result.index
result['OverallScore'] = Y_test
result['Id'] = result.index+1
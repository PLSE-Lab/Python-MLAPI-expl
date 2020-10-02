import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import xgboost as xgb
from sklearn.model_selection import KFold
from ml_metrics import rmsle
from sklearn.linear_model import LinearRegression

train =  pd.read_csv('../input/train.tsv', sep='\t')
#train = pd.read_csv('../input/train.tsv', sep='	')
y_train = train['price']
test =  pd.read_csv('../input/test.tsv', sep='\t')
#test = pd.read_csv('../input/test.tsv', sep='	')

ans = pd.DataFrame()
ans['test_id'] = test['test_id']
'''
train = train.drop(['train_id', 'name', 'item_description', 'price'], axis=1)
#train['category_name'] = train['category_name'].str.split(' ').str[-1]

test = pd.read_csv('test.tsv', sep='	')
train['category_name'] = train['category_name'].fillna('unnamed')
train['brand_name'] = train['brand_name'].fillna('unnamed')
test['category_name'] = test['category_name'].fillna('unnamed')
test['brand_name'] = test['brand_name'].fillna('unnamed')
b = test['test_id']
test = test.drop(['test_id', 'item_condition_id', 'item_description'], axis=1)
train['brand_name_c'] = y_train
train['brand_name_b'] = y_train
test['brand_name_c'] = np.float()
test['brand_name_b'] = np.float()
a = y_train

i=0
print(len(pd.unique(train['brand_name'])), len(pd.unique(train['category_name'])))
for string in pd.unique(train['brand_name']):
    a[train['brand_name'] == string] = np.mean(y_train[train['brand_name'] == string])
    b[test['brand_name'] == string] = np.mean(y_train[train['brand_name'] == string])
    i += 1
    print(i)
a.to_csv('train_brand_name_c.csv')
b.to_csv('test_brand_name_c.csv')
i=0
for category in pd.unique(train['category_name']):
    i+=1
    print(i)
    a[train['category_name'] == category] = np.mean(y_train[train['category_name'] == category])
    b[test['category_name'] == category] = np.mean(y_train[train['category_name'] == category])
a.to_csv('train_category_name.csv')
b.to_csv('test_category_name.csv')
train.to_csv('train_with.csv')
test.to_csv('test_with.csv')
print(train['brand_name_c'])
'''
train1 = pd.read_csv('../input/train_category_name.csv', encoding='ISO-8859-1')
train2 = pd.read_csv('../input/train_brand_name_c.csv')
train1.columns = ['2', '1']
train2.columns = ['1', '2']
X_train = np.concatenate((np.array(train1['1']).reshape((len(train1), 1)), np.array(train2['2']).reshape((len(train1), 1))), axis=1)
test1 = pd.read_csv('../input/test_category_name.csv', encoding='ISO-8859-1')
test2 = pd.read_csv('../input/test_brand_name_c.csv')
test1.columns = ['2', '1']
test2.columns = ['1', '2']
X_test = np.concatenate((np.array(test1['1']).reshape((len(test1), 1)), np.array(test2['2']).reshape((len(test2), 1))), axis=1)
K = 5
kf = KFold(n_splits=K, random_state=42, shuffle=True)
# 5 Cross Validation
results = []
i = 0
lr = LinearRegression()
lr.fit(X_train, y_train[1:])
ee = lr.predict(X_test)
ans['price'] = ee
ans.to_csv('ans.csv',  index=False)
'''for train_index, test_index in kf.split(X_train):
    i += 1
    train_X, valid_X = X_train[train_index], X_train[test_index]
    train_y, valid_y = y_train[train_index], y_train[test_index]
    watchlist = [(xgb.DMatrix(train_X, train_y), 'train'), (xgb.DMatrix(valid_X, valid_y), 'valid')]
    params = {'eta': 0.03, 'max_depth': 4, 'objective': 'reg:linear', 'seed': 42, 'silent': True}

    model = xgb.train(params, xgb.DMatrix(train_X, train_y), 1500, watchlist, maximize=False, verbose_eval=5,
                      early_stopping_rounds=100)
    resy = pd.DataFrame(model.predict(xgb.DMatrix(X_test)))
    resy.to_csv(str(i)+'mod.csv')
'''
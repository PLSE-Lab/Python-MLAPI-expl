from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
#now = datetime.datetime.now()

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
macro = pd.read_csv('../input/macro.csv')
id_test = test.id

y_train = train["price_doc"] * .9691 + 10.08
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

#can't merge train with test because the kernel run for very long time

df_tot = pd.concat([x_train,x_test])

for c in df_tot.columns:
    if df_tot[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_tot[c].values)) 
        df_tot[c] = lbl.transform(list(df_tot[c].values))
        
x_train = df_tot.head(len(x_train))
x_test = df_tot.tail(len(x_test))


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'alpha' : 0.9,
    'lambda' : 10,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_output = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=1000, 
                   early_stopping_rounds=20,
                   verbose_eval=20, 
                   show_stdv=False)

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)


y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

output.to_csv('xgbSub.csv', index=False)

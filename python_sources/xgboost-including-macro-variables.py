#!/usr/bin/env python
# coding: utf-8

# ### Consider the influence of seasonality

# ### Consider the year of the transaction

# ### Consider smaller set of features in macro
# * eliminate two similar features in macro:"micex_rgbi_tr", "micex_cbi_tr", and keep "micex".
# * eliminate "balance_trade_growth", and keep "balance_trade". 
# * optimize the macro features

# # Plot some figures

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
#now = datetime.datetime.now()

# From here: https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity/notebook
#macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
#"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
#"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build","bandwidth_sports", 
#             "micex", "net_capital_export", "oil_urals"]

macro_cols = ["balance_trade_growth", "eurrub", "average_provision_of_build_contract","deposits_rate", "mortgage_value", "mortgage_rate",
"income_per_cap", "apartment_build","bandwidth_sports", 
             "micex", "net_capital_export", "oil_urals"]


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#macro = pd.read_csv('./input/macro.csv')

macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)


id_test = test.id
train.sample(3)
print(train.shape)
print(test.shape)
# Any results you write to the current directory are saved as output.


# In[ ]:


#It seems that this doen't improve anything. 

train["timestamp"] = pd.to_datetime(train["timestamp"])
#train["year"] = train["timestamp"].dt.year
#train["month"] = train["timestamp"].dt.month

#train["year"], train["month"], train["day"] = train["timestamp"].dt.year,train["timestamp"].dt.month,train["timestamp"].dt.day

test["timestamp"] = pd.to_datetime(test["timestamp"])
#test["year"] = test["timestamp"].dt.year
#test["month"] = test["timestamp"].dt.month
#test["year"], test["month"], test["day"] = test["timestamp"].dt.year,test["timestamp"].dt.month,test["timestamp"].dt.day


# In[ ]:


train_all = pd.merge_ordered(train, macro, on='timestamp', how='left')
test_all = pd.merge_ordered(test, macro, on='timestamp', how = 'left')

print(train_all.shape)
print(test_all.shape)


# In[ ]:


plt.plot(train_all.mortgage_rate, train_all.eurrub, '.')


# In[ ]:


# Other feature engineering
#train_all['rel_floor'] = train_all['floor'] / train_all['max_floor'].astype(float)
#train_all['rel_kitch_sq'] = train_all['kitch_sq'] / train_all['full_sq'].astype(float)
#train_all['rel_life_sq'] = train_all['life_sq'] / train_all['full_sq'].astype(float)

#test_all['rel_floor'] = test_all['floor'] / test_all['max_floor'].astype(float)
#test_all['rel_kitch_sq'] = test_all['kitch_sq'] / test_all['full_sq'].astype(float)
#test_all['rel_life_sq'] = test_all['life_sq'] / test_all['full_sq'].astype(float)


# In[ ]:


#train_all.rel_life_sq > 1
#train_all.loc[(train_all.rel_life_sq > 1.)]


# In[ ]:


y_train = train_all["price_doc"]
x_train = train_all.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test_all.drop(["id", "timestamp"], axis=1)

#can't merge train with test because the kernel run for very long time

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        #x_train.drop(c,axis=1,inplace=True)
        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
        #x_test.drop(c,axis=1,inplace=True)       


# In[ ]:


xgb_params = {
    'eta': 0.05,
    'max_depth': 4,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)


# In[ ]:


cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()


# In[ ]:


num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(8, 20))
xgb.plot_importance(model, max_num_features=100, height=0.8, ax=ax)


# In[ ]:


y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()


# In[ ]:


output.to_csv('sub10.csv', index=False)


# In[ ]:





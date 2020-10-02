#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[ ]:


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    w[y != 0] = 1./(y[y != 0]**2)
    return w

def rmspe(yhat, y):
    return np.sqrt(np.mean( ToWeight(y) * (y - yhat)**2 ))

def rmspe_xgb(yhat, y):
    y = np.exp(y.get_label()) - 1
    yhat = np.exp(yhat) - 1
    return "rmspe", np.sqrt(np.mean(ToWeight(y) * (y - yhat)**2))


# In[ ]:


def showCorr(df):
    fig = plt.subplots(figsize = (10,10))
    sb.set(font_scale=1.5)
    sb.heatmap(df.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})
    plt.show()


# In[ ]:


def plotItem(df, group, column, title='', label='', setFigure=True, size=3):
    if setFigure:
        plt.figure(figsize=(30, size))
        
    if title == '':
        title = column +' x '+ group
    plot = df.groupby(group)[column].max().plot(legend=True, marker='X', label=label+' max', title=title)
    plot = df.groupby(group)[column].mean().plot(legend=True, marker='o', label=label+' mean')
    plot = df.groupby(group)[column].min().plot(legend=True, marker='x', label=label+' min')  


# In[ ]:


features = ['Store', 'State', 'StoreType', 'Assortment', 'Holiday', 'Promo', 'Day', 'DayOfWeek', 'WeekOfMonth', 'Month',  'Year']
featuresTrain = features + ['Sales']


# In[ ]:


def excludeStoreMaintence(df):
    excludeStore = []
    for store in df.Store.unique():
        excludeStore.append({'Store': store, 'Exclude': df.loc[df.Store == store].Store.count() < 750})    
    df = pd.merge(df, pd.DataFrame(excludeStore), on='Store')
    return df.loc[df.Exclude == False]


# In[ ]:


monthStr = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

def checkPromoInterval(item):
    if isinstance(item['PromoInterval'],str) and item['MonthStr'] in item['PromoInterval']:
        return 1
    else:
        return 0

def changeFeatureSelection(df, features, checkOpen=False, removeStoreMaintence=False):
        
    for column in ['StoreType', 'Assortment', 'StateHoliday', 'State']:
        df[column] = df[column].replace(0,'0')
        labels = df[column].unique()
        map_labels = dict(zip(labels, range(0,len(labels))))
        df[column] = df[column].map(map_labels)       
    
    df['Holiday'] = (df.SchoolHoliday == 1) | (df.StateHoliday > 0)
    df['Holiday'] = df.Holiday.astype(int)
    
    Date = pd.DatetimeIndex(df.Date)
    df['Day'] = Date.day
    df['Month'] = Date.month
    df['Week'] = Date.week
    df['WeekOfMonth'] = (Date.day-1)/7+1
    df['WeekOfMonth'] = df['WeekOfMonth'].astype(int)
    df['Year'] = Date.year
    df['MonthStr'] = df.Month.map(monthStr)

    df['PromoMonth'] = df.apply(lambda item: checkPromoInterval(item), axis=1)
    df['Promo'] = (df.Promo == 1) | (df.PromoMonth == 1)
    df['Promo'] = df.Promo.astype(int)
        
    df['CompetitionDistance'] = np.where(df.CompetitionDistance == 0, 0, df.CompetitionDistance / 1000)     
    
    competitionOpenBase = 12*2013
    df['CompetitionOpen'] = np.where(df.CompetitionOpenSinceYear < 2013, 0, 12*df.CompetitionOpenSinceYear + df.CompetitionOpenSinceMonth - competitionOpenBase)
    
    if (checkOpen):
        df = df.loc[df.Open == 1] 
        
    if 'Customers' in df.columns:
        df = df.loc[df.Customers > 0]
        
    if 'Sales' in df.columns:
        df = df.loc[(df.Sales >= 1500) & (df.Sales <= 35000)]
        
    if (removeStoreMaintence):
        df = excludeStoreMaintence(df)
        
    for column in features:
        df[column] = df[column].astype(float)
    return df


# In[ ]:


train = pd.read_csv("../input/competicao-dsa-machine-learning-mar-2019/dataset_treino.csv")
test = pd.read_csv("../input/competicao-dsa-machine-learning-mar-2019/dataset_teste.csv")
store = pd.read_csv("../input/competicao-dsa-machine-learning-mar-2019/lojas.csv")
store_state = pd.read_csv("../input/store-states/store_states.csv")


# In[ ]:


store.fillna(0, inplace=True)
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)


# In[ ]:


store = pd.merge(store, store_state, on='Store')
store['StateName'] = store['State']
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')


# In[ ]:


train = changeFeatureSelection(train, features, True, True)
test = changeFeatureSelection(test, features)


# In[ ]:


plotItem(train, 'Store', 'Sales', '', '', True, 10)
plotItem(train, 'StateName', 'Sales')


# In[ ]:


plotItem(train, 'Year', 'Sales')
plotItem(train, 'StateName', 'Sales')
plotItem(train, ['Year', 'Month'], 'Sales', 'Sales x [Year-Month]', '', True, 3)


# In[ ]:


showCorr(train[featuresTrain])
showCorr(test[features])


# In[ ]:


params = {"objective": "reg:linear", "eta": 0.2, "max_depth": 10, "subsample": 0.7, "colsample_bytree": 0.7, "silent": 1}
num_rounds = 1000


# In[ ]:


X_train, X_test = train_test_split(train[featuresTrain], test_size=0.2, random_state=10)
dtrain = xgb.DMatrix(X_train[features], np.log(X_train.Sales + 1))
dvalid = xgb.DMatrix(X_test[features], np.log(X_test.Sales + 1))
dtest = xgb.DMatrix(test[features])


# In[ ]:


evallist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_rounds, evals=evallist, early_stopping_rounds=100, feval=rmspe_xgb, verbose_eval=True)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(model, max_num_features= 10, height=0.5, ax=ax)
plt.show()


# In[ ]:


predict = model.predict(xgb.DMatrix(X_test[features]))
predict[predict < 0] = 0
error = rmspe(np.exp(predict) - 1, X_test.Sales.values)
print('rmspe:', error)


# In[ ]:


predict = model.predict(xgb.DMatrix(test[features]))
predict[test.Open==0] = 0


# In[ ]:


submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(predict) - 1})
submission.Sales = submission.Sales.astype(int)
submission.sort_values('Id', inplace=True)
test['Sales'] = submission.Sales
submission.to_csv("submission.csv", index=False)


# In[ ]:


test = test.loc[test.Open == 1].copy()
train = train.loc[(train.Open == 1) & (train.Month >= test.Month.min()) & (train.Month <= test.Month.max())].copy()


# In[ ]:


plotItem(train, 'Day', 'Sales')
plotItem(test, 'Day', 'Sales', '', 'predict', False)

plotItem(train, 'Month', 'Sales')
plotItem(test, 'Month', 'Sales', '', 'predict', False)

plotItem(train, 'DayOfWeek', 'Sales')
plotItem(test, 'DayOfWeek', 'Sales', '', 'predict', False)

plotItem(train, 'WeekOfMonth', 'Sales')
plotItem(test, 'WeekOfMonth', 'Sales', '', 'predict', False)

plotItem(train, 'Year', 'Sales')
plotItem(test, 'Year', 'Sales', '', 'predict', False)


# In[ ]:


plotItem(train, 'Store', 'Sales', '', '', True, 5)
plotItem(test, 'Store', 'Sales', '', 'predict', True, 5)


# In[ ]:


plotItem(train, 'StateName', 'Sales', '', '', True, 10)
plotItem(test, 'StateName', 'Sales', '', 'predict', False)


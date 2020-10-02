#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


burakhans = "../input/burakhans/assesment-train-data-and-result-file/"

os.listdir(burakhans)

bf = pd.read_csv(os.path.join(burakhans, "train.csv"))
bf.drop(bf.columns[:1],axis=1,inplace=True)


# In[ ]:


bf.columns


# In[ ]:


bf.Segment


# In[ ]:


df["month"] = df.Departure_YMD_LMT.map(lambda date:int(str(date)[4:6]))

df.describe()

df.columns

sns.pairplot(df)

df.count()

df.apply(pd.value_counts)[:10]

df['SWC_Baggage'] = df['Passenger_Baggage_Count'].map(lambda x: 0  if x is 0 else 1)

df['is_1'] = df['Operation_Count'].map(lambda x: 1  if x is 1 else 0)

df.drop(["Passenger_Baggage_Count"],axis=1,inplace=True)

plt.matshow(df.corr())
plt.show()


df.drop(["Operation_Count"],axis=1, inplace=True)


# In[ ]:


datadir = "../input/datathon/assessment/assessment/Assessment Data/"

train_path = os.path.join(datadir,"Assessment Train Data.csv")
test_path = os.path.join(datadir,"Assessment Result File.csv")

df = pd.read_csv(train_path)


# In[ ]:


def map_channels(df):
    channels = {
        "JW": "online",
        "TW":"online",
        "TS":"mobile",
        "JM":"mobile",
        "TY":"kontuar",
        "QC":"kontuar",
        "SC":"kiosk"
    }

    df['Operation_Channel'] = df['Operation_Channel'].map(channels) 
    df['Operation_Channel'].fillna("other", inplace=True)
    return df 


# In[ ]:


channel, gender = df.Operation_Channel, df.Passenger_Gender
df.drop(df.columns[:13],axis=1,inplace=True)
df["Operation_Channel"] = channel
df["gender"] = gender
df['Cabin_Class'] = df['Cabin_Class'].replace(['?'], 'Y')
df.Passenger_Baggage_Weight /= df.Passenger_Baggage_Weight.max()
df.Passenger_Baggage_Count /= df.Passenger_Baggage_Count.max()
df = map_channels(df)


# In[ ]:


df = pd.get_dummies(df)
df.head()


# In[ ]:


z = np.log(y)
sns.distplot(z)


# In[ ]:


z = y/y.max()


# In[ ]:


z.value_counts().plot()


# In[ ]:


df = df.groupby(by='Operation_Count').filter(lambda x: len(x) > 1)


# In[ ]:


df.Operation_Count.value_counts().plot()


# In[ ]:


df["log_y"] = np.log(df.Operation_Count.values)
df.log_y.value_counts().plot()


# In[ ]:


max_op_count = df.Operation_Count.max()
df.Operation_Count = normalize(df.Operation_Count) 


# In[ ]:


df.Operation_Count.value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split

X = df.loc[:, df.columns != "Operation_Count"]
y = df.Operation_Count
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11, stratify=y)


# In[ ]:


from sklearn.metrics import (mean_squared_error, r2_score,
                             explained_variance_score,
                             median_absolute_error,
                             mean_absolute_error,
                             max_error,
                             mean_squared_log_error
                            )

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    

def score(reg):
    y_pred = reg.predict(X_test)
    # y_pred = np.exp(y_pred)
    y_pred = [int(pred) for pred in y_pred]
    # y_pred = [int(pred*max_op_count) for pred in y_pred]
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    print('Explained variance score: %.2f' % explained_variance_score(y_test, y_pred))
    print('Score: %.2f' % reg.score(X_test,y_test))
    print('median_absolute_error: %.2f' % median_absolute_error(y_test,y_pred))
    print('max_error: %.2f' % max_error(y_test,y_pred))
    print('mean_squared_log_error: %.2f' % mean_squared_log_error(y_test,y_pred))
    print('mean_absolute_error: %.2f' % mean_absolute_error(y_test,y_pred))

    print("Mean absolute percentage error: %.2f" % mean_absolute_percentage_error(y_test, y_pred))


# In[ ]:


def plot_feature_importances(model):
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importances.nlargest(12).plot(kind='barh')


# In[ ]:


from sklearn.compose import TransformedTargetRegressor

def train(reg):
    regr_trans = TransformedTargetRegressor(regressor=reg,
                                            func=np.log1p,
                                            inverse_func=np.expm1)
    reg.fit(X_train, y_train)
    score(reg)
    return reg


# In[ ]:


from sklearn.dummy import DummyRegressor

dummy = DummyRegressor()
train(dummy)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
train(lr)


# In[ ]:


from xgboost import XGBRegressor

xgbr = XGBRegressor(n_estimators=10, n_jobs=12)
train(xgbr)


# In[ ]:


plot_feature_importances(xgbr)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, n_jobs=12)
rf.fit(X_train, y_train)
score(rf)
plot_feature_importances(rf)


# In[ ]:


sns.pairplot(df, diag_kind="kde")
plt.show()


# In[ ]:


xgbr = XGBRegressor(n_estimators=10, n_jobs=12)
bilbobaggins = BaggingRegressor(base_estimator=xgbr, max_features=6)
bilbobaggins.fit(X_train, y_train)
score(bilbobaggins)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gbr = GradientBoostingRegressor(**params)
gbr.fit(X_train, y_train)
score(gbr)


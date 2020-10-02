#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


get_ipython().run_line_magic('ls', '-lrth ../input/m5-forecasting-accuracy/')


# In[ ]:


INPUT_PATH='/kaggle/input/m5-forecasting-accuracy'


# In[ ]:


df_calendar = pd.read_csv(f'{INPUT_PATH}/calendar.csv')


# In[ ]:


df_sales_train = pd.read_csv(f'{INPUT_PATH}/sales_train_validation.csv')


# In[ ]:


df_sell_prices = pd.read_csv(f'{INPUT_PATH}/sell_prices.csv')


# In[ ]:


df_sample_sub = pd.read_csv(f'{INPUT_PATH}/sample_submission.csv')


# ### This is a Sample Submission which actually copies the last 28 days of data to Future. And this is my first submission. Exploring the solution futher, I can predict by using Moving Average, Time Series, Regression Algo, etc.

# In[ ]:


for day in range(1914, 1914+28):
    df_sales_train[f"d_{day}"] = df_sales_train[df_sales_train.columns[-5:]].mean(axis=1)

#df[df.columns[-33:]]


# In[ ]:


date_cols = [col for col in df_sales_train.columns if 'd_' in col]
temp_sample = df_sales_train.set_index('id')[date_cols[-28:]].reset_index()


# In[ ]:


temp_sample.columns = df_sample_sub.columns
temp_sample = temp_sample + df_sample_sub
temp_sample.id = df_sample_sub.id
temp_sample.fillna(0, inplace=True)
print(temp_sample)
temp_sample.to_csv('submission.csv', index=False)


# df_calendar
# df_sales_train
# df_sell_prices
# df_sample_sub

# In[ ]:


df_sales_train.head()


# In[ ]:


#np.r_[0,-28:0]


# In[ ]:


df_calendar.head()


# In[ ]:


df_sell_prices.head()


# In[ ]:


#df_sell_prices[(df_sell_prices.store_id == 'CA_1') & (df_sell_prices.item_id == 'HOBBIES_1_001')].sort_values(by='wm_yr_wk').plot(kind='scatter', x='wm_yr_wk', y='sell_price')


# In[ ]:


#np.finfo(np.float16)


# In[ ]:


#np.iinfo(np.int16)


# In[ ]:


df_sell_prices.describe()


# In[ ]:


df_sell_prices.info()


# In[ ]:


df_calendar.info()


# In[ ]:


df_calendar.describe()


# In[ ]:


df_sales_train.info()


# In[ ]:


df_sales_train.describe()#.apply(lambda x: format(x, 'f'))


# In[ ]:


# df_sales_train.drop(columns='id', inplace=True, axis=1)


# In[ ]:


def updateDataType(df):
    datatypes = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    columns = df.columns
    for column in columns:
        column_type = df[column].dtype
        if column_type in datatypes:
            minVal, maxVal = df[column].min(), df[column].max()
            if str(column_type).startswith('int'):
                if minVal > np.iinfo(np.int8).min and maxVal < np.iinfo(np.int8).max:
                    df[column] = df[column].astype('int8')
                elif minVal > np.iinfo(np.int16).min and maxVal < np.iinfo(np.int16).max:
                    df[column] = df[column].astype('int16')
                elif minVal > np.iinfo(np.int32).min and maxVal < np.iinfo(np.int32).max:
                    df[column] = df[column].astype('int32')
                else: pass
            else:
                if minVal > np.finfo(np.float16).min and maxVal < np.finfo(np.float16).max:
                    df[column] = df[column].astype('float16')
                elif minVal > np.finfo(np.float32).min and maxVal < np.finfo(np.float32).max:
                    df[column] = df[column].astype('float32')
                else: pass
    return df        


# In[ ]:


df_sales_train_l = df_sales_train.select_dtypes(exclude='int64')


# In[ ]:


df_sales_train_r = df_sales_train.select_dtypes(include='int64').astype('int8')


# In[ ]:


df_sales_train = df_sales_train_l.join(df_sales_train_r)


# In[ ]:


df_sales_train.info()


# In[ ]:


df_calendar = updateDataType(df_calendar)
# df_sales_train = updateDataType(df_sales_train) # As we already converted the dtypes
df_sell_prices = updateDataType(df_sell_prices)


# In[ ]:


df_calendar.info()


# In[ ]:


df_sales_train.info()


# In[ ]:


df_sell_prices.info()


# In[ ]:


import plotly.express as px


# In[ ]:


df_calendar.shape


# In[ ]:


df_sales_train.shape


# In[ ]:


df_sell_prices.shape


# ### No.of Items in the Dataset

# In[ ]:


print(df_sales_train.item_id.nunique())
print(df_sales_train.item_id.unique())


# ### No.of Departments in the Dataset

# In[ ]:


print(df_sales_train.dept_id.nunique())
print(df_sales_train.dept_id.unique())


# ### No.of Catagories in the Dataset

# In[ ]:


print(df_sales_train.cat_id.nunique())
print(df_sales_train.cat_id.unique())


# ### No.of Stores in the Dataset

# In[ ]:


print(df_sales_train.store_id.nunique())
print(df_sales_train.store_id.unique())


# ### No.of States in the Dataset

# In[ ]:


print(df_sales_train.state_id.nunique())
print(df_sales_train.state_id.unique())


# ### No.of Event Names in the Calendar Dataset

# In[ ]:


print(df_calendar.event_name_1.nunique())
print(df_calendar.event_name_1.unique())

print(df_calendar.event_name_2.nunique())
print(df_calendar.event_name_2.unique())


# ### No.of Event Types in the Calandar Dataset

# In[ ]:


print(df_calendar.event_type_1.nunique())
print(df_calendar.event_type_1.unique())

print(df_calendar.event_type_2.nunique())
print(df_calendar.event_type_2.unique())


# In[ ]:


values = {'event_name_1': 'No_Event', 'event_name_2': 'No_Event', 'event_type_1': 'Regular', 'event_type_2': 'Regular'}
df_calendar.fillna(value=values, inplace=True)


# In[ ]:


df_calendar.head()


# In[ ]:


df_sales_train.head()


# df_calendar_t = df_calendar.copy()
# 
# categories = df_sales_train.cat_id.unique()
# days = [day for day in df_sales_train.columns if 'd_' in day]
# for day in days:
#     for cat in categories:
#         df_calendar_t.loc[df_calendar_t.d == day, cat] = df_sales_train.loc[df_sales_train.cat_id == cat, day].sum()

# df_calendar_t.head()

# In[ ]:


# fig = px.bar(df_calendar, x="event_type_1", y="sales", barmode="group", facet_row="year", facet_col="weekday",
#             category_orders={"weekday": ["Monday", "Tuesday", "Wednesday", "Thrusday", "Friday", "Saturday", "Sunday"]})
# fig.show()


# fig1 = px.line(df_calendar_t, x="date", y="HOBBIES", color="year", line_group="weekday", hover_name="month",
#         line_shape="spline", render_mode="svg")
# fig2 = px.line(df_calendar_t, x="date", y="HOUSEHOLD", color="month", line_group="year", hover_name="weekday",
#         line_shape="vh", render_mode="svg")
# fig3 = px.line(df_calendar_t, x="date", y="FOODS", color="weekday", line_group="month", hover_name="weekday",
#         line_shape="linear", render_mode="svg")
# 
# fig1.show()
# fig2.show()
# fig3.show()

# From the above, every Christmas, the sales are dropped to single digit value almost. It is clear that People wish to celebrate rather purchase.

# ### No.of Items in the Sell Prices Dataset

# In[ ]:


print(df_sell_prices.item_id.nunique())
print(df_sell_prices.item_id.unique())


# In[ ]:


df_sell_prices.head()


# In[ ]:


# %%time
# df_sell_prices.item_id.str.split("_").str[0]

# This takes 14sec to fetch result


# ### No.of Item Types

# In[ ]:


get_ipython().run_cell_magic('time', '', 'pd.Series(df_sell_prices.item_id.unique().tolist()).str.split("_").str[0].unique()')


# In[ ]:


# hobbiesCount = df_sell_prices.item_id.str.match('HOBBIES').sum()
# householdCount = df_sell_prices.item_id.str.match('HOUSEHOLD').sum()
# fooddCount = df_sell_prices.item_id.str.match('FOODS').sum()


# In[ ]:


pd.Series(df_sell_prices.item_id.unique().tolist()).str.split("_").str[0].value_counts()


# In[ ]:


#import matplotlib.pyplot as plt
#import seaborn as sb


# In[ ]:


#sb.countplot(df_sell_prices.item_id.str.split("_").str[0])


# In[ ]:


#df_sell_prices.sell_price.describe().apply(lambda x: format(x, 'f'))


# f,ax=plt.subplots(1,2,figsize=(16,9))
# 
# sb.countplot('state_id', data=df_sales_train, ax=ax[0])
# df_sales_train.state_id.value_counts().plot.pie(autopct='%1.1f%%',ax=ax[1],shadow=True)
# 
# ax[0].set_title('Sales per State')
# ax[1].set_title('Statewise Sales')
# 
# plt.show()

# In[ ]:


#df_sales_train.id.str.split("_")


# In[ ]:


#df_sales_train_ca = df_sales_train[df_sales_train.state_id.str.startswith('CA')]
#df_sales_train_ca.shape


# In[ ]:


#df_sales_train_tx = df_sales_train[df_sales_train.state_id.str.startswith('TX')]
#df_sales_train_tx.shape


# In[ ]:


#df_sales_train_wi = df_sales_train[df_sales_train.state_id.str.startswith('WI')]
#df_sales_train_wi.shape


# In[ ]:


#df_sample_sub.head()


# In[ ]:


#df_sample_sub.id.str.split("_").str[3].unique()


# In[ ]:


#id_vars=['A'], value_vars=['B'], var_name='myVarname', value_name='myValname'


# In[ ]:


import gc


# df_sales_train_tx_melt = pd.melt(df_sales_train_tx, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'd', value_name = 'demand')

# In[ ]:


df_sales_train_tx_melt.info()


# In[ ]:


df_sales_train_tx_melt.head()


# In[ ]:


df_calendar.head()


# In[ ]:


df_sales_train_last28 = df_sales_train.iloc[:, np.r_[0,-28:0]]
df_sales_train_last28.tail()


# In[ ]:


df_sales_train_last28_melt = df_sales_train_last28.melt('id', var_name='d', value_name='demand')
df_sales_train_last28_melt.head()


# In[ ]:


df_sales_train_byweek = df_sales_train_last28_melt.merge(df_calendar).groupby(['id', 'wday'])['demand'].mean()


# In[ ]:


sub = df_sample_sub.copy()
# change the column names to match the last 28 days
sub.columns = ['id'] + ['d_' + str(1914+x) for x in range(28)]
# select only the rows with an id with the validation tag
sub = sub.loc[sub.id.str.contains('validation')]


# In[ ]:


# melt this dataframe and merge it with the calendar so we can join it with by_weekday dataframe
sub = sub.melt('id', var_name='d', value_name='demand')
sub = sub.merge(df_calendar)[['id', 'd', 'wday']]
df = sub.join(df_sales_train_byweek, on=['id', 'wday'])
df.head()


# In[ ]:


df = df.pivot(index='id', columns='d', values='demand')
df.reset_index(inplace=True)
df.head()


# In[ ]:


submission = df_sample_sub[['id']].copy()
submission = submission.merge(df)
submission = pd.concat([submission, submission], axis=0)
submission['id'] = df_sample_sub.id.values
submission.columns = columns = ['id'] + ['F' + str(i) for i in range(1,29)]


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.shape


# In[ ]:





# df_sales_train_tx_melt_merge = df_sales_train_tx_melt.merge(df_calendar, on='d')

# In[ ]:


del df_sales_train, df_sales_train_tx, df_sales_train_tx_melt, df_calendar
gc.collect()


# In[ ]:


df_sales_train_tx_melt_merge.head()


# In[ ]:


df_sell_prices.head()


# df_merge1 = df_sales_train_tx_melt_merge.merge(df_sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'])

# In[ ]:


del df_sales_train_tx_melt_merge, df_sell_prices
gc.collect()


# In[ ]:


df_merge1.head()


# In[ ]:


from sklearn import preprocessing
def transform(data):
    
    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in nan_features:
        data[feature].fillna('unknown', inplace = True)
    
    encoder = preprocessing.LabelEncoder()
    data['id_encode'] = encoder.fit_transform(data['id'])
    
    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'weekday']
    for feature in cat:
        encoder = preprocessing.LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])
    
    return data


# df_merge1 = transform(df_merge1)

# In[ ]:


df_merge1.info()


# df_merge1 = updateDataType(df_merge1)

# In[ ]:


df_merge1.info()


# In[ ]:


df_merge1.head()


# In[ ]:


#df_merge1[df_merge1.id == 'HOBBIES_1_004_TX_1_validation']


# df_merge1['5day_avg'] = df_merge1.groupby('id')['demand'].transform(lambda x: x.shift(5).rolling(5).mean())
# df_merge1['10day_avg'] = df_merge1.groupby('id')['demand'].transform(lambda x: x.shift(10).rolling(10).mean())
# df_merge1['20day_avg'] = df_merge1.groupby('id')['demand'].transform(lambda x: x.shift(20).rolling(20).mean())
# df_merge1['50day_avg'] = df_merge1.groupby('id')['demand'].transform(lambda x: x.shift(50).rolling(50).mean())

# df_merge1['7day_std'] = df_merge1.groupby('id')['demand'].transform(lambda x: x.shift(7).rolling(7).std())
# df_merge1['15day_std'] = df_merge1.groupby('id')['demand'].transform(lambda x: x.shift(15).rolling(15).std())
# df_merge1['28day_std'] = df_merge1.groupby('id')['demand'].transform(lambda x: x.shift(28).rolling(28).std())
# df_merge1['100day_std'] = df_merge1.groupby('id')['demand'].transform(lambda x: x.shift(100).rolling(100).std())

# df_merge1['1day_change'] = df_merge1.groupby('id')['demand'].transform(lambda x: x - x.shift(1))
# df_merge1['2day_change'] = df_merge1.groupby('id')['demand'].transform(lambda x: x - x.shift(2))
# df_merge1['5day_change'] = df_merge1.groupby('id')['demand'].transform(lambda x: x - x.shift(5))
# df_merge1['10day_change'] = df_merge1.groupby('id')['demand'].transform(lambda x: x - x.shift(10))
# df_merge1['28day_change'] = df_merge1.groupby('id')['demand'].transform(lambda x: x - x.shift(28))

# df_merge1['7day_min'] = df_merge1.groupby('id')['demand'].transform(lambda x: x.rolling(7).min())
# df_merge1['7day_max'] = df_merge1.groupby('id')['demand'].transform(lambda x: x.rolling(7).max())
# df_merge1['28day_min'] = df_merge1.groupby('id')['demand'].transform(lambda x: x.rolling(28).min())
# df_merge1['28day_max'] = df_merge1.groupby('id')['demand'].transform(lambda x: x.rolling(28).max())

# df_merge1["date"] = pd.to_datetime(df_merge1["date"])
# df_merge1["year"] = df_merge1["date"].dt.year
# df_merge1["month"] = df_merge1["date"].dt.month
# df_merge1["week"] = df_merge1["date"].dt.week
# df_merge1["day"] = df_merge1["date"].dt.day
# df_merge1["dayofweek"] = df_merge1["date"].dt.dayofweek
# df_merge1["is_weekend"] = df_merge1["dayofweek"].isin([5, 6])

# df_merge1 = updateDataType(df_merge1)

# In[ ]:


df_merge1.info()


# In[ ]:


#df_merge1.date.max()


# In[ ]:


#df_merge1.drop(columns='id', axis=1, inplace=True)


# In[ ]:


features = df_merge1.select_dtypes(exclude=['object', 'datetime64[ns]']).columns.tolist()
features


# In[ ]:


X_train = df_merge1[df_merge1.date <= '2016-01-29'][features]
y_train = X_train['demand']

X_test = df_merge1[(df_merge1['date'] > '2016-03-27') & (df_merge1['date'] <= '2016-04-24')][features]
y_test = X_test.demand


# In[ ]:


del df_merge1
gc.collect()


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# In[ ]:


ad


# In[ ]:


from sklearn import linear_model, metrics


# In[ ]:


X_train.drop(columns = 'demand', axis=1, inplace=True)
X_test.drop(columns = 'demand', inplace=True)


# linear_reg = linear_model.LinearRegression()
# linear_reg.fit(X_train, y_train)
# print(linear_reg.score(X_test, y_test))

# In[ ]:


import lightgbm as lgb


# In[ ]:


best_params = {
    "boosting_type": "gbdt",
    "metric": "rmse",
    "objective": "regression",
    "n_jobs": -1,
    "seed": 42,
    "learning_rate": 0.1,
    "bagging_fraction": 0.75,
    "bagging_freq": 10,
    "colsample_bytree": 0.75,
}
fit_params = {
    "num_boost_round": 100_000,
    "early_stopping_rounds": 50,
    "verbose_eval": 100,
}


# lgb_train = lgb.Dataset(X_train,
#                         label=y_train,
#                         free_raw_data=False)
# lgb_test = lgb.Dataset(X_test,
#                        label=y_test,
#                        free_raw_data=False)
#     
# del X_train, X_test
# gc.collect()

# In[ ]:


print(type(lgb_train))


# In[ ]:


from datetime import datetime


# start=datetime.now()
# 
# model = lgb.train(best_params,
#                 lgb_train,
#                 valid_sets=lgb_test,
#                 **fit_params,
#                 )
# 
# stop=datetime.now()

# In[ ]:


execution_time_lgbm = stop-start
execution_time_lgbm


# In[ ]:


z


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test = y_test.astype('float16')


# from sklearn.metrics import mean_squared_error 
# 
# mse_lgbm = mean_squared_error(y_pred,y_test)
# print(mse_lgbm)
# print(np.sqrt(mse_lgbm))

# In[ ]:





# In[ ]:


sfdsefs


# #### Fit regression model
# params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
#           'learning_rate': 0.01, 'loss': 'ls'}
# model = ensemble.GradientBoostingRegressor(**params)

# model.fit(X_train, y_train)

# In[ ]:


y_pred = model.predict(X_test)


# mse = mean_squared_error(y_test, y_pred)
# print("MSE: %.4f" % mse)

# In[ ]:





# In[ ]:





# To make analysis of data in table easier, we can reshape the data into a more computer-friendly form using Pandas in Python. Pandas.melt() is one of the function to do so..
# 
# Pandas.melt() unpivots a DataFrame from wide format to long format.

# In[ ]:


fsjaslk;f


# In[ ]:


df_sales_train_melt = pd.melt(df_sales_train, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')


# In[ ]:


df_sales_train_melt.head()


# In[ ]:


df_sales_train_melt.id.nunique()


# In[ ]:


df_sales_train.head()


# In[ ]:


df_sell_prices.head()


# In[ ]:


df_calendar.head()


# In[ ]:


df_sales_train.columns


# In[ ]:


df_sell_prices.columns


# In[ ]:


df_calendar.columns


# In[ ]:


import gc


# In[ ]:


del df_calendar_t
del df_sales_train
gc.collect()


# In[ ]:


df_sales_train_melt.head()


# In[ ]:


df_sales_train_melt.tail()


# In[ ]:


# merged_data = pd.merge(df_sales_train_melt, df_sell_prices, on=['store_id', 'item_id'])


# In[ ]:


#df_merge1 = df_sell_prices.merge(df_calendar, on=['wm_yr_wk'])


# In[ ]:


df_merge1.shape


# In[ ]:


df_merge1.head()


# In[ ]:


del df_sell_prices
del df_calendar
gc.collect()


# In[ ]:


#df_sales_train_melt = df_sales_train_melt.loc[df_sales_train_melt.demand != 0]


# In[ ]:


#df_sales = df_sales_train_melt[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()


# In[ ]:


#df_merge2 = df_sales.merge(df_merge1, on=['store_id', 'item_id'])


# In[ ]:


gc.collect()


# In[ ]:


df_merge2.shape


# In[ ]:


df_merge2.head()


# In[ ]:


pd.Series(df_merge1.id.unique().tolist()).str.split("_").str[-1].unique()


# In[ ]:


del df_merge1
del df_sales_train_melt
gc.collect()


# In[ ]:


df_plot = df_merge2.loc[df_merge2.item_id == 'FOODS_3_611', ['state_id', 'date', 'sell_price']].sort_values(by='sell_price')
df_plot.head()


# In[ ]:


fig = px.scatter(df_plot, x="date", y="sell_price", color="state_id", 
                 marginal_y="violin", marginal_x="box", trendline="ols")
                 #marginal_y="rug", marginal_x="histogram")
fig.show()


# In[ ]:





# df_calendar
# 
# df_sales_train
# 
# df_sell_prices
# 
# df_sample_sub

# In[ ]:


df_sample_sub.id.head()


# In[ ]:


df_sales_train.id.head()


# In[ ]:


df_sales_train.groupby(['cat_id']).sum().T.reset_index(drop = True)


# In[ ]:





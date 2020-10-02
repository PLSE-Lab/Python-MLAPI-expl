#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


# # **Load data**

# In[ ]:


INPUT_DIR = '../input/m5-forecasting-accuracy/'
train_df = pd.read_csv(os.path.join(INPUT_DIR, 'sales_train_evaluation.csv'))
price_df = pd.read_csv(os.path.join(INPUT_DIR, 'sell_prices.csv'))
calender_df = pd.read_csv(os.path.join(INPUT_DIR, 'calendar.csv'))

calender_df['date'] = pd.to_datetime(calender_df['date'])


# In[ ]:


train_df.head(3)


# In[ ]:


calender_df


# In[ ]:


price_df.head(3)


# Split *price_df* into 10 dataframes (one dataframe for each store) just to speed up the function *get_item_price* (see below):

# In[ ]:


price_dfs = {
    'CA_1': price_df[price_df['store_id'] == 'CA_1'],
    'CA_2': price_df[price_df['store_id'] == 'CA_2'],
    'CA_3': price_df[price_df['store_id'] == 'CA_3'],
    'CA_4': price_df[price_df['store_id'] == 'CA_4'],
    'TX_1': price_df[price_df['store_id'] == 'TX_1'],
    'TX_2': price_df[price_df['store_id'] == 'TX_2'],
    'TX_3': price_df[price_df['store_id'] == 'TX_3'],
    'WI_1': price_df[price_df['store_id'] == 'WI_1'],
    'WI_2': price_df[price_df['store_id'] == 'WI_2'],
    'WI_3': price_df[price_df['store_id'] == 'WI_3']
}


# Drop the variable which we don't need anymore:

# In[ ]:


price_df = None


# Some functions to prepare a dataframe for one item:

# In[ ]:


def transform_d_dates_to_dates(d_dates):
    return calender_df.set_index('d').loc[d_dates]['date']


def transform_dates_to_d_dates(dates):
    return calender_df.set_index('date').loc[dates]['d']


def transform_dates_to_wm_yr_wk(dates):
    return calender_df.set_index('date').loc[dates]['wm_yr_wk']


def get_avg_item_n_sold_prev_month(item_id, store_id, dates_df):
    assert len(item_id.split('_')) == 3
    assert store_id in ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    assert dates_df.shape == (1604, 14)
    
    d_dates_month_ago = transform_dates_to_d_dates(dates_df['date'] - pd.DateOffset(days=28))
    assert d_dates_month_ago.shape == (dates_df.shape[0],)
    
    first_day = int(d_dates_month_ago.iloc[0].split('_')[1]) - 28
    last_day = int(d_dates_month_ago.iloc[-1].split('_')[1])
    assert first_day == 310 and last_day == 1941
    
    tmp = train_df[
        (train_df['item_id'] == item_id) &
        (train_df['store_id'] == store_id)
    ][['d_' + str(i) for i in range(first_day, last_day + 1)]]
    assert tmp.shape == (1, dates_df.shape[0] + 28)
    return tmp.iloc[0].rolling(28).mean().iloc[28:].to_numpy()


def get_item_n_sold_year_ago(item_id, store_id, dates_df):
    assert len(item_id.split('_')) == 3
    assert store_id in ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    assert dates_df.shape == (1604, 14)
    
    d_dates_year_ago = transform_dates_to_d_dates(dates_df['date'] - pd.DateOffset(years=1))
    assert d_dates_year_ago.shape == (dates_df.shape[0],)
    
    tmp = train_df[
        (train_df['item_id'] == item_id) &
        (train_df['store_id'] == store_id)
    ][d_dates_year_ago]
    assert tmp.shape == (1, dates_df.shape[0])
    return tmp.iloc[0].to_numpy()


def get_item_price(item_id, store_id, dates_df):
    assert len(item_id.split('_')) == 3
    assert store_id in ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    assert dates_df.shape == (1604, 14)
    
    wm_yr_wk = transform_dates_to_wm_yr_wk(dates_df['date']).to_numpy()
    assert wm_yr_wk.shape == (dates_df.shape[0],)
    
    week_to_price = price_dfs[store_id][
        (price_dfs[store_id]['item_id'] == item_id)
    ].set_index('wm_yr_wk')['sell_price'].to_dict()
    
    price = np.full(dates_df.shape[0], np.nan)
    for i in range(wm_yr_wk.shape[0]):
        week = wm_yr_wk[i]
        if week in week_to_price:
            price[i] = week_to_price[week]
    
    item_price_df = pd.DataFrame(data={'price': price})
    item_price_df = item_price_df.fillna(method='ffill').fillna(method='bfill')

    assert item_price_df['price'].isna().sum() == 0
    assert item_price_df.shape == (dates_df.shape[0], 1)

    # norm_price = item_price_df['price']
    # item_price_df['price'] /= np.linspace(1.00, 1.05, num=item_price_df.shape[0])  # inflation
    norm_price = item_price_df['price'].to_numpy()
    
    assert norm_price.shape == (dates_df.shape[0],)
    return norm_price


def get_is_snap(item_id, store_id, dates_df):
    assert 'FOODS' in item_id
    assert len(item_id.split('_')) == 3
    assert store_id in ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    assert dates_df.shape == (1604, 14)
    
    if store_id in ['CA_1', 'CA_2', 'CA_3', 'CA_4']:
        return dates_df['snap_CA'].to_numpy()
    elif store_id in ['TX_1', 'TX_2', 'TX_3']:
        return dates_df['snap_TX'].to_numpy()
    elif store_id in ['WI_1', 'WI_2', 'WI_3']:
        return dates_df['snap_WI'].to_numpy()

    assert False
    return None
    
    
def get_week_days_features(item_id, store_id, dates_df):
    assert len(item_id.split('_')) == 3
    assert store_id in ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    assert dates_df.shape == (1604, 14)
    
    return pd.DataFrame(
        index=dates_df['d'],
        data={
            'is_Monday': (dates_df['weekday'] == 'Monday').astype(int).to_numpy(),
            'is_Tuesday': (dates_df['weekday'] == 'Tuesday').astype(int).to_numpy(),
            'is_Wednesday': (dates_df['weekday'] == 'Wednesday').astype(int).to_numpy(),
            'is_Thursday': (dates_df['weekday'] == 'Thursday').astype(int).to_numpy(),
            'is_Friday': (dates_df['weekday'] == 'Friday').astype(int).to_numpy(),
            'is_Saturday': (dates_df['weekday'] == 'Saturday').astype(int).to_numpy()
        }
    )
    
    
def get_event_features(item_id, store_id, dates_df):
    assert len(item_id.split('_')) == 3
    assert store_id in ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    assert dates_df.shape == (1604, 14)
    
    events_df = dates_df[['date', 'd', 'event_type_1', 'event_type_2']].copy()
    events_df['tomorrow_event_type_1'] = events_df['event_type_1'].shift(periods=-1)
    events_df['tomorrow_event_type_2'] = events_df['event_type_2'].shift(periods=-1)
    
    return pd.DataFrame(
        index=events_df['d'],
        data={
            'is_today_religious': (
                (events_df['event_type_1'] == 'Religious') |
                (events_df['event_type_2'] == 'Religious')
            ).astype(int).to_numpy(),
            'is_today_national': (
                (events_df['event_type_1'] == 'National') |
                (events_df['event_type_2'] == 'National')
            ).astype(int).to_numpy(),
            'is_today_cultural': (
                (events_df['event_type_1'] == 'Cultural') |
                (events_df['event_type_2'] == 'Cultural')
            ).astype(int).to_numpy(),
            'is_today_sporting': (
                (events_df['event_type_1'] == 'Sporting') |
                (events_df['event_type_2'] == 'Sporting')
            ).astype(int).to_numpy(),
            'is_tomorrow_religious': (
                (events_df['tomorrow_event_type_1'] == 'Religious') |
                (events_df['tomorrow_event_type_2'] == 'Religious')
            ).astype(int).to_numpy(),
            'is_tomorrow_national': (
                (events_df['tomorrow_event_type_1'] == 'National') |
                (events_df['tomorrow_event_type_2'] == 'National')
            ).astype(int).to_numpy(),
            'is_tomorrow_cultural': (
                (events_df['tomorrow_event_type_1'] == 'Cultural') |
                (events_df['tomorrow_event_type_2'] == 'Cultural')
            ).astype(int).to_numpy(),
            'is_tomorrow_sporting': (
                (events_df['tomorrow_event_type_1'] == 'Sporting') |
                (events_df['tomorrow_event_type_2'] == 'Sporting')
            ).astype(int).to_numpy()
        }
    )


def get_item_X_y(item, is_debug):
    assert len(item.split('_')) == 5
    
    dates_df = calender_df.iloc[365:]
    
    item_parts = item.split('_')
    item_id = item_parts[0] + '_' + item_parts[1] + '_' + item_parts[2]
    store_id = item_parts[3] + '_' + item_parts[4]
    
    df = pd.DataFrame(
        index=dates_df['d'].to_numpy(),
        data={
            'avg_item_n_sold_prev_month': get_avg_item_n_sold_prev_month(item_id, store_id, dates_df),
            'item_n_sold_year_ago': get_item_n_sold_year_ago(item_id, store_id, dates_df),
            'price': get_item_price(item_id, store_id, dates_df)
        }
    )
    
    if 'FOODS' in item:
        df['is_snap'] = get_is_snap(item_id, store_id, dates_df)
        
    df = pd.concat([
        df,
        get_week_days_features(item_id, store_id, dates_df),
        get_event_features(item_id, store_id, dates_df)
    ], axis=1)
    assert df.isna().sum().sum() == 0
    
    features_to_drop = []
    for feature in df.columns:
        if len(df[feature].unique()) <= 1:
            features_to_drop.append(feature)
    df = df.drop(features_to_drop, axis=1)
    if is_debug:
        print('Features', features_to_drop, 'have been dropped')
    
    target = train_df[
        train_df['id'] == item + '_evaluation'
    ][
        ['d_' + str(i) for i in range(366, 1942)]
    ].to_numpy()[0]
    assert target.shape == (1576,)
    target = np.concatenate([target, np.full(28, np.nan)])
    df['target'] = target

    return df


# Functions to train a model (one model for each item):

# In[ ]:


def plot_feature_importances(model, features):
    assert len(model.coef_) == len(features)
    plt.figure(figsize=(12, 4))
    plt.title('FEATURE IMPORTANCES')
    sns.barplot(x=model.coef_, y=features)
    
    
def plot_public_test(y_true, y_pred):
    assert y_true.shape == y_pred.shape == (28,)
    plt.figure(figsize=(14, 3))
    plt.title('PUBLIC TEST')
    plt.plot([i for i in range(1, 29)], y_true, label='true')
    plt.plot([i for i in range(1, 29)], y_pred, label='pred')
    plt.legend()


def train_item_model(item, is_debug):
    df = get_item_X_y(item, is_debug)
    assert df.shape[0] == 1604

    X = df.drop(['target'], axis=1)
    y = df['target']
    
    X_train = X.loc[['d_' + str(i) for i in range(366, 1914)]]  # train
    y_train = y.loc[['d_' + str(i) for i in range(366, 1914)]]  # train
    assert X_train.shape[0] == y_train.shape[0] == 1548
    
    X_valid = X.loc[['d_' + str(i) for i in range(1914, 1942)]]  # public test
    y_valid = y.loc[['d_' + str(i) for i in range(1914, 1942)]]  # public test
    X_test = X.loc[['d_' + str(i) for i in range(1942, 1970)]]  # private test
    assert X_valid.shape[0] == y_valid.shape[0] == X_test.shape[0] == 28
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # train
    X_valid = scaler.transform(X_valid)  # public test
    X_test = scaler.transform(X_test)  # private test
    
    model = Ridge()
    model.fit(X_train, y_train)
    
    y_valid_pred = model.predict(X_valid)
    y_valid_pred[y_valid_pred < 0] = 0
    y_valid_pred[y_valid_pred > y.max()] = y.max()
    
    y_test_pred = model.predict(X_test)
    y_test_pred[y_test_pred < 0] = 0
    y_test_pred[y_test_pred > y.max()] = y.max()
    
    if is_debug:
        plot_feature_importances(model, X.columns)
        plot_public_test(y_valid, y_valid_pred)
        print('PUBLIC TEST: mean_absolute_error =', mean_absolute_error(y_valid, y_valid_pred))
    
    return y_valid_pred, y_test_pred


# Let's take a look at one random dataframe:

# In[ ]:


get_item_X_y('HOBBIES_1_004_CA_1', is_debug=True)


# # **Model for one random item**

# In[ ]:


train_item_model('HOBBIES_1_004_CA_1', is_debug=True)


# # **Models for all items**

# In[ ]:


submission = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))


# In[ ]:


for i in tqdm.tqdm(range(train_df.shape[0])):
    item = train_df.iloc[i]['item_id'] + '_' + train_df.iloc[i]['store_id']
    public_test_y, private_test_y = train_item_model(item, is_debug=False)

    submission.loc[
        submission[submission['id'] == item + '_validation'].index,
        ['F' + str(i) for i in range(1, 29)]
    ] = public_test_y

    submission.loc[
        submission[submission['id'] == item + '_evaluation'].index,
        ['F' + str(i) for i in range(1, 29)]
    ] = private_test_y


# In[ ]:


submission


# In[ ]:


submission.to_csv('submission.csv', index=False)


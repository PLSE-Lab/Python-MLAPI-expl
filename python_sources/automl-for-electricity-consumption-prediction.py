import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, make_union
from copy import copy
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pylab as plt
from xgboost import plot_importance
from pandas import concat

## Yuri Shendryk / Electricity Prediction 

#################################### TRAINING ##############################

## 1. Read data
df = pd.read_csv("../input/electricity/train_electricity.csv")

## 2. Add datetime related features
def add_datetime_features(df):
    features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",
                "Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start",
                "Hour", "Minute",]

    datetime = pd.to_datetime(df.Date * (10 ** 9))

    df['Datetime'] = datetime  # We won't use this for training, but we'll remove it later

    for feature in features:
        new_column = getattr(datetime.dt, feature.lower())
        df[feature] = new_column
    return df
df = add_datetime_features(df)

## 3. Preprocessing
label_col = "Consumption_MW"  # The target values are in this column
to_drop = [label_col, "Date"]  # Columns we do not need for training

df.Is_month_start = df.Is_month_start.astype(int) ## Convert boolean to int
df.Is_year_end = df.Is_year_end.astype(int) ## Convert boolean to int
df.Is_year_start = df.Is_year_start.astype(int) ## Convert boolean to int
df.Is_month_end = df.Is_month_end.astype(int) ## Convert boolean to int
df = df.set_index('Datetime') ## Set datetime index

## 4. Remove outliers (hardcoded)
df['Coal_MW'].mask(df['Coal_MW'] < 0, inplace=True)
df['Gas_MW'].mask(df['Gas_MW'] < 0, inplace=True)
df['Wind_MW'].mask(df['Wind_MW'] < 0, inplace=True)
df['Solar_MW'].mask(df['Solar_MW'] < 0, inplace=True)
df['Consumption_MW'].mask(df['Consumption_MW'] > 10000, inplace=True)
df['Consumption_MW'].mask(df['Consumption_MW'] < 3000, inplace=True)
df['Nuclear_MW'].mask(df['Nuclear_MW'] < 1200, inplace=True)
df['Biomass_MW'].mask(df['Biomass_MW'] > 80, inplace=True)
df['Wind_MW'].mask(df['Wind_MW'] > 3000, inplace=True)

## 5. Interpolate data
df['Consumption_MW'].interpolate(method='linear', limit_direction='both', inplace=True)
df['Coal_MW'].interpolate(method='linear',  limit_direction='both', inplace=True)
df['Gas_MW'].interpolate(method='linear', limit_direction='both', inplace=True)
df['Hidroelectric_MW'].interpolate(method='linear', limit_direction='both', inplace=True)
df['Nuclear_MW'].interpolate(method='linear', limit_direction='both', inplace=True)
df['Wind_MW'].interpolate(method='linear', limit_direction='both', inplace=True)
df['Solar_MW'].interpolate(method='linear', limit_direction='both', inplace=True)
df['Biomass_MW'].interpolate(method='linear', limit_direction='both', inplace=True)
df['Production_MW'].interpolate(method='linear', limit_direction='both', inplace=True)

## 6. Add a 'weekday' feature
df['Weekday'] = ((df.index.dayofweek) // 5 == 1).astype(int)

## 7. Add weather related features and interpolated them
## The data is downloaded from: http://rp5.ru/archive.php?wmo_id=15422&lang=en)
df_weather = pd.read_csv('../input/weather/weather.csv', header=0)
df_weather['Time'] = pd.to_datetime(df_weather['Time'])
df_weather = df_weather.set_index('Time') ## Set datetime index
# T, air temperature
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['T']))
df['T'] = df.index.floor('H').map(df_comb)
df['T'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['Po']))
# P0, atmospheric pressure at weather station level (millimeters of mercury)
df['Po'] = df.index.floor('H').map(df_comb)
df['Po'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['P']))
# P, atmospheric pressure reduced to mean sea level (millimeters of mercury)
df['P'] = df.index.floor('H').map(df_comb)
df['P'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['Pa']))
df['Pa'] = df.index.floor('H').map(df_comb)
df['Pa'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['U']))
# U, relative humidity (%) at a height of 2 metres above the earth's surface
df['U'] = df.index.floor('H').map(df_comb)
df['U'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['Ff']))
# FF, mean wind speed at a height of 10-12 metres above the earth’s surface over the 10-minute period immediately preceding the observation (meters per second)
df['Ff'] = df.index.floor('H').map(df_comb)
df['Ff'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['VV']))
# VV, horizontal visibility (km)
df['VV'] = df.index.floor('H').map(df_comb)
df['VV'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['Td']))
df['Td'] = df.index.floor('H').map(df_comb)
df['Td'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['RRR']))
df['RRR'] = df.index.floor('H').map(df_comb)
df['RRR'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['Tn']))
df['Tn'] = df.index.floor('H').map(df_comb)
df['Tn'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['Tx']))
df['Tx'] = df.index.floor('H').map(df_comb)
df['Tx'].interpolate(method='linear', limit_direction='both', inplace=True)

## 8. Add cyclic features:
df['min_sin'] = np.sin(df['Minute']*(2.*np.pi/59))
df['min_cos'] = np.cos(df['Minute']*(2.*np.pi/59))
df['hr_sin'] = np.sin(df['Hour']*(2.*np.pi/23))
df['hr_cos'] = np.cos(df['Hour']*(2.*np.pi/23))
df['mnth_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['mnth_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
df['day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
df['week_sin'] = np.sin(2 * np.pi * df['Week'] / 53)
df['week_cos'] = np.cos(2 * np.pi * df['Week'] / 53)
df['doy_sin'] = np.sin(2 * np.pi * df['Dayofyear'] / 366)
df['doy_cos'] = np.cos(2 * np.pi * df['Dayofyear'] / 366)

## 9. Add time-shift features (12- and 24- hours)
df_shift = concat([df['Production_MW'].shift(144), df['Production_MW'].shift(72)], axis=1)
df_shift.columns = ['Prod-2', 'Prod-1']
df_shift['Prod-2'].interpolate(method='nearest', limit_direction='both', inplace=True)
df_shift['Prod-1'].interpolate(method='nearest', limit_direction='both', inplace=True)
df = concat([df, df_shift], axis=1)
del df_shift
df_shift = concat([df['T'].shift(144), df['T'].shift(72)], axis=1)
df_shift.columns = ['T-2', 'T-1']
df_shift['T-2'].interpolate(method='nearest', limit_direction='both', inplace=True)
df_shift['T-1'].interpolate(method='nearest', limit_direction='both', inplace=True)
df = concat([df, df_shift], axis=1)
del df_shift

print("Dataset has", len(df), "entries.")
print(f"\n\t{'Column':20s} | {'Type':8s} | {'Min':12s} | {'Max':12s}\n")
for col_name in df.columns:
    col = df[col_name]
    print(f"\t{col_name:20s} | {str(col.dtype):8s} | {col.min():12.1f} | {col.max():12.1f}")


## 9. Train best model (from TPOP optimization) using all data
x_train = df.drop(columns=to_drop).values
y_train = df[label_col].values

print('XGBoost training started...')
exported_pipeline = make_pipeline(XGBRegressor(learning_rate=0.1, max_depth=6, min_child_weight=7,
                 n_estimators=5000, nthread=50, subsample=0.1))

exported_pipeline.fit(x_train, y_train)

#################################### INFERENCE ##############################

## 1. Read test dataset, use the best model (from TPOT) for prediction, save submission csv
test_df = pd.read_csv("../input/electricity/test_electricity.csv")
test_df = add_datetime_features(test_df)

test_df.Is_month_start = test_df.Is_month_start.astype(int)
test_df.Is_year_end = test_df.Is_year_end.astype(int)
test_df.Is_year_start = test_df.Is_year_start.astype(int)
test_df.Is_month_end = test_df.Is_month_end.astype(int)
test_df = test_df.set_index('Datetime')

## 2. Remove outliers
test_df['Coal_MW'].mask(test_df['Coal_MW'] < 0, inplace=True)
test_df['Gas_MW'].mask(test_df['Gas_MW'] < 0, inplace=True)
test_df['Wind_MW'].mask(test_df['Wind_MW'] < 0, inplace=True)
test_df['Solar_MW'].mask(test_df['Solar_MW'] < 0, inplace=True)
test_df['Nuclear_MW'].mask(test_df['Nuclear_MW'] < 1200, inplace=True)
test_df['Biomass_MW'].mask(test_df['Biomass_MW'] > 80, inplace=True)
test_df['Wind_MW'].mask(test_df['Wind_MW'] > 3000, inplace=True)

## 3. Interpolate
test_df['Coal_MW'].interpolate(method='linear',  limit_direction='both', inplace=True)
test_df['Gas_MW'].interpolate(method='linear', limit_direction='both', inplace=True)
test_df['Hidroelectric_MW'].interpolate(method='linear', limit_direction='both', inplace=True)
test_df['Nuclear_MW'].interpolate(method='linear', limit_direction='both', inplace=True)
test_df['Wind_MW'].interpolate(method='linear', limit_direction='both', inplace=True)
test_df['Solar_MW'].interpolate(method='linear', limit_direction='both', inplace=True)
test_df['Biomass_MW'].interpolate(method='linear', limit_direction='both', inplace=True)
test_df['Production_MW'].interpolate(method='linear', limit_direction='both', inplace=True)

## 4. Add a 'weekday' feature
test_df['Weekday'] = ((test_df.index.dayofweek) // 5 == 1).astype(int)

## 5. Add weather features
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['T']))
test_df['T'] = test_df.index.floor('H').map(df_comb)
test_df['T'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['Po']))
test_df['Po'] = test_df.index.floor('H').map(df_comb)
test_df['Po'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['P']))
test_df['P'] = test_df.index.floor('H').map(df_comb)
test_df['P'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['Pa']))
test_df['Pa'] = test_df.index.floor('H').map(df_comb)
test_df['Pa'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['U']))
test_df['U'] = test_df.index.floor('H').map(df_comb)
test_df['U'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['Ff']))
test_df['Ff'] = test_df.index.floor('H').map(df_comb)
test_df['Ff'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['VV']))
test_df['VV'] = test_df.index.floor('H').map(df_comb)
test_df['VV'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['Td']))
test_df['Td'] = test_df.index.floor('H').map(df_comb)
test_df['Td'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['RRR']))
test_df['RRR'] = test_df.index.floor('H').map(df_comb)
test_df['RRR'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['Tn']))
test_df['Tn'] = test_df.index.floor('H').map(df_comb)
test_df['Tn'].interpolate(method='linear', limit_direction='both', inplace=True)
df_comb = dict(zip(df_weather.index.floor('H'), df_weather['Tx']))
test_df['Tx'] = test_df.index.floor('H').map(df_comb)
test_df['Tx'].interpolate(method='linear', limit_direction='both', inplace=True)

## 6. Add cyclic features:
test_df['min_sin'] = np.sin(test_df['Minute']*(2.*np.pi/59))
test_df['min_cos'] = np.cos(test_df['Minute']*(2.*np.pi/59))
test_df['hr_sin'] = np.sin(test_df['Hour']*(2.*np.pi/23))
test_df['hr_cos'] = np.cos(test_df['Hour']*(2.*np.pi/23))
test_df['mnth_sin'] = np.sin(2 * np.pi * test_df['Month'] / 12)
test_df['mnth_cos'] = np.cos(2 * np.pi * test_df['Month'] / 12)
test_df['day_sin'] = np.sin(2 * np.pi * test_df['Day'] / 31)
test_df['day_cos'] = np.cos(2 * np.pi * test_df['Day'] / 31)
test_df['week_sin'] = np.sin(2 * np.pi * test_df['Week'] / 53)
test_df['week_cos'] = np.cos(2 * np.pi * test_df['Week'] / 53)
test_df['doy_sin'] = np.sin(2 * np.pi * test_df['Dayofyear'] / 366)
test_df['doy_cos'] = np.cos(2 * np.pi * test_df['Dayofyear'] / 366)

## 7. Add time-shift features for 'Production_MW' and 'T'
df_shift = concat([test_df['Production_MW'].shift(144), test_df['Production_MW'].shift(72)], axis=1)
df_shift.columns = ['Prod-2', 'Prod-1']
df_shift['Prod-2'].interpolate(method='nearest', limit_direction='both', inplace=True)
df_shift['Prod-1'].interpolate(method='nearest', limit_direction='both', inplace=True)
test_df = concat([test_df, df_shift], axis=1)
del df_shift
df_shift = concat([test_df['T'].shift(144), test_df['T'].shift(72)], axis=1)
df_shift.columns = ['T-2', 'T-1']
df_shift['T-2'].interpolate(method='nearest', limit_direction='both', inplace=True)
df_shift['T-1'].interpolate(method='nearest', limit_direction='both', inplace=True)
test_df = concat([test_df, df_shift], axis=1)
del df_shift

print("Dataset has", len(test_df), "entries.")
print(f"\n\t{'Column':20s} | {'Type':8s} | {'Min':12s} | {'Max':12s}\n")
for col_name in test_df.columns:
    col = test_df[col_name]
    print(f"\t{col_name:20s} | {str(col.dtype):8s} | {col.min():12.1f} | {col.max():12.1f}")

test_data = test_df.drop(columns=["Date"]).values

solution_df = pd.DataFrame(test_df["Date"])
solution_df["Consumption_MW"] = exported_pipeline.predict(test_data)

solution_df.to_csv("final_submission.csv", index=False) # RMSE of ~308 MW
print("Done!")

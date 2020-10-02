#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
plt.style.use('bmh')
# mpl.rcParams['axes.labelcolor'] = 'grey'
# mpl.rcParams['xtick.color'] = 'grey'
mpl.rcParams['xtick.labelsize'] = 'large'
# mpl.rcParams['ytick.color'] = 'grey'
# mpl.rcParams['axes.labelcolor'] = 'grey'
# mpl.rcParams['text.color'] = 'grey'
mpl.rcParams["legend.loc"] = 'best'


# In[ ]:


buildings = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
weather_train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")
weather_test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_test.csv")


# # BUILDING

# In[ ]:


buildings['age'] = 2016 - buildings['year_built']


# In[ ]:


sns.distplot(buildings.age.dropna());


# In[ ]:


sns.distplot(buildings.square_feet.dropna());


# In[ ]:


def show_counts(df, col, n):
    cnt = df[col].value_counts().nlargest(n)[::-1]
    fig, ax = plt.subplots()
    ax.barh(cnt.index.astype(str), cnt.values)
    ax.set_xlabel('count')
    ax.set_title(col, color='white')
    for i, v in enumerate(cnt):
        ax.text(v + 3, i, str(int(v/cnt.sum() * 100)) + "%", color='grey', fontweight='bold')


# In[ ]:


show_counts(buildings, 'primary_use', 5);


# In[ ]:


show_counts(buildings, 'floor_count', 7)


# # age

# In[ ]:


def compare_kde(df, col, n):
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(9, 10), sharex=True)
    most_use_case = buildings.primary_use.value_counts().nlargest(n).index
    for i, use_case in enumerate(most_use_case):
        sns.distplot(buildings.loc[buildings.primary_use==use_case, col].dropna(), ax=axes[i]);
        axes[i].set_title(use_case)
    plt.tight_layout()


# In[ ]:


compare_kde(buildings, 'age', 5)


# # floor count

# In[ ]:


compare_kde(buildings, 'floor_count', 5)


# # square feet

# In[ ]:


compare_kde(buildings, 'square_feet', 5)


# # WEATHER (We'll draw Train set in Blue, and Test Set in Green)

# In[ ]:


weather_train.head()


# In[ ]:


weather_sites = weather_train.site_id.unique()


# In[ ]:


calculate_nans = ['site_id', 'air_temperature', 'cloud_coverage', 'dew_temperature',
                  'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
                  'wind_speed']
weather_concat = pd.concat([weather_train, weather_test], axis=0)
calculate_nans_df = weather_concat[calculate_nans].isna()
calculate_nans_df['site_id'] = weather_concat.site_id
calculate_nans_df = calculate_nans_df.groupby('site_id').sum() / (24 * 365 * 3) * 100


# In[ ]:


# add time attribute
# Only Month, HourofDay have something to do with weather
weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])
weather_test['timestamp'] = pd.to_datetime(weather_test['timestamp'])

weather_train['hour'] = weather_train.timestamp.dt.hour
weather_test['hour'] = weather_test.timestamp.dt.hour

weather_train['month'] = weather_train.timestamp.dt.month
weather_test['month'] = weather_test.timestamp.dt.month

weather_train = weather_train.set_index('timestamp')
weather_test = weather_test.set_index('timestamp')


# In[ ]:


def plot_weather_attribute(col, rolling_window=60):
    fig, axes = plt.subplots(nrows=6, ncols=3, sharex=True, sharey=True, figsize=(18, 12))
    exclude_site = [c for c in weather_train.columns if c != 'site_id']
    rolling_weather_train = weather_train[exclude_site].rolling(rolling_window).mean()
    rolling_weather_train['site_id'] = weather_train.site_id

    rolling_weather_test = weather_test[exclude_site].rolling(rolling_window).mean()
    rolling_weather_test['site_id'] = weather_test.site_id

    for i in range(6):
        for j in range(3):
            site_idx = i * 3 + j
            if (site_idx) > 15:
                break
            rolling_weather_train.loc[rolling_weather_train.site_id==(site_idx), col].plot(ax=axes[i, j])
            rolling_weather_test.loc[rolling_weather_test.site_id==(site_idx), col].plot(ax=axes[i, j], color='green')

            axes[i, j].set_title(f"Site {site_idx} NULL: {int(calculate_nans_df.loc[site_idx, col])}%")
    plt.tight_layout()


# # air_temperature:

# In[ ]:


plot_weather_attribute('air_temperature', rolling_window=60)


# # Site 4 is the most stable

# # cloud_coverage:

# In[ ]:


plot_weather_attribute('cloud_coverage', rolling_window=7)


# # Only Site 12 has the full data

# # dew_temperature:

# In[ ]:


plot_weather_attribute('dew_temperature', rolling_window=60)


# # Look pretty similar to air_temperature

# # precip_depth_1_hr:

# In[ ]:


plot_weather_attribute('precip_depth_1_hr', rolling_window=7)


# # sea_level_pressure:

# In[ ]:


plot_weather_attribute('sea_level_pressure', rolling_window=7)


# # wind_direction:

# In[ ]:


plot_weather_attribute('wind_direction', rolling_window=60)


# # Site 4 has same pattern with temparature (monsoons?)
# # Should be 0 to 360 degree

# # wind_speed:

# In[ ]:


plot_weather_attribute('wind_speed', rolling_window=60)


# # Then we do EDA by HOUR OF THE DAY......

# In[ ]:


train_site_hour_mean = weather_train.groupby(['site_id', 'hour']).mean()
test_site_hour_mean = weather_test.groupby(['site_id', 'hour']).mean()


# In[ ]:


def plot_weather_by_hour(col):
    fig, axes = plt.subplots(nrows=6, ncols=3, sharex=True, sharey=True, figsize=(18, 12))
    for i in range(6):
        for j in range(3):
            site_idx = (i * 3 + j)
            if site_idx > 15:
                break
            train_site_data = train_site_hour_mean.loc[(site_idx, slice(None)), col]
            train_site_data.index = train_site_data.index.droplevel(0)

            test_site_data = test_site_hour_mean.loc[(site_idx, slice(None)), col]
            test_site_data.index = test_site_data.index.droplevel(0)

            train_site_data.plot(ax=axes[i, j])
            test_site_data.plot(ax=axes[i, j], color='green')

            axes[i, j].set_title(f"Site {site_idx} NULL: {int(calculate_nans_df.loc[site_idx, col])}%")
    plt.tight_layout()


# In[ ]:


weather_train.columns


# # air_temperature:

# In[ ]:


plot_weather_by_hour('air_temperature')


# # cloud_coverage:

# In[ ]:


plot_weather_by_hour('cloud_coverage')


# # There are some weird spikes...
# # Let's check if it's related to the NULLs

# In[ ]:


check_cloud_na = weather_train[['site_id', 'hour', 'cloud_coverage']].copy()
check_cloud_na.loc[:, 'cloud_coverage'] = check_cloud_na['cloud_coverage'].isna()
check_cloud_na.groupby(['site_id', 'hour']).sum().unstack('hour')


# # --> Yes, the spikes is just because of the Nulls

# # dew_temperature:

# In[ ]:


plot_weather_by_hour('dew_temperature')


# # precip_depth_1_hr:

# In[ ]:


plot_weather_by_hour('precip_depth_1_hr')


# # Though Site 7, 11 look weird
# # It's because there are around 90% null in them
# # ---> It's biased

# # sea_level_pressure:

# In[ ]:


plot_weather_by_hour('sea_level_pressure')


# # wind_direction:

# In[ ]:


plot_weather_by_hour('wind_direction')


# # Site 2, 4, 15 might be near the ocean
# # as the wind direction is opposite in day and night

# # wind_speed:

# In[ ]:


plot_weather_by_hour('wind_speed')


# # Then we do EDA by Month....

# In[ ]:


train_site_month_mean = weather_train.groupby(['site_id', 'month']).mean()
test_site_month_mean = weather_test.groupby(['site_id', 'month']).mean()


# In[ ]:


def plot_weather_by_month(col):
    fig, axes = plt.subplots(nrows=6, ncols=3, sharex=True, sharey=True, figsize=(18, 12))
    for i in range(6):
        for j in range(3):
            site_idx = (i * 3 + j)
            if site_idx > 15:
                break
            train_site_data = train_site_month_mean.loc[(site_idx, slice(None)), col]
            train_site_data.index = train_site_data.index.droplevel(0)

            test_site_data = test_site_month_mean.loc[(site_idx, slice(None)), col]
            test_site_data.index = test_site_data.index.droplevel(0)

            train_site_data.plot(ax=axes[i, j])
            test_site_data.plot(ax=axes[i, j], color='green')

            axes[i, j].set_title(f"Site {site_idx} NULL: {int(calculate_nans_df.loc[site_idx, col])}%")
    plt.tight_layout()


# # air_temperature:

# In[ ]:


plot_weather_by_month('air_temperature')


# # cloud_coverage:

# In[ ]:


plot_weather_by_month('cloud_coverage')


# # dew_temperature:

# In[ ]:


plot_weather_by_month('dew_temperature')


# # precip_depth_1_hr:

# In[ ]:


plot_weather_by_month('precip_depth_1_hr')


# # sea_level_pressure:

# In[ ]:


plot_weather_by_month('sea_level_pressure')


# # wind_direction:

# In[ ]:


plot_weather_by_month('wind_direction')


# # wind_speed:

# In[ ]:


plot_weather_by_month('wind_speed')


# # Then Let's see how attributes interact with each other

# In[ ]:


def plot_weather_correlation(df):
    fig, axes = plt.subplots(nrows=6, ncols=3, sharex=True, sharey=True, figsize=(18, 24))
    for i in range(6):
        for j in range(3):
            site_idx = (i * 3 + j)
            if site_idx > 15:
                for tick in axes[i, j].get_xticklabels():
                    tick.set_rotation(90)
                continue
                
            sns.heatmap(df.loc[df.site_id==site_idx, 
                                          ['air_temperature', 'cloud_coverage', 'dew_temperature',
                                          'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
                                          'wind_speed']
                                         ]\
                                         .corr(), ax=axes[i, j], annot=True, fmt ='.1f')
            axes[i, j].set_title(f"Site {site_idx}")
    plt.tight_layout()


# # In Train set

# In[ ]:


sns.heatmap(weather_train[['air_temperature', 'cloud_coverage', 'dew_temperature',
               'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
               'wind_speed']].corr(), annot=True, fmt ='.2f');


# # In Test Set

# In[ ]:


sns.heatmap(weather_test[['air_temperature', 'cloud_coverage', 'dew_temperature',
               'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
               'wind_speed']].corr(), annot=True, fmt ='.2f');


# # In Train Set by Site

# In[ ]:


plot_weather_correlation(weather_train)


# In[ ]:


# In Test Set by Site


# In[ ]:


plot_weather_correlation(weather_test)


# # Conclusions on Buildings:
# # 1. Not sure what's the cause of the NULLs
# # 2. The Top5 most common building type might be important when performing further analysis
# 
# # Conclusions on weather:
# # 1. All sites are from Northern Hemisphere
# # 2. Mostly Train and Test data share the same pattern
# # 3. Beware of the NULLs
# # 4. Sites can be identified if correctly identify the temperature, wind direction hourly/monthly
# 
# # **Conclusion:**
# # Thank you for reading. Hope this help.
# # Please let me know if there's any mistake and have a good day!

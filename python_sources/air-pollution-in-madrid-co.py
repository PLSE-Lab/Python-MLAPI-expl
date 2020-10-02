import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with pd.HDFStore('../input/madrid.h5') as data:
    df_stations = data['master']

partials = list()

with pd.HDFStore('../input/madrid.h5') as data:
    stations = [k[1:] for k in data.keys() if k != '/master']
    for station in stations:
        df = data[station]
        df['station'] = station
        partials.append(df)

df = pd.concat(partials).sort_index()
df = df.fillna(0)

# get list of unique timestamps
one_station = df[df['station'] == '28079017']
x = one_station.index.to_series()

# find mean value for each hour among stations
# if there is better and faster solution, please let me know
mean_values = np.array([])
for i in range(len(x)):
    val = df['CO'][df.index == x[i]].mean()
    mean_values = np.append(mean_values, val)
    if i % 100 == 0:
        print('Iteration number: ' + str(i))

# as previous operation took a lot of time I decided not to risk and exported the new data
df_mean = pd.DataFrame(data=mean_values, index=x, columns=['CO'])
df_mean.to_csv('CO_mean_values.csv')

# group data by different time periods
df_mean = df_mean.fillna(0)
df_gr_D = df_mean.groupby(pd.Grouper(freq='D')).transform(np.mean).resample('D').mean()
df_gr_D = df_gr_D.fillna(0)
df_gr_M = df_gr_D.groupby(pd.Grouper(freq='M')).transform(np.mean).resample('M').mean()
df_gr_M = df_gr_M['CO'].fillna(0)

# prepare final dataset
df_detailed = df_gr_D.copy()
df_detailed['year'] = df_detailed.index.year
df_detailed['month'] = df_detailed.index.month
df_detailed['day'] = df_detailed.index.day
df_detailed = df_detailed.fillna(0)

# finding max and min values per each month
max_vals_df = pd.DataFrame(columns=['year', 'month', 'volume'])
for i in range(2001, 2019):
    max_val = max(df_detailed['CO'][df_detailed['year'] == i])
    if max_val > 0.0:
        month = df_detailed[(df_detailed['CO'] == max_val) & (df_detailed['year'] == i)]['month'].values[0]
        to_add = pd.DataFrame([[i, month, max_val]], columns=['year', 'month', 'volume'])
        max_vals_df = max_vals_df.append(to_add)

min_vals_df = pd.DataFrame(columns=['year', 'month', 'volume'])
for i in range(2001, 2019):
    min_val = min(df_detailed['CO'][df_detailed['year'] == i])
    if min_val > 0.0:
        month = df_detailed[(df_detailed['CO'] == min_val) & (df_detailed['year'] == i)]['month'].values[0]
        to_add = pd.DataFrame([[i, month, min_val]], columns=['year', 'month', 'volume'])
        min_vals_df = min_vals_df.append(to_add)


# really useful function
def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x == 0 else x for x in values]


df_detailed['CO'] = zero_to_nan(df_detailed['CO'])

# plot 1 - difference between max and min values through time
plt.plot(min_vals_df['year'], min_vals_df['volume'], 'b')
plt.plot(max_vals_df['year'], max_vals_df['volume'], 'r')
plt.yscale('log')
plt.xlabel('Year')
plt.ylabel('Volume')
plt.show()

# plot 2 - changes in air pollution through years
years = []
for i in range(2001, 2019, 2):
    years.append(i)

plt.plot(df_detailed.index, df_detailed['CO'], c='r')
plt.xlabel('Year')
plt.ylabel('mg/m3')
plt.title('CO pollution in Madrid 2001-2018')
plt.show()

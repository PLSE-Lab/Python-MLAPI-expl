import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from sklearn.neighbors import KNeighborsRegressor

X = pd.read_csv('../input/CLIWOC15.csv')

nationalities = ['ES', 'FR', 'NL', 'UK']
wind_directions = {}

# Read Wind Direction data
for nat in nationalities:
    wind_directions[nat] = pd.read_csv('../input/Lookup_%s_WindDirection.csv' % nat)
    wind_directions[nat] = pd.Series(data=wind_directions[nat].ProbWindDD.values, index=wind_directions[nat].WindDirection)
    

# Replace special symbols with angles
nat = 'Nationality'
X.loc[X[nat] == 'Spanish', 'WindDirection'] = X.loc[X[nat] == 'Spanish'].replace({'WindDirection': wind_directions['ES']})
X.loc[X[nat] == 'British', 'WindDirection'] = X.loc[X[nat] == 'British'].replace({'WindDirection': wind_directions['UK']})
X.loc[X[nat] == 'French', 'WindDirection'] = X.loc[X[nat] == 'French'].replace({'WindDirection': wind_directions['FR']})
X.loc[X[nat] == 'Dutch', 'WindDirection'] = X.loc[X[nat] == 'Dutch'].replace({'WindDirection': wind_directions['NL']})
X = X.loc[X[nat].isin(['Spanish', 'British', 'French', 'Dutch']), :]

# Drop NaN, Variable Directions, No Wind and other nonvalid values
X = X[(X['WindDirection'] != 500) & (X['WindDirection'] != 999) & (X['WindDirection'] != 0) & (X['WindDirection'].isin(range(0, 1000)))]

needed_data = X[['Year', 'Month', 'Lon3', 'Lat3', 'WindDirection']].dropna()

winter_data = needed_data[needed_data['Month'].isin([11 ,12, 1, 2, 3]) & needed_data['Year'].isin(range(1825, 1851))]
winter_temp = X[X['Month'].isin([12, 1, 2]) & X['Year'].isin(range(1825, 1851))][['Lon3', 'Lat3', 'ProbTair']].dropna()
summer_data = needed_data[needed_data['Month'].isin([5, 6, 7, 8, 9]) & needed_data['Year'].isin(range(1825, 1851))]
summer_temp = X[X['Month'].isin([6, 7, 8]) & X['Year'].isin(range(1825, 1851))][['Lon3', 'Lat3', 'ProbTair']].dropna()

# Grid boundaries
min_lat = -58
max_lat = 58
min_lon = -100
max_lon = 35

k = 1
for wind, temp in [(winter_data, winter_temp), (summer_data, summer_temp)]:
    fig = plt.figure(figsize=(10, 10), dpi=100)
    
    map = Basemap(width=13000000, height=14000000, resolution='l', projection='stere', lat_ts=5, lat_0=0, lon_0=-33)
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='#bd7728')
    map.drawcoastlines()

    # Temperatures
    g = KNeighborsRegressor(n_neighbors=100)
    g.fit(temp[['Lon3', 'Lat3']], temp['ProbTair'])
    xx, yy = np.meshgrid(np.linspace(min_lon, max_lon, 250), np.linspace(min_lat, max_lat, 250))
    predictions = np.zeros_like(xx)
    
    for i in range(xx.shape[1]):
        predictions[:, i] += g.predict(np.vstack((xx[:, i], yy[:, i])).T)
        
    lon, lat = map(xx, yy)
    pc = plt.pcolor(lon, lat, predictions, cmap=sns.diverging_palette(220, 20, as_cmap=True), snap=True, vmin=0, vmax=30)
    cbar = plt.colorbar(pc, drawedges=True, spacing='proportional', ticks=np.arange(0, 31, 5), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['%d C$^{\circ}$' % t for t in np.arange(0, 31, 5)])

    # Wind Directions
    g = KNeighborsRegressor(n_neighbors=100)
    g.fit(wind[['Lon3', 'Lat3']], wind['WindDirection'].astype(int))
    xx, yy = np.meshgrid(np.linspace(min_lon, max_lon, 20), np.linspace(min_lat, max_lat, 20))
    predictions = np.zeros_like(xx)
    
    for i in range(xx.shape[1]):
        predictions[:, i] += g.predict(np.vstack((xx[:, i], yy[:, i])).T)
        
    # Transform angles into 2-d vectors
    WD_X = np.cos(np.deg2rad(270 - predictions))
    WD_Y = np.sin(np.deg2rad(270 - predictions))

    lat, lon = map(xx, yy)
    map.quiver(lat, lon, WD_X, WD_Y, scale=40, width=0.002)
    map.drawparallels(range(-90, 100, 30), linewidth=1, dashes=[4, 2], labels=[1,0,0,1], color='#707070')
    map.drawmeridians(np.arange(-180.,181.,30.),  linewidth=1, dashes=[4, 2], labels=[1,0,0,1], color='#707070')
    
    if k == 2:
        plt.title('Atlantic Ocean Temperature Distribution \nWith Wind Directions (May-September 1825-1851)', fontdict={'fontsize': 15, 'family' : 'serif'})
    else:
        plt.title('Atlantic Ocean Temperature Distribution \nWith Wind Directions (November-March 1825-1851)', fontdict={'fontsize': 15, 'family' : 'serif'})
        
    plt.savefig('picture%d.png' % k)
    k += 1


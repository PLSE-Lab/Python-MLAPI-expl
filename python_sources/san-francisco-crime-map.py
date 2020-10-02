# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import zipfile

data_path = '../input/'
z = zipfile.ZipFile(data_path+'train.csv.zip')
data = pd.read_csv(z.open('train.csv'))
data = data[(data.X > -123) & (data.X < -122) &
            (data.Y > -37) & (data.Y < 38)
            & data.Category.notnull()]
data = data[['Dates', 'Category', 'DayOfWeek', 'X', 'Y']]

def basic_sf_map(ax=None, lllat=37.699, urlat=37.8299, lllon=-122.5247, urlon=-122.3366):
   
    m = Basemap(ax=ax, projection='stere',
                lon_0=(urlon + lllon) / 2,
                lat_0=(urlat + lllat) / 2,
                llcrnrlat=lllat, urcrnrlat=urlat,
                llcrnrlon=lllon, urcrnrlon=urlon,
                resolution='f')   
    m.drawcoastlines()
    m.drawcountries()    
    return m
    
cate_counts = data['Category'].value_counts()
categories = list(cate_counts.index)

colors = ['r', 'w', 'y', 'g', 'b', 'c', 'm']
days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

for cat in categories[10:20]:    
    cate_data = data[data['Category'] == cat]
    cat = cat.replace('/', ' ')
    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(35, 5))
    for d, c, ax in zip(days, colors, axes.flat):
        m = basic_sf_map(ax, lllat=37.699, urlat=37.8299, lllon=-122.5247, urlon=-122.3366)
        shapefile_path = '../input/san-francisco_california_osm_roads'
        m.readshapefile(shapefile_path, 'roads')
        day_data = cate_data[cate_data['DayOfWeek'] == d]
        x, y = m(day_data.X.values, day_data.Y.values)     
        m.plot(x, y, c+'.', alpha=0.5)    
        ax.set_title('%s: %s' % (cat, d))
    plt.savefig('%s.png' %cat)
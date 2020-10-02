import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from mpl_toolkits.basemap import Basemap


crime = pd.read_csv("../input/SouthAfricaCrimeStats_v2.csv")

year = ['2005-2006', '2006-2007', '2007-2008', '2008-2009',
       '2009-2010', '2010-2011', '2011-2012', '2012-2013', '2013-2014',
       '2014-2015', '2015-2016']

mapping = dict(zip(year, [x.split("-")[0] for x in year]))

crime.rename(columns=mapping,inplace=True)
crime['Station'] = crime['Station'].str.upper()

num_colors = 15
shapefile = '../input/Police_bounds'

def animation_map(category):
            
    group = crime[crime['Category'] == category ][['Station','Category', '2005', '2006', '2007', '2008', '2009', 
        '2010','2011', '2012', '2013','2014', '2015']].groupby(['Station','Category'])['2005', '2006', '2007',
        '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'].sum()


    fig = plt.figure(figsize=(22, 12))
    ax = fig.add_subplot(111)


    m = Basemap(llcrnrlon=11.5, llcrnrlat= -36., urcrnrlon=37., urcrnrlat=-20,
                lat_0=-25, projection='merc')

    m.drawcoastlines()
    m.drawcountries()
    m.readshapefile(shapefile,'units', color='#444444', linewidth=.2)

    draw = plt.plot([], [])[0]

    for shape in m.units:
        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        ax.add_collection(pc)

    def animate(x):
        fig.suptitle('Count of {} in South Afrika in {} per district.'.format(category, x), fontsize=25, y=.93, x=.512)
        values = group[x].values
        cm = plt.get_cmap('Oranges')
        scheme = [cm(i / num_colors) for i in range(num_colors)]
        bins = np.linspace(values.min(), values.max(), num_colors)
        group['bin'] = np.digitize(values, bins) - 1           

        # Colored each district
        for info, shape in zip(m.units_info, m.units):
            name = info['COMPNT_NM']
            if name not in group.index.levels[0]:
                color = '#dddddd'
            else:
                color = scheme[group[group.index.levels[0] == name]['bin'].values]
            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches)
            pc.set_facecolor(color)
            ax.add_collection(pc)

        # Draw color legend.
        ax_legend = fig.add_axes([0.217, 0.125, 0.591, 0.03])
        cmap = mpl.colors.ListedColormap(scheme)
        cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, boundaries=bins, orientation='horizontal')
        cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])

        return draw,

    output = animation.FuncAnimation(plt.gcf(), animate, group.columns, interval=150, blit=True, repeat=True)
    output.save('crime_{}.gif'.format(category), writer='imagemagick')



animation_map('Truck hijacking')
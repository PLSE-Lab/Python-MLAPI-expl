#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import zipfile
import csv
import sys

def find_nearest_ind(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

# http://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib/17479417#17479417
def add_subplot_axes(ax,rect):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],frameon=False)
    return subax

"""
    Generate the data structure
    {state: {puma: []}}
"""
def fillData():
    for i in range(row_count):
        row=csvf.__next__()
        state=row[stCol]
        puma=row[pumCol]
        if (row[header.index(columnName)] == ''):
            continue

        col=int(row[header.index(columnName)])
        if (state not in data):
            data.update({state: {puma: np.array([col])}})
        elif (puma not in data[state]):
            data[state].update({puma: np.array([col])})
        else:
            data[state][puma] = np.append(data[state][puma],col)

# state codes from http://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict13.txt
# note that areas outside of the conus have been commented out
state_codes = {'01': 'Alabama',
               '02': 'Alaska',
               '15': 'Hawaii',
               '04': 'Arizona',
               '05': 'Arkansas',
               '06': 'California',
               '08': 'Colorado',
               '09': 'Connecticut',
               '10': 'Delaware',
            #   '11': 'District of Columbia',
               '12': 'Florida',
               '13': 'Georgia',

               '16': 'Idaho',
               '17': 'Illinois',
               '18': 'Indiana',
               '19': 'Iowa',
               '20': 'Kansas',
               '21': 'Kentucky',
               '22': 'Louisiana',
               '23': 'Maine',
               '24': 'Maryland',
               '25': 'Massachusetts',
               '26': 'Michigan',
               '27': 'Minnesota',
               '28': 'Mississippi',
               '29': 'Missouri',
               '30': 'Montana',
               '31': 'Nebraska',
               '32': 'Nevada',
               '33': 'New Hampshire',
               '34': 'New Jersey',
               '35': 'New Mexico',
               '36': 'New York',
               '37': 'North Carolina',
               '38': 'North Dakota',
               '39': 'Ohio',
               '40': 'Oklahoma',
               '41': 'Oregon',
               '42': 'Pennsylvania',
               '44': 'Rhode Island',
               '45': 'South Carolina',
               '46': 'South Dakota',
               '47': 'Tennessee',
               '48': 'Texas',
               '49': 'Utah',
               '50': 'Vermont',
               '51': 'Virginia',
               '53': 'Washington',
               '54': 'West Virginia',
               '55': 'Wisconsin',
               '56': 'Wyoming',
            #   '72': 'Puerto Rico'
               }
stCol = 5
pumCol = 3
columnName = "SSMC"

csvf = csv.reader(open('../input/pums/ss13husa.csv', 'rU'))
header = csvf.__next__()

row_count = 756065

data = {}

fillData()


csvf = csv.reader(open('../input/pums/ss13husb.csv', 'rU'))
header = csvf.__next__()

row_count = 720248

fillData()

"""
    Use three subplots (mainland,Hawaii,Alaska)
"""
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
rect = [0.08,0.05,0.35,0.35]
axAlaska = add_subplot_axes(ax,rect)
rect = [0.3,0.02,0.2,0.2]
axHawaii = add_subplot_axes(ax,rect)

fig.suptitle('Census 2013: Same-sex married couple households', fontsize=20)


# create a map object with the Albers Equal Areas projection.
# This projection tends to look nice for the contiguous us.
mNormal = Basemap(width=5000000,height=3500000,
            resolution='l',projection='aea',\
            ax=ax, \
            lon_0=-96,lat_0=38)

mAlaska = Basemap(width=5000000,height=3500000,
            resolution='l',projection='aea',\
            ax=axAlaska, \
            lon_0=-155,lat_0=65)

mHawaii = Basemap(width=1000000,height=700000,
            resolution='l',projection='aea',\
            ax=axHawaii, \
            lon_0=-157,lat_0=20)

# state_codes = {'02': 'Alaska'}

# define a colorramp

num_colors = 21
cm = plt.get_cmap('RdYlGn')
blues = [cm(1.*i/num_colors) for i in range(num_colors)]

# read each states shapefile
for key in state_codes.keys():
    print(state_codes[key])
    if (state_codes[key] == "Alaska"):
        mAlaska.readshapefile('../input/shapefiles/pums/tl_2013_{0}_puma10'.format(key),name='state', drawbounds=True)
        m = mAlaska
    elif (state_codes[key] == "Hawaii"):
        mHawaii.readshapefile('../input/shapefiles/pums/tl_2013_{0}_puma10'.format(key),name='state', drawbounds=True)
        m = mHawaii
    else:
        mNormal.readshapefile('../input/shapefiles/pums/tl_2013_{0}_puma10'.format(key),name='state', default_encoding='latin-1', drawbounds=True)
        m = mNormal

    # loop through each PUMA and assign a random color from our colorramp
    for info, shape in zip(m.state_info, m.state):
        # print(info['PUMACE10'])
        dataForStPuma = data[key][info['PUMACE10']]
        
        # get the number of 1 inside dataForStPuma
        zeros = (dataForStPuma == 0)
        zerosPerc = 20*(1-(sum(zeros)/(1.0*len(dataForStPuma)))) # using range 0-5%


        colorInd = min(int(round(zerosPerc*num_colors)),num_colors-1)

        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches, edgecolor='k', linewidths=1., zorder=2)
        pc.set_color(blues[colorInd])
        if (state_codes[key] == "Alaska"):
            axAlaska.add_collection(pc)
        elif (state_codes[key] == "Hawaii"):
            axHawaii.add_collection(pc)
        else:
            ax.add_collection(pc)


# add colorbar legend
cmap = mpl.colors.ListedColormap(blues)
# define the bins
bounds = np.linspace(0,10,11) 


# create a second axes for the colorbar
ax2 = fig.add_axes([0.82, 0.1, 0.03, 0.8])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, ticks=bounds, boundaries=bounds, format='%1i')
legend_str = [str(float(i)/2)+"%" for i in bounds] # divide by 2 for 0-5%
legend_str[-1] = ">= "+legend_str[-1] 
cb.ax.set_yticklabels(legend_str)# vertically oriented colorbar



plt.savefig('scripts-map-{0}.png'.format(columnName))

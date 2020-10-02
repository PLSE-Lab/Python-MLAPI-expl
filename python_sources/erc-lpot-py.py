# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import pandas
import datetime
import numpy as np
from sqlalchemy import create_engine, MetaData
import pandas.io.sql as psql
import sqlite3
plt.style.use('ggplot')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
PSAs = ['P_CentralTX','P_CoastalPlns','P_HighPlns','P_HillCountry','P_LowerCoast','P_NETX','P_NorthTX',
        'P_RioGrandPlns','P_RollingPlns','P_SETX','P_SouthernPlns','P_TransPecos','P_UpperCoast','P_WPineywoods']

PSAinDB = {'P_CentralTX':'CENTRLTX',
            'P_CoastalPlns':'COASTLPL',
            'P_HighPlns':'HIGHPLAN',
            'P_HillCountry':'HILLCNTY',
            'P_LowerCoast':'LOWCOAST',
            'P_NETX':'NETX',
            'P_NorthTX':'NTXS',
            'P_RioGrandPlns':'RIOGRAND',
            'P_RollingPlns':'ROLLPLN',
            'P_SETX':'SETX',
            'P_SouthernPlns':'SOUTHPLN',
            'P_TransPecos':'TRANSPEC',
            'P_UpperCoast':'UPRCOAST',
            'P_WPineywoods':'WESTPINE'
            }

PSAwithLongname ={'CENTRLTX':'Central TX',
                'NTXS':'North TX',
                'TRANSPEC':"Trans Pecos",
                'ROLLPLN':"Rolling Plains",
                'SETX':"SE TX",
                'NETX':"NE TX",
                'COASTLPL':"Coastal Plains",
                'HIGHPLAN':"High Plains",
                'HILLCNTY':"Hill Country",
                'LOWCOAST':"Lower Coast",
                'RIOGRAND':"Rio Grande Pla",
                'SOUTHPLN':"Southern Plain",
                'UPRCOAST':"Upper Coast",
                'WESTPINE':"W Pinewoods"}
                
producetime = datetime.datetime.now()
date_today = producetime.strftime("%Y%m%d")

PSA="TRANSPEC"
PSA_longname="Trans Pecos"

table_erc = 'PSA_ERC_AVG'
#Table for the historical analyses (including Max, Avg and last year ERC) for each PSA
#table_hist =PSA + '_ERC_HIST2015'
#The historial should be 2016 for 2017
table_hist =PSA + '_ERC_HIST2016' #Updated 12/31/2016
#Table for daily historical erc for all the years for each PSA
table_hist_full = PSA + '_ERC_ALLYEAR'

#retrive the plotting data from database(PostgreSQL)
try:

    #sqlite through sqlite3 on Windows
    engine = create_engine('sqlite:///../input/database.sqlite')
    #Try Retrieving the data form the data
    df_ERC = pandas.read_sql_table(table_erc,engine)
    df_HIST = pandas.read_sql_table(table_hist,engine)
    df_HIST_FULL = pandas.read_sql_table(table_hist_full,engine)
    #print(df_ERC)
    engine.dispose()
except:
    print('there is a problem in connecting to database')
    exit(0)
    
date_todate = df_ERC.loc[:,'index'].values
erc = df_ERC.loc[:,PSA].values
date_2016 = df_HIST.loc[:,'index'].values
avg_erc = df_HIST.loc[:,'ercAvg'].values
max_erc = df_HIST.loc[:,'ercMax'].values
lastyear_erc = df_HIST.loc[:,'lastYear'].values
#df_HIST_FULL.index = pandas.to_datetime(df_HIST_FULL.loc[:,'DATE'])
ts_ERC = df_HIST_FULL.loc[:,'ERC'].values

firstYear = df_HIST_FULL.loc[0,'DATE'].year
latestYear = '2016'
latestYear = '2017' #For year 2017
#print firstYear,latestYear
#ts= pandas.Series(df_HIST_FULL.loc[:,'ERC'].values, index=df_HIST_FULL.loc[:,'DATE'].values)
# percentile97 = np.percentile(erc_allyear,97)
Percentile97 = pandas.Series(np.nanpercentile(ts_ERC,97), index=date_2016 )
Percentile90 = pandas.Series(np.nanpercentile(ts_ERC,90), index=date_2016 )
dayofObs = len(ts_ERC) + len(erc)
#print(Percentile97)


formatter = mdates.DateFormatter('%b %d')
matplotlib.rc('xtick', labelsize=9)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.xaxis.set_major_formatter(formatter)

#Historical Average ERC
ax.plot_date(date_2016, avg_erc, '-',c='grey',lw=1.5, label='Avg')
#Historical Maximum ERC
ax.plot_date(date_2016, max_erc, '-',c='red',lw=1.5, label='Max')
#Last year ERC
ax.plot_date(date_2016, lastyear_erc, ':',c='blue',lw=1.3, label='2016')
#97 and 90 Percentile for all the Previous years
##        ax.plot_date(date_2016, Percentile97, '-',c='purple',lw=1,label='97%')
##        ax.plot_date(date_2016, Percentile90, '-',c='green',lw=1,label='90%')
ax.plot_date(date_2016, Percentile97, '-',c='purple',lw=1,label='97% ['+ str(int(Percentile97[0])) + ']')
ax.plot_date(date_2016, Percentile90, '-',c='green',lw=1,label='90% ['+ str(int(Percentile90[0])) + ']')
#Uptodate ERC
ax.plot_date(date_todate, erc,'-',c='black',lw=1.5, label='2017')

#add titles and legends,etc
plt.xlabel('1 Day Periods',fontsize='xx-large')
plt.ylabel('Energy Release Component',fontsize='x-large')
#    PSA_longname = PSA_longname[PSA]
subtitlename = "PSA - " + PSA_longname
plt.suptitle(subtitlename,fontsize='x-large')
title_year = str(firstYear) + '-' + str(latestYear)
plt.title(title_year)
leg = plt.legend(loc='lower center',ncol=2,fontsize='small')
bb = leg.legendPatch.get_bbox().inverse_transformed(ax.transAxes)
xOffset = 0.305
yOffset = 0.15
newX0 = bb.x0 + xOffset
newX1 = bb.x1 + xOffset
newY0 = bb.y0 - yOffset
newY1 = bb.y1 - yOffset
bb.set_points([[newX0, newY0], [newX1, newY1]])
leg.set_bbox_to_anchor(bb)

#Text to show the fuel model used and the date generated
fuelmodel = 'Fuel Model: 8G' #Either 8G or 7G
#Need to create a dictionary for the fuel model definition(it has all 8G, use 8G, otherwise will be 7G), it can be based on each PSA
#fuelmodel = 'Fuel Model: ' +  fuelmodel
fuelmodel = 'Fuel Model: G' ##Meeting on Jan 26 2015, no difference will be made between 7G or 8G for the signature
plt.figtext(0.9, 0.09, fuelmodel, horizontalalignment='right')
observationtext = str(dayofObs) + ' Wx Observations'
plt.figtext(0.9, 0.055, observationtext , horizontalalignment='right')
producetime = datetime.datetime.now()
producetext = 'Generated on ' + producetime.strftime("%m/%d/%Y-%H:%M")
plt.figtext(0.9, 0.02, producetext , horizontalalignment='right')

fig.autofmt_xdate()
plt.show()
#plt.savefig("erc.png")
#plt.show()




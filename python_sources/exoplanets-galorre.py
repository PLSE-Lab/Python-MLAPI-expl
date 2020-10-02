# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pylab as plt 
#%pylab inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

### Import and check data
df = pd.read_csv('../input/oec.csv', delimiter=',')
df.head()
#print(df.columns)

### Get the methods
methods=set(df['DiscoveryMethod']) #planet discovery method
#print(methods)

#I'll pick 2 of the most famous discovery methods and compare
idx1 = df['DiscoveryMethod'] == 'transit' #method[4]
df_transit = df[idx1]

idx2 = df['DiscoveryMethod'] == 'RV' #method[2]
df_RV = df[idx2]


set(df_transit['DiscoveryYear'])
set(df_RV['DiscoveryYear'])

df_transit['DiscoveryYear'].value_counts(sort='False')


### Plot
#create a new 2-column dataframe
df2= pd.DataFrame(df_RV['DiscoveryYear'].value_counts(sort=False)) 
df2['transit'] = df_transit['DiscoveryYear'].value_counts(sort=False)
df2.columns = ['RV','transit']

fig, ax = plt.subplots()

ax = df2.sort_index(axis=0).plot.bar()
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_ylim([0,800])

#the interval [1995, 2015] can't be used since it is an array of year; 0=1989 and len(df2)=2016
ax.set_xlim([1,len(df2)-1]) 
ax.set_ylabel('Count')
ax.set_xlabel('Year of discovery')
plt.show()

### There is something happended in 2009 so let's divide the discovery year into 3 epochs
### groupA = pre-2000
idx1_RV = df_RV['DiscoveryYear'] <2000
groupA_RV = df_RV[idx1_RV]

idx1_tr= df_transit['DiscoveryYear'] <2000
groupA_transit = df_transit[idx1_tr]

### groupB = 2000-2008
groupB_RV = df_RV.query('2000 <= DiscoveryYear <= 2008')
groupB_transit = df_transit.query('2000 <= DiscoveryYear <= 2008')

### groupC = post-2008 = since 2009
idx3_RV = df_RV['DiscoveryYear'] >2008
groupC_RV = df_RV[idx3_RV]

idx3_tr= df_transit['DiscoveryYear'] >2008
groupC_transit = df_transit[idx3_tr]

print('Number of transiting planets between 2000 and 2008: {}'.format(len(groupB_transit)))
print('Number of transiting planets since 2009: {}'.format(len(groupC_transit)))


d = {'transit' : pd.Series([len(groupA_transit), len(groupB_transit), len(groupC_transit)], index=['<2000', '2000-8', '>2008']),
     'RV' : pd.Series([len(groupA_RV), len(groupB_RV), len(groupC_RV)], index=['<2000', '2000-8', '>2008'])}
table = pd.DataFrame(d)    
print(table)


table.plot.bar()
plt.title('Number of Discovered Planets');
plt.xlabel('Year of Discovery')
plt.xticks(rotation=0)
plt.ylabel('Count')

### Stellar properties
DF1 = pd.DataFrame({'<2000': groupA_transit['HostStarTempK'], 
                   '2000-8': groupB_transit['HostStarTempK'], 
                   '>2009': groupC_transit['HostStarTempK']})
DF2 = pd.DataFrame({'<2000': groupA_RV['HostStarTempK'], 
                   '2000-8': groupB_RV['HostStarTempK'], 
                   '>2009': groupC_RV['HostStarTempK']})

DF1.columns.tolist()

DF1 = DF1[['>2009', '2000-8', '<2000']]
DF2 = DF2[['>2009', '2000-8', '<2000']]

ax = plt.figure()
DF1.plot.hist(stacked=True, bins=20, alpha=0.7)
plt.title('Transiting Planets')
plt.xlabel('Stellar Teff (K)')
DF2.plot.hist(stacked=True, bins=20, alpha=0.7)
plt.title('RV Planets')
plt.xlabel('Stellar Teff (K)')


### semi-major axis of planets
#planets discovered pre-2000
#groupA_transit

##no transit before 2000
#groupA_RV

dfA_RV_set = pd.DataFrame(groupA_RV['SemiMajorAxisAU'].dropna())
dfA_RV_set['M'] = groupA_RV['PlanetaryMassJpt'].dropna()
dfA_RV_set.columns = ['P','M']

###planets discovered between 2000 and 2008
#groupB_transit
#groupB_RV

dfB_tr_set = pd.DataFrame(groupB_transit['SemiMajorAxisAU'].dropna())
dfB_tr_set['M'] = groupB_transit['PlanetaryMassJpt'].dropna()
dfB_tr_set.columns = ['P','M']

dfB_RV_set = pd.DataFrame(groupB_RV['SemiMajorAxisAU'].dropna())
dfB_RV_set['M'] = groupB_RV['PlanetaryMassJpt'].dropna()
dfB_RV_set.columns = ['P','M']

###planets discovered since 2009
#groupC_transit
#groupC_RV

dfC_tr_set = pd.DataFrame(groupC_transit['SemiMajorAxisAU'].dropna())
dfC_tr_set['M'] = groupC_transit['PlanetaryMassJpt'].dropna()
dfC_tr_set.columns = ['P','M']

dfC_RV_set = pd.DataFrame(groupC_RV['SemiMajorAxisAU'].dropna())
dfC_RV_set['M'] = groupC_RV['PlanetaryMassJpt'].dropna()
dfC_RV_set.columns = ['P','M']

import seaborn as sb

with sb.axes_style('whitegrid'):
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    ##transit-no fill
    #ax.scatter(dfA_tr_set.P.values,dfA_tr_set.M.values, edgecolor='r',facecolor='none') #no data
    ax.scatter(dfB_tr_set.P.values,dfB_tr_set.M.values, c='b', alpha=0.5, marker='s')
    ax.scatter(dfC_tr_set.P.values,dfC_tr_set.M.values, c='g', alpha=0.5, marker='s')

    ##RV-filled
    ax.scatter(dfA_RV_set.P.values,dfA_RV_set.M.values, label='pre-2000', c='r')
    ax.scatter(dfB_RV_set.P.values,dfB_RV_set.M.values, label='2000-2008', c='b')
    ax.scatter(dfC_RV_set.P.values,dfC_RV_set.M.values, label='post-2008', c='g')
    
    ax.set_ylabel('Mass (M$_{Jup}$)')
    ax.set_xlabel('Semi-major axis (AU)')
    ax.set_xscale('log')
    ax.set_ylim(0,30)
    ax.set_xlim(0.01,10)
    plt.legend(loc='upper left')
    plt.show()



###
from matplotlib import pyplot as plt
with sb.axes_style('white'):
    ##transit-red ; Rv-blue
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Mass (M$_{Jup}$)')
    #ax.set_xlabel('Period (day)')
    #ax.set_xscale('log')
    ax.tick_params(labelbottom='off')
    ax.yaxis.labelpad = 15
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    #ax.grid('off')
    
    ax1 = fig.add_subplot(311)
    #ax1.scatter(dfA_tr_set.P.values,dfA_tr_set.M.values, label='Transit', c='b')
    ax1.scatter(dfA_RV_set.P.values,dfA_RV_set.M.values, label='RV', c='r')
    ax1.tick_params(labelbottom='off')
    ax1.set_ylim(0,30)
    ax1.set_xlim(0.01,30)
    ax1.set_xscale('log')
    plt.title('pre-2000')
    
    ax2 = fig.add_subplot(312, sharex=ax1,sharey=ax1)
    ax2.scatter(dfB_tr_set.P.values,dfB_tr_set.M.values, label='Transit', c='b')
    ax2.scatter(dfB_RV_set.P.values,dfB_RV_set.M.values, label='RV', c='r')
    ax2.tick_params(labelbottom='off')
    ax2.set_ylim(0,30)
    ax2.set_xlim(0.01,30)
    plt.legend()
    plt.title('2000-2008')
    
    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.scatter(dfC_RV_set.P.values,dfC_RV_set.M.values, label='RV', c='r')
    ax3.scatter(dfC_tr_set.P.values,dfC_tr_set.M.values, label='Transit', c='b')
    ax3.set_xlabel('Semi-major axis (AU)')
    ax3.set_ylim(0,30)
    ax3.set_xlim(0.01,30)
    plt.title('post-2008')
    
    plt.show()













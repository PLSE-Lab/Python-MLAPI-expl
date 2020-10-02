#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib import gridspec
import matplotlib as mpl
import seaborn as sns
 
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['legend.fontsize'] = 14

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


"""OK, so in the second version of this notebook I have cleaned my act up and added some more structure to my take on the Dutch electricity landscape. While later versions might include gas as well, I think it is interesting to start out with electricity as there are a number of interesting dynamics transforming this landscape at the moment. Many thanks to Luca Basanisi for putting together this dataset, because most of us dealing with large datasets know that getting the data into the right format is one of the most time-consuming steps towards making sense out of the information at hand.  """


# Version 3 of the notebook, new features are: 
# *  a more elaborate feature selection in the dimensional reduction section
# *  data visualization by projecting the data on a map 
# 
# Took some time to figure out, and there are definitely many more possible approaches than this, but it's a good first try. Let me know what you think!

# # Analysis items:
# ## [1. Introduction: Country-wide picture](#intro)<br>
# ## [2. Transition to the smart grid](#smartgrid)<br>
# ## [3. Non-centralized electricity production by households](#selfprod)<br>
# ## [4. Some individual city stats](#cityviolins)<br>
# ## [5. Global analysis - feature selection and dimensional reduction](#tsne)<br>
# ## [6. Geographical projections ](#geomapping)<br>
# ## [7. Conclusions](#conclusions)<br>

# In[ ]:


def load_and_reindex(path,filelist):
    start_time = datetime.now()
    df = None
    for file in filelist:
        year = file[-8:-4]
        manager = file.split('_')[0]
        if df is None:
            df = pd.read_csv(path+file)
            df['year'] = year
            df.index = manager+'_'+year+'_'+df.index.astype(str)
        else:
            temp = pd.read_csv(path+file)
            temp['year'] = year
            temp.index = manager+'_'+year+'_'+temp.index.astype(str)
            df = df.append(temp)
    # adding columns of interest
    df['low_tarif_consumption'] = df['annual_consume'].multiply(df['annual_consume_lowtarif_perc']/100)
    df['num_active_connections'] = df['num_connections'].multiply(df['perc_of_active_connections']/100).astype(int)
    try:
        df['num_smartmeters'] = df['num_connections'].multiply(df['smartmeter_perc']/100).astype(int)
    except ValueError:
        df['num_smartmeters'] = df['num_connections'].multiply(df['smartmeter_perc']/100)
        #print('Number of smartmeters could not be calculated')
    df['net_annual_consumption'] = df['annual_consume'].multiply(df['delivery_perc']/100)
    df['self_production'] = df['annual_consume'] - df['net_annual_consumption']
    df['self_prod_perc'] = df['self_production'].divide(df['annual_consume']/100)
    
    time_elapsed = datetime.now() - start_time
    print('Made main dataframe, time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    return(df)


# In[ ]:


path = '../input/dutch-energy/dutch-energy/Electricity/'
files_all = [f for f in os.listdir(path)]
elec_all = load_and_reindex(path,files_all)


# <a id="intro"></a>
# # 1. Introduction
# As a first approach it suffices to look at the energy parameters at the city level. This is convenient, as we then do not have to start messing around with postal code areas (some of which have been created or shuffled around in the intervening years). We can increase the resolution at a later point if that is of interest, but so for now we can make pivot tables containing the aggregate of an energy parameter of interest, per city per year:

# In[ ]:


# make pivot tables of relevant parameter such that we have total per city per year
annual_consume = pd.pivot_table(elec_all,values='annual_consume',index='city',columns='year',aggfunc=np.sum)
num_connections = pd.pivot_table(elec_all,values='num_connections',index='city',columns='year',aggfunc=np.sum)
num_active_connections = pd.pivot_table(elec_all,values='num_active_connections',index='city',columns='year',aggfunc=np.sum)
perc_active_connections = pd.pivot_table(elec_all,values='perc_of_active_connections',index='city',columns='year',aggfunc=np.mean)
smartmeter_perc = pd.pivot_table(elec_all,values='smartmeter_perc',index='city',columns='year',aggfunc=np.mean)
smartmeter_perc_median = pd.pivot_table(elec_all,values='smartmeter_perc',index='city',columns='year',aggfunc=np.median)
num_smartmeters = pd.pivot_table(elec_all,values='num_smartmeters',index='city',columns='year',aggfunc=np.sum)
self_production = pd.pivot_table(elec_all,values='self_production',index='city',columns='year',aggfunc=np.sum)
self_prod_perc_mean = pd.pivot_table(elec_all,values='self_prod_perc',index='city',columns='year',aggfunc=np.mean)
net_annu_consume = pd.pivot_table(elec_all,values='net_annual_consumption',index='city',columns='year',aggfunc=np.sum)
annu_cons_lowtarif_perc = pd.pivot_table(elec_all,values='annual_consume_lowtarif_perc',index='city',columns='year',aggfunc=np.mean)


# In[ ]:


#print(annual_consume.sort_values('2018',ascending=False).head())
specs = {'markersize':20,'markerfacecolor':'w','linewidth':2}
plt.plot(annual_consume.columns.astype(int)-1,annual_consume.sum(),'-o',**specs)
#plt.yscale('log')
plt.ylabel('Energy consumption (kWh)')
plt.xlabel('Year')
plt.title('Total yearly energy consumption')


# Ok, so the 'year' 2009 - which allegedly covers the consumption in the year before (which is why I changed the x-axis in the plot) - seems to be incomplete. I will therefore drop this year in all following electricity analyses.
# 
# First, let's have a look at the country as a whole:

# In[ ]:


annual_consume.drop('2009',axis=1,inplace=True)
num_connections.drop('2009',axis=1,inplace=True)
smartmeter_perc.drop('2009',axis=1,inplace=True)
num_smartmeters.drop('2009',axis=1,inplace=True)
self_production.drop('2009',axis=1,inplace=True)
net_annu_consume.drop('2009',axis=1,inplace=True)


# In[ ]:


annual_consume.sum()/annual_consume.sum()[0]*100


# In[ ]:


f = plt.figure()

gs = gridspec.GridSpec(5,2)
f.set_figwidth(13)
f.set_figheight(10)
plt.suptitle('Electricity Netherlands',fontsize=20)

ax1 = f.add_subplot(gs[0:2,0])
ax1.plot(annual_consume.columns.astype(int),annual_consume.sum(),'-o',**specs)
ax1.plot(annual_consume.columns.astype(int),net_annu_consume.sum(),'-o',**specs)
lgnd = plt.legend(['gross','net'])

#plt.yscale('log')
plt.ylabel('Energy consumption (kWh)')
plt.xlabel('');plt.xticks([])
plt.title('Total yearly energy consumption')

ax11 = f.add_subplot(gs[2,0])
ax11.fill_between(annual_consume.columns.astype(int),0,annual_consume.sum()/annual_consume.sum()[0]*100-100)#,'-o',**specs)
#plt.yscale('log')
plt.ylabel('as % of 2009')
plt.ylim(-4,3)
plt.xlabel('');plt.xticks([])
#plt.title('Total yearly energy consumption')

ax2 = f.add_subplot(gs[0:2,1])
ax2.plot(num_connections.columns.astype(int),num_connections.sum(),'-og',**specs)
#plt.yscale('log')
plt.ylabel('Count')
plt.xlabel('');plt.xticks([])
plt.title('Total number of connections')

ax21 = f.add_subplot(gs[2,1])
ax21.fill_between(annual_consume.columns.astype(int),0,num_connections.sum()/num_connections.sum()[0]*100-100,color='g')#,'-o',**specs)
#plt.yscale('log')
#plt.ylabel('as % of 2008')
plt.xlabel('');plt.xticks([])
#plt.title('Total yearly energy consumption')

ax3 = f.add_subplot(gs[3:,0])
ax3.plot(self_production.columns.astype(int),self_production.sum(),'-oc',**specs)
#plt.yscale('log')
plt.ylabel('Energy  (kWh)')
plt.xlabel('Year')
plt.title('Total yearly self-production')

ax4 = f.add_subplot(gs[3:,1])
ax4.plot(num_smartmeters.columns.astype(int),num_smartmeters.sum(),'-om',**specs)
#plt.yscale('log')
plt.ylabel('Count')
plt.xlabel('Year')
plt.title('Total number of smartmeters')

gs.update(wspace=.51,hspace=.3)




# What we see is that the yearly total energy consumption has remained roughly the same, while there is a linear increase in the number of connections. This implies that on average the annual amount of electricity used by a connection (=~household) is going down. In addition, the amount of electricity produced by households has been increasing linearly since 2012. At the same time we see that the number of smartmeters is increasing exponentially and one could expect that the number of smartmeters will start approaching the number of connections in the not too far future. 

# <a id="smartgrid"></a>
# # 2. Transition to the smart grid
# 
# Let's look into the evolution of smartmeters per city for a bit first.

# In[ ]:


from scipy.optimize import curve_fit
x = num_smartmeters.columns.astype(int)-1
# exponential fit
def exponenial_func(x, a, b, c):
    return 1/(a*np.exp(-b*x)+c)
#p_opt, p_cov = curve_fit(exponenial_func, x, y, p0=(1e-16, 1e-6, ))
xfit = np.linspace(2009,2021,20)
#yfit = exponenial_func(xfit,*p_opt)
# this doesn't seem to work well, since we're working with large numbers here. 
# lin fit in lin-log space does the trick for here (although this is not the most elegant solution):
me,be = np.polyfit(x,np.log(num_smartmeters.sum()),1)
# linear fit
m,b = np.polyfit(x,num_connections.sum(),1)

f,ax = plt.subplots(1,2,figsize=(16,8))

ax[0].plot(num_smartmeters.columns.astype(int)-1,num_connections.sum(),'og',**specs)
ax[0].plot(num_smartmeters.columns.astype(int)-1,num_smartmeters.sum(),'om',**specs)
ax[0].plot(xfit,m*xfit+b,'g')#,'linewidth'=3)
ax[0].plot(xfit,np.exp(me*xfit+be),'m')#,'linewidth'=3)
#plt.yscale('log')
ax[0].set_ylabel('Count')
ax[0].set_xlabel('Year')
ax[0].legend(['# connections','# smartmeters','lin fit','exp fit'],loc=4)
ax[0].set_ylim(bottom=-1e6,top=1e7)
#plt.xlim(0,10)
ax[0].set_title('Smartmeters vs. connections')

specs2 = {'markersize':20,'fillstyle':'none','linewidth':2}
# this is the mean of the mean per city:
ax[1].plot(num_smartmeters.columns.astype(int), smartmeter_perc.mean(),'om',**specs2)
# this is the median of the mean per city:
ax[1].plot(num_smartmeters.columns.astype(int), smartmeter_perc.median(),'og',**specs2)
ax[1].set_title('Smartmeters, percentage of total')
ax[1].set_xlabel('Year')
ax[1].set_ylabel('% of meters being smart')
ax[1].legend(['mean','median'], fontsize=14)


# Fitting the data shows that the number of smartmeters should reach the total number of connections in mid 2018. We probably could also have gotten this from the percent of smartmeters parameter in the first place, but let's look at this in greater detail, because simply taking the countrywide mean here oversimplifies what is actually going on. 
# 
# For instance, one can notice an increasing discrepancy between the mean and median (of the mean) percentage of smartmeters. This is an indication that we are not dealing with a normal distribution of these things across the cities, which is not a surprise in itself.

# In[ ]:


import seaborn as sns

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def gauss2(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

y = '2018'
X_dataG = np.linspace(-1,100,102)
# extract y_data
counts,bins = np.histogram(smartmeter_perc[y],bins=X_dataG,density=True)#[y].hist(bins=40)
y_data = counts
# initial guesses p0
# initialize them differently (one at 20 and the other at 80 %) so the optimization algorithm works better
# this can be circumvented using simulated annealing or sth in the like
p0 = [.03, 20, 1.,.01, 80, 1.]

#optimize and in the end you will have 6 coeff (3 for each gaussian)
coeff, var_matrix = curve_fit(gauss2, X_dataG[1:], y_data, p0=p0)

#plot each gaussian separately 
pg1 = coeff[0:3]
pg2 = coeff[3:]
# using the single gauss function
g1 = gauss(X_dataG, *pg1)
g2 = gauss(X_dataG, *pg2)



f,axs = plt.subplots(3,2,figsize=(16,20))
axs = axs.ravel()
smartmeter_perc['2018'].hist(bins=40,alpha=.3,color='r',ax=axs[0])
#axs[0].set_xlabel('Smartmeters (% of total in a city)')
axs[0].set_ylabel('count')
axs[0].set_title('Distribution of smartmeters per city 2018')
X_data = np.linspace(0,100,40)
elec_all['provider'] = ['liander' if 'liander' in f else 'stedin' if 'stedin' in f else 'enexis' if 'enexis' in f else 'none' for f in elec_all.index]
smart_provider_2018 = pd.pivot_table(elec_all[elec_all.year=='2018'],values='smartmeter_perc',index='city',columns='provider',aggfunc=np.mean)
pleg = []
for provider in smart_provider_2018.columns:
    smart_provider_2018[provider].hist(bins=40,alpha=.3,ax=axs[1])
    pleg.append(provider)
axs[1].legend(pleg)
axs[1].set_title('Smartmeters per city per provider,2018')
leg = []

for i in range(2014,2019):
    y=str(i)
    smartmeter_perc[y].hist(bins=40,alpha=.3,ax=axs[2])
    axs[3].hist(smartmeter_perc[y],bins=X_data,density=True,alpha=.3)
    sns.distplot(smartmeter_perc[y],bins=X_data,hist=False,kde=True,ax=axs[4])
    leg.append(y)
axs[5].hist(smartmeter_perc['2018'],bins=X_data,alpha=.3,color='r',density=True)
axs[5].plot(X_dataG, g1, label='Gaussian1',linewidth=3)
axs[5].plot(X_dataG, g2, label='Gaussian2',linewidth=3)
axs[5].legend()

#ax[2].set_xlabel('Smartmeters (% of total in a city)')
axs[2].set_ylabel('count')
axs[2].set_title('Distribution of smartmeters per city 2014-2018')
axs[2].legend(leg)
axs[3].set_ylabel('Density')
axs[3].set_title('Normalized distribution of smartmeters per city 2014-2018')
axs[4].set_xlabel('Smartmeters (% of total in a city)')
axs[4].legend(leg)
axs[4].set_ylabel('Density')
plt.suptitle('Mean number of smartmeters per city', fontsize=20)


# First of all, we see that the distribution is bimodal (2018 shown here): a population of early adopters and a lagging population. It is interesting to look into this - as in: what type of city can be typically found in one or the other - further below. First let's see at what year, given the decline rate of the lagging mode, we expect all cities to have ~80% smartmeters. If we normalize the area under the curve we can fit the data.
# 
# Another observation would be that of the three large electricity providers, Enexis is leading the transition to the smart grid.

# In[ ]:


gausparams = pd.DataFrame(index=range(2014,2019),columns=['A1','mu1','s1','A2','mu2','s2'])
p0 = [.12, 20, 1.,.01, 80, 1.]
for year in gausparams.index:
    counts,bins = np.histogram(smartmeter_perc[str(year)],bins=X_dataG,density=True)#[y].hist(bins=40)
    y_data = counts
    if year<=2016: #fit with 1 gaussian
        coeff, var_matrix = curve_fit(gauss, X_dataG[1:], y_data, p0=p0[:3])
        gausparams.loc[year,:3] = coeff
    else: #fit with 2
        coeff, var_matrix = curve_fit(gauss2, X_dataG[1:], y_data, p0=p0)
        gausparams.loc[year,:] = coeff

xfit = list(gausparams.index)
y = list(gausparams['A1'])
xpred = np.linspace(2014,2020,20)
m,b = np.polyfit(xfit,y,1)

plt.plot(xfit,y,'og',**specs2,label='Data')
plt.plot(xpred,m*xpred+b,'-g',linewidth=2,label='fit')
plt.axhline(0)
plt.title('Decline of lagging smartmeter population')
plt.xlabel('Year')
plt.ylabel('Gauss fit amplitude')
plt.legend()


# Another convenient way to present this kind of data - I find - is to save a sequence of images like those above as a gif (or movie), for instance using the [imageio](http://https://imageio.github.io) package, like so (code commented out below):
# ![](https://i.imgur.com/MsOXTk7.gif)

# In[ ]:


"""f,axs = plt.subplots(2,2,figsize=(15,15))
axs = axs.ravel()
leg = []
for i in range(2014,2019):
    j=i-2014
    y=str(i)
    smartmeter_perc[y].hist(bins=40,alpha=.3,ax=axs[0])
    
    sns.distplot(smartmeter_perc[y],bins=X_data,hist=False,kde=True,ax=axs[1])

    leg.append(y)
    axs[2].clear()
    axs[2].hist(smartmeter_perc[y],bins=X_data,alpha=.3,color='r',density=True)
    gy = list(gausparams.loc[i,['A1','mu1','s1']])
    axs[3].plot(xfit[j],gy[0],'og',**specs2,label='Data')
    g1 = gauss(X_dataG,*gy)
    axs[2].plot(X_dataG, g1, label='Gaussian1',linewidth=3)
    #axs[2].plot(X_dataG, g2, label='Gaussian2',linewidth=3)

    axs[0].legend(leg)

    #ax[2].set_xlabel('Smartmeters (% of total in a city)')
    axs[0].set_ylabel('count')
    axs[0].set_title('Distribution of smartmeters per city')
    axs[0].set_xlim(-5,105)
    axs[1].set_xlim(-5,105)
    axs[2].set_xlim(-5,105);axs[2].set_ylim(0,.13)
    axs[3].set_xlim(2013.5,2020.5);axs[3].set_ylim(-.01,.13)
    axs[0].legend(leg)
    axs[1].set_ylabel('Density')
    axs[1].set_title('Normalized distribution of smartmeters')
    axs[2].set_xlabel('Smartmeters (% of total in a city)')

    axs[2].set_ylabel('Density')
    axs[3].axhline(0)
    axs[3].set_title('Lagging smartmeter population')
    axs[3].set_xlabel('Year')
    axs[3].set_ylabel('Gauss fit amplitude')
    #plt.suptitle('Mean number of smartmeters per city', fontsize=20)
    f.savefig('smartmeters_'+y+'.png') 
    if y=='2018':
        axs[3].plot(xpred,m*xpred+b,'-g',linewidth=2,label='fit')
        f.savefig('smartmeters2019.png')
        
import imageio
import glob
files = glob.glob('*.png')
files = np.sort(files)
# make a copy of each image to slow down gif by factor 2
from shutil import copyfile
for file in files:
    copyfile(file, file.split('.')[0]+'_1.png')
files = glob.glob('*.png')
files = np.sort(files)
images = []
for file in files:
    # imageio.imread(file) creates a numpy matrix array
    # In this case a 200 x 200 matrix for every file, since the files are 200 x 200 pixels.
    images.append(imageio.imread(file))
    print(file)
imageio.mimsave('smartmeter-laggingpop.gif', images)

# step 3: prep the gif for display in notebook 
from IPython.display import Image
Image("smartmeter-laggingpop.gif")"""


# By fitting the "lagging" population with a single gaussian and plotting the (normalized) amplitude vs. year, we see a linear decrease. A simple linear fit shows that this peak is likely to become zero within 2 years of the 2018 timepoint, so around the start of 2020.  So in 2020, one can expect the average city to have around 80% smartmeters. This nicely follows Pareto's law (80% of the work done in 20% of the time and vice versa). If we then assume that the remaining 20% will take 80% of the time, and that the transition started around 2012, the Netherlands will be a 100% smartgrid in 8years/2x8 = 32+2020 = 2052. But this is just a back-of-the-envelope calculation of course. 
# 
# Does connection to the smart grid also mean a smarter/more economical use of electricity? Let's see what the correlation between percent smartmeter and percent lowtarif consumption tells us, per city, with a different color for each provider. Many things can be observed here. In general, we see the smartgrid transition started with a handful of small cities for each provider. The percentage of lowtarif consumption tends to cluster per provider, indicating that there might be different definitions of what lowtarif means. I wonder what's going on with Enexis there.

# In[ ]:


f, axs = plt.subplots(2,3, figsize=(18, 12))#, facecolor='w', edgecolor='k')
#f.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
for i in range(2013,2019):
    j=i-2013
    y=str(i)
    axs[j].scatter(elec_all[(elec_all.year==y)&(elec_all.index.str.contains('enexis'))].groupby('city').median()['smartmeter_perc']
             ,elec_all[(elec_all.year==y)&(elec_all.index.str.contains('enexis'))].groupby('city').median()['annual_consume_lowtarif_perc']
             ,s=elec_all[(elec_all.year==y)&(elec_all.index.str.contains('enexis'))].groupby('city').sum()['num_connections'].divide(1e2),alpha=.2)
    axs[j].scatter(elec_all[(elec_all.year==y)&(elec_all.index.str.contains('liander'))].groupby('city').median()['smartmeter_perc']
             ,elec_all[(elec_all.year==y)&(elec_all.index.str.contains('liander'))].groupby('city').median()['annual_consume_lowtarif_perc']
             ,s=elec_all[(elec_all.year==y)&(elec_all.index.str.contains('liander'))].groupby('city').sum()['num_connections'].divide(1e2),alpha=.2)
    axs[j].scatter(elec_all[(elec_all.year==y)&(elec_all.index.str.contains('stedin'))].groupby('city').median()['smartmeter_perc']
             ,elec_all[(elec_all.year==y)&(elec_all.index.str.contains('stedin'))].groupby('city').median()['annual_consume_lowtarif_perc']
             ,s=elec_all[(elec_all.year==y)&(elec_all.index.str.contains('stedin'))].groupby('city').sum()['num_connections'].divide(1e2),alpha=.2)
    axs[j].set_title(y,fontsize=18)
    if (i==2013) | (i==2016):
        axs[j].set_ylabel('mean % of lowtarif consumption')
    if i>=2016:
        axs[j].set_xlabel('mean % of smartmeters')
    if i==2018:
        lgnd = plt.legend(['Enexis','Liander','Stedin'],loc='lower right')
        lgnd.legendHandles[0]._sizes = [50]
        lgnd.legendHandles[1]._sizes = [50]
        lgnd.legendHandles[2]._sizes = [50]
plt.suptitle('Evolution of smartmeter-lowtarif per city\n(sphere radius correlates with city size)',fontsize=25)


# Squinting at the data one can see that on average, connection to the smartgrid is positively correlated with the percentage of lowtarif consumption. This effect is most prominent for Liander cities, where for instance Amsterdam (biggest orange circle) moves from the bottom left towards the top right as the years pass. That being said, correlation needs not imply causation, as we all know:
# 
# ![](https://imgs.xkcd.com/comics/cell_phones.png)
# 
# Or here is the animated verison again (code commented out below).
# ![](https://i.imgur.com/teiW0ug.gif)

# In[ ]:


"""#step 1: make a sequence of images

for i in range(2013,2019):
    f = plt.figure(figsize=(7,7))
    y=str(i)
    plt.scatter(elec_all[(elec_all.year==y)&(elec_all.index.str.contains('enexis'))].groupby('city').median()['smartmeter_perc']
             ,elec_all[(elec_all.year==y)&(elec_all.index.str.contains('enexis'))].groupby('city').median()['annual_consume_lowtarif_perc']
             ,s=elec_all[(elec_all.year==y)&(elec_all.index.str.contains('enexis'))].groupby('city').sum()['num_connections'].divide(1e2),alpha=.2)
    plt.scatter(elec_all[(elec_all.year==y)&(elec_all.index.str.contains('liander'))].groupby('city').median()['smartmeter_perc']
             ,elec_all[(elec_all.year==y)&(elec_all.index.str.contains('liander'))].groupby('city').median()['annual_consume_lowtarif_perc']
             ,s=elec_all[(elec_all.year==y)&(elec_all.index.str.contains('liander'))].groupby('city').sum()['num_connections'].divide(1e2),alpha=.2)
    plt.scatter(elec_all[(elec_all.year==y)&(elec_all.index.str.contains('stedin'))].groupby('city').median()['smartmeter_perc']
             ,elec_all[(elec_all.year==y)&(elec_all.index.str.contains('stedin'))].groupby('city').median()['annual_consume_lowtarif_perc']
             ,s=elec_all[(elec_all.year==y)&(elec_all.index.str.contains('stedin'))].groupby('city').sum()['num_connections'].divide(1e2),alpha=.2)
    plt.title(y,fontsize=18)
    plt.ylabel('mean % of lowtarif consumption')
    plt.ylim(-5,107)
    plt.xlabel('mean % of smartmeters')
    plt.xlim(-10,105)
    lgnd = plt.legend(['Enexis','Liander','Stedin'],loc='lower right')
    lgnd.legendHandles[0]._sizes = [50]
    lgnd.legendHandles[1]._sizes = [50]
    lgnd.legendHandles[2]._sizes = [50]
    f.savefig(y+'.png')
    plt.close(f)

# step 2: making a gif with the sequence of images:

import imageio
import glob
files = glob.glob('*.png')
files = np.sort(files)
# make a copy of each image to slow down gif by factor 2
from shutil import copyfile
for file in files:
    copyfile(file, file.split('.')[0]+'_1.png')
files = glob.glob('*.png')
files = np.sort(files)
images = []
for file in files:
    # imageio.imread(file) creates a numpy matrix array
    # In this case a 200 x 200 matrix for every file, since the files are 200 x 200 pixels.
    images.append(imageio.imread(file))
    print(file)
imageio.mimsave('smartmeter-lotarif.gif', images)

# step 3: prep the gif for display in notebook 
from IPython.display import Image
Image("smartmeter-lotarif.gif")"""


# <a id="selfprod"></a>
# # 3. Non-centralized electricity production by households 
# 
# Let's look at electricity production per city for a bit now.

# In[ ]:


f = plt.figure()
gs = gridspec.GridSpec(1,5)

ax1 = f.add_subplot(gs[0,0])
elec_all[elec_all.year=='2018'].groupby('city').sum()['self_production'].divide(1e3).sort_values(ascending=False)[:20].plot.barh(color='darkblue',width=.9,alpha=.7,ax=ax1)
# could have achieved the same with self_production.sort_values('2018',ascending=False).divide(1e3)[:20]
plt.gca().invert_yaxis()
plt.xlabel('Energy produced (MWh)');plt.ylabel('')
plt.title('Self production top 20, 2018')

ax2 = f.add_subplot(gs[0,2])
self_production.divide(num_active_connections).loc[self_production.sort_values('2018',ascending=False).divide(1e3)[:20].index,:].sort_values('2018',ascending=False)['2018'].plot.barh(color='m',width=.9,alpha=.7,ax=ax2)
plt.gca().invert_yaxis()
plt.xlabel('Energy produced per connection (kWh)');plt.ylabel('')
plt.title('Self production per connection, for the top 20 biggest producers, 2018')

ax3 = f.add_subplot(gs[0,4])
self_production.divide(annual_consume/100).loc[self_production.sort_values('2018',ascending=False).divide(1e3)[:20].index,:].sort_values('2018',ascending=False)['2018'].plot.barh(color='g',width=.9,alpha=.7,ax=ax3)
plt.gca().invert_yaxis()
plt.xlabel('Energy produced (% of used)');plt.ylabel('')
plt.title('Self production, % of consumption, top 20 2018')

f.set_figheight(7)
f.set_figwidth(20)


# We see that in absolute terms Almere is the biggest producer of energy. When normalized by the number of active connections (which is - I think -  a proxy for the number of inhabitants) this yields the average electricity produced per household per city. The larger cities in the 'randstad' move down the ladder. If we take the actual top 20 cities with the highest average electricity production per household, we expect this ranking to be dominated by small towns in windy (rural) places:

# In[ ]:


f = plt.figure()
gs = gridspec.GridSpec(1,3)

ax1 = f.add_subplot(gs[0,0])
self_production.divide(num_active_connections).sort_values('2018',ascending=False)['2018'][:20].plot.barh(color='m',width=.9,alpha=.7,ax=ax1)
plt.gca().invert_yaxis()
plt.xlabel('Energy produced per connection (kWh)');plt.ylabel('')
plt.title('Self production per connection, top 20 2018')

ax2 = f.add_subplot(gs[0,2])
self_production.divide(annual_consume/100).sort_values('2018',ascending=False)['2018'][:20].plot.barh(color='g',width=.9,alpha=.7,ax=ax2)
plt.gca().invert_yaxis()
plt.xlabel('Energy produced (% of consumption)');plt.ylabel('')
plt.title('Self production, % of consumption, top 20 2018')

f.set_figheight(7)
f.set_figwidth(13)


# These mostly seem to be rural places in the (north)east. But let's look at the current top 20 and how they ranked over the past years:

# In[ ]:


from collections import defaultdict
from scipy import interpolate


def streamgraph(dataframe, **kwargs):
    """ Wrapper around stackplot to make a streamgraph """
    X = dataframe.columns
    Xs = np.linspace(dataframe.columns[0], dataframe.columns[-1], num=1024)
    Ys = [interpolate.PchipInterpolator(X, y)(Xs) for y in dataframe.values]
    return plt.stackplot(Xs, Ys, labels=dataframe.index, **kwargs)

def add_widths(x, y, width=1):
    """ Adds flat parts to widths """
    new_x = []
    new_y = []
    for i,j in zip(x,y):
        new_x += [i-width, i, i+width]
        new_y += [j, j, j]
    return new_x, new_y

def bumpsplot(dataframe, color_dict=defaultdict(lambda: "k"), 
                         linewidth_dict=defaultdict(lambda: 4),
                         labels=[]):
    r = dataframe.rank(method="first")
    r = (r - r.max() + r.max().max()).fillna(0) # Sets NAs to 0 in rank
    for j in r.index:
        x = np.arange(r.shape[1])
        y = r.loc[j].values
        color = color_dict[j]
        lw = linewidth_dict[j]
        x, y = add_widths(x, y, width=0.1)
        xs = np.linspace(0, x[-1], num=1024)
        plt.plot(xs, interpolate.PchipInterpolator(x, y)(xs), color=color, linewidth=lw, alpha=0.5)
        if j in labels:
            plt.text(x[0] - 0.1, y[0], s=j, horizontalalignment="right", verticalalignment="center", color=color)
            plt.text(x[-1] + 0.1, y[-1], s=j, horizontalalignment="left", verticalalignment="center", color=color)
    plt.xticks(np.arange(r.shape[1]), dataframe.columns)
    
    
userank = self_production.sort_values('2010',ascending=False)
cities = list(userank[:75].index)
top_cities = list(self_production['2018'].sort_values(ascending=False).index[:20])
finallist = top_cities+list(set(cities).difference(set(top_cities)))
winter_colors = defaultdict(lambda: "grey")
lw = defaultdict(lambda: 1)

top_cities = list(self_production['2018'].sort_values(ascending=False).index[:20])
for i,c in enumerate(top_cities):
    winter_colors[c] = sns.color_palette("husl", n_colors=len(top_cities))[i]
    lw[c] = 4

f = plt.figure(figsize=(18,18))
bumpsplot(userank.loc[finallist,:],color_dict=winter_colors,labels=top_cities)
plt.gca().get_yaxis().set_visible(False)

#plt.gcf().subplots_adjust(left=0.25,bottom=.05,right=.75,top=.95)
sns.despine(left=True)  
plt.title('The road to becoming the top 20 electricity producers 2018')
plt.show()


# We see that the early adopters in self-production of electricity - like Amersfoort, The Hague, Rotterdam, Zoetermeer - are still in the top absolute energy producers of 2018, but we also see several other big cities (Eindhoven (something is wrong with pre-2017 data), Tilburg, Zwolle, Maastricht) catch up, starting around 2012.  What is the mean or median fraction of self-produced energy (as a function of the electricity consumed) for the Dutch cities in 2018? We can see that for all Dutch cities combined, the percentage self-produced follows a log-normal distribution centered around 10% in the most recent year of this dataset (below, left), and that the median of this distribution shifts upward linearly with time, starting around 2012. 

# In[ ]:


from matplotlib.ticker import MaxNLocator



f = plt.figure()
gs = gridspec.GridSpec(1,2)

ax1 = f.add_subplot(gs[0,0])
self_production.divide(annual_consume/100)['2012'].plot.hist(bins=np.logspace(-2,2,40),alpha=.3,ax=ax1)
self_production.divide(annual_consume/100)['2014'].plot.hist(bins=np.logspace(-2,2,40),alpha=.3,ax=ax1)
self_production.divide(annual_consume/100)['2018'].plot.hist(bins=np.logspace(-2,2,40),alpha=.3,ax=ax1)
#plt.axvline(self_production.divide(annual_consume/100)['2018'].median(),color='k')
plt.xscale('log')
plt.legend(['2012','2014','2018'])#,'2018 median'])
plt.xlabel('Energy production per city (% of consumption)')

ax2 = f.add_subplot(gs[0,1])
ax2.plot(list(range(2010,2019)),self_production.divide(annual_consume/100).median(),'o',**specs2)
xfit = list(range(2013,2019))
yfit = self_production.divide(annual_consume/100).median()[3:]
m,b = np.polyfit(xfit,yfit,1)
xpred = np.linspace(2013,2022,20)
ax2.plot(xpred,m*xpred+b)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel('percentage of total consumption')
plt.xlabel('Year')
plt.legend(['Data', 'fit'])
plt.suptitle('Self-production per city',fontsize=18)

f.set_figheight(7)
f.set_figwidth(13)


# Extrapolation of the fit $y=mx+b$ to the data (below, right) would suggest that the median reaches 50% self-production in $\frac{50-b}{m}\approx2041$. But this might level off way before that due to technical hurdles, revoking of subsidies or other regulations in the meantime. It's hard for me to assess what factors put a cap on (or speed up?) this trend at the moment.
# 
# By looking at the data per provider, we can get an idea of how different regions of the Netherlands are performing with respect to one-another. [Amsterdam=liander, Rotterdam=stedin, the east and north = enexis] We can see that self-production is a growing trend in all regions, but that the Enexis region is outperforming the other regions likely because there is more space to place windmills or solar panels. Self production became a thing roughly in 2012.

# In[ ]:


provider_self_prod = pd.pivot_table(elec_all,values='self_production',index='provider',columns='year',aggfunc=np.sum)
provider_prod = pd.pivot_table(elec_all,values='annual_consume',index='provider',columns='year',aggfunc=np.sum)
provider_net_prod = pd.pivot_table(elec_all,values='net_annual_consumption',index='provider',columns='year',aggfunc=np.sum)

f = plt.figure()
gs = gridspec.GridSpec(2,3)

ax1 = f.add_subplot(gs[0,0])
provider_prod.T.plot.bar(ax=ax1,legend=False)
#ax[0].set_yscale('log')
#ax[0].set_ylim(1e8,1e9)
ax1.set_ylabel('Annual production (kWh)')
ax1.set_title('Gross annual production')
ax1.set_xlabel('');ax1.set_xticks([])

ax2 = f.add_subplot(gs[0,1])
provider_self_prod.divide(provider_prod/100).T.plot(ax=ax2)
ax2.set_ylabel('Self-production (% of total)')
plt.suptitle('Electricity production per provider',fontsize=18)
ax2.set_title('Self-production annual')
ax2.set_xlabel('')

ax3 = f.add_subplot(gs[0,2])
provider_prod['2018'].plot.pie(ax=ax3)
ax3.set_title('Market share 2018\n(based on gross prod.)')
ax3.set_ylabel('')

ax4 = f.add_subplot(gs[1,0])
provider_net_prod.T.plot.bar(ax=ax4,legend=False)
#ax[0].set_yscale('log')
#ax[0].set_ylim(1e8,1e9)
ax4.set_ylabel('Annual production (kWh)')
ax4.set_title('Net annual production')

ax5 = f.add_subplot(gs[1,1])
provider_self_prod.T.plot.bar(ax=ax5,legend=False)
#ax[0].set_yscale('log')
#ax[0].set_ylim(1e8,1e9)
ax5.set_ylabel('Self-production (kWh)')
ax5.set_title('Annual self-production')

ax6 = f.add_subplot(gs[1,2])
provider_self_prod['2018'].plot.pie(ax=ax6)
ax6.set_title('Market share 2018\n(based on self-prod.)')
ax6.set_ylabel('')

f.set_figheight(12)
f.set_figwidth(18)


# Ok, let's look at the correlation between self production and city size for a bit more, as we have seen above that it is easier for smaller, more rural places to become more self-reliant when it comes to electricity. Perhaps the city size is a reasonable proxy as to what degree a city can produce its own energy (though I suspect building structure, climate etc also play a crucial role), as for instance it might turn out that in the limit of large cities, it is just more efficient to have a majority of the electricity produced in a centralized manner.

# In[ ]:


from scipy import stats

def make_log_df(x_data,y_data,year,offset):
    x = np.log10(x_data[year].dropna())
    y = np.log10(y_data[year].dropna())
    xfit = x.loc[y[y!=-np.inf].index]
    yfit = y[y!=-np.inf]
    # df for regression in log space
    data_for_reg = yfit.to_frame()
    data_for_reg.rename(index=str,columns={year:'y_'+year},inplace=True)
    data_for_reg = data_for_reg.join(xfit.to_frame())
    data_for_reg.rename(index=str,columns={year:'x_'+year},inplace=True)
    
    # set the cities with zero (-inf) to arbitrary dropout value
    y.loc[y[y==-np.inf].index] = -1
    # this includes dropout values intended not to be included in the regression,
    # but for plotting purposes only
    plot_data = y.to_frame().rename(index=str,columns={year:'y_'+year})
    plot_data = plot_data.join(x.to_frame().rename(index=str,columns={year:'x_'+year}))
    data_for_reg,fitparams = classify(data_for_reg,offset,year)
    return(data_for_reg,plot_data,fitparams) 

def classify(df,offset,year):
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['x_'+year],df['y_'+year])
    df['offset_'+year] = df['y_'+year] - (df['x_'+year]*slope+intercept)
    df['cat_'+year] = ['over' if f>offset else 'under' if f<-offset else 'in' for f in df['offset_'+year]]
    fit = [slope, intercept, r_value, p_value, std_err]
    return(df,fit)
    
y1 = np.log10(self_prod_perc_mean['2018'].dropna())
x1 = np.log10(annual_consume['2018'].dropna())
regdata,pldata,fit = make_log_df(annual_consume,self_prod_perc_mean,'2018',.15) 
#plt.scatter(x1,y1,alpha=.2)
sns.jointplot('x_2018','y_2018',data=regdata,kind='reg',scatter_kws={'alpha':0.1,'s':80},height=7)
plt.ylabel('Self-production (%)')
plt.xlabel('Number of active connections')
loc,label = plt.yticks()
yticks = [round(10**float(f),2) for f in loc]
plt.yticks(loc,yticks)
plt.suptitle('Self-production vs city size 2018')


# What we see here is that there is indeed a negative correlation between city size and the percentage of energy that is self-produced. 

# In[ ]:


f,ax = plt.subplots(1,2,figsize=(16,8))
leg = [];slope_evol = [];offset = []
for i in range(2012,2019):
    y=str(i)
    regdata,plotdata,fit = make_log_df(num_active_connections,self_prod_perc_mean,y,.15)
    leg.append(y)
    sns.regplot('x_'+y,'y_'+y,regdata,x_ci='ci',scatter=True,scatter_kws={'alpha':0.05},ax=ax[0])
    slope_evol.append(fit[0])
    offset.append(fit[1])
    ax[0].set_ylim(-2,2) # 0.01 to 100%
    ax[0].legend(leg)
    ax[0].set_ylabel('Self-production (%)')
    ax[0].set_xlabel('Log$_{10}$ Num active connections')

"""    abel = [round(10**(item.get_position()[]),2) for item in ax[0].get_yticklabels()]
    ax[0].set_yticklabels(abel)

    ax[0].set_yticklabels(abel)
"""
ax[1].plot(range(2012,2019),slope_evol,'ob',**specs)
ax[1].set_xlim(2011,2019)
ax[1].set_ylim(-.4,0)
ax[1].set_ylabel('Slope of fit')
ax[1].set_xlabel('Year')
plt.suptitle('City size vs. self-production percentage')


# What we see is that over the past 7 years the slope of the fit became less negative. The negative initial slope (at 2012) makes sense as it is much easier for small towns to get a relatively high percentage of self-produced electricity than it is for large places. As self-production is a nation-wide trend, it is also logical that the slope of the fit becomes less negative, as the larger places catch up as the years pass. However, the rate at which the slope increases is slowing down and it seems like this slope-of-slopes might be asymtotic at a negative value. This could be logical, as it is probably impossible for large cities to be as self-reliant when it comes to electricity production as smaller towns are (i.e. in the limit of 100% urbanization, which might mean 100% high-rise buildings, it is likely impossible for each apartment building to produce its own electricity (with current technologies at hand, that is.)).
# 
# Let's look at the largest Dutch cities and where they stand. We could for instance classify cities into over- or underperforming when above or below the linear regression by a certain amount, as shown below (gray is neither over or underperforming (arbitrary offset chosen here). 

# In[ ]:



regdata,plotdata,fit = make_log_df(num_active_connections,self_prod_perc_mean,'2018',.15)
lut = {'in':[.5,.5,.5],'over':'g','under':'r'}
color = regdata['cat_2018'].map(lut)

f = plt.figure(figsize=(14,12))
gs = gridspec.GridSpec(2,2)
xregr = np.linspace(1,6,10)
#slope, intercept, r_value, p_value, std_err = stats.linregress(regdata['x'],regdata['y'])
#plt.plot(num_active_connections['2015'],self_prod_perc_mean['2015'],'o',ms=10,alpha=.2)
ax1 = f.add_subplot(gs[0,0])
#sns.regplot('x','y',regdata,x_ci='ci',scatter=False,scatter_kws={'alpha':0.1,'s':75})#,ax=ax)
#sns.jointplot('x','y',data=data,kind='reg')
plt.scatter(regdata['x_2018'],regdata['y_2018'],alpha=.31,color=color)
#plt.plot(x,y,'o',ms=10,alpha=.1,color=[.5,.5,.5])
plt.plot(xregr,fit[0]*xregr+fit[1],'k')
#plt.plot(xregr,slope*xregr+intercept+.15,'.-k')
#plt.plot(xregr,slope*xregr+intercept-.15,'.-k')
#plt.xscale('log');plt.yscale('log')
plt.ylabel('Self-production (%)')
plt.xlabel('Number of active connections (x1000)')
plt.ylim(-1,2)
loc,label = plt.yticks()
yticks = [round(10**float(f),2) for f in loc]
plt.yticks(loc,yticks)
loc,label = plt.xticks()
xticks = [round((10**float(f))/1e3,3) for f in loc]
plt.xticks(loc,xticks,fontsize=13)
#plt.text(2.5,-.5,"$y=$"+str(round(slope,3))+"$x+$"+str(round(10**intercept,3)),fontsize=15)
#plt.grid()
plt.xlim(0,6)
#plt.axvline(np.log10(5e4))

ax2 = f.add_subplot(gs[0,1])
ax2.scatter(regdata['x_2018'],regdata['y_2018'],s=num_active_connections.loc[regdata.index,'2018']/1e3,alpha=.4,color=color)
#plt.plot(x,y,'o',ms=10,alpha=.1,color=[.5,.5,.5])
ax2.plot(xregr,fit[0]*xregr+fit[1],'k')
#plt.ylabel('Self-production (%)')
plt.xlabel('Number of active connections (x1000)')
plt.ylim(0,1.)
plt.xlim(left=np.log10(4e4),right=6)
loc,label = plt.yticks()
yticks = [round(10**float(f),2) for f in loc]
plt.yticks(loc,yticks)
loc,label = plt.xticks()
xticks = [round((10**float(f))/1e3) for f in loc]
plt.xticks(loc,xticks,fontsize=13)
twenty18 = num_active_connections['2018'].to_frame()
twenty18_L = twenty18[twenty18>5e4].dropna()
for cit in twenty18_L.index:
    x = regdata.loc[cit,'x_2018']
    y = regdata.loc[cit,'y_2018']
    plt.text(x+.02,y,cit)
    
plt.suptitle('Self-production vs city size 2018, over & underperformers')
all_offset = pd.DataFrame(index=twenty18_L.index,columns=[0])
for year in range(2012,2019):
    year = str(year)
    regdata,plotdata,fit = make_log_df(num_active_connections,self_prod_perc_mean,year,.15)
    if year=='2012':
        all_offset = regdata.loc[twenty18_L.index,:]
    else:
        all_offset = all_offset.join(regdata.loc[twenty18_L.index,:])
ax3 = f.add_subplot(gs[1,:])
values = sns.color_palette('husl',len(all_offset));keys = all_offset.index
lut = dict(zip(keys,values))
colors = all_offset.index.map(lut)
all_offset.loc[:,['offset_2012','offset_2013','offset_2014','offset_2015','offset_2016','offset_2017','offset_2018']].T.plot(ax=ax3,color=colors)
plt.axhline(0,color='k',linewidth=3)
loc,label = plt.xticks()
plt.ylabel('Percentage points above or below regression')
plt.xticks(loc,[int(f+2012) for f in loc])

loc,label = plt.yticks()
yticks = [round(10**float(abs(f)),2) if f>0 else -round(10**float(abs(f)),2) if f<0 else 0 for f in loc]
plt.yticks(loc,yticks)
plt.legend(bbox_to_anchor=(1.5, 1), loc='upper right', ncol=2)


# Ok, so we see above which of the large cities (num_active_connections>5e4) are over/underperforming compared to the regression performed on all Dutch cities. The largest cities seem to be doing worst in terms of becoming self-reliant, and that mostly has to do with the rapid growth of self-production in smaller towns. Over the years (bottom panel) the growth of self-production in large cities is lower than that of the bigger places, turning most large cities into places that are underperforming compared to the fit. This likely has to do with there being a limit to which urbanized places can become self-reliant (e.g. a flat in a high-rise building will not have the option to place solar panels). On a positive note, three Dutch cities seem to be doing a pretty good job at showing that others could do better: **Almere, Apeldoorn** and **Zwolle**.
# 
# My guess is also that Enexis will have a larger fraction of out/overperformers than the other two providers. (And as a side note: Eindhoven's data is off, as we see again)

# <a id="cityviolin"></a>
# # 4. City stats

# In[ ]:


f = plt.figure()
gs = gridspec.GridSpec(1,2)
ax1 = f.add_subplot(gs[0,0])
labels = ['smartmeter_perc','annual_consume_lowtarif_perc','delivery_perc','self_prod_perc']
dat = elec_all[(elec_all.city.isin(['ROTTERDAM','AMSTERDAM']))&(elec_all.year=='2018')].melt(id_vars='city',value_vars=labels) ##'num_active_connections', 'num_smartmeters'])#
g = sns.violinplot(x='variable',y='value',hue=dat.city,split=True,data=dat,ax=ax1)
plt.xlabel('');plt.ylabel('percentage')
g.set_xticklabels(labels,rotation=30,ha='right')
#plt.yscale('log')
ax2 = f.add_subplot(gs[0,1])
dat = elec_all[(elec_all.city.isin(['UTRECHT',"MAASTRICHT"]))&(elec_all.year=='2018')].melt(id_vars='city',value_vars=labels) ##'num_active_connections', 'num_smartmeters'])#
g = sns.violinplot(x='variable',y='value',hue=dat.city,split=True,data=dat,ax=ax2)
plt.xlabel('');plt.ylabel('')
g.set_xticklabels(labels,rotation=30,ha='right') 
f.set_figheight(7)
f.set_figwidth(14)
plt.suptitle('Comparison of individual cities, 2018.')


# Above we see that each city can have its own specific dynamics. Here I chose to display only those parameters that are on the percentage scale, mainly out of laziness. One could also normalize other data to visualize it in this manner of course. As delivery % = 100 - self-production %, these graphs have a bit of redundancy in them.

# <a id="tsne"></a>
# # 5. Global analysis - feature selection and dimensional reduction
# 
# How similar or different are cities from eachother in terms of electricity consumption and behavior? Are there different classes of cities based on energy parameters? Does the average street in Rotterdam behave differently than the average street in Eindhoven? These are questions one could attempt to answer using principle component analysis (PCA) or other dimensional reduction techniques (e.g. TSNE, UMAP), or at least make classifying easier. Admittedly, there are not too many dimensions that are independent of each other, so one can also answer these questions by juggling around with the data in different ways, but this might become less obvious if you add more parameters such as the gas data or other metadata such as income, house prices, ... That and I just want to apply some things that I use in my genomics research and see how they pan out. 

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# In[ ]:


#.pivot_table(elec2018,index='city',columns=sumcols,aggfunc=np.sum)
elec_all.loc[:,(elec_all.dtypes=='float64')|(elec_all.dtypes=='int64')]
sumcols = ['annual_consume','num_connections','low_tarif_consumption','num_active_connections','num_smartmeters','net_annual_consumption','self_production']
meancols = ['annual_consume_lowtarif_perc','delivery_perc','perc_of_active_connections','smartmeter_perc','self_prod_perc']
df = elec_all[elec_all.year=='2018'].groupby('city').sum()[sumcols]
df = df.join(elec_all[elec_all.year=='2018'].groupby('city').mean()[meancols])

x = StandardScaler().fit_transform(df) # transform the variables (mean=0, var=1)
df_standard = pd.DataFrame(x,index=df.index,columns=df.columns)

x_emb = TSNE(n_components=2,perplexity=80,random_state=23944).fit_transform(x) # actual t-SNE (reduction from 11 dims to 2)
tsne_cities = pd.DataFrame(x_emb,index=df.index)
tsne_cities = tsne_cities.join(df['num_active_connections'])

# color by provider (to see if the cities cluster by provider)
enexis = elec_all[(elec_all.year=='2018')&(elec_all.index.str.contains('enexis'))].groupby('city').count().index
liander = elec_all[(elec_all.year=='2018')&(elec_all.index.str.contains('liander'))].groupby('city').count().index
stedin = elec_all[(elec_all.year=='2018')&(elec_all.index.str.contains('stedin'))].groupby('city').count().index
tsne_cities.loc[enexis,'provider'] = 'enexis'
tsne_cities.loc[liander,'provider'] = 'liander'
tsne_cities.loc[stedin,'provider'] = 'stedin'
keys = ['enexis','liander','stedin']
lut = dict(zip(keys,sns.color_palette('dark',3)))
colors = tsne_cities.provider.map(lut)


# In[ ]:


f = plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(1,1)
ax1 = f.add_subplot(gs[0,0])
tsne_cities.loc[enexis].plot.scatter(0,1,s=df.loc[enexis,'num_active_connections'].divide(1e2),alpha=.1,figsize=(8,8),color=colors.loc[enexis],ax=ax1)
tsne_cities.loc[liander].plot.scatter(0,1,s=df.loc[liander,'num_active_connections'].divide(1e2),alpha=.1,figsize=(8,8),color=colors.loc[liander],ax=ax1)
tsne_cities.loc[stedin].plot.scatter(0,1,s=df.loc[stedin,'num_active_connections'].divide(1e2),alpha=.1,figsize=(8,8),color=colors.loc[stedin],ax=ax1)
lgnd = plt.legend(['Enexis','Liander','Stedin'],loc='lower right')
lgnd.legendHandles[0]._sizes = [50]
lgnd.legendHandles[1]._sizes = [50]
lgnd.legendHandles[2]._sizes = [50]
plt.title('Electricity consumption-based\nt-SNE of all Dutch cities, 2018')
plt.xlabel('t-SNE1');plt.ylabel('t-SNE2')


# We see in the t-SNE above that cities cluster together by their size (as the marker size scales with city size) and the provider they have. This because the provider influences parameters such as self-production, smartmeter percentage and lowtarif consumption. This can be an active influence: the amount of effort put into smartmeter implementation - or passive: the geographic location that is fixed and some areas just being more amenable to self-production than others.

# In[ ]:


tsne_cities[(tsne_cities[0]>=0)&(tsne_cities[1]<-30)].sort_values('num_active_connections',ascending=False)
tsne_cities[(tsne_cities[0]>=15)&(tsne_cities[1]<0)&(tsne_cities[0]<25)]

col1 =  self_prod_perc_mean.loc[tsne_cities.index,'2018'].to_frame()
col2 =  smartmeter_perc.loc[tsne_cities.index,'2018'].to_frame()
col3 =  annu_cons_lowtarif_perc.loc[tsne_cities.index,'2018'].to_frame()
col4 =  np.log10(annual_consume.loc[tsne_cities.index,'2018'].to_frame())


# In[ ]:


f = plt.figure(figsize=(18,15))

gs = gridspec.GridSpec(2,2)
ax1 = f.add_subplot(gs[0,0])
tsne_cities.plot.scatter(0,1,alpha=.5,c=col1['2018'],cmap='RdBu_r',ax=ax1)
ax1.arrow(4,-20,-10,-3,width=.5,length_includes_head=True)
ax1.text(5,-20,'region 1')
plt.title('Self production percentage')
plt.xlabel('');plt.ylabel('')

ax2 = f.add_subplot(gs[0,1])
tsne_cities.plot.scatter(0,1,alpha=.5,c=col2['2018'],cmap='RdBu_r',ax=ax2)
ax2.arrow(25,-18,0,8,width=.5,length_includes_head=True)
ax2.text(20,-20,'region 2')
ax2.arrow(9,-30,-5,-5,width=.5,length_includes_head=True)
ax2.text(10,-30,'region 3')
plt.title('Smartmeter percentage')
plt.xlabel('');plt.ylabel('')

ax3 = f.add_subplot(gs[1,0])
tsne_cities.plot.scatter(0,1,alpha=.5,c=col3['2018'],cmap='RdBu_r',ax=ax3)
plt.title('Annual lowtarif cons. percentage')
plt.xlabel('');plt.ylabel('')

ax4 = f.add_subplot(gs[1,1])
tsne_cities.plot.scatter(0,1,alpha=.5,c=col4['2018'],cmap='RdBu_r',ax=ax4)
plt.title('Log10 Annual consumption')
plt.xlabel('');plt.ylabel('')


# Ok, so coloring the t-SNE by the values of the independent parameters, we get an idea of why the cities are grouped together in a certain way. For instance region 1 (indicated in top left) is a cluster of tiny cities that have a very high percentage of self-production, which we saw earlier. There are also some other small towns with a high percentage of self-production (towards upper right corner of plot), but these are different in other ways, as the remaining subplots show.  Two groups containing fairly large cities (region 2 and 3) do not cluster together with the other large cities in the middle. It is clear from the smartmeter subplot that these are the leading cities in terms of smartmeters, but they are different from each other in terms of lowtarif consumption percentage (which tends to be provider-specific).

# In[ ]:


region1_idx = tsne_cities[(tsne_cities[0]>-15)&(tsne_cities[0]<-5)&(tsne_cities[1]<-20)&(tsne_cities[1]>-30)].index
print('region 1 head')
tsne_cities.join(self_prod_perc_mean.loc[region1_idx,'2018']).sort_values('2018',ascending=False).head()


# Perhaps it is more interesting to not let city size weigh in so much, so let's select the features that are normalized (the percentages), these also happen to be the features with the highest variance, which implies these will be more defining of a given city than features with less variance. 

# In[ ]:


df_norm = df[sumcols].divide(df[sumcols].sum(),axis=1)
df_norm = df_norm.join(df[meancols].divide(100))
df_norm.var()


# In[ ]:


x = StandardScaler().fit_transform(df_norm[meancols]) # transform the variables (mean=0, var=1)


x_emb = TSNE(n_components=2,perplexity=80,random_state=23944).fit_transform(x) # actual t-SNE (reduction from 11 dims to 2)
tsne_norm_cities = pd.DataFrame(x_emb,index=df.index)
tsne_norm_cities = tsne_norm_cities.join(df['num_active_connections'])


# In[ ]:


f = plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(1,1)
ax1 = f.add_subplot(gs[0,0])
tsne_norm_cities.loc[enexis].plot.scatter(0,1,s=df.loc[enexis,'num_active_connections'].divide(1e2),alpha=.1,figsize=(8,8),color=colors.loc[enexis],ax=ax1)
tsne_norm_cities.loc[liander].plot.scatter(0,1,s=df.loc[liander,'num_active_connections'].divide(1e2),alpha=.1,figsize=(8,8),color=colors.loc[liander],ax=ax1)
tsne_norm_cities.loc[stedin].plot.scatter(0,1,s=df.loc[stedin,'num_active_connections'].divide(1e2),alpha=.1,figsize=(8,8),color=colors.loc[stedin],ax=ax1)
lgnd = plt.legend(['Enexis','Liander','Stedin'],loc='lower right')
lgnd.legendHandles[0]._sizes = [50]
lgnd.legendHandles[1]._sizes = [50]
lgnd.legendHandles[2]._sizes = [50]
plt.title('Electricity consumption-based\nt-SNE of all Dutch cities, 2018, normalized')
plt.xlabel('t-SNE1');plt.ylabel('t-SNE2')


# We see now that the cities cluster less by size than in the first round, and still largely cluster by provider, and more specifically by Enexis vs. non-Enexis. Why that is we'll see in the multi-panel figure below. Interesting to note is that there are a couple of large non-Enexis cities deep within Enexis territory, let's see who they are:

# In[ ]:


tsne_norm_cities[(tsne_norm_cities[0]<-18)&(tsne_norm_cities[1]<-30)&((tsne_norm_cities.index.isin(liander))|(tsne_norm_cities.index.isin(stedin)))]


# Interesting, these are all cities literally deep within Enexis territory, so perhaps there is some overlap?

# In[ ]:


#'DEVENTER' in enexis
print(set(enexis).intersection(set(liander)))
print(set(enexis).intersection(set(stedin)))


# Indeed there is, strange. I will leave this as is and continue with the global analysis below.

# In[ ]:


f = plt.figure(figsize=(20,11))

gs = gridspec.GridSpec(2,3)
ax1 = f.add_subplot(gs[0,1])
tsne_norm_cities.plot.scatter(0,1,s=tsne_norm_cities['num_active_connections'].divide(2e2),alpha=.5,c=col1['2018'],cmap='RdBu_r',vmax=30,ax=ax1)
#ax1.arrow(4,-20,-10,-3,width=.5,length_includes_head=True)
#ax1.text(5,-20,'region 1')
plt.title('Self production percentage')
plt.xlabel('');plt.ylabel('')

ax2 = f.add_subplot(gs[0,2])
tsne_norm_cities.plot.scatter(0,1,s=tsne_norm_cities['num_active_connections'].divide(2e2),alpha=.5,c=col2['2018'],cmap='RdBu_r',ax=ax2)
#ax2.arrow(25,-18,0,8,width=.5,length_includes_head=True)
#ax2.text(20,-20,'region 2')
#ax2.arrow(9,-30,-5,-5,width=.5,length_includes_head=True)
#ax2.text(10,-30,'region 3')
plt.title('Smartmeter percentage')
plt.xlabel('');plt.ylabel('')

ax3 = f.add_subplot(gs[1,0])
tsne_norm_cities.plot.scatter(0,1,s=tsne_norm_cities['num_active_connections'].divide(2e2),alpha=.5,c=col3['2018'],cmap='RdBu_r',ax=ax3)
plt.title('Annual lowtarif cons. percentage')
plt.xlabel('');plt.ylabel('')

ax4 = f.add_subplot(gs[0,0])
tsne_norm_cities.plot.scatter(0,1,s=tsne_norm_cities['num_active_connections'].divide(2e2),alpha=.5,c=col4['2018'],cmap='RdBu_r',ax=ax4)
plt.title('Log10 Annual consumption')
plt.xlabel('');plt.ylabel('')

col5 =  perc_active_connections.loc[tsne_cities.index,'2018'].to_frame()
ax5 = f.add_subplot(gs[1,1])
tsne_norm_cities.plot.scatter(0,1,s=tsne_norm_cities['num_active_connections'].divide(2e2),alpha=.5,c=col5['2018'],cmap='RdBu_r',vmin=60,ax=ax5)
plt.title('Percent of active connections')
plt.xlabel('');plt.ylabel('')


# Alright, so here we see the large cities cluster into roughly 4 or 5 regions, and we see (especially clear from the first panel) that they group based on other parameters besides size. These other parameters are of course the features we used to perform our dimensional reduction on, the percentages of: lowtarif consumption, self-production, active connections and smartmeters. From the top middle we see that the large cities do not really distinguish themselves in terms of self-production. **So the defining features are: lowtarif consumption, active connections and smartmeter percentage**. Top right shows that two groups have a high fraction of smartmeters compared to the rest. Lowtarif consumption (bottom left panel) we see the approximate Enexis -  non-Enexis divide again, but splitting one of the 4 major clusters down the middle (hence I think it is justified to call 5 clusters of large cities here). Bottom middle we see the percentage of active connections is higher among the Enexis (left two) clusters. 
# 
# This is all great, but it would be insightful to see which cities are representative of these 5 clusters, and perhaps color them geographically to gain more insight as to why certain cities have certain feature values. 
# 
# Bottom left Enexis cluster:

# In[ ]:



print('high smartmeter %; high % active connections; low lowtarif consumption %')
list(tsne_norm_cities[(tsne_norm_cities[0]<-18)&(tsne_norm_cities[1]<-30)].sort_values('num_active_connections',ascending=False).head().index)


# Middle left Enexis cluster:

# In[ ]:



print('low smartmeter %; high % active connections; low lowtarif consumption %')
list(tsne_norm_cities[(tsne_norm_cities[0]<-20)&(tsne_norm_cities[1]>-10)].sort_values('num_active_connections',ascending=False).head().index)


# Top center cluster, left side:

# In[ ]:



print('low smartmeter %; medium % active connections; medium-low lowtarif consumption %')
list(tsne_norm_cities[(tsne_norm_cities[0]<5)&(tsne_norm_cities[1]>10)].sort_values('num_active_connections',ascending=False).head().index)


# Top center cluster, right side:

# In[ ]:



print('low smartmeter %; medium % active connections; high lowtarif consumption %')
list(tsne_norm_cities[(tsne_norm_cities[0]>5)&(tsne_norm_cities[1]>10)].sort_values('num_active_connections',ascending=False).head().index)


# Right side cluster:

# In[ ]:



print('high smartmeter %; medium % active connections; high lowtarif consumption %')
list(tsne_norm_cities[(tsne_norm_cities[0]>25)].sort_values('num_active_connections',ascending=False).head().index)


# 
# Perhaps adding some gas parameters would make it more interesting.

# In[ ]:


path = '../input/dutch-energy/dutch-energy/Gas/'
files_all = [f for f in os.listdir(path)]
gas_all = load_and_reindex(path,files_all)


# In[ ]:


# make pivot tables of relevant parameter such that we have total per city per year
gas_annual_consume = pd.pivot_table(gas_all,values='annual_consume',index='city',columns='year',aggfunc=np.sum)
gas_num_connections = pd.pivot_table(gas_all,values='num_connections',index='city',columns='year',aggfunc=np.sum)
gas_num_active_connections = pd.pivot_table(gas_all,values='num_active_connections',index='city',columns='year',aggfunc=np.sum)
gas_smartmeter_perc = pd.pivot_table(gas_all,values='smartmeter_perc',index='city',columns='year',aggfunc=np.mean)
gas_smartmeter_perc_median = pd.pivot_table(gas_all,values='smartmeter_perc',index='city',columns='year',aggfunc=np.median)
gas_num_smartmeters = pd.pivot_table(gas_all,values='num_smartmeters',index='city',columns='year',aggfunc=np.sum)
gas_self_production = pd.pivot_table(gas_all,values='self_production',index='city',columns='year',aggfunc=np.sum)
gas_self_prod_perc_mean = pd.pivot_table(gas_all,values='self_prod_perc',index='city',columns='year',aggfunc=np.mean)
gas_net_annu_consume = pd.pivot_table(gas_all,values='net_annual_consumption',index='city',columns='year',aggfunc=np.sum)
gas_annu_cons_lowtarif_perc = pd.pivot_table(gas_all,values='annual_consume_lowtarif_perc',index='city',columns='year',aggfunc=np.mean)


# In[ ]:


gas_self_production[gas_self_production.T.sum()!=0]


# Peculiar to see that a very limited number of (random?) towns produce a limited amount of gas?

# In[ ]:


df_all = elec_all[elec_all.year=='2018'].groupby('city').sum()[sumcols]
df_all = df_all.join(elec_all[elec_all.year=='2018'].groupby('city').mean()[meancols])
values = ['elec_'+f for f in sumcols+meancols]
df_all.rename(index=str,columns=dict(zip(sumcols+meancols,values)),inplace=True)
df_all = df_all.join(gas_all[gas_all.year=='2018'].groupby('city').sum()[sumcols])
df_all = df_all.join(gas_all[gas_all.year=='2018'].groupby('city').mean()[meancols])
values_g = ['gas_'+f for f in sumcols+meancols]
df_all.rename(index=str,columns=dict(zip(sumcols+meancols,values_g)),inplace=True)
df_all['gas_smartmeter_perc'] = df_all['gas_smartmeter_perc'].fillna(0)
df_all.sort_values('elec_annual_consume',ascending=False).head()


# By just joining these two datasets, I have noticed there is an outersect between the cities in gas and electricity:

# In[ ]:


print('Cities in gas dataset but not in electricity:'+str(set(gas_all.city).difference(set(elec_all.city))))
print('Cities in electricity dataset but not in gas:'+str(set(elec_all.city).difference(set(gas_all.city))))


# I'm going to ignore that for now. 

# In[ ]:


df_all = df_all.dropna(how='any',axis=0)
x = StandardScaler().fit_transform(df_all) # transform the variables (mean=0, var=1)
df_standard = pd.DataFrame(x,index=df_all.index,columns=df_all.columns)

x_emb = TSNE(n_components=2,perplexity=80,random_state=23944).fit_transform(x) # actual t-SNE (reduction from 11 dims to 2)
tsne_cities = pd.DataFrame(x_emb,index=df_all.index)
tsne_cities = tsne_cities.join(df_all['elec_num_active_connections'])

# color by provider (ignoring the discrepancies between the gas/elec cities)
provider_city = pd.DataFrame(index=elec_all[(elec_all.year=='2018')].groupby('city').count().index,columns=['provider'])
provider_city.loc[enexis,'provider'] = 'enexis'
provider_city.loc[liander,'provider'] = 'liander'
provider_city.loc[stedin,'provider'] = 'stedin'
tsne_cities['provider'] = provider_city.loc[tsne_cities.index,'provider']
keys = ['enexis','liander','stedin']
lut = dict(zip(keys,sns.color_palette('dark',3)))
colors = tsne_cities.provider.map(lut)


# In[ ]:


f = plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(1,1)
ax1 = f.add_subplot(gs[0,0])
#tsne_cities.plot.scatter(0,1,s=df_all['elec_num_active_connections'].divide(1e2),alpha=.1,ax=ax1)
tsne_cities.loc[tsne_cities[tsne_cities['provider']=='enexis'].index].plot.scatter(0,1,s=tsne_cities.loc[tsne_cities[tsne_cities['provider']=='enexis'].index,'elec_num_active_connections'].divide(1e2),alpha=.1,figsize=(8,8),color=colors.loc[tsne_cities[tsne_cities['provider']=='enexis'].index],ax=ax1)
tsne_cities.loc[tsne_cities[tsne_cities['provider']=='liander'].index].plot.scatter(0,1,s=tsne_cities.loc[tsne_cities[tsne_cities['provider']=='liander'].index,'elec_num_active_connections'].divide(1e2),alpha=.1,figsize=(8,8),color=colors.loc[tsne_cities[tsne_cities['provider']=='liander'].index],ax=ax1)
tsne_cities.loc[tsne_cities[tsne_cities['provider']=='stedin'].index].plot.scatter(0,1,s=tsne_cities.loc[tsne_cities[tsne_cities['provider']=='stedin'].index,'elec_num_active_connections'].divide(1e2),alpha=.1,figsize=(8,8),color=colors.loc[tsne_cities[tsne_cities['provider']=='stedin'].index],ax=ax1)
lgnd = plt.legend(['Enexis','Liander','Stedin'],loc='lower right')
lgnd.legendHandles[0]._sizes = [50]
lgnd.legendHandles[1]._sizes = [50]
lgnd.legendHandles[2]._sizes = [50]
plt.title('All energy consumption-based\nt-SNE of all Dutch cities, 2018')
plt.xlabel('t-SNE1');plt.ylabel('t-SNE2')


# Again we see cities clustering by size and the parameters being provider-specific. This might be an artifact (providers have different definitions/ ways of classifying things) or represent the different ways in which the companies operate. We could also do this for the individual postal code areas in two cities that are of similar size but divergent otherwise, or cities that are very alike. But I'll leave it as is for now.

# In[ ]:


"""bli = elec_all[elec_all.city.isin(['ALMERE','DORDRECHT'])][meancols+[sumcols[0]]+[sumcols[3]]]
xc = StandardScaler().fit_transform(bli) # transform the variables (mean=0, var=1)
df_almdor = pd.DataFrame(x,index=df.index,columns=df.columns)

xc_emb = TSNE(n_components=2,perplexity=80,random_state=23944).fit_transform(xc) # actual t-SNE (reduction from 11 dims to 2)
tsne_almdo = pd.DataFrame(xc_emb,index=bli.index)
tsne_almdo['city'] = elec_all.loc[tsne_almdo.index,'city']"""


# In[ ]:


"""lut = {'ALMERE':'r','DORDRECHT':'b'}
coloer = tsne_almdo['city'].map(lut)
tsne_almdo.plot.scatter(0,1,alpha=.01,figsize=(8,8),color=coloer)"""


# We can also look at the importance of parameters using PCA:

# In[ ]:


elec_all.loc[:,(elec_all.dtypes=='float64')|(elec_all.dtypes=='int64')]
sumcols = ['annual_consume','num_connections','low_tarif_consumption','num_active_connections','num_smartmeters','net_annual_consumption','self_production']
meancols = ['annual_consume_lowtarif_perc','delivery_perc','perc_of_active_connections','smartmeter_perc','self_prod_perc']

df = elec_all[elec_all.year=='2018'].groupby('city').sum()[sumcols]
df = df.join(elec_all[elec_all.year=='2018'].groupby('city').mean()[meancols])
x = StandardScaler().fit_transform(df) 
pca = PCA(n_components=4)
principalComp = pca.fit_transform(x)
principaldf = pd.DataFrame(data=principalComp,columns=['PC1','PC2','PC3','PC4'])
#finaldf = pd.concat([principaldf,df[target]],axis=1)


# In[ ]:


col1 =  self_prod_perc_mean.loc[df.index,'2018'].to_frame()
col2 =  smartmeter_perc.loc[df.index,'2018'].to_frame()
col3 =  annu_cons_lowtarif_perc.loc[df.index,'2018'].to_frame()
col4 =  np.log10(annual_consume.loc[df.index,'2018'].to_frame())

f = plt.figure(figsize=(28,7))
gs = gridspec.GridSpec(1,4)

ax1 = f.add_subplot(gs[0,0])
principaldf.plot.scatter('PC1','PC2',s=df['num_active_connections'].divide(1e2),alpha=.1,c=col4['2018'],cmap='RdBu_r',ax=ax1) #pc2 ~ size 
plt.xlabel('PC1')
plt.title('Color: size')

ax2 = f.add_subplot(gs[0,1])
principaldf.plot.scatter('PC1','PC2',s=np.log10(df['num_active_connections'])*50,alpha=.1,c=col1['2018'],cmap='RdBu_r',ax=ax2) #pc2 ~ self prod 
plt.xlabel('PC1')
plt.title('Color: self-production')

ax3 = f.add_subplot(gs[0,2])
principaldf.plot.scatter('PC2','PC3',s=df['num_active_connections'].divide(1e2),alpha=.1,c=col3['2018'],cmap='RdBu_r',ax=ax3) # pc3 ~ lowtarif
plt.xlabel('PC2')
plt.title('Color: lowtarif consumption')

ax4 = f.add_subplot(gs[0,3])
principaldf.plot.scatter('PC3','PC4',s=np.log10(df['num_active_connections'])*50,alpha=.1,c=col2['2018'],cmap='RdBu_r',ax=ax4) # pc4 ~ smartmeter perc
plt.xlabel('PC3')
plt.title('Color: smartmeter percentage')


# In[ ]:


"""leaders2015 = elec_all[elec_all.year=='2015'].groupby('city').mean()#['smartmetaer_perc']
leaders2015size = elec_all[elec_all.year=='2015'].groupby('city').sum()
idx = leaders2015[leaders2015['smartmeter_perc']>60].index
leaders2015size.loc[idx,:].sort_values('num_connections',ascending=False)
#elec_all[elec_all.city.isin(idx)]"""


# <a id="geomapping"></a>
# # 6. Geographical projections
# 
# Alright, so let's try and project the information at onto a map of the Netherlands, as this could make regional differences clear at a glance. What we have at hand is the polygon structure of the 491 municipalities of the Netherlands which we can project onto a map and e.g. color per province, like such:

# In[ ]:


from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon as PG
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import geopandas as gpd
import folium
from matplotlib.patches import Patch
from shapely.geometry import Point, Polygon
import shapely.speedups
shapely.speedups.enable()
from geopandas import GeoDataFrame


# In[ ]:


municip_geo = gpd.read_file('/kaggle/input/municip_shapefile/NLD_adm2.shp')
municip_geo.head(2)


# In[ ]:


f, ax = plt.subplots(figsize = (10,15))
m = Basemap(resolution='h', # c, l, i, h, f or None (courseness)
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=3.15, llcrnrlat= 50.7, urcrnrlon=7.3, urcrnrlat=53.84, ax=ax)
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.readshapefile('/kaggle/input/municip_shapefile/NLD_adm2','geometry')
m.drawcoastlines()

# this seems like a very roundabout way
provinces = municip_geo.NAME_1.unique()
colors = sns.color_palette('Paired',len(provinces))
lut = dict(zip(provinces,colors))
legend_elements = []
for province in provinces: 
    prov = []
    for info, shape in zip(m.geometry_info, m.geometry):
        if (info['NAME_1'] == province) & (province not in ['Zeeuwse meren','IJsselmeer']):
            prov.append( PG(np.array(shape), True) )    
    ax.add_collection(PatchCollection(prov, facecolor= lut[province], edgecolor='k', linewidths=.1, zorder=2))
    if province not in ['Zeeuwse meren','IJsselmeer']:
        legend_elements.append(Patch(facecolor=lut[province], edgecolor='k',
                         label=province))
ax.legend(handles=legend_elements)



# Now comes the hard part: aggregating the information from the 2469 unique cities in the dataset into the 491 municipalities. Without going into a manual adding of geolocation and municipality each city belongs to of course. The easiest way of doing this I found was by using the 4-digit postal codes. I obtained a list of the 4-digit postal code location [here](https://github.com/bobdenotter/4pp). This allows me to easily merge the energy dataframe (grouped by the 4-digit postal codes with relevant parameters summed or averaged) with geolocation per postal code. We get:

# In[ ]:


# first get the data into the right form
elec_all['zipcode_from_int'] = elec_all['zipcode_from'].str[:-2].astype(int)
elec_all['zipcode_to_int'] = elec_all['zipcode_to'].str[:-2].astype(int)
elec_all['zidiff'] = elec_all['zipcode_to_int'] - elec_all['zipcode_from_int']
sumcols = ['annual_consume','num_connections','low_tarif_consumption','num_active_connections','num_smartmeters','net_annual_consumption','self_production']
meancols = ['annual_consume_lowtarif_perc','delivery_perc','perc_of_active_connections','smartmeter_perc','self_prod_perc']
elec_all['provider'] = [f[0] for f in elec_all.index.str.split('_')]

# add postalcode geolocations 
postalcode_geoloc = pd.read_csv('/kaggle/input/postalcodegeolocation/4pp.csv').set_index('postcode',drop=False)

"""
All these functions assume the existence of lists and dataframes loaded above
It's a bit messy, but fairly obvious what I am using here

Dataframes:     postalcode_geoloc
                elec_all
                municip_geo
                
Lists:          meancols
                sumcols
"""

def make_group_zipcode(year,df):
    df = df[df.year==year]
    pc_df = df.groupby('zipcode_from_int').sum()[sumcols]#.count()['city']
    pc_df = pc_df.join(df.groupby('zipcode_from_int').mean()[meancols])
    pc_df = pc_df.join(df.groupby('zipcode_from_int').count()['city']).rename({'city':'code_count'},axis=1)
    pc_df = pc_df.join(df.groupby('zipcode_from_int').first()[['city','provider']])
    values = [year+'_'+f for f in pc_df.columns]
    pc_df.rename(dict(zip(pc_df.columns,values)),axis=1,inplace=True)
    pc_df = pc_df.join(postalcode_geoloc,how='left') #add geolocation
    
    geometry = [Point(xy) for xy in zip(pc_df.longitude, pc_df.latitude)]
    crs = {'init': 'epsg:4326'}
    gdf = GeoDataFrame(pc_df, crs=crs, geometry=geometry) # make geopandas df
    pc_df = attach_municip(pc_df,gdf)
    return pc_df

def attach_municip(df,geodf):
    for municip in municip_geo.ID_2:
        idx = municip_geo[municip_geo.ID_2==municip].index[0]
        municip_name = municip_geo.loc[idx,'NAME_2']
        ingroup = list(geodf[geodf['geometry'].within(municip_geo.loc[idx,'geometry'])==True].index)
        df.loc[ingroup,'ID_2'] = municip
        df.loc[ingroup,'NAME_2'] = municip_name
    return(df)

def merge_energy_data_into_municip(df,year):
    pc_df = make_group_zipcode(year,df)
    firstcols = ['NAME_2','provincie','netnummer',year+'_provider']
    meancols_y = [year+'_'+f for f in meancols]
    sumcols_y = [year+'_'+f for f in sumcols]
    sumcols_y.append(year+'_code_count')
    final_frame = pc_df.groupby('ID_2').first()[firstcols]
    final_frame['ID_2'] = final_frame.index
    final_frame = final_frame.join(pc_df.groupby('ID_2').mean()[meancols_y])
    final_frame = final_frame.join(pc_df.groupby('ID_2').sum()[sumcols_y])
    polygon_frame = pd.DataFrame({
        'shapes': [PG(np.array(shape),True) for shape in m.geometry],
        'ID_2': [area['ID_2'] for area in m.geometry_info]})
    polygon_frame = polygon_frame.merge(final_frame, on='ID_2', how='left')
    return final_frame, polygon_frame
    
    

    


# In[ ]:


pc_2018 = make_group_zipcode('2018',elec_all)
f,ax = plt.subplots(figsize=(11,13))
#pc2018.plot.scatter('longitude','latitude',s=pc2018['2018_num_active_connections'].astype(float).divide(2e2)
#                    ,alpha=.4,ax=ax)
pc_2018.plot.scatter('longitude','latitude',s=pc_2018['2018_num_active_connections'].astype(float).divide(2e2)
                    ,alpha=.4,ax=ax)
plt.title('Number of active connections per 4-digit postal code, 2018')


# For those who have stared at a map of the Netherlands more than once, it becomes pretty clear that the entire province of Zeeland is missing(!). It turns out **Zeeland is not part of the electricity dataset**. They might have a different provider.  Anyway, ignoring this, I think we now have at hand the right information to start binning the postal code based information into the polygons of the 491 municipalities minus Zeeland. This involves determining which postal codes belong to which polygon. Exellent documentation on how to determine wheter a point falls within a polygon can be found [here](http://https://automating-gis-processes.github.io/CSC18/lessons/L4/point-in-polygon.html), and everything is summarized into the functions I wrote above.

# In[ ]:


# let's make polygon dataframes for all years
fin_2011, poly2011 = merge_energy_data_into_municip(elec_all,'2011')
fin_2012, poly2012 = merge_energy_data_into_municip(elec_all,'2012')
fin_2013, poly2013 = merge_energy_data_into_municip(elec_all,'2013')
fin_2014, poly2014 = merge_energy_data_into_municip(elec_all,'2014')
fin_2015, poly2015 = merge_energy_data_into_municip(elec_all,'2015')
fin_2016, poly2016 = merge_energy_data_into_municip(elec_all,'2016')
fin_2017, poly2017 = merge_energy_data_into_municip(elec_all,'2017')
fin_2018, poly2018 = merge_energy_data_into_municip(elec_all,'2018')


# In[ ]:


f, ax = plt.subplots(1,2,figsize = (20,24))
m = Basemap(resolution='h', # c, l, i, h, f or None (courseness)
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=3.15, llcrnrlat= 50.7, urcrnrlon=7.3, urcrnrlat=53.84, ax=ax[0])
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.readshapefile('/kaggle/input/municip_shapefile/NLD_adm2','geometry')
m.drawcoastlines()


providers = ['enexis','liander','stedin']
colors = sns.color_palette('dark',len(providers))
lut = dict(zip(providers,colors))

# there must be a cleaner way to do all this, but I'm leaving it for now
pc_lian = poly2018[poly2018['2018_provider']=='liander'] 
pc_enex = poly2018[poly2018['2018_provider']=='enexis'] 
pc_sted = poly2018[poly2018['2018_provider']=='stedin'] 
pcna = poly2018[poly2018['provincie'].isna()] # municipalities not having data
pcij = poly2018[(poly2018.ID_2==146)|(poly2018.ID_2==400)] #polygons ijsselmeer, zeeuwse meren


pcl = PatchCollection(pc_lian.shapes, zorder=2)
pce = PatchCollection(pc_enex.shapes, zorder=2)
pcs = PatchCollection(pc_sted.shapes, zorder=2)
pna = PatchCollection(pcna.shapes, zorder=2)
pij = PatchCollection(pcij.shapes, zorder=2)

pcl.set_facecolor(lut['liander'])
pce.set_facecolor(lut['enexis'])
pcs.set_facecolor(lut['stedin'])
pna.set_facecolor('gray')
pij.set_facecolor('#46bcec')

pna.set_edgecolor('white')
pij.set_edgecolor('white')
pcs.set_edgecolor('white')
pce.set_edgecolor('white')
pcl.set_edgecolor('white')


ax[0].add_collection(pcl)
ax[0].add_collection(pce)
ax[0].add_collection(pcs)

ax[0].add_collection(pna)
ax[0].add_collection(pij)

legend_elements = []
for provider in providers: 
    legend_elements.append(Patch(facecolor=lut[provider], edgecolor='white',
                         label=provider))
legend_elements.append(Patch(facecolor='gray',edgecolor='white',label='no data (2018)'))
ax[0].legend(handles=legend_elements)
ax[0].set_title('Electricity providers per municipality')


m = Basemap(resolution='h', # c, l, i, h, f or None (courseness)
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=3.15, llcrnrlat= 50.7, urcrnrlon=7.3, urcrnrlat=53.84, ax=ax[1])
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.readshapefile('/kaggle/input/municip_shapefile/NLD_adm2','geometry')
m.drawcoastlines()


pc2 = poly2018[~poly2018['provincie'].isna()] # all others 

norm = Normalize()
cmap = plt.get_cmap('RdBu_r') 
#cmap = plt.get_cmap('Oranges') 
pc = PatchCollection(pc2.shapes,zorder=2)
pna2 = PatchCollection(pcna.shapes, zorder=2)
pij2 = PatchCollection(pcij.shapes, zorder=2)

pc.set_facecolor(cmap(norm(np.log10(pc2['2018_annual_consume'].values))))
pna2.set_facecolor('gray')
pij2.set_facecolor('#46bcec')

pna2.set_edgecolor('white')
pij2.set_edgecolor('white')

ax[1].add_collection(pc)
ax[1].add_collection(pna2)
ax[1].add_collection(pij2)
ax[1].set_title('Log10 annual electricity consumption, 2018')
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
 
mapper.set_array(poly2018['2018_annual_consume'])
#f.colorbar(mapper, shrink=0.4,ax=ax[1],la)


# In[ ]:


# join all poly_dfs at the hip

# there is a smarter way but im getting tired
excludecols = ['shapes', 'ID_2', 'NAME_2', 'provincie', 'netnummer']
incols = [f for f in poly2011.columns if f not in excludecols]
combineddf = poly2018.join(poly2011[incols],how='left')
year = '2012'
incols = [year+f[4:] for f in incols]
combineddf = combineddf.join(poly2012[incols],how='left')

year = '2013'
incols = [year+f[4:] for f in incols]
combineddf = combineddf.join(poly2013[incols],how='left')

year = '2014'
incols = [year+f[4:] for f in incols]
combineddf = combineddf.join(poly2014[incols],how='left')

year = '2015'
incols = [year+f[4:] for f in incols]
combineddf = combineddf.join(poly2015[incols],how='left')

year = '2016'
incols = [year+f[4:] for f in incols]
combineddf = combineddf.join(poly2016[incols],how='left')

year = '2017'
incols = [year+f[4:] for f in incols]
combineddf = combineddf.join(poly2017[incols],how='left')


# In[ ]:


"""# alrighty, there's probably a way to do this in a loop, but I'll refrain to the more elaborate way here
param = 'smartmeter_perc'

f = plt.figure(figsize = (20,35))
gs = gridspec.GridSpec(4,2)

patch1 = combineddf[~combineddf['provincie'].isna()] # all others 
patch2 = combineddf[combineddf['provincie'].isna()] # municipalities not having data
patch3 = combineddf[(combineddf.ID_2==146)|(combineddf.ID_2==400)] #polygons ijsselmeer&zeeuwse meren

y2011 = f.add_subplot(gs[0,0]);year='2011'
m = Basemap(resolution='h', # c, l, i, h, f or None (courseness)
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=3.15, llcrnrlat= 50.7, urcrnrlon=7.3, urcrnrlat=53.84, ax=y2011)
m.drawmapboundary(fill_color='#46bcec') #46bcec
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.readshapefile('/kaggle/input/municip_shapefile/NLD_adm2','geometry')
m.drawcoastlines()

norm = Normalize()
#norm = mpl.colors.Normalize(vmin=0, vmax=100)
cmap = plt.get_cmap('RdBu_r') 
cmap = plt.get_cmap('Oranges') 
pc = PatchCollection(patch1.shapes,zorder=2)
pna = PatchCollection(patch2.shapes, zorder=2)
pij = PatchCollection(patch3.shapes, zorder=2)
pc.set_facecolor(cmap(norm(patch1[year+'_'+param].values)))
pna.set_facecolor('gray')
pij.set_facecolor('#46bcec')
y2011.add_collection(pc)
y2011.add_collection(pna)
y2011.add_collection(pij)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
mapper.set_array(combineddf[year+'_'+param])
plt.colorbar(mapper, shrink=0.4)
plt.title('Smartmeter percentage, '+year)




y2012 = f.add_subplot(gs[0,1]);year='2012'
m = Basemap(resolution='h', # c, l, i, h, f or None (courseness)
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=3.15, llcrnrlat= 50.7, urcrnrlon=7.3, urcrnrlat=53.84, ax=y2012)
m.drawmapboundary(fill_color='#46bcec') #46bcec
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.readshapefile('/kaggle/input/municip_shapefile/NLD_adm2','geometry')
m.drawcoastlines()

norm = Normalize()
cmap = plt.get_cmap('RdBu_r') 
cmap = plt.get_cmap('Oranges') 
pc = PatchCollection(patch1.shapes,zorder=2)
pna = PatchCollection(patch2.shapes, zorder=2)
pij = PatchCollection(patch3.shapes, zorder=2)
pc.set_facecolor(cmap(norm(patch1[year+'_'+param].values)))
pna.set_facecolor('gray')
pij.set_facecolor('#46bcec')
y2012.add_collection(pc)
y2012.add_collection(pna)
y2012.add_collection(pij)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
mapper.set_array(combineddf[year+'_'+param])
plt.colorbar(mapper, shrink=0.4)
plt.title('Smartmeter percentage, '+year)


y2013 = f.add_subplot(gs[1,0]);year='2013'
m = Basemap(resolution='h', # c, l, i, h, f or None (courseness)
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=3.15, llcrnrlat= 50.7, urcrnrlon=7.3, urcrnrlat=53.84, ax=y2013)
m.drawmapboundary(fill_color='#46bcec') #46bcec
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.readshapefile('/kaggle/input/municip_shapefile/NLD_adm2','geometry')
m.drawcoastlines()

norm = Normalize()
cmap = plt.get_cmap('RdBu_r') 
cmap = plt.get_cmap('Oranges') 
pc = PatchCollection(patch1.shapes,zorder=2)
pna = PatchCollection(patch2.shapes, zorder=2)
pij = PatchCollection(patch3.shapes, zorder=2)
pc.set_facecolor(cmap(norm(patch1[year+'_'+param].values)))
pna.set_facecolor('gray')
pij.set_facecolor('#46bcec')
y2013.add_collection(pc)
y2013.add_collection(pna)
y2013.add_collection(pij)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
mapper.set_array(combineddf[year+'_'+param])
plt.colorbar(mapper, shrink=0.4)
plt.title('Smartmeter percentage, '+year)


y2014 = f.add_subplot(gs[1,1]);year='2014'
m = Basemap(resolution='h', # c, l, i, h, f or None (courseness)
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=3.15, llcrnrlat= 50.7, urcrnrlon=7.3, urcrnrlat=53.84, ax=y2014)
m.drawmapboundary(fill_color='#46bcec') #46bcec
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.readshapefile('/kaggle/input/municip_shapefile/NLD_adm2','geometry')
m.drawcoastlines()

norm = Normalize()
cmap = plt.get_cmap('RdBu_r') 
cmap = plt.get_cmap('Oranges') 
pc = PatchCollection(patch1.shapes,zorder=2)
pna = PatchCollection(patch2.shapes, zorder=2)
pij = PatchCollection(patch3.shapes, zorder=2)
pc.set_facecolor(cmap(norm(patch1[year+'_'+param].values)))
pna.set_facecolor('gray')
pij.set_facecolor('#46bcec')
y2014.add_collection(pc)
y2014.add_collection(pna)
y2014.add_collection(pij)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
mapper.set_array(combineddf[year+'_'+param])
plt.colorbar(mapper, shrink=0.4)
plt.title('Smartmeter percentage, '+year)

y2015 = f.add_subplot(gs[2,0]);year='2015'
m = Basemap(resolution='h', # c, l, i, h, f or None (courseness)
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=3.15, llcrnrlat= 50.7, urcrnrlon=7.3, urcrnrlat=53.84, ax=y2015)
m.drawmapboundary(fill_color='#46bcec') #46bcec
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.readshapefile('/kaggle/input/municip_shapefile/NLD_adm2','geometry')
m.drawcoastlines()

norm = Normalize()
cmap = plt.get_cmap('RdBu_r') 
cmap = plt.get_cmap('Oranges') 
pc = PatchCollection(patch1.shapes,zorder=2)
pna = PatchCollection(patch2.shapes, zorder=2)
pij = PatchCollection(patch3.shapes, zorder=2)
pc.set_facecolor(cmap(norm(patch1[year+'_'+param].values)))
pna.set_facecolor('gray')
pij.set_facecolor('#46bcec')
y2015.add_collection(pc)
y2015.add_collection(pna)
y2015.add_collection(pij)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
mapper.set_array(combineddf[year+'_'+param])
plt.colorbar(mapper, shrink=0.4)
plt.title('Smartmeter percentage, '+year)


y2016 = f.add_subplot(gs[2,1]);year='2016'
m = Basemap(resolution='h', # c, l, i, h, f or None (courseness)
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=3.15, llcrnrlat= 50.7, urcrnrlon=7.3, urcrnrlat=53.84, ax=y2016)
m.drawmapboundary(fill_color='#46bcec') #46bcec
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.readshapefile('/kaggle/input/municip_shapefile/NLD_adm2','geometry')
m.drawcoastlines()

norm = Normalize()
cmap = plt.get_cmap('RdBu_r') 
cmap = plt.get_cmap('Oranges') 
pc = PatchCollection(patch1.shapes,zorder=2)
pna = PatchCollection(patch2.shapes, zorder=2)
pij = PatchCollection(patch3.shapes, zorder=2)
pc.set_facecolor(cmap(norm(patch1[year+'_'+param].values)))
pna.set_facecolor('gray')
pij.set_facecolor('#46bcec')
y2016.add_collection(pc)
y2016.add_collection(pna)
y2016.add_collection(pij)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
mapper.set_array(combineddf[year+'_'+param])
plt.colorbar(mapper, shrink=0.4)
plt.title('Smartmeter percentage, '+year)


y2017 = f.add_subplot(gs[3,0]);year='2017'
m = Basemap(resolution='h', # c, l, i, h, f or None (courseness)
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=3.15, llcrnrlat= 50.7, urcrnrlon=7.3, urcrnrlat=53.84, ax=y2017)
m.drawmapboundary(fill_color='#46bcec') #46bcec
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.readshapefile('/kaggle/input/municip_shapefile/NLD_adm2','geometry')
m.drawcoastlines()

norm = Normalize()
cmap = plt.get_cmap('RdBu_r') 
cmap = plt.get_cmap('Oranges') 
pc = PatchCollection(patch1.shapes,zorder=2)
pna = PatchCollection(patch2.shapes, zorder=2)
pij = PatchCollection(patch3.shapes, zorder=2)
pc.set_facecolor(cmap(norm(patch1[year+'_'+param].values)))
pna.set_facecolor('gray')
pij.set_facecolor('#46bcec')
y2017.add_collection(pc)
y2017.add_collection(pna)
y2017.add_collection(pij)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
mapper.set_array(combineddf[year+'_'+param])
plt.colorbar(mapper, shrink=0.4)
plt.title('Smartmeter percentage, '+year)


y2018 = f.add_subplot(gs[3,1]);year='2018'
m = Basemap(resolution='h', # c, l, i, h, f or None (courseness)
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=3.15, llcrnrlat= 50.7, urcrnrlon=7.3, urcrnrlat=53.84, ax=y2018)
m.drawmapboundary(fill_color='#46bcec') #46bcec
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.readshapefile('/kaggle/input/municip_shapefile/NLD_adm2','geometry')
m.drawcoastlines()

norm = Normalize()
cmap = plt.get_cmap('RdBu_r') 
cmap = plt.get_cmap('Oranges') 
pc = PatchCollection(patch1.shapes,zorder=2)
pna = PatchCollection(patch2.shapes, zorder=2)
pij = PatchCollection(patch3.shapes, zorder=2)
pc.set_facecolor(cmap(norm(patch1[year+'_'+param].values)))
pna.set_facecolor('gray')
pij.set_facecolor('#46bcec')
y2018.add_collection(pc)
y2018.add_collection(pna)
y2018.add_collection(pij)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
mapper.set_array(combineddf[year+'_'+param])
plt.colorbar(mapper, shrink=0.4)
plt.title('Smartmeter percentage, '+year)"""


# In[ ]:


"""# gonna save some figs here for gifs, 
param = 'smartmeter_perc';fancy_name = 'Smartmeter percentage'
param = 'self_prod_perc';fancy_name = 'Self-production percentage'
param = 'annual_consume_lowtarif_perc';fancy_name = 'Lowtarif consumption percentage'
param = 'perc_of_active_connections';fancy_name = 'Percentage of active connections'
param = 'annual_consume';fancy_name = 'Log10 annual consumption (kWh)'
param = 'self_production';fancy_name = 'Log10 self-production (kWh)'
for i in range(2011,2019):
    year = str(i)
    f = plt.figure(figsize = (10,10))
    gs = gridspec.GridSpec(1,1)

    patch1 = combineddf[~combineddf['provincie'].isna()] # all others 
    patch2 = combineddf[combineddf['provincie'].isna()] # municipalities not having data
    patch3 = combineddf[(combineddf.ID_2==146)|(combineddf.ID_2==400)] #polygons ijsselmeer&zeeuwse meren

    y2011 = f.add_subplot(gs[0,0]);
    m = Basemap(resolution='h', # c, l, i, h, f or None (courseness)
                projection='merc',
                lat_0=54.5, lon_0=-4.36,
                llcrnrlon=3.15, llcrnrlat= 50.7, urcrnrlon=7.3, urcrnrlat=53.84, ax=y2011)
    m.drawmapboundary(fill_color='#46bcec') #46bcec
    m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
    m.readshapefile('/kaggle/input/municip_shapefile/NLD_adm2','geometry')
    m.drawcoastlines()

    #norm = Normalize()
    norm = mpl.colors.Normalize(vmin=0, vmax=7)
    cmap = plt.get_cmap('RdBu_r') 
    #cmap = plt.get_cmap('Oranges') 
    pc = PatchCollection(patch1.shapes,zorder=2)
    pna = PatchCollection(patch2.shapes, zorder=2)
    pij = PatchCollection(patch3.shapes, zorder=2)
    pc.set_facecolor(cmap(norm(np.log10(patch1[year+'_'+param].values+0.00001))))
    pna.set_facecolor('gray')
    pij.set_facecolor('#46bcec')
    y2011.add_collection(pc)
    y2011.add_collection(pna)
    y2011.add_collection(pij)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array(combineddf[year+'_'+param])
    plt.colorbar(mapper, shrink=0.4)
    plt.title(fancy_name+', '+year)
    
    f.savefig(param+'_'+year+'_rdbu.png')
    plt.close(f)

"""


# In[ ]:


"""
# step 2: making a gif with the sequence of images:

import imageio
import glob
files = glob.glob(param+'*')
files = np.sort(files)
# make a copy of each image to slow down gif by factor 2
from shutil import copyfile
for file in files:
    copyfile(file, file.split('.')[0]+'_1.png')
    copyfile(file, file.split('.')[0]+'_2.png')
    if '2018' in file:
        copyfile(file, file.split('.')[0]+'_3.png')
        copyfile(file, file.split('.')[0]+'_4.png')
files = glob.glob(param+'*')
files = np.sort(files)
images = []
for file in files:
    # imageio.imread(file) creates a numpy matrix array
    # In this case a 200 x 200 matrix for every file, since the files are 200 x 200 pixels.
    images.append(imageio.imread(file))
    print(file)
imageio.mimsave('map_selfprod.gif', images)

# step 3: prep the gif for display in notebook 
from IPython.display import Image
Image("map_selfprod.gif")"""


# Alright, so now we have all the things at hand to start plotting a bunch of stuff, and below are the maps I thought were worth making. Sequences over time, in spirit of the other images in this analysis. For these maps this gives a more qualitative than a quantitative expression, but insightful nonetheless. Also in a way more informative to see the aggregate information of a municipality instead of for every town and village. 
# ![](https://i.imgur.com/sn27hXs.gif)
# ![](https://i.imgur.com/CuDeBa6.gif)
# ![](https://i.imgur.com/KXjwZXZ.gif)
# ![](https://i.imgur.com/pY6ICM4.gif)
# ![](https://i.imgur.com/yx5aBAn.gif)
# ![](https://i.imgur.com/Hn9l5xx.gif)

# <a id="conclusions"></a>
# # 7. Conclusions:
# * Total electricity consumption seems have been roughly constant over the past decade, or even a slight decline in the "randstad" (Liander&Stedin).
# * This while the number of connections is growing linearly, **so the consumption per connection ($\approx$household) seeminly decreased**, possibly by large-scale introduction of more energy-efficient products such as the LED light. **
# * There are other exciting developments unfolding now, such as **transition to a smart grid** as well as the emergence of a significant fraction of **self-production of electricity by households**. 
# * Of the three providers, Enexis seems to be leading these transitions - this might be due to the different makeup of the Enexis territory compared to that of Liander and Stedin (the latter two cover the more densely populated randstad).
# * Within two years of the 2018 dataset it is expected that almost all cities will have >50% (or actually around 80%) smartmeter percentages. **It seems like the 80% smartmeters by 2020 is within reach** The complete transition (to 100%) might take an additional 3 decades.
# * The self-production fraction (as the median of the % of total consumption for all cities combined) has been increasing linearly from <1% in 2012 to $\sim$10% now. At this rate a median of 50% is expected to be reached by 2041.
# * Leaders in the self-production transition are smaller places in rural areas, **especially in Groningen**, but the large cities at the top of the list in absolute terms, with many of them having been in the top 10 absolute self-producers since the beginning of this revolution in 2012.
# 
# ** Though the percentage of active connections seems to decrease as well, as we see in the last map directly above.
# 
# 
# Technical issues with the dataset:
# * The province of Zeeland seems to be missing entirely, but this could be due to there being a fourth provider 
# * Something is wrong with the historical (pre-2017) data for Eindhoven
# * Some cities seem to have multiple providers. This might actually be true, but seems unlikely for some larger cities in Enexis territory like Weert and Deventer
# * Some cities seem to be producing gas, this seems unlikely
# * There are many minor errors, such as 's Gravenhage appearing more than once (but overall I ignore this since there is one 's Gravenhage that is the right size, the others are almost negligibly small.
# 
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ## Introduction ##
# 
# In this notebook, I will implement a simple Neural Network with Theras to predict weather patterns based on the information contained on the Ocean Ships Logbooks. Obviously, real weather prediction require much complex algorithms and the information found in the logbooks is very limited, but it can still be interesting to see what kind of results we can get out of it.
# 
# Before starting with the neural network, however, let's start having a look and see what kind of data we are working with.

# Ship's Courses
# --------------
# 
# The simplest analysis we can apply to this interesting dataset is trying to plot the courses followed by the different ships. Mapping it as a function of different features, such as the nationality of the ship, its origin port and its destination, allows us to infer the story of these vessels' voyages. 

# In[ ]:


#Let's start by importing a few libraries we will need for the analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs 
from geopy.geocoders import Nominatim 
 
#And then import the database
df = pd.read_csv('../input/CLIWOC15.csv', low_memory=False)
#we remove Nan entries for the columns we are interested in studying
df1=df.dropna(subset = ['Lon3', 'Lat3', 'UTC', 'VoyageIni'])
df2=df.dropna(subset = ['VoyageFrom', 'VoyageTo', 'VoyageIni'])


# Now we define a function that will plot all the courses for a given ship on a map built with Cartopy, after grouping them according to the starting date of the trip. 
# We also define a function to plot the ports of origin and destination as spheres, with their size proportional to the number of visits, unfortunately, the geolocator is not working at the moment, so we'll skip this part of the analysis.

# In[ ]:


def plotPath(ship,df,ax,col='#444444',alp=0.5):
    path=df[df['ShipName']==ship][['Lon3', 'Lat3', 'UTC', 'VoyageIni']]
    #Grouping the paths according to the start date of the Voyage helps splitting them
    #and plotting them properly, to be even more accurate one should introduce a function 
    #to split the voyages that present "jumps" from one part to another of the globe
    #which give rise to unpleasant lines crossing the land portions of the map,
    #but since this is just an exploratory plot we will ignore this for now
    groupedPath=path.groupby('VoyageIni')
    for name, group in groupedPath:
        if group.size<2:
            continue
        group.sort_values(by='UTC', inplace=True)
        #draw path on the background
        x,y=group['Lon3'].tolist(),group['Lat3'].tolist()
        ax.plot(x,y,color=col,alpha=alp,transform=ccrs.Geodetic(),linewidth=0.5)
        
#we define a function that finds all origin and destination ports and put them in a dictionary with 
#a count of how many ships departed from/arrived there

def plotPorts(ship,df,ax,col1='#444444',col2='#444444'):
    dictFrom, dictTo={},{}
    path=df[df['ShipName']==ship][['VoyageFrom', 'VoyageTo', 'VoyageIni']]
    groupedFrom=path.groupby('VoyageFrom')
    groupedTo=path.groupby('VoyageTo')
    for name, group in groupedFrom:
        place=group['VoyageFrom'].iloc[0]
        if place not in dictFrom:
            #here we locate the origin port from its name where possible
            location = geolocator.geocode(place)
            if location is not None:
                dictFrom[place]= [group.VoyageIni.nunique(), location.longitude, location.latitude]
            else:
                dictFrom[place][0]=dictFrom[place][0]+group.VoyageIni.nunique()
        for name, group in groupedTo:
            place=group['VoyageTo'].iloc[0]
            if place not in dictTo:
                #here we locate the arrival port from its name where possible
                location = geolocator.geocode(place)
                if location is not None:	
                    dictTo[place]= [group.VoyageIni.nunique(), location.longitude, location.latitude]
            else:
                dictTo[place][0]=dictTo[place][0]+group.VoyageIni.nunique()

    #now we generate the plot, with the ports as circles whose radius is proportional to their 
    #relative number of visits
    sumF= sum([v[0] for k,v in dictFrom.items()])
    sumT= sum([v[0] for k,v in dictTo.items()])
    maxA= max([v[0] for k,v in dictFrom.items()]+[v[0] for k,v in dictTo.items()])
        
    for el in dictFrom[:10]:
	    ax.plot(dictFrom[el][1],dictFrom[el][2],color=col1,transform=ccrs.Geodetic(),marker='o', ms=dictFrom[el][0]*20/maxA , mec='#222222', mew=0.5)
    for el in dictTo[:10]:
        ax.plot(dictTo[el][1],dictTo[el][2],color=col2,transform=ccrs.Geodetic(),marker='o', ms=dictTo[el][0]*20/maxA , mec='#222222', mew=0.5)


# Now we test our function, plotting, for example, all the voyages by Spanish, British and Dutch vessels.

# In[ ]:


#here we initalise the geolocator
geolocator = Nominatim()

#we set up the plots
fig = plt.figure(figsize=(8, 12))
gs1 = gridspec.GridSpec(4, 1)
ax1, ax2, ax3, ax4 = fig.add_subplot(gs1[0],projection=ccrs.Robinson()),                fig.add_subplot(gs1[1],projection=ccrs.Robinson()),                fig.add_subplot(gs1[2],projection=ccrs.Robinson()),                fig.add_subplot(gs1[3],projection=ccrs.Robinson())

#this function helps initialise the different plots with Cartopy according to our preferences
def ax_init(ax):
    ax.set_global()
    ax.outline_patch.set_edgecolor('#3e2c16')
    ax.outline_patch.set_linewidth(0.5)
    ax.outline_patch.set_alpha(0.5)
    #the following commands require Cartopy to contact the server and download some information
    #like the coastlines shape, but unfortunately it doesn't seem to be working at the moment, 
    #so I deactivated them, we'll need a bit of fantasy to see the shape of the landmasses 
    #but that's ok, if you download the code locally it should work fine
    #ax.stock_img()
    ##ax.add_feature(cartopy.feature.LAND, facecolor='#dbc79c', alpha=0.5)
    #ax.add_feature(cartopy.feature.OCEAN, facecolor='#dbc79c', alpha=0.5)
    #ax.coastlines(color='#3e2c16', linewidth=0.5, alpha=0.5)
    
ax_init(ax1), ax_init(ax2), ax_init(ax3), ax_init(ax4)

ax1.set_title("Spanish Travels")
#ships = df1[df1['Nationality']!='Spanish']['ShipName'].unique()
#for ship in ships[:]:
#	plotPath(ship,df1,ax1,'#444444',0.1)

ships = df1[df1['Nationality']=='Spanish']['ShipName'].unique()
for ship in ships[:]:
    plotPath(ship,df1,ax1,'#BB3333',0.3)
    #plotPorts(ship,df2,ax1,'#BB3333','#BB3333')
    
ax2.set_title("British Travels")    
#ships = df1[df1['Nationality']!='British']['ShipName'].unique()
#for ship in ships[:]:
#	plotPath(ship,df1,ax2,'#444444',0.1)

ships = df1[df1['Nationality']=='British']['ShipName'].unique()
for ship in ships[:]:
    plotPath(ship,df1,ax2,'#3333BB',0.3)
    #plotPorts(ship,df2,ax2,'#3333BB','#3333BB')
    
ax3.set_title("Dutch Travels")    
#ships = df1[df1['Nationality']!='Dutch']['ShipName'].unique()
#for ship in ships[:]:
#	plotPath(ship,df1,ax3,'#444444',0.1)

ships = df1[df1['Nationality']=='Dutch']['ShipName'].unique()
for ship in ships[:]:
    plotPath(ship,df1,ax3,'#BBBB33',0.3)
    #plotPorts(ship,df2,ax3,'#3333BB','#3333BB')

ax4.set_title("French Travels")    
#ships = df1[df1['Nationality']!='Dutch']['ShipName'].unique()
#for ship in ships[:]:
#	plotPath(ship,df1,ax3,'#444444',0.1)

ships = df1[df1['Nationality']=='French']['ShipName'].unique()
for ship in ships[:]:
    plotPath(ship,df1,ax4,'#44AA44',0.3)
    #plotPorts(ship,df2,ax3,'#33BB33','#33BB33')


# From these plots, one can then try to make sense of the huge amount of data contained in the logbooks dataset.
# 
# While the Spanish empire mostly focused on central and south America, the British went further north and repeatedly visited India and south-east Asia. Similarly, the Dutch navy interacted with most of south America, central Africa and south-east Asia, while the French limited themselves to central and north America. These assumptions are relative to the number of entries in the logbook (higher density of lines suggests that most of the entries come from Dutch travels) and represent an incomplete description of the intercontinental voyages of these maritime powers, but they are in line with what we all learned from history books. One could then further analyse the evolution of these naval routes over time, the growth of commercial ports, battles between rival nations and more. This logbook is indeed a historical testimony of one of the most prosperous periods of maritime exploration and commerce for European nations. 
# 
# Putting aside all this interesting historical data, we will now focus on the weather logs and see what we can learn.

# In[ ]:


#we import the library
from wordcloud import WordCloud
from functools import reduce

fig = plt.figure(figsize=(6, 12))
gs1 = gridspec.GridSpec(3, 1)
ax1, ax2, ax3 = fig.add_subplot(gs1[0]),                fig.add_subplot(gs1[1]),                fig.add_subplot(gs1[2])
        
df3=df.dropna(subset = ['WindForce','StateSea','Weather'])
df3=df3[df3['Nationality']=='British']

def plotWordCloud(column, df, ax):
    repls = ("weather" , ""), ("sea" , ""), ("water", ""), ("air", ""), ("  ", " ")
    text=" ".join(df[column].tolist())
    text=reduce(lambda a, kv: a.replace(*kv), repls, text)
    wordcloud = WordCloud(background_color="white").generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")

plotWordCloud('Weather', df3, ax1)
plotWordCloud('StateSea', df3, ax2)
plotWordCloud('WindForce', df3, ax3)


# ## Weather Logbooks ##
# 
# Information on the climate conditions met by the sailors can be found in different columns of the .csv file. We can get information on the sea (StateSea), the winds (WindForce) and general weather (Weather) We can first try to extact the information we need directly from there. A qualitative analysis of the  frequency of appearance of specific words will tell us if this is a good idea and something more. Here we will use word cloud, a library that does exactly that with just a few lines of code, but we will use only English words for simplicity (although armed with a translator algorithm or a dictionary one could repeat the analysis for other languages).

# Although visually interesting, this doesn't tell us much about the climate during the voyages, but it shows that the task ahead is harder than it seemed at the beginning. For this analysis, we don't really care if a ship met a light breeze or a fresh one, and we need a way to join different synonyms and apply a more systematic classification. The task becomes even more daunting when considering that most of the logbook is in different languages!
# 
# Luckily for us, somebody did most of the work already and categorised part this information in a few columns, where the presence or absence of weather phenomena like rain, snow or storms is registered for each logbook entry.

# ## KDE ##
# 
# Mapping directly all the entries on the map could result in a very confusing plot, thus we will use instead a Kernel Density Estimation algorithm to infer the probability density of such phenomena in different geographic and temporal points and extract a density map. This method simply estimates the probability density function of a population starting from a limited subset of data: the histogram of such points as a function of the features of interest is smoothed by kernel functions. Different parameters have to be chosen and can affect the quality of the result, such as the bandwidth (which tunes the level of smoothing) and the shape of the kernel functions, which need to be positive, with zero mean and to integrate to one.

# In[ ]:


#we will use kde as implemented in sklearn
import types
from matplotlib import animation
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
df4=df.dropna(subset = ['Lon3', 'Lat3', 'Year'])


# In[ ]:


#we define a KDE function, following the implementation found at 
#https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

def plotKDE(x, y, xbins=100j, ybins=100j, ax, cmap=plt.get_cmap('Blues'), **kwargs): 
	# first we create grid of sample locations (default: 100x100)
	xx, yy = np.mgrid[np.min(x):np.max(x):xbins, np.min(y):np.max(y):ybins]
	#we reshape the data 		
	xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
	xy_train  = np.vstack([y, x]).T
	#now if an adequate bandwidth is known it can be used directly, otherwise
	#gridsearch can find the best value for each dataset     
	kde_skl = KernelDensity(bandwidth=1, **kwargs)
	#20-fold cross-validation
	#kde_skl = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 30)}, cv=20) 
	kde_skl.fit(xy_train)
	# score_samples() returns the log-likelihood of the samples
	#print('Best Bandwidth: '+str(kde_skl.best_params_['bandwidth']))
	#kde_skl = kde_skl.best_estimator_
	z = np.exp(kde_skl.score_samples(xy_sample))
    data=np.reshape(z, xx.shape)
    clevs=np.arange(np.max(data)/10,np.max(data),np.max(data)/10)
    cf=ax.contourf(xx, yy, data, clevs, alpha=0.5, cmap=cmap,                   extend='max', transform=ccrs.PlateCarree())


# In[ ]:


#This part is in case you want to plot how the density changes every X years
#as an animation

"""
#we define a function that only calculates the kde and does not plot it
#since it will be plotted with the animation
def kde2D(x, y, xbins=100j, ybins=100j, **kwargs): 
	# first we create grid of sample locations (default: 100x100)
	xx, yy = np.mgrid[np.min(x):np.max(x):xbins, np.min(y):np.max(y):ybins]
	#we reshape the data 		
	xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
	xy_train  = np.vstack([y, x]).T
	#now if an adequate bandwidth is known it can be used directly, otherwise
	#gridsearch can find the best value for each dataset     
	kde_skl = KernelDensity(bandwidth=1, **kwargs)
	#20-fold cross-validation
	#kde_skl = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 30)}, cv=20) 
	kde_skl.fit(xy_train)
	# score_samples() returns the log-likelihood of the samples
	#print('Best Bandwidth: '+str(kde_skl.best_params_['bandwidth']))
	#kde_skl = kde_skl.best_estimator_
	z = np.exp(kde_skl.score_samples(xy_sample))
	return xx, yy, np.reshape(z, xx.shape)


maxyear=df['Year'].max()
minyear=df['Year'].min()
def year_xform(dl):
	year=minyear
	while year<maxyear:
	    if dl >= year and dl<year+5 : return year
	    year=year+5	

df4['5years'] = df4['Year'].map(year_xform)
grouped=df4.groupby('5years')
gustsX,gustsY=[],[]
groupname=[]
for name,group in grouped:
	groupname.append(name)
	gustsX.append(group[group['Rain'] ==1]['Lon3'].tolist())
	gustsY.append(group[group['Rain'] ==1]['Lat3'].tolist())
#	print len(group[group['Gusts'] ==1]['Lon3'].tolist()), name

cmap = plt.get_cmap('Blues')

listXX,listYY,listData=[],[],[]
for i in range(len(gustsX)):
	xx,yy,data=kde2D(gustsX[i], gustsY[i])
	listXX.append(xx)
	listYY.append(yy)
	listData.append(data)
    
def init():
    cf=ax.contourf([],[],[])	
#    cf.set_data([], [], [])
    return cf

def animate(i):
	xx=listXX[i]
	yy=listYY[i]
	data=listData[i]
	clevs=np.arange(np.max(data)/10,np.max(data),np.max(data)/10)
    #Unfortunately countorf and animation don't really work together and 
    #it is not possible to remove the previous plot for every new
    #frame of the animation, so we'll have to replot everyhing
    #including the background map for each frame (extremely slow)
    #for coll in cf.collections:
    # 	plt.gca().collections.remove(coll) 
    #for obj in ax.findobj(match=None, include_self=True):
    #   ax.findobj(match = type(ax.contourf(xx,yy,data))):
    #  		obj.remove()
	plt.cla()
	ax = plt.axes(projection=ccrs.Robinson())
	ax.set_global()
	ax.outline_patch.set_edgecolor('#3e2c16')
	ax.outline_patch.set_linewidth(0.5)
	ax.outline_patch.set_alpha(0.5)
	#ax.stock_img()
	#ax.add_feature(cartopy.feature.OCEAN, facecolor='#dbc79c', alpha=0.5)
	#ax.add_feature(cartopy.feature.LAND, facecolor='#dbc79c', alpha=0.5)
	#ax.coastlines(color='#3e2c16', linewidth=0.5, alpha=0.5)

	cf=ax.contourf(xx, yy, data, clevs, alpha=0.5, cmap=cmap, extend='max', transform=ccrs.PlateCarree())	
	plt.title('%d' % groupname[i])
	return cf    

output = animation.FuncAnimation(plt.gcf(), animate, frames=len(listData), interval=50, repeat=True)
"""


# ## TO BE CONTINUED...##

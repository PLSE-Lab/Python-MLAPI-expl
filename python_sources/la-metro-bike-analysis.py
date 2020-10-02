#!/usr/bin/env python
# coding: utf-8

# START BY IMPORTING IMPORTANT LIBRARIES...

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# READ CSV FILE...

# In[ ]:


df=pd.read_csv(r"../input/metro-bike-share-trip-data.csv")


# EXPLORE THE COLUMNS AND THEIR TYPES...

# In[ ]:


df.info()


# DISPLAY THE FIRST FEW ROW...

# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


print("shape:",df.shape)


# In[ ]:


df['Start Time']=pd.to_datetime(df['Start Time'])
df['End Time']=pd.to_datetime(df['End Time'])


# CONCLUSION DRAWN BY JUST LOOKING INTO FIRST FEW ROWS
# 
# 1.Trip id provide no additional info and is just like index.
# 
# 2.Info provided by Starting Lat-Long and Ending Lat-Long is redundant.
# 
# 3.Plan duration provide no extra info since passholder type already convey the plan duration.
# 
# 4.Seperating date and time of Start Time and End Time can give further insights about data.

# In[ ]:


# Since Starting Lat-Long and Ending Lat-Long don't provide any extra info so we will drop it
df=df.drop(['Starting Lat-Long','Ending Lat-Long'],axis=1)


# In[ ]:


# Here I am seperating date and time of Start Time and End Time to do some further analysis 
#which are present at bottom
new_dates, new_times = zip(*[(d.date(),d.time()) for d in df['Start Time']])
df = df.assign(new_start_date=new_dates, new_start_time=new_times)
new_dates, new_times = zip(*[(d.date(),d.time()) for d in df['End Time']])
df = df.assign(new_end_date=new_dates, new_end_time=new_times)


# In[ ]:


# Now droping Start Time and End Time
df=df.drop(['Start Time','End Time'],axis=1)


# In[ ]:


# Time to handle missing data
df.isnull().sum()


# In[ ]:


# removing rows with null value
print("shape before:",df.shape)
df=df.dropna()
print("shape after:",df.shape)


# In[ ]:


df.describe()


# LETS DO ANALYSIS ON TRIP DURATION ...

# In[ ]:


# converting duration in min
df['Duration']/=60
df['Duration'].hist(bins=80,range=(0,60))
plt.show()


# It seems that majority of the trips are 5-30 minutes long beacuse according to la metro rides less than 30 min are free for all pass type and after that they charge 1.75 dollar for each 30 min.
# 

# In[ ]:


#out of all pass type walk-up pass make max trips which is longer than 30 mins(78.172%)
df[(df['Duration']>30)][['Passholder Type','Trip ID']].groupby(['Passholder Type']).agg(['count'])/len(df[df['Duration']>30])*100


# ANALYSIS OF WHICH PASS IS FAMOUS...

# In[ ]:


#From the analysis it seems that the percentage of people using monthly pass are maximum.
df[['Trip ID','Passholder Type','Plan Duration']].groupby(['Passholder Type','Plan Duration']).agg(['count'])/len(df)*100


# In[ ]:


#Even exploring further it was found that people using monthly pass for one way are more than
#round trip.
df[['Trip ID','Trip Route Category','Passholder Type']].groupby(['Trip Route Category','Passholder Type']).agg(['count'])/len(df)*100


# ANALYSIS ON STATIONS...

# In[ ]:


gdf=df[['Starting Station ID','Starting Station Latitude','Starting Station Longitude']].groupby('Starting Station ID').max()
print('Unique stations:',gdf.shape[0])
gdf


# In[ ]:


#It is clear that station 4108's longitude is missing.
#here I am removing 4108 in order to get min max lat-long
gdf=gdf[(gdf['Starting Station Latitude']!=0.0) & (gdf['Starting Station Longitude']!=0.0)]
minlatitude=np.min(gdf['Starting Station Latitude'])
maxlatitude=np.max(gdf['Starting Station Latitude'])
minlongitude=np.min(gdf['Starting Station Longitude'])
maxlongitude=np.max(gdf['Starting Station Longitude'])
print("min and max latitude:",minlatitude,maxlatitude)
print("min and max longitude:",minlongitude,maxlongitude)


# In[ ]:


#ploting station to get a better look
s=30
alpha=1
fig, axs = plt.subplots(1, 1,figsize=(20,10))
axs.scatter(gdf['Starting Station Longitude'], gdf['Starting Station Latitude'], zorder=1, alpha=alpha, c='r', s=s)
axs.set_title('Station location ')
plt.show()


# These two stations are farthest from the closely knit cluster of other stations. 

# In[ ]:


# I am not sure why I am getting 3045, any suggestion will be helpful but for now i will not
#consider it
gdf[(gdf['Starting Station Latitude']<34.03) & (gdf['Starting Station Longitude']<-114.38)]


# From latitude and longitude of starting and ending station we can kind of estimate distance between the station.After seeing the statistic of distance i found that max distance is about 12580,and this is mainly because station 4108's longitude value is 0.0.

# In[ ]:


lon1, lat1, lon2, lat2 = map(np.radians, [df['Starting Station Longitude'], df['Starting Station Latitude'],df['Ending Station Longitude'], df['Ending Station Latitude']])
dlat = lat2 - lat1 
dlon = lon2 - lon1
a = (np.sin(dlat/2))**2 +np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2 
c = 2 * np.arcsin( np.sqrt(a)) 
d = 6373  * c 
df['distance in km']=d
df['distance in km'].hist(bins=100,range=(0,5))
plt.show()


# From the plot we can see that 0km has frequency near about 14000, they are actually depicting frequency of round trip.Then all the stations are generally 0-4 km away from others.. 

# I wanted to know which ending station has maximum one way trip from a certain starting station.Even amongst all the popular routes,route from station 3031 to 3014 is mostly used for one way with 927 trips in between.(yellow dot)

# In[ ]:


ndf=df[['Starting Station ID','Ending Station ID','Trip ID']]
mask=df['Trip Route Category']=='One Way'
kndf=ndf[mask].groupby(['Starting Station ID','Ending Station ID']).agg('count')
kndf=kndf.reset_index()
klistofindex=[kndf[kndf['Starting Station ID']==x]['Trip ID'].idxmax() for x in kndf['Starting Station ID'].unique()]
kndf=kndf.loc[klistofindex]   
plt.figure(figsize=(25,15))
plt.scatter(x=kndf['Starting Station ID'],y=kndf['Ending Station ID'],c=kndf['Trip ID'],s=kndf['Trip ID'])
plt.axis([3000.0,3093.0,3000.0,3093.0])
plt.xlabel('Starting Station ID')
plt.ylabel('Ending Station ID')
plt.colorbar()
plt.show()


# In[ ]:


#incase scatter plot is not understandable we can look the table.
listofdistance=[]
for x in zip(kndf['Starting Station ID'],kndf['Ending Station ID']):
    mask=((df['Starting Station ID']==x[0])& (df['Ending Station ID']==x[1]))
    listofdistance.append(np.mean(df[mask]['distance in km']))
kndf['distance']=listofdistance
mask=(kndf['Starting Station ID']==3009.0)|(kndf['Starting Station ID']==3039.0)
kndf[mask]


# From station 3009 maximum no. of one way trips are made to 3039and vice-versa, may be because it is closest in comaprison to other station.

# In[ ]:


kndf


# The no of round trip taken from each station.

# In[ ]:


rndf=df[['Starting Station ID','Trip ID']]
mask1=df['Trip Route Category']=='Round Trip'
rndf=rndf[mask1].groupby('Starting Station ID').count()
rndf=rndf.reset_index()
print('Station which is popular or has max round trip',rndf.iloc[np.argmax(rndf['Trip ID'])])
plt.figure(figsize=(25,10))
rndf.plot(x='Starting Station ID', y='Trip ID')
plt.axis([3000.0,3093.0,0,600])
plt.xlabel('Starting Station ID')
plt.ylabel('No of round Trip')
plt.show()


# ANALYSIS OF DATE AND TIME ...

# In[ ]:


#Total no of trip which ended next day.(duration less than 24hr)
notendedsameday=(df['new_start_date']!=df['new_end_date'])
notendedsameday.sum()


# One way and round trips occured from time 0 to 11.59.59 and from time 12 to 23.59.59,it was found that there 29% one way and 23% round trip taken during night and ealy morning. and  stations which are popular for one way and round trip during each segment are stated below:

# In[ ]:


startdaytime=pd.to_datetime('00:00:00',format='%H:%M:%S').time()
enddaytime=pd.to_datetime('11:59:59',format='%H:%M:%S').time()
startnighttime=pd.to_datetime('12:00:00',format='%H:%M:%S').time()
endnighttime=pd.to_datetime('23:59:59',format='%H:%M:%S').time()
mask1=(df['Trip Route Category']=='One Way')
mask2=(df['Trip Route Category']=='Round Trip')
mask3 = (df['new_start_time'] > startdaytime) & (df['new_start_time'] <= enddaytime )& mask1
mask4 = (df['new_start_time'] > startdaytime) & (df['new_start_time'] <= enddaytime )& mask2
mask5 = (df['new_start_time'] > startnighttime) & (df['new_start_time'] <= endnighttime)& mask1
mask6= (df['new_start_time'] > startnighttime) & (df['new_start_time'] <= endnighttime)& mask2
print('Percentage of one way trip in first 12 hr:',df[mask3]['Trip ID'].count()/df[mask1]['Trip ID'].count()*100)
print('Percentage of one way trip in next 12 hr:',df[mask5]['Trip ID'].count()/df[mask1]['Trip ID'].count()*100)
print('Percentage of round trip in first 12hr:',df[mask4]['Trip ID'].count()/df[mask2]['Trip ID'].count()*100)
print('Percentage of round trip in next 12 hr:',df[mask6]['Trip ID'].count()/df[mask2]['Trip ID'].count()*100)
print('maximum one way trip in first 12hr happens from station:', df[mask3].groupby('Starting Station ID').count()['Trip ID'].idxmax()) 
print('maximum round trip in first 12hr happens from station:',df[mask4].groupby('Starting Station ID').count()['Trip ID'].idxmax())
print('maximum one way trip in next 12hr happens from station:',df[mask5].groupby('Starting Station ID').count()['Trip ID'].idxmax()) 
print('maximum round trip in next 12hr happens from station:',df[mask6].groupby('Starting Station ID').count()['Trip ID'].idxmax())

                                                                                         


# In[ ]:


#The peak time for all the pass type.
nnst=pd.DataFrame([(d.hour+d.minute/60) for d in df['new_start_time']])
nnst['Passholder Type']=df['Passholder Type']
for ph in ['Monthly Pass','Walk-up','Flex Pass']:
    (nnst[nnst['Passholder Type']==ph][0]).hist(bins=200,figsize=(25,10))
    plt.xlabel('hour of day')
    plt.ylabel('count')
    plt.title(ph)
    plt.show()


# In[ ]:


dnst=pd.DataFrame([(d.day) for d in df['new_start_date']])
dnst['Passholder Type']=df['Passholder Type']
for ph in ['Monthly Pass','Walk-up','Flex Pass']:
    (dnst[dnst['Passholder Type']==ph][0]).hist(bins=100,figsize=(25,10))
    plt.xlabel('days of month')
    plt.ylabel('count')
    plt.title(ph)
    plt.show()


# In[ ]:


mnst=pd.DataFrame([(d.month) for d in df['new_start_date']])
mnst['Passholder Type']=df['Passholder Type']
for ph in ['Monthly Pass','Walk-up','Flex Pass']:
    (mnst[mnst['Passholder Type']==ph][0]).hist(bins=200,figsize=(25,10))
    plt.xlabel('months')
    plt.ylabel('count')
    plt.title(ph)
    plt.show()


#  In above graphs for april to june there is no data.

# In[ ]:


#The bike ID which has max trip
bdf=df[['Bike ID','Trip ID']].groupby('Bike ID').count().unstack()
bdf.plot(figsize=(25,10))
plt.show()
print("bike mostly used:",bdf.idxmax()[1])


# If there are any issues with the kernel please do comment, maybe I can improve it.

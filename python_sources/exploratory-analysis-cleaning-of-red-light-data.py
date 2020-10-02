#!/usr/bin/env python
# coding: utf-8

# # Red Light Camera Locations dataset
# ## Exploratory Data Analysis

# In[ ]:


#import EDA libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


#Camera locations df
df = pd.read_csv('../input/red-light-camera-locations.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


#tidying of intersection names
import re
def subber(inp):
    ret = re.sub('[\/]+', '-', inp)
    return re.sub('[^0-9a-zA-Z\-]+', ' ',ret).strip()

df['INTERSECTION'] = df['INTERSECTION'].apply(subber).str.upper()

# list of lists from street names at each intersection
intersections = df['INTERSECTION'].str.split('-')
intersections[:5]


# In[ ]:


street_list = []

for inter in intersections:
    for street in inter:
        street_list.append(street)
        
street_list = pd.Series(street_list)


# #### Which streets have the most red light cams on them?

# In[ ]:


sns.set_style('whitegrid')
#create a DF of street_list value counts
sort = pd.DataFrame(street_list.value_counts(),columns=['Count']).head(10)

#create seaborn barplot of highest counts by street
plt.figure(figsize=(12,6))
sns.barplot(x = sort.index, y='Count', data=sort, estimator=sum, palette='plasma')
plt.title('Number of red light cameras on each street')
plt.xlabel('Street Name')
plt.ylabel('Red Light Cam Count')
plt.yticks(range(0,21,2))
plt.show()


# <br>
# 
# ### Where are the red light cameras exactly??

# In[ ]:


#turn LATITUDE, LONGITUDE columns into lat,lon lists to feed folium
locations = df[['LATITUDE','LONGITUDE']].values.tolist()

locations[:5]


# In[ ]:


#CREATING THE MAP
import folium

#build pop-ups
from folium import IFrame

def make_popup(cam_number, data, append_s=''):
    approaches = []
    
    if 'FIRST APPROACH' in data.columns.values:
        for col in data[['FIRST APPROACH', 'SECOND APPROACH', 'THIRD APPROACH']].columns:
            if isinstance(data[col][cam_number],str):
                approaches.append(data[col][cam_number])
    topstring = data['INTERSECTION'][cam_number]
    midstring = 'Traffic Directions: {}'.format(', '.join(approaches))
    if 'GO LIVE DATE' in data.columns.values:
        botstring = 'Active since: {}'.format(str(data['GO LIVE DATE'][cam_number])[:10])
    else:
        botstring = 'N/A'
    html = """
    <h4 style="margin-bottom:5px">{}: {}</h4>
    <p style="margin-top:0px">{}</br>
    <i>{}</i></p>
    <p>{}</p>""".format(cam_number, topstring, midstring, botstring, append_s)
    pop_iframe = folium.Popup(IFrame(html=html, width=380, height=120))
    return pop_iframe



#locate map
m = folium.Map(location=locations[72], zoom_start=11)

#add points to map
from folium.plugins import MarkerCluster

cluster = MarkerCluster().add_to(m)
for cam in range(len(locations)):
    folium.Marker(locations[cam], popup=make_popup(cam,df)).add_to(cluster)

#show map
m


# <br>
# 
# ### When did these cameras go live?

# In[ ]:


#create a new time series of 'go live' dates
df_timeseries = pd.DataFrame(df[['INTERSECTION', 'GO LIVE DATE']])

df_timeseries['GO LIVE DATE'] = df_timeseries['GO LIVE DATE'].apply(lambda s: pd.to_datetime(s[:10]))
df_timeseries.set_index('GO LIVE DATE', inplace=True)

df_timeseries.head()


# In[ ]:


#create an ECDF plot

#calculate ECDF plot values

def ecdf(data):
    n = len(data)
    x = data
    y = np.arange(1, n+1) / n
    return x,y

sort = df_timeseries.index.sort_values()
x, y = ecdf(sort)

#plot ECDF
plt.figure(figsize=(12,8))
plt.plot(x, y, marker='.', linestyle='none')
plt.ylim(min(y), max(y))

plt.ylabel('ECDF')
plt.xlabel('GO LIVE DATES')
plt.title('Empirical Cumulative Distribution Function for GO LIVE DATES')
plt.show()


# In[ ]:


series_sort = pd.Series(sort)
print('First: {}'.format(series_sort.iloc[0].strftime('%Y-%m-%d')))
print('Latest: {}'.format(series_sort.iloc[-1].strftime('%Y-%m-%d')))


# In[ ]:


print(series_sort.quantile([0.25, 0.75]))


# ###### 50% of all red light cameras in Chicago were installed within a 2 year period!

# <br>
# 
# # Red Light Camera Violations dataset
# ## Data Cleaning

# In[ ]:


df2 = pd.read_csv('../input/red-light-camera-violations.csv')
df2.head(5)


# In[ ]:


df2.tail()


# In[ ]:


df2.info()


# There appears to be several thousand violation records that are missing latitude and longitude fields.  Will use intersection names to match with locations dataframe.

# In[ ]:


#regex string formatting for intersection column
df2['INTERSECTION'] = df2['INTERSECTION'].apply(subber)


# In[ ]:


#format 'INTERSECTION' column to match formatting of 1st df.
def format_int(inp):
    return '-'.join(inp.split(' AND '))
def inverse_format(inp):
    return '-'.join(inp.split(' AND ')[::-1])
#applying inverse format function as defined a few cells down
df2['INVERSE INTERSECTION'] = df2['INTERSECTION'].apply(inverse_format)
df2['INTERSECTION'] = df2['INTERSECTION'].apply(format_int)


# In[ ]:


df2.head()


# In[ ]:


print('Total violations length:\n{}\n'.format(len(df2.index)))
match_inter = df2['INTERSECTION'].isin(df['INTERSECTION'])
print('Total INTERSECTION exact matches from locations DF:\n{}\n'.format(sum(match_inter)))
match_latlon = (df2['LATITUDE'].isin(df['LATITUDE'])) & (df2['LONGITUDE'].isin(df['LONGITUDE']))
print('Total records with a latitudinal co-ord AND a longitudinal co-ord matching any locations:\n{}'      .format(sum(match_latlon)))


# ###### A little over half of the violations records have an exact intersection match from our locations DF.
# 
# ###### Zero records have an exact match for latitude and longitude values.

# In[ ]:


get_ipython().run_cell_magic('time', '', "#rounding both dataframes lat/lon values to 6 decimal places, then checking again for match\n\nfor i in range(6,1,-1):\n    lat1 = df['LATITUDE'].round(i)\n    lon1 = df['LONGITUDE'].round(i)\n    lat2 = df2['LATITUDE'].round(i)\n    lon2 = df2['LONGITUDE'].round(i)\n    print('Rounded to {} decimals.'.format(i))\n    print(sum(lat2.isin(lat1) & lon2.isin(lon1)))\nprint('\\n')")


# ###### By rounding latitude and longitude, it's possible to match more violation records by co-ordinates.  However, we are still unsure if this would be an accurate method. We will first try to find more matches by manipulating the 'INTERSECTION' column.

# In[ ]:


#create function to remove suffixes
def suff_remove(s):
    suff_list = [' STREET', ' ST', ' ROAD', ' RD', ' DRIVE', ' DR', ' CIRCLE', ' CR', ' HIGHWAY',' HWY']
    for suff in suff_list:
        s = s.replace(suff, '')
    return s
df2['INTERSECTION'] = df2['INTERSECTION'].apply(suff_remove)
df['INTERSECTION'] = df['INTERSECTION'].apply(suff_remove)


# In[ ]:


print('Total violations length:\n{}\n'.format(len(df2.index)))
match_inter = df2['INTERSECTION'].isin(df['INTERSECTION'])
print('Total INTERSECTION exact matches from locations DF:\n{}\n'.format(sum(match_inter)))


# ###### By removing common suffixes for street names from the violations dataframe, we were able to increase exact matches by approx 3,500.

# In[ ]:


#create a column in violations dataframe to show whether or not intersection is an exact match from locations DF
df2['MATCHING INTERSECTION'] = df2['INTERSECTION'].isin(df['INTERSECTION'])

#create a new dataframe of all records where 'MATCHING INTERSECTION' is false
df2_nomatch = df2[df2['MATCHING INTERSECTION'] == False]


# In[ ]:


df2.head()


# In[ ]:


#non matching intersections
df2_nomatch['INTERSECTION'].value_counts()


# ###### The unmatched violations appear to be from commonly occuring intersection names.
# Let's create a new column, intersection named with streets in reversed order.

# In[ ]:


#create new column in df2- 'INVERSE INTERSECTION',
#initialize it above, before call to format_int function

#update 'MATCHING INTERSECTION' column, using 'INVERSE INTERSECTION' as well
df2['MATCHING INTERSECTION'] = df2['INTERSECTION'].isin(df['INTERSECTION']) |                                df2['INVERSE INTERSECTION'].isin(df['INTERSECTION'])

df2['MATCHING INVERSE'] = df2['INVERSE INTERSECTION'].isin(df['INTERSECTION'])

df2_nomatch = df2[df2['MATCHING INTERSECTION'] == False]
len(df2_nomatch['INTERSECTION'])


# In[ ]:


sum(df2['MATCHING INTERSECTION'])


# ###### By inversing the order that streets were listed in the 'INTERSECTION' column of the violations dataframe, we were able to find approx 117,000 additional matches for locations!

# In[ ]:


#Lets find LOCATIONS that have no matching VIOLATIONS instead, and compare these locations to df2_nomatch
df['MATCHING INTERSECTION'] = df['INTERSECTION'].isin(df2['INTERSECTION']) |                                df['INTERSECTION'].isin(df2['INVERSE INTERSECTION'])

df_nomatch = df[df['MATCHING INTERSECTION'] ==False]
list(df_nomatch['INTERSECTION'].values)


# In[ ]:


#labels for remaining unmatched intersections
df2_nomatch['INTERSECTION'].value_counts()


# In[ ]:


#Create a column in the violations DF that shows the final spelling of the intersection used.
#this final definition of the 'intersection' will match a field in the locations dataframe.
def final(data):
    if data['MATCHING INVERSE'] == True:
        return data['INVERSE INTERSECTION']
    else:
        return data['INTERSECTION']

df2['FINAL INTERSECTION'] = df2.apply(final, axis=1)


# In[ ]:


#create a dict to map mismatched intersection names

df_nomatch_dict={'KIMBALL-DIVERSEY':'MILWAUKEE-DIVERSEY',
                'STONEY ISLAND-79TH':'STONY ISLAND-79TH-SOUTH CHICAGO',
                'KOSTNER-NORTH':'KOSTNER-GRAND-NORTH',
                '31ST-MARTIN LUTHER KING':'DR MARTIN LUTHER KING-31ST',
                'DAMEN-DIVERSEY':'DAMEN-DIVERSEY-CLYBOURN',
                'PULASKI-ARCHER':'PULASKI-ARCHER-50TH',
                '55TH and PULASKI':'PULASKI-55TH',
                'COTTAGE GROVE-71ST':'COTTAGE GROVE-71ST-SOUTH CHICAGO',
                'HALSTED-FULLERTON':'HALSTED-FULLERTON-LINCOLN',
                '4700 WESTERN':'WESTERN-47TH',
                '79TH-KEDZIE':'KEDZIE-79TH-COLUMBUS',
                'CICERO-I55':'CICERO-STEVENSON NB SOUTH INTERSECTION',
                'DIVERSEY-WESTERN':'WESTERN-DIVERSEY-ELSTON',
                'LAKE-UPPER WACKER':'WACKER-LAKE',
                'STONEY ISLAND-76TH':'STONY ISLAND-76TH'}

df2['FINAL INTERSECTION']=df2['FINAL INTERSECTION'].map(df_nomatch_dict, na_action=None)                        .fillna(df2['FINAL INTERSECTION'])

print(len(df_nomatch.index))
print(len(df_nomatch_dict))


# In[ ]:


#create a column to show manual matching
df2['MATCHING MANUAL'] = df2['INTERSECTION'].isin(df_nomatch_dict.keys())

#update nomatch df
df2_nomatch = df2[(df2['MATCHING INTERSECTION'] == False) & (df2['MATCHING MANUAL'] == False)]
df2_nomatch['INTERSECTION'].value_counts()


# In[ ]:


total_matched = sum(df2['MATCHING INTERSECTION'])+sum(df2['MATCHING MANUAL'])
print('Violation records with matching camera locations:\n{} / {}'.format(total_matched, len(df2.index)))
print('{}%'.format(np.round(total_matched/len(df2.index)*100,4)))


# ###### 94% of violations records had a matching record under red light camera locations.

# In[ ]:


#locate more cameras by using lat/lon co-ordinates of violations
has_lat = df2_nomatch['LATITUDE'].notna()
has_lon = df2_nomatch['LONGITUDE'].notna()

missing_violations = df2_nomatch[has_lat & has_lon]
missing_locations = missing_violations.groupby('INTERSECTION').mean()[['LATITUDE','LONGITUDE']]

missing_violations.head()


# In[ ]:


#create a clean violations dataframe
df_violations = df2[(df2['MATCHING INTERSECTION'] == True) | (df2['MATCHING MANUAL'] == True)]            [['FINAL INTERSECTION','VIOLATION DATE', 'VIOLATIONS',]]


#concat missing_violations to df_violations
missing_violations= missing_violations[['FINAL INTERSECTION','VIOLATION DATE','VIOLATIONS']]

df_violations = pd.concat([df_violations, missing_violations], sort=False) 
df_violations.rename(columns={'FINAL INTERSECTION':'INTERSECTION',
                              'VIOLATIONS':'COUNT'}, inplace=True)

#set violation date as index for time series analysis
df_violations['VIOLATION DATE'] = pd.to_datetime(df_violations['VIOLATION DATE'].str.slice(0,10))
df_violations.set_index('VIOLATION DATE', inplace=True)
df_violations.sort_index(inplace=True)
df_violations.index.name = 'DATE'

print('Violations accounted for, after generating missing locations:')
print('{} / {}'.format(len(df_violations.index),len(df2.index)))
print('{}%'.format(np.round(len(df_violations.index)/len(df2.index)*100,4)))


# ###### >99% of violations have a matching location, after generating missing locations from mean LATITUDE and LONGITUDE of violations grouped by intersection

# In[ ]:


missing_locations.head()


# In[ ]:


#create clean locations dataframe
df_locations = df.drop(['LOCATION', 'MATCHING INTERSECTION'], axis=1)

#set 'GO LIVE DATE' as index, for any future time series analysis
go_live_mask = df_locations['GO LIVE DATE'].notna()
df_locations['GO LIVE DATE'] = pd.to_datetime(df_locations[go_live_mask]['GO LIVE DATE'].str.slice(0,10))

#concat missing_locations to df_locations
df_locations = pd.concat([df_locations,missing_locations.reset_index()], sort=False)


# In[ ]:


#plot missing locations lat/lon to verify accuracy
#locate map
missing_reset = missing_locations.reset_index()
latlon = missing_reset[['LATITUDE','LONGITUDE']].values.tolist()
m2 = folium.Map(location=latlon[23], zoom_start=11)

#add points to map
from folium.plugins import MarkerCluster

#add points to map
for i in range(len(missing_reset)):
    folium.Marker(latlon[i], popup=make_popup(i, missing_reset)).add_to(m2)

#show map
m2


# ###### A plot of the generated observations for the intersection data. The points show the accuracy of using the mean lat/lon co-ordinates from the violations dataset, in our attempt to locate these new intersections.  Unfortunately, we are unable to extrapolate approach directions and 'go live' dates.

# ## Exploratory Data Analysis of Violations

# In[ ]:


df_violations.head()


# In[ ]:


df_violations.tail()


# ###### Red light camera violation records are only dated back as far as 2014-07-01. The first red light camera was installed more than 10 years earlier, on 2003-11-01. This represents approximately 4 years of provided data, and 11 years of missing data.

# In[ ]:


day1 = df_violations.iloc[0].name
before_day1 = len(df_locations[df_locations['GO LIVE DATE'] < day1])
print('{} / {} cameras live before start date of violations dataset.\n'.format(before_day1, len(df)))
print('{} additional cameras live date N/A.  Camera locations generated from violations data.'         .format(len(df_locations[df_locations['GO LIVE DATE'].isnull()])))


# ###### A vast majority of the red light cameras were already running before the start date of the violations dataset.

# In[ ]:


df_violations.info()


# In[ ]:


df_violations.loc['Jan-01-2016'].head()


# ###### The violations data is formatted to have repeat values for datetime index.  Each intersection has a datetime index, and so calling the .loc method with a date as parameter would yield results for numerous intersections.  But we are uncertain if any intersections have missing dates.  We can split each intersection into it's own dataframe, then reindex, and fill a 0 count for missing days, then concat back into df_violations.

# In[ ]:


df_violations[df_violations['INTERSECTION'] == 'WESTERN-VAN BUREN'].index


# ###### Note multiple entries for single intersection for the same datetime index value.

# In[ ]:


df_violations[(df_violations['COUNT'] == 0) | df_violations['COUNT'].isna()]


# ###### Note no 0 entries.
# ###### There are cases of multiple entries for a single intersection and a specific date.  There are also possibly missing datetimes for intersections which had no violations on a specific day (Since we see there are no observations of a 0 count, or NaN).  We need to create a dataframe for each individual intersection, then perform .groupby(datetime_index).sum() that will aggregate multiple entries for the same datetime.  Then we must reindex, by creating an index containing all dates between obersvation [0] and observation [-1], filling NA with 0 (for count).  Finally, we will concat all intersection DFs back into 1 large DF of all violations, then sort_index

# In[ ]:


#time process
from datetime import datetime as dt
time_start = dt.now()

#multiprocessing imports
from multiprocessing import Pool, Manager
import os
#kernel will use all cpu cores except for 1 when assigning work
cpu_count= os.cpu_count() -1

print(df_violations.shape)

#create dict and list manager for pool work
manager = Manager()
d = manager.dict()
l = manager.list()

#create function to pass to workers
def pool_work(value):
    #select by intersection
    temp_df = df_violations[df_violations['INTERSECTION'] == value]
    #sum by date
    temp_df = temp_df.groupby(temp_df.index).sum()
    
    #create new index
    idx = temp_df.index
    new_idx = pd.date_range(idx[0], idx[-1])
    temp_df = temp_df.reindex(new_idx, fill_value=0)
    
    temp_df.insert(0, 'INTERSECTION', value)
    d[value] = temp_df
    #list of # of missing dates per intersection
    l.append(len(new_idx) - len(idx))

    
#create iterator for pool
iterator = df_violations['INTERSECTION'].unique().tolist()
def mp_mapper(iterator):
#map
    if __name__ == '__main__':
        #instantiate a pool
        p = Pool(cpu_count)
        #create a chunksize that will equally distribute all work at outset.
        c_size = (len(iterator)//cpu_count)+1
        #assign work
        p.map(pool_work, iterator, chunksize=c_size)
        p.close()
        p.join()
        p.terminate()

#call mapper
mp_mapper(iterator)

mean_missing_dates = np.round(np.mean(l),2)

df_violations = pd.concat([d[k] for k in d.keys()])
df_violations.sort_index(inplace=True)


print(df_violations.shape)
print('The mean number of missing dates per intersection was {}\n'.format(mean_missing_dates))

#timer output
seconds, microseconds = divmod((dt.now() - time_start).microseconds, 10**6)
milliseconds = microseconds//1000
time_delta = '{}.{} seconds'.format(seconds,milliseconds)
print('\n\nFinished in {}.'.format(time_delta))


# In[ ]:


df_violations.index.name='DATE'
df_violations.head(10)


# In[ ]:


#violations per day / biweekly average
plt.figure(figsize=(14,6))
import matplotlib.cm as cm
cmap = plt.cm.plasma
df_violations.groupby('DATE')['COUNT'].sum().plot(alpha=0.15, color=cmap(0.5))
df_violations.groupby('DATE')['COUNT'].sum().rolling(window=14).mean().plot(color=cmap(0.3))
plt.ylabel('Violations')
plt.title('Total Violations Per Day', fontsize=16)
plt.legend(['DAILY','14D ROLLING'])
plt.show()


# ###### There appears to be some seasonal trending with violations, showing middle months have higher correlation with violations-per-day

# ### Violations per day, for each intersection?

# In[ ]:


#get mean of 'count' column, sorted by 'intersection' column, in descending order.
int_daily_mean = df_violations.groupby('INTERSECTION')['COUNT'].mean().sort_values(ascending=False)

mean_df = pd.DataFrame(int_daily_mean).reset_index()

if 'MEAN DAILY VIOLATIONS' not in df_locations.columns.values:
    df_locations = df_locations.merge(mean_df, how='left',on='INTERSECTION')                    .rename(columns={'COUNT':'MEAN DAILY VIOLATIONS'})


# ### Where are these located on a map?

# In[ ]:


#sort data (sorting in descending value will create smaller bubbles last
#this allows popup to function for all locations)
sort_loc = df_locations.sort_values('MEAN DAILY VIOLATIONS', ascending=False).reset_index(drop=True)

#build list of lat/lon co-ords
latlon=sort_loc[['LATITUDE','LONGITUDE']].values.tolist()

#locate map
m4 = folium.Map(location=latlon[12], zoom_start=11)

#create new append function
def append(i):
    res = np.round(sort_loc.reset_index()['MEAN DAILY VIOLATIONS'].iloc[i],2)
    return 'Mean Daily Violations: {}'.format(res)

#define colormap
import branca.colormap as colormap
cmap_b = colormap.linear.YlOrBr_08

#add points to map
for i in range(len(sort_loc)):
    append_s= append(i)
    mean_daily = sort_loc['MEAN DAILY VIOLATIONS']
    #determine intensity of cmap for color (from 0 to 1)
    #using log of mean daily volumes to create a larger spread in colors
    c_intensity = cmap_b(np.log(mean_daily.iloc[i]) / np.log(mean_daily.max()))
    #determine radius based on mean_daily
    radius = np.log(mean_daily.iloc[i]) * 250
    #draw a small dot at center of each circle
    folium.CircleMarker(latlon[i], color=c_intensity, radius=1).add_to(m4)
    #draw custom circle for each location
    folium.Circle(latlon[i], popup=make_popup(i,sort_loc, append_s),
                 color=c_intensity, radius = radius, fill=True).add_to(m4)
    
#show map
m4


# ### Which intersections have the most daily violations?

# In[ ]:


#create bar plot of highest mean counts by intersection
sns.set_palette('plasma_r', n_colors=10)
plt.figure(figsize=(12,8))

#plot background mean
pd.Series(int_daily_mean.mean()).plot(kind='bar', alpha=0.25, width=40, cmap='plasma')
#plot n bars
barcount=10
int_daily_mean[:barcount].plot(kind='bar', lw=0)

#labelling
plt.title('Mean Daily Violations by Intersection', fontsize=16)
plt.xlabel('INTERSECTION',fontsize=16)
plt.ylabel('Mean Daily Violations',fontsize=16)
plt.xticks([])

#create custom legend
from matplotlib.patches import Patch
cmap = plt.cm.plasma
#build handles
custom_handles = []
lins = np.linspace(0.9,0,barcount)
for i in range(barcount):
    custom_handles.append(Patch(color= cmap(lins[i])))
custom_handles.append(Patch(color=cmap(0), alpha = 0.25))
#build labels
custom_labels = []
for i in range(barcount):
    custom_labels.append(int_daily_mean.index.values[i])
custom_labels.append('ALL INTERSECTIONS')    
#build legend box
plt.legend(custom_handles, custom_labels, bbox_to_anchor=(1.05,1),
           loc=2, fontsize=12, labelspacing=1)
    
plt.show()


# In[ ]:


#What date did these cameras 'go live'?
pd.DataFrame(df_locations[df_locations['INTERSECTION'].isin(int_daily_mean.head(barcount).index.values)]            [['GO LIVE DATE','INTERSECTION']])


# ###### The only 'GO LIVE DATE' outlier is 'WACKER-LAKE', which has been operating for less than 1 year.  All the others appear to be long-established red light camera locations.

# In[ ]:


#update locations DF of mean daily violations
high_v_selector = df_locations['INTERSECTION'].isin(int_daily_mean.head(barcount).index.values)
high_vmean = pd.DataFrame(df_locations[high_v_selector]['INTERSECTION'])

#df_locations slicing
df_loc_select = df_locations[df_locations['INTERSECTION'].isin(high_vmean['INTERSECTION'])]
df_loc_select = df_loc_select.sort_values('MEAN DAILY VIOLATIONS', ascending=False).reset_index(drop=True)
df_loc_select


# ### Show a map of the top 10 intersections for 'mean daily violations'

# In[ ]:


#locate map
latlon = df_loc_select[['LATITUDE','LONGITUDE']].values.tolist()
m3 = folium.Map(location=(41.830848,-87.646351), zoom_start=11)

#create an 'append' function, to add an extra line of info to the map popups.
def append(i):
    res = np.round(df_loc_select.reset_index()['MEAN DAILY VIOLATIONS'].iloc[i],2)
    return 'Mean Daily Violations: {}'.format(res)

mean_daily = df_loc_select['MEAN DAILY VIOLATIONS']
#add points to map
for i in range(len(df_loc_select)):
    append_s= append(i)
    #determine intensity of cmap for color (from 0 to 1)
    intensity= np.log(mean_daily.iloc[i])-np.log(mean_daily.min())
    c_intensity = cmap_b(intensity)
    #determine radius based on mean_daily
    radius = np.log(mean_daily.iloc[i]) * 250
    #draw a small dot at center of each circle
    folium.CircleMarker(latlon[i], color=c_intensity, radius=1).add_to(m3)
    #draw custom circle for each location
    folium.Circle(latlon[i], popup=make_popup(i,df_loc_select, append_s),
                 color=c_intensity, radius = radius, fill=True).add_to(m3)

#show map
m3


# ###### 7 of the 10 highest daily means are at intersections for on/off ramps for major highways, or are running directly parallel to these highways.
# 
# ###### 'Wacker-Lake', the newest camera, and highest daily mean, is not one of these 7.  This intersection appears to be an outlier.  Let's investigate it further.

# In[ ]:


#wacker_lake slice
wacker_lake = df_violations[df_violations['INTERSECTION'] == 'WACKER-LAKE']

#plot a 14-period moving average (mean) of the 'count' column
plt.figure(figsize=(12,6))
wacker_lake['COUNT'].rolling(window=14).mean().plot(color=cmap(0.9))
plt.ylabel('Violations 14 day mean')
plt.yticks([i*10 for i in range(0,11)])
plt.title('WACKER-LAKE 14 day moving average')

plt.show()


# ###### This plot appears to show a similar trend to the entire dataset:  Higher values in the middle of the year.
# ### How does this plot compare to all post 2016 cameras?

# In[ ]:


post_2016_mask = df_locations['GO LIVE DATE'] > '2016'


# In[ ]:


post_2016_mask.value_counts()


# In[ ]:


post_2016 = df_locations[post_2016_mask].sort_values('MEAN DAILY VIOLATIONS', ascending=False)
post_2016


# In[ ]:


#plot a 14-period moving average (mean) of the 'count' column
plt.figure(figsize=(12,6))

for i,intersect in enumerate(list(post_2016['INTERSECTION'])):
    lins = np.linspace(0,0.95,len(post_2016))[-(i+1)]
    loc_mask = df_violations['INTERSECTION'] == intersect
    #ignor error for trying to calc np.log(0), result will be NaN(and unplotted)
    with np.errstate(divide='ignore'):
        np.log(df_violations[loc_mask]['COUNT'].rolling(window=14).mean())            .plot(label=intersect, color=cmap(lins))


plt.xlabel('DATE')
plt.ylabel('LOG OF 14D ROLLING MEAN')
plt.title('LOG OF 14D ROLLING MEANS vs DATE *for all cameras installed after 2016')
plt.legend()
plt.show()


# ###### The 'WACKER-LAKE' numbers are much higher, so we've plotted log of rolling means to preserve details in the plot of the smaller observations.

# In[ ]:


int_daily_mean[:10].index.values


# ### Plot of the top 5 intersection (by mean violations count)

# In[ ]:


plt.figure(figsize=(12,6))
for i,intersect in enumerate(list(int_daily_mean[:5].index.values)):
    lins = np.linspace(0,0.95,5)[-(i+1)]
    loc_mask = df_violations['INTERSECTION'] == intersect
    df_violations[loc_mask]['COUNT'].rolling(window=14).mean()        .plot(label=intersect, color=cmap(lins))

plt.legend()
plt.show()


# ###### It appears there was a period of time, the majority of 2015, that 'CICERO-STEVENSON...' was not capturing any violations.  This results in the mean and median # of violations for this intersection being brought down.
# ###### How many '0' counts exist for each of these intersections?

# In[ ]:


#list of top 5 violation count means
top5 = list(int_daily_mean[:5].index.values)
#select only intersections from the top 5
loc_mask = df_violations['INTERSECTION'].isin(top5)
#select only dates which have a count of 0
zcount = df_violations['COUNT'] == 0

#count of zero-count dates for each intersection
top5_zcount = df_violations[loc_mask & zcount].groupby('INTERSECTION').count()
#include any top5 intersections that have no zcount
top5_zcount = top5_zcount.reindex(top5, fill_value=0)                .sort_values('COUNT', ascending=False)
top5_zcount


# In[ ]:


#count NON-zero observations for each
top5_nozcount = df_violations[loc_mask & ~zcount].groupby('INTERSECTION').count()

#set index to match top5_zcount
top5_nozcount = top5_nozcount.reindex(top5_zcount.index, fill_value=0)

top5_nozcount


# In[ ]:


#percentage of dates that have 0 violations
perc_missing = (top5_zcount / (top5_nozcount + top5_zcount) * 100).rename(columns={'COUNT':'PERCENT MISSING'})
perc_missing


# ###### 'CICERO-STEVENSON...' has 16% missing data.
# ### Are there more intersection that are missing several days of data?

# In[ ]:


missing_dates = df_violations[df_violations['COUNT']==0].groupby('INTERSECTION')                    .count().sort_values('COUNT', ascending=False)

missing_dates.head(10)


# ###### Some of these counts are very high.  If we can confirm that the count of red light violations is a Poisson process, we can infer a certain number of expected 0 count days.

# ### Is this a Poisson distribution?

# In[ ]:


sns.set_palette('tab10', n_colors=10)

data = df_violations['COUNT']
#build Poisson, normal, exponential distributions same size as len(data
poisson_dist = np.random.poisson(np.mean(data),size=len(data))
normal_dist = np.random.normal(np.mean(data), scale= data.std(), size=len(data))

#build bins
steps=1
max_range = range(-24,48,steps)
bins = [x for x in max_range]

#build iterator
d_list = ['Normal','Poisson']
iterator = zip([normal_dist, poisson_dist],d_list)

#create 3 subplots
n_plots= len(d_list)
f, axes = plt.subplots(n_plots,1, figsize=(12,4*(n_plots)))


#plot each
for i,(data_dist,dist_label) in enumerate(iterator):
    axes[i].hist(data, bins=bins, alpha=1, label='COUNT')
    axes[i].hist(data_dist, bins=bins, alpha=0.5, label=dist_label.upper())
    axes[i].set_xticks(max_range)
    axes[i].set_title('Count vs {} Distribution'.format(dist_label))
    axes[i].set_xlabel('Histogram of Violations per Day')
    axes[i].legend()

plt.tight_layout()
plt.show()


# ### Use Poisson distribution and binomial testing to determine likelihood of null hypothesis.
# ###### The null hypothesis we will be testing is: the missing dates are all dates with 0 violations.

# In[ ]:


#define a function to sample from poisson distribution
def bs_zeros(mean, samples=10000, size=1000):
    zeros = []
    for i in range(size):
        #sample is number of times out of 100 that 0 will occur, given the median
        sample = np.sum((np.random.poisson(mean, size=samples)) == 0)
        #create list of size=size samples
        zeros.append(sample)
    zeros = np.array(zeros)
    #return 95% confidence interval as a percentage
    return np.percentile(zeros,[2.5,97.5,50])/samples


# In[ ]:


from scipy.stats import binom_test
data = df_violations[df_violations['INTERSECTION'] == 'DAMEN-FULLERTON']
missing = (data['COUNT']==0).sum()
total = len(data.index)

bs_test = bs_zeros(data['COUNT'].mean(),total)
binom_test(missing, total,bs_test[1])


# ###### The p-value for our binomial test on 'DAMEN-FULLERTON' is below 0.05, which is strong evidence to reject our null hypothesis.  These n/a values most likely are not all 0 counts.
# 
# ### Let's perform this test on all intersections without making any assumptions about the values of the missing data being 0.

# In[ ]:


#time it
time_start = dt.now()

m=Manager()
l = m.list()

def pool_work(idx):
    
    data = df_violations[df_violations['INTERSECTION'] == idx]
    missing = (data['COUNT']==0).sum()
    total = len(data.index)
    
    #calc mean of all counts
    #mean = data['COUNT'].mean()
    #calc mean of all non-zero counts
    mean= data[data['COUNT']>0]['COUNT'].mean()
    
    bs = bs_zeros(mean, samples = total,size=10000)
    #perform binomial test on p= 97.5th percentile
    res = binom_test(missing,total,bs[1])
    #strf 90% confidence interval
    expected = str(int(bs[0]*total))+'-'+str(int(bs[1]*total))
    #median expected count of zeros
    median_expected = bs[2]*total
    
    l.append([idx,mean,expected,int(median_expected),missing,total,res])
    
    
#create iterator for pool
iterator = missing_dates.index.values
#call mp_mapper for pool processing
mp_mapper(iterator)

#timer output
seconds = (dt.now()-time_start).seconds
micro = (dt.now()-time_start).microseconds
milliseconds = microseconds//1000
time_delta = '{}.{}'.format(seconds,milliseconds)
print('\n\nFinished in {} seconds.'.format(time_delta))


# In[ ]:


df_missing = (pd.DataFrame([x for x in l],
            columns=['INTERSECTION','MEAN VIOLATIONS','EXPECTED ZEROS','EXPECTED MEDIAN','MISSING','TOTAL','P-VALUE']))\
            .sort_values('P-VALUE', ascending=False).reset_index(drop=True)

df_missing.head(20)


# ###### After sorting by p-value we see extremely low values for a majority of p-vals for the dataset. Our p-value very quickly drops below 0.05.  This is strong evidence against our original null hypothesis.  This means it is highly unlikely that all these missing data are 0 count days.  It is not evidence against ANY of the missing data being 0 counts, but only evidence against ALL missing data being 0 counts.

# In[ ]:


#mean p-value for dataset
mean_p = df_missing['P-VALUE'].mean()

mean_p


# ###### Our missing data is exactly that - missing. We have not been given any information about why the data are missing, and why there are no 0 counts in our set.  Although it is likely that many 0 counts naturally would occur in these missing data (as shown with our Poisson distributions), it is also likely that a majority of these missing data would be missing for other reasons.  It is important to impute these missing data as accurately as possible to build successful machine learning models.

# In[ ]:


#build imputation function
#some dates will retain the original 0 we assigned to missing data
#the rest will be imputed as a sample from a poisson distribution for mean count for each intersection

df_violations_imp = df_violations.copy()
#selectors
ZEROS = df_violations['COUNT'] == 0  #our original missing data
NO_ZEROS = df_violations['COUNT'] > 0 #all valid data


from scipy.stats import mode

for idx in df_violations['INTERSECTION'].unique().tolist():
    #selector
    inter = df_violations['INTERSECTION'] == idx
    if len(df_violations[ZEROS & inter].index) == 0:
        #if there are no missing data
        pass
    else:
        #count of missing for this date
        missing = len(df_violations[ZEROS & inter].index)
        #expected count of 0s for this intersection
        exp_zeros = df_missing[df_missing['INTERSECTION'] == idx]['EXPECTED MEDIAN'].values[0]
        #build array of exp_zeros number of 0s.
        imp_zeros_count = np.zeros(exp_zeros)
        #find most commonly occuring count
        mode_= mode(df_violations[NO_ZEROS & inter]['COUNT'])[0][0]
        #build array of n samples from Poisson distribution
        #where n = missing-exp_zeros
        imp_mode_count = np.array([np.random.poisson(mode_) for _ in range(missing-exp_zeros)])
        #append lists to create imputation_list
        imputation_list= np.append(imp_zeros_count, imp_mode_count)
        
        #randomize list for cold deck imputation
        cold_deck_imp = np.random.choice(imputation_list, len(imputation_list))
        
        #use cold deck imputation on all missing data
        df_violations_imp.loc[ZEROS & inter,'COUNT'] = cold_deck_imp
    


# ###### We have kept the expected number of zeros from our previous model, also we have simulated a Poisson process for the remainder of the days missing data.  We have then randomized the order of these zeros and simulations, and assigned those values to the missing values for our entire dataset (which we were previously representing as 0).

# In[ ]:


plt.figure(figsize=(12,6))
for i,intersect in enumerate(list(int_daily_mean[:5].index.values)):
    lins = np.linspace(0,0.95,5)[-(i+1)]
    loc_mask = df_violations_imp['INTERSECTION'] == intersect
    df_violations_imp[loc_mask]['COUNT'].rolling(window=14).mean()        .plot(label=intersect, color=cmap(lins))

plt.ylabel('Violations')
plt.title('Top 5 Intersections: Violations per day (after imputation)')
plt.legend()
plt.show()


# ###### As we can see now, the 'CICERO-STEVENSON...' data has been imputed successfully.

# In[ ]:


#create an updated 'df_locations' dataframe with daily mean including imputed data

#merge locations with new column
left = df_locations.drop('MEAN DAILY VIOLATIONS', axis=1)
right = pd.DataFrame(df_violations_imp.groupby('INTERSECTION')['COUNT'].mean())
df_locations_imp = pd.merge(left, right, on='INTERSECTION')                    .rename(columns={'COUNT':'MEAN DAILY VIOLATIONS'})


# ### Our data has been cleaned, and is now ready to be used for the machine learning process.
# 
# #### The cleaned data are located in the following DataFrames: `df_locations_imp` and `df_violations_imp`

#!/usr/bin/env python
# coding: utf-8

# # Emergency - 911 Calls : Data Visualization using Python

# ### About 911 Emergency :

# A 911 emergency is when someone needs help right away because of an injury or an immediate danger. For example, call 911 if:
# - there's a fire
# - someone has passed out
# - someone suddenly seems very sick and is having a hard time speaking or breathing or turns blue
# - someone is choking
# - you see a crime happening, like a break-in
# - you are in or see a serious car accident

# ### 911 Calls : Dataset 

# In this project we will be analyzing 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data-  contains the following fields:
# 
# - lat : String variable, Latitude
# - lng: String variable, Longitude
# - desc: String variable, Description of the Emergency Call
# - zip: String variable, Zipcode
# - title: String variable, Title
# - timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# - twp: String variable, Township
# - addr: String variable, Address
# - e: String variable, Dummy variable (always 1)

# # Data Visualization using Python

# #### 1. Importing all the required library for Visualization

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as plt
import plotly.plotly as py
import geopandas
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### 2. Reading the dataset and storing it in dataframe(df)

# In[ ]:


df=pd.read_csv("../input/911.csv")


# #### 3. Some more details & describtion about Dataset ( total entries,data type )

# In[ ]:


# total entries , Data types , Memory Usage...
df.info()


# In[ ]:


# Stastical describtion about 
df.describe()


# #### 4. Printing the sample dataframe values

# In[ ]:


#prints the head of dataframe- top 5 values by default
df.head()


# #### 5. Data cleaning

# Data cleaning is the process of identifying and removing (or correcting) inaccurate records from a dataset, table, or database and refers to recognising unfinished, unreliable, inaccurate or non-relevant parts of the data and then restoring, remodelling, or removing the dirty or crude data.
# examples:
# - Get Rid of Extra Spaces
# - Select and Treat All Blank Cells
# - Convert Numbers Stored as Text into Numbers
# - Remove Duplicates
# - Highlight Errors
# - Change Text to Lower/Upper/Proper Case
# - Spell Check
# - Delete all Formatting

# #### In the given 911 dataset

# ### 5.a) Cleaning of given dataset:

# #####    1 ) Removing the dummy Column "e" which has all entires equals 1.

# In[ ]:


# Dropping the dummy column 'e' which can done in below 2 ways.
#df.drop('e', axis=1, inplace=True)
del df['e']


# ####  2 ) Converting of timeStamp object (99492,non-null,object) to DateTime objects

# In[ ]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# ### 5.b ) Creating new variables:

# #### 3 ) to extract more valuable information from dataset,we Create type and subtype based on the title column

# In[ ]:


#Splitting the Title
df['reason']=df['title'].apply(lambda i:i.split(':')[0])


# In[ ]:


df['Detail reason']=df['title'].apply(lambda i:i.split(':')[1])


# #### 4 ) Creating the Year and Month from the timestemp column to Visualization based on time and months

# In[ ]:


df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
df['Year'] = df['timeStamp'].apply(lambda t: t.year)
df['Date'] = df['timeStamp'].apply(lambda t: t.day)


# #### 5 ) Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week:
# 
# dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)


# ### 6 ) Extraction more details from Desc column, here using matching function i am extraction Station Number as below. 

# In[ ]:


#getting details of station 
df['Station'] = df['desc'].str.extract('(Station.+?);',expand=False).str.strip()


# In[ ]:


df["day/night"] = df["timeStamp"].apply(lambda x : "night" if int(x.strftime("%H")) > 18 else "day")


# #### After creating all new columns for analysis, our dataframe looks as below

# In[ ]:


df.head()


# # Data Visualizing and Analysing

# - visualizations really helps make things clearer and easier to understand, especially with larger, high dimensional datasets.
# - Matplotlib,seaborn,plotly is a popular Python library that can be used to create your Data Visualizations quite easily.
# - Here you learn about plots and how to create them with Matplotlib, histograms,bar charts,pie charts,box plots,scatter plots and bubble plots etc

# #### 1 ) Total Number of calls during " Day " time and " Night " Time

# In[ ]:


sns.countplot(x='day/night',data=df)
sns.set_style("darkgrid")


# #### 2 ) Calculating top 10 Station with highest number of calls.

# In[ ]:


df.Station.value_counts().head(10)


# #### 3 ) Countplot for Top 10 station with highest calls

# In[ ]:


plt.figure(figsize=(20,10))
sns.set_context("paper", font_scale = 2)
sns.countplot(y='Station', data=df, palette="bright", order=df['Station'].value_counts().index[:20])
plt.title("Top 10 Station with highest call")
sns.set_style("darkgrid")
plt.show()


# #### 4 ) Top reasons to call 911

# In[ ]:


sns.countplot(x='reason',data=df,palette='magma')
sns.set_style("darkgrid")


# #### 5 ) countplot of the Day of Week column with the hue based off of the Reason column

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(x='Day of Week',data=df,hue='reason',palette='cividis')
plt.title("Calls on each days of the week")
sns.set_style("darkgrid")
plt.show()


# #### 6 ) countplot of the Month column with the hue based off of the Reason colum

# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(x='Month',data=df,hue='reason',palette='hot')
plt.title("Calls Count during each Month ")
sns.set_style("darkgrid")
plt.show()


# #### 7 ) Plot for calls recieved monthly combined of all years:

# In[ ]:


# Plot for calls recieved monthly combined of all years:
plt.figure(figsize=(12,6))
sns.countplot(x='Month',data=df,palette='spring')
plt.title("Total Calls recieved Monthly For all Years")
sns.set_style("darkgrid")
plt.show()


# #### 8) Plot for calls recieved on yearly basis:

# In[ ]:


# Plot for calls recieved yearly:
sns.countplot(x= "Year", data= df,palette='RdYlGn_r')
plt.title("calls recieved on yearly basis")
sns.set_style("darkgrid")
plt.show()


# #### 9 )  Count for all Calls Reason  Yearly

# In[ ]:


plt.figure(figsize=(14,7))
sns.set_context("paper", font_scale = 2)
sns.countplot(x= "reason", data= df, palette="bright" ,hue= "Year")
plt.title(" Calls Reason Yearly")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.set_style("darkgrid")
plt.show()


# #### 10 ) Calls Reason Yearly having the hue of reasons

# In[ ]:


plt.figure(figsize=(14,7))
sns.set_context("paper", font_scale = 2)
sns.countplot(x= "Year", data= df, palette="Paired", hue = "reason")
plt.title(" Calls Reason Yearly having the hue of reasons")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.set_style("ticks")
plt.show()


# #### 11 ) Daily calls every year on days of week basis.

# In[ ]:


plt.figure(figsize=(14,7))
sns.set_context("paper", font_scale = 2)
sns.countplot(x= "Day of Week", data= df, palette="cubehelix", hue= "Year" )     
plt.title(" Daily Calls By Year ")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.set_style("white")
plt.show()


# #### 12 ) Daily calls on every days of the week with Reason basis.

# In[ ]:


plt.figure(figsize=(14,7))
sns.set_context("paper", font_scale = 2)
sns.countplot(x= "Day of Week", data= df, palette="autumn", hue= ("reason") )     
plt.title(" Day Calls By Reason ")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.set_style("darkgrid")
plt.show()


# #### 13 ) Monthly call rates in each year

# In[ ]:


plt.figure(figsize = (14,7))

sns.set_context("paper", font_scale=2)
sns.countplot(data= df, x= "Month", hue= "Year", palette="gist_earth")

plt.title(" Monthly Calls Yearly")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.set_style("ticks")
plt.show()


# #### 14 ) Monthly calls category combined all years 

# In[ ]:


plt.figure(figsize=(14,7))
sns.set_context("paper", font_scale = 2)
sns.countplot(x= "Month", data= df, palette="copper", hue= "reason")
plt.title(" Monthly Calls Category Combined All Years")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.set_style("darkgrid")
plt.show()


# #### 15 ) top Cases registered in detail reasons..

# In[ ]:


df['Detail reason'].value_counts().head(20)


# #### 16 ) Count plot for Cases registered. / Top 10 Cases registered.

# In[ ]:


plt.figure(figsize=(25,10))
sns.set_context("paper", font_scale = 2)
sns.countplot(y='Detail reason', data=df, palette="bright", order=df['Detail reason'].value_counts().index[:20])
plt.title("Top 10 Cases registered")
sns.set_style("darkgrid")
plt.show()


# #### 17 ) top 5 zipcodes for 911 calls

# In[ ]:


#df.groupby(['zip']).nunique()
df.zip.value_counts().head(5)


# #### 18 ) Count plot for Top 20 zip code in 911 emergnecy .

# In[ ]:


plt.figure(figsize=(25,10))
sns.set_context("paper", font_scale = 2)
sns.countplot(x='zip', data=df, palette="seismic_r", order=df['zip'].value_counts().index[:20])
plt.title("top 5 zipcodes for 911 calls")
sns.set_style("darkgrid")
plt.show()


# #### 19 ) Count plot for Top 20 zip code in 911 emergnecy with hue as reason

# In[ ]:


plt.figure(figsize=(25,10))
sns.set_context("paper", font_scale = 2)
sns.countplot(x='zip', data=df, palette="seismic_r", order=df['zip'].value_counts().index[:20],hue='reason')
plt.title("top 5 zipcodes for 911 calls")
sns.set_style("darkgrid")
plt.show()


# #### 20 ) Count plot for Top 20 township in 911 emergnecy .

# In[ ]:


df.twp.value_counts().head(10)


# #### 21 ) visulization plot top 20 township for 911 calls

# In[ ]:


plt.figure(figsize=(25,10))
sns.set_context("paper", font_scale = 2)
sns.countplot(y='twp', data=df, palette="spring", order=df['twp'].value_counts().index[:20])
plt.title("top 20 township for 911 calls")
sns.set_style("darkgrid")
plt.show()


# #### 22 ) top 20 township for 911 calls with Hue Reason

# In[ ]:


plt.figure(figsize=(25,20))
sns.set_context("paper", font_scale = 2)
sns.countplot(y='twp', data=df, palette="gist_heat", order=df['twp'].value_counts().index[:10],hue='reason')
plt.title("top 20 township for 911 calls with Hue Reason")
sns.set_style("darkgrid")
plt.show()


# #### 23 ) Top 5 township and with count of Cases registered

# In[ ]:


plt.figure(figsize=(25,100))
sns.set_context("paper", font_scale = 2)
sns.countplot(y='twp', data=df, palette="bright", order=df['twp'].value_counts().index[:5],hue='Detail reason')
plt.title("Top 10 Cases registered")
sns.set_style("darkgrid")
plt.show()


# # Creating a GeoDataFrame from a DataFrame with coordinates

# ### 24 ) A GeoDataFrame needs a shapely object, so we create a new column Coordinates as a tuple of Longitude and Latitude :

# In[ ]:


df['Coordinates'] = list(zip(df.lng, df.lat))


# Then, we transform tuples to Point :

# In[ ]:


df['Coordinates'] = df['Coordinates'].apply(Point)


# Now, we can create the GeoDataFrame by setting geometry with the coordinates created previously.

# In[ ]:


gdf = geopandas.GeoDataFrame(df, geometry='Coordinates')


# now we have a new column called Co-ordinate, with all points

# In[ ]:


gdf.head()


# Finally, we plot the coordinates over a country-level map.

# In[ ]:


world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

# We restrict to South America.
ax = world[world.continent == 'North America'].plot(
    color='white', edgecolor='black')

# We can now plot our GeoDataFrame.
gdf.plot(ax=ax, color='red')
plt.show()


# ### 25 ) Plotting the same graph using Plotly library. 

# - plotly maps are interactive based map, where you can zoom at the given co-ordinates
# - here we are using only top 500 values, since data is not spreaded apart

# In[ ]:


import pandas as pd
from  plotly.offline import plot
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [ dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df['lng'].head(500),
        lat = df['lat'].head(500),
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            colorbar=dict(
                title="Coordinates points "
            )
        ))]

layout = dict(
        title = '911 calls Location <br>(Hover for co ordinate names)',
        colorbar = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )

iplot( fig, validate=False, filename='d3-airports' )


# In[ ]:





# # Heat Maps:

# - creating heatmaps with seaborn and 911 data.
# - first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week.
# - combine groupby with an unstack method.

# In[ ]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['reason'].unstack()
dayHour.head()


# #### 26 ) Creating Heat hap

# In[ ]:


plt.figure(figsize=(8,4))
sns.heatmap(dayHour,cmap='inferno')
plt.show()


# #### 27 ) Creating Clustermap with the given data

# In[ ]:


plt.figure(figsize=(8,8))
sns.clustermap(dayHour,cmap='inferno_r')
plt.show()


# #### 27 ) same plots and operations, for a DataFrame that shows the Month as the column

# In[ ]:


dayMonth = df.groupby(by=['Day of Week','Month']).count()['reason'].unstack()
dayMonth.head()


# #### 28 ) Creating Heat hap with months and days of week

# In[ ]:


sns.heatmap(dayMonth,cmap='Oranges')


# In[ ]:


sns.clustermap(dayMonth,cmap='Purples')


# In[ ]:





# In[ ]:





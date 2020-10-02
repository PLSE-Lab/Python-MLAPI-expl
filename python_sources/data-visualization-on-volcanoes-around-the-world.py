#!/usr/bin/env python
# coding: utf-8

# **Load Libraries**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as animation
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')


# **Loading Volcano Dataset**

# In[2]:


# Input data files are available in the "../input/" directory.
volcanoes = pd.read_csv("../input/volcanic-eruptions/database.csv")


# In[3]:


#List of Columns on Valcano dataset
volcanoes.columns


# In[4]:


#list of first 5 rows
volcanoes.head()


# **Loading Earthquake Dataset**

# In[5]:


eq = pd.read_csv("../input/earthquake-database/database.csv")


# In[6]:


eq.columns


# In[7]:


#list of first 5 rows
eq.head()


# **Data Cleaning**
# 
# The next step is to clean the data. I decided to take the classification for granted and simply remove question marks. Also, use one token for each type and change the alternative spellings accordingly. Finally remove superfluous whitespace and start all words with capital letter.

# In[8]:


#Changing the Column name just for easier handling purpose (ofcourse its not mandatory!)
volcanoes = volcanoes.rename(columns={'Elevation (Meters)': 'Elevation'})
volcanoes.head(2)


# In[9]:


#Check the types of valcanoes thats existing
volcanoes['Type'].value_counts() 


# From the above list, we see that a single type of valcano for eg: if you could take Stratovolcano, its available with different names of Stratovolcano? ,Stratovolcano(es). Lets try grouping these together by assigning a single token as "Stratovolcano" to reference all these types.

# In[10]:


def cleanup_type(s):
    if not isinstance(s, str):
        return s
    s = s.replace('?', '').replace('  ', ' ')
    s = s.replace('Stratovolcano(es)', 'Stratovolcano')
    s = s.replace('Shield(s)', 'Shield')
    s = s.replace('Submarine(es)', 'Submarine')
    s = s.replace('Pyroclastic cone(s)', 'Pyroclastic cone')
    s = s.replace('Volcanic field(s)', 'Volcanic field')
    s = s.replace('Caldera(s)', 'Caldera')
    s = s.replace('Complex(es)', 'Complex')
    s = s.replace('Lava dome(s)', 'Lava dome')
    s = s.replace('Maar(s)', 'Maar')
    s = s.replace('Tuff cone(s)', 'Tuff cone')
    s = s.replace('Tuff ring(s)', 'Tuff ring')
    s = s.replace('Fissure vent(s)', 'Fissure vent')
    s = s.replace('Lava cone(s)', 'Lava cone')
    return s.strip().title()

volcanoes['Type'] = volcanoes['Type'].map(cleanup_type)
volcanoes['Type'].value_counts() 


# In[11]:


#Now, lets check for any null values present
volcanoes.isnull().sum()


# Now let's get rid of incomplete records.

# In[12]:


volcanoes.dropna(inplace=True)
len(volcanoes)


# Volcanoes will be plotted as red triangles, whose sizes depend on the elevation values, that's why I'll only consider positive elevations, i. e. remove submarine volcanoes from the data frame.

# In[13]:


# A general check on why we are removing submarine alone..
volcanoes[volcanoes['Type'] == 'Submarine'].head()


# In[14]:


volcanoes = volcanoes[volcanoes['Elevation'] >= 0]
len(volcanoes)


# **Map Creation**
# 
# Next I define a function that will plot a volcano map for the given parameters. Lists of longitudes, latitudes and elevations, that all need to have the same lengths, are mandatory, the other parameters have defaults set.
# 
# As mentioned above, sizes correspond to elevations, i. e. a higher volcano will be plotted as a larger triangle. To achieve this the 1st line in the plot_map function creates an array of bins and the 2nd line maps the individual elevation values to these bins.
# 
# Next a Basemap object is created, coastlines and boundaries will be drawn and continents filled in the given color. Then the volcanoes are plotted. The 3rd parameter of the plot method is set to ^r, the circumflex stands for triangle and r for red. For more details, see the documentation for plot.
# 
# The Basemap object will be returned so it can be manipulated after the function finishes and before the map is plotted.

# In[15]:


def plot_map(Longitude , Latitude, Elevation, projection='mill', llcrnrlat=-80, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='i', min_marker_size=2):
    bins = np.linspace(0, Elevation.max(), 10)
    marker_sizes = np.digitize(Elevation, bins) + min_marker_size

    m = Basemap(projection=projection, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution=resolution)
    m.drawcoastlines()
    m.drawmapboundary()
    m.fillcontinents(color = '#333333')

    for lon, lat, msize in zip(Longitude , Latitude, marker_sizes):
        x, y = m(lon, lat)
        m.plot(x, y, '^r', markersize=msize, alpha=.7)

    return m


# Now Lets View the entire list of **Vocanoes of the world**

# In[19]:


from mpl_toolkits.basemap import Basemap
plt.figure(figsize=(16, 8))
plot_map(volcanoes['Longitude'], volcanoes['Latitude'], volcanoes['Elevation'])
plt.title('Volcanoes of the World', color='#000000', fontsize=20)


# Now, lets try extracting a particular type of Volcano which gives us a maximum of count
# 
# **Map of Stratovolcanos**
# 
#  Let us analyse this particular type of Volcano. The below picture provides us the overview of distribution of  all 722 stratovolcanoes over the world.

# In[20]:


from mpl_toolkits.basemap import Basemap
plt.figure(figsize=(16, 8))
Vol = volcanoes[volcanoes['Type'] == 'Stratovolcano']
plot_map(Vol['Longitude'], Vol['Latitude'], Vol['Elevation'])


# Inference: We can clearly see that most Stratovolcanoes are located, where tectonic plates meet. 
# So, Let's now dig into research on some of these regions where it is widely located.

# Visually, we can see the presence of these Volcanoes more towards the coast of North America, South America, Japan, Indonesia and Phillipines. So, let us now take the region of North America and find out its lowest and highest peak for all  Stratovolcanoes present there.

# **Volcanoes of North America**
# 
# We are extracting the Stratovolcanoes for the region of North America alone. 

# In[21]:


#List of Stratovalcanoes present in US.
Vol_US = volcanoes.loc[(volcanoes['Country'] == 'United States') & (volcanoes['Type'] == 'Stratovolcano')]
Vol_US.head()


# In[22]:


len(Vol_US)


# In[23]:


# Finding the highest peak among the list
Vol_US['Elevation'].max()


# In[24]:


Vol_US.loc[Vol_US['Elevation'] == 5005]


# In[25]:


# Finding the lowest peak among the list
Vol_US['Elevation'].min()


# In[26]:


Vol_US.loc[Vol_US['Elevation'] == 0]


# In[27]:


# Displaying the two particular rows of min and Max values
Vol_US = Vol_US.loc[(675,967),:]
Vol_US


# **Plotting the highest and lowest peak in US for the type = Stratovolcano**

# In[28]:


plt.figure(figsize=(12, 10))
plot_map(Vol_US['Longitude'], Vol_US['Latitude'],Vol_US['Elevation'],min_marker_size=10)


# The next map shows entire list of all North American volcanoes in the data frame.

# In[29]:


plt.figure(figsize=(12, 10))
plot_map(volcanoes['Longitude'], volcanoes['Latitude'],volcanoes['Elevation'],
         llcrnrlat=5.5, urcrnrlat=83.2, llcrnrlon=-180, urcrnrlon=-52.3, min_marker_size=4)


# Inference: We can clearly see that most Stratovolcanoes are located, where tectonic plates meet.
# 
# Now, lets try to find a relationship between earthquake and volcano presence.  

# In[30]:


m = Basemap(projection='mill',llcrnrlat=5.5, urcrnrlat=83.2, llcrnrlon=-180, urcrnrlon=-52.3,resolution='c')
fig = plt.figure(figsize=(12,10))

longitudes_vol = volcanoes['Longitude'].tolist()
latitudes_vol = volcanoes['Latitude'].tolist()

longitudes_eq = eq['Longitude'].tolist()
latitudes_eq = eq['Latitude'].tolist()

x,y = m(longitudes_vol,latitudes_vol)
a,b= m(longitudes_eq,latitudes_eq)

plt.title("Volcanos areas (red) Earthquakes (green)", color='#000000', fontsize=20)
m.plot(x, y, '^r', markersize = 5, color = 'red')
m.plot(a, b, "o", markersize = 3, color = 'green')

m.drawcoastlines()
m.drawcountries()
#m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.show()


# Some, but not all, earthquakes are related to volcanoes. For example, most earthquakes are along the edges of tectonic plates. This is where most volcanoes are too. However, most earthquakes are caused by the interaction of the plates not the movement of magma. 

# **Casuse of Earthquake**: Most earthquakes directly beneath a volcano are caused by the movement of magma. The magma exerts pressure on the rocks until it cracks the rock. Then the magma squirts into the crack and starts building pressure again. Every time the rock cracks it makes a small earthquake.  This proves that there is a evident relation betwen volcanoes and Earthquake.

# **Stratovolcanoes Volcanoes of South America**

# In[31]:


plt.figure(figsize=(12, 10))
plot_map(volcanoes['Longitude'], volcanoes['Latitude'], volcanoes['Elevation'],
         llcrnrlat=-57, urcrnrlat=15, llcrnrlon=-87, urcrnrlon=-32, min_marker_size=4)


# **Stratovolcanoes Volcanoes of Indonesia**
# 
# Another region with many volcanoes is the Indonesian archipelago. Some of them like the Krakatoa and Mount Tambora have undergone catastrophic eruptions with tens of thousands of victims in the past 200 years.

# In[32]:


plt.figure(figsize=(18, 8))
plot_map(volcanoes['Longitude'], volcanoes['Latitude'], volcanoes['Elevation'],
         llcrnrlat=-11.1, urcrnrlat=6.1, llcrnrlon=95, urcrnrlon=141.1, min_marker_size=4)


# **Earthquakes in the world and its affected regions**

# In[33]:


m1 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
fig = plt.figure(figsize=(12,10))

longitudes_eq = eq['Longitude'].tolist()
latitudes_eq = eq['Latitude'].tolist()
k,j= m1(longitudes_eq,latitudes_eq)

plt.title("Earthquake of the World", color='#000000', fontsize=20)
m1.plot(k, j, "o", markersize = 3, color = 'green')

m1.drawcoastlines()
m1.drawcountries()
m1.drawmapboundary()
plt.show()


# **Ring of Fire Analysis**

# It seems that majority of earthquakes are concentrated in the Indonesian, Sino Pacific and the Japanese area. Why is this so? Before we look at magnitude of earthquakes and how it relates to the geographical distribution of significant earthqauakes, let's try to answer this question. According to National Geographic, the Pacific Ring of Fire, technically called the Circum-Pacific belt, is the world's greatest earthquake belt, according to the U.S. Geological Survey (USGS), due to its series of fault lines stretching 25,000 miles (40,000 kilometers) from Chile in the Western Hemisphere through Japan and Southeast Asia. The magazine states that
# 
# * Roughly 90 percent of all the world's earthquakes, and 80 percent of the world's largest earthquakes, strike along the Ring of Fire
# Is statistics true? Let's find out!
# 
# (Refer to: [https://earthquake.usgs.gov/learn/facts.php] for more details)

# In[34]:


#Selecting the necessary columns for Analysis
eq_analysis = eq[["Date", "Time", "Latitude","Longitude","Magnitude", "Depth"]]
len(eq_analysis)


# In[35]:


eq_analysis.head()


# In[36]:


m2 = Basemap(projection='mill')
rof_lat = [-61.270, 56.632]
rof_long = [-70, 120]
ringoffire = eq_analysis[((eq_analysis.Latitude < rof_lat[1]) & 
                    (eq_analysis.Latitude > rof_lat[0]) & 
                     ~((eq_analysis.Longitude < rof_long[1]) & 
                       (eq_analysis.Longitude > rof_long[0])))]
f,g = m2([longs for longs in ringoffire["Longitude"]],
         [lats for lats in ringoffire["Latitude"]])
fig3 = plt.figure(figsize=(20,20))
plt.title("Earthquakes in the Ring of Fire Area")
m2.plot(f,g, "o", markersize = 3, color = 'green')
m2.drawcoastlines()
m2.drawmapboundary()
m2.drawcountries()
m2.fillcontinents(color='lightsteelblue',lake_color='skyblue')

plt.show()


# In[37]:


print("Total number of data on world's Earthquakes:", len(eq_analysis))
print("Total number of Earthquakes in the Ring of Fire Area:",len(ringoffire))


# Inference: When comparing the Earthquakes in the Ring of Fire Area map and Earthquakes of the world, around 75% of World's earthquakes are on the rings of fire region..

# **Where was the Maximum / Minimum Hit? ** 
# Lets find out this by measuring the magnitude of the earth quake.

# In[38]:


eq_max = eq_analysis['Magnitude'].max()
eq_min = eq_analysis['Magnitude'].min()
print (eq_max)
print (eq_min)


# In[39]:


max_eq = eq_analysis.loc[eq_analysis['Magnitude'] == 9.1] 
len(max_eq)


# In[40]:


min_eq = eq_analysis.loc[eq_analysis['Magnitude'] == 5.5]
len(min_eq)


# In[42]:


#Scroll in find the two most horrible earthquakes being hit.
import folium
map = folium.Map(location = [eq_analysis['Latitude'].mean(), eq_analysis['Longitude'].mean()], zoom_start = 4, tiles = 'Mapbox Bright' )
folium.Marker(
    location=[3.295, 95.982],
    popup='Indonesia',
    icon=folium.Icon(icon='cloud')
).add_to(map)
folium.Marker(
    location=[38.297, 142.373],
    popup='Japan',
    icon=folium.Icon(color='green')
).add_to(map)

map


# Let's consider the major earthquake disaster zone-  Japan. 

# In[43]:


eq_Japan = eq[eq['Location Source'] == 'Japan'] 
eq_Japan_list = eq.loc[:,('Latitude','Longitude')]


# Refer to : [http://www.scientificamerican.com/article/japan-s-volcanoes-made-more-jittery-by-2011-quake/]
# 
# 
# Japan may well be moving into a period of increased volcanic activity touched off by the 9.0 magnitude earthquake of March 11, 2011, said Toshitsugu Fujii, a volcanologist and professor emeritus at the University of Tokyo.
# 
# According to him, "The 2011 quake convulsed all of underground Japan quite sharply, and due to that influence Japan's volcanoes may also become much more active. It has been much too quiet here over the last century, so we can reasonably expect that there will be a number of large eruptions in the near future."
# 
# So, We will view the distribution of Volcanoes in this region and a histogram view of its occurences to understand how far the statement is true.

# In[44]:


Vol_Japan = volcanoes.loc[volcanoes['Country'] == 'Japan'] 
Vol_Japan_list = Vol_Japan.loc[:,('Latitude','Longitude')]
len(Vol_Japan)


# In[45]:


plt.figure(figsize=(18, 8))
plot_map(Vol_Japan['Longitude'], Vol_Japan['Latitude'], Vol_Japan['Elevation'],
         llcrnrlat=25, urcrnrlat=46, llcrnrlon=125, urcrnrlon=150, min_marker_size=4)


# Hence, its eveident that Earthquakes of magnitude 9.0 or greater around the world have led to repeated volcanic eruptions in the last 50 years, sometimes within days.

# Inspired by one of the tutorial I had come across, tried using the same code to our logic

# In[46]:


n = pd.read_csv("../input/earthquake-database/database.csv",encoding='ISO-8859-1')


# In[47]:


n=n[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
n.head()


# In[53]:


import time
#getting the year from Date Attribute
n['Year']= n['Date'].str[6:]

fig = plt.figure(figsize=(10, 10))
fig.text(.8, .3, 'L.Rajesh', ha='right')
cmap = plt.get_cmap('coolwarm')

m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='burlywood',lake_color='lightblue', zorder = 10)
m.drawmapboundary(fill_color='lightblue')


START_YEAR = 1965
LAST_YEAR = 2016

points = n[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']][n['Year']==str(START_YEAR)]

x, y= m(list(points['Longitude']), list(points['Latitude']))
scat = m.scatter(x, y, s = points['Magnitude']*points['Depth']*0.3, marker='o', alpha=0.3, zorder=10, cmap = cmap)
year_text = plt.text(-170, 80, str(START_YEAR),fontsize=15)
plt.title("Earthquake visualisation (1965 - 2016)")
plt.close()

start = time.time()
def update(frame_number):
    current_year = START_YEAR + (frame_number % (LAST_YEAR - START_YEAR + 1))
    year_text.set_text(str(current_year))
    points = n[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']][n['Year']==str(current_year)]
    x, y= m(list(points['Longitude']), list(points['Latitude']))
    color = points['Depth']*points['Magnitude'];
    scat.set_offsets(np.dstack((x, y)))
    scat.set_sizes(points['Magnitude']*points['Depth']*0.3)

ani = animation.FuncAnimation(fig, update, interval=500, repeat_delay=0, frames=LAST_YEAR - START_YEAR + 1,blit=False)
ani.save('animation.gif', writer='imagemagick') #, writer='imagemagick'
plt.show()

end = time.time()
print("Time taken by above cell is {}".format(end-start))
    
#ani = animation.FuncAnimation(fig, update, interval=750, frames=LAST_YEAR - START_YEAR + 1)
#ani.save('animation.gif', writer='imagemagick', fps=5)


# In[54]:


import io
import base64

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# In[ ]:





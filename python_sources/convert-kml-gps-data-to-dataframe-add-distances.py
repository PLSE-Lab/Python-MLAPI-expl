#!/usr/bin/env python
# coding: utf-8

# In this notebook i will read Gps file which comes as .kml ( xml ) and convert the  data to pandas Dataframe. Additionally, i will calculate distances between  gps point to point(Lat, Long)  and accumulated distance for whole dataset. 
# 
# Let's Begin !

# To be able to read and parse .kml file, i will use beautifulsoup lib

# In[ ]:


from bs4 import BeautifulSoup


# In[ ]:



infile = open(str("../input/utmb.kml"),"r")
contents = infile.read()


# To get coordinates from  .xml file between releated tag  
# 
# 
# 

# In[ ]:


soup = BeautifulSoup(contents,'xml')
titles = soup.find_all(str("LineString"))


# There could be more than one LineString tag in .kml file. Therefore, we need to merge data 

# In[ ]:


coor=[]
for title in titles:
            coor.append(title.get_text())


# At the moment, our data is just one string in corr list

# In[ ]:


coor[0][0:1000]


# To split text values by  space. If we look at our data we have three value seperated by "," and those sets separated eachother by space. To split text by space we will use shlex lib.

# In[ ]:


import shlex


# In[ ]:


liste_0 =[]
for i in range(len(coor)):
    a = shlex.split(coor[i], posix=False)
    liste_0.append(a)


# Let's see our splited data in list

# In[ ]:


liste_0[0][0:10]


# Make it Pandas dataframe and transpose it for next steps.

# In[ ]:


import pandas as pd


# In[ ]:


df_0 = pd.DataFrame(liste_0).transpose()


# Now, we have dataframe with one column. each row has a string data. actually "," in this string is a sperator first part is represents Lat second represents Long and Last one represent Alt. (see  Lat Long Alt https://en.wikipedia.org/wiki/Geographic_coordinate_system )

# In[ ]:


df_0.head(10)


# 
# To split Values by ","

# In[ ]:


df_0.columns = ["column"]
df_0[['Lat', 'Long', 'Alt']] = df_0.column.str.split(",",expand=True)


# In[ ]:


df_0.head(10)


# To drop first column and change the other columns' data type to float from string

# In[ ]:


df_0 = df_0[['Lat', 'Long', 'Alt']].astype({"Lat": float , "Long": float,"Alt":float})


# In[ ]:


df_0.head(10)


# To eliminate singular 0 values from altitude data. But, if our dataset doesn't include altitude values, it means we have whole column with 0 values for that case i don't want to drop zeros.

# In[ ]:


if df_0.Alt.sum() < 15:
    df_0 = df_0.dropna()
    df_0 = df_0.reset_index(drop=True)
else:
    df_0 = df_0.dropna()
    df_0 = df_0[~(df_0 == 0).any(axis=1)]
    df_0 = df_0.reset_index(drop=True)


# In[ ]:


df_0.head(10)


# Now, our dataset ready to add it distances

# In[ ]:


from math import sin, cos, sqrt, atan2, radians


# In[ ]:


distances = []
for i in range(len(df_0)):
    if i == 0:
        distances.append(0)
    else:
        lat1 = radians(df_0.Lat[i])
        lon1 = radians(df_0.Long[i])
        lat2 = radians(df_0.Lat[i-1])
        lon2 = radians(df_0.Long[i-1])

        dlon = lon1 - lon2
        dlat = lat1 - lat2
#to calculate distace between two points on the earth
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        radius_of_world = 6371 #kms
        distance = radius_of_world * c

        distances.append(distance*1000)
distances = pd.DataFrame(distances)
c_distances = distances.cumsum()
distances.columns = ["Dist"]
c_distances.columns = ["Cdist"]
dframe=pd.concat([df_0, distances,c_distances], axis=1)


# Now, we have distance datas

# In[ ]:


dframe.head(10)


# Let's import matplotlib and plot the values in line plot as a topo of gps data

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.xlabel('Distance meters')
plt.ylabel('Altitude meters')
plt.title('Route Topo')
plt.plot(dframe.Cdist, dframe.Alt, "r-" )


# In[ ]:


plt.scatter(dframe.Lat,dframe.Long)


# In[ ]:





# In[ ]:





# In[ ]:





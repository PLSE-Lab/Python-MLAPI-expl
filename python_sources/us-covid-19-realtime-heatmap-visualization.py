#!/usr/bin/env python
# coding: utf-8

# # US Covid-19 Realtime Heatmap Visualization
# ## Sources
# - us geojson : https://github.com/PublicaMundi/MappingAPI/blob/master/data/geojson/us-states.json
# - github : https://github.com/andy971022/us-covid-19-visualization
# 
# ## Final Output
# <img src="./uscovid19/us_covid-19_visualization_cases.gif">
# 
# ## My Personal Blog : [http://andy971022.com/](http://andy971022.com/)

# In[ ]:


get_ipython().system('cp -r ../input/uscovid19 .')


# The following imports our necessary packages

# In[ ]:


import pandas as pd
import pandasql as ps
import json
from folium import Map, LayerControl, Choropleth
import numpy
import os
from PIL import Image
import glob
import subprocess


# This function creates a map object  

# In[ ]:


def plot_map(date, df):
    maximum = df[df["date"] == date]["cases"].max()
    m = Map(location = [37,-98],
        zoom_start = 4)
    Choropleth(geo_data = GEOJSON_PATH + GEOJSON_FILE,
                name = 'choropleth',
                data = df[df["date"] == date],
                columns = ['state','cases'],
                key_on = 'feature.properties.name',
                fill_opacity = 0.7,
                line_opacity = 0.2,
                line_color = 'red',
                fill_color = 'YlOrRd',
                bins = [0,2,4,6,9,12],
              ).add_to(m)
    LayerControl().add_to(m)
    return m


# We then instantiate our variables

# In[ ]:


GEOJSON_PATH = "./uscovid19/GeoJson/"
CSV_DIRECTORY = "../input/us-counties-covid-19-dataset/"
FILE_NAME = "us-counties.csv"
GEOJSON_FILE = "us-states.json"
df = pd.read_csv(f"{CSV_DIRECTORY}{FILE_NAME}")


# We define an SQL query via string and run it using pandasql.
# 
# It basically treats the dataframe as a sql table and runs the defined query.
# 
# For visualizing conveniences, we convert the scale into a logarithmic scale by applying the natural log to the number of cases.

# In[ ]:



q1 = """
select distinct date, state, sum(cases) as cases from df group by date, state
"""

cases_df = ps.sqldf(q1, locals())

cases_df["cases"] = cases_df["cases"].apply(lambda x : numpy.log(x))
print(cases_df.columns)


# We read the GeoJson file to get the boudaries of the states in The US.

# In[ ]:


gj = json.load(open(f"{GEOJSON_PATH}{GEOJSON_FILE}"))


# We then plot the data onto the map and save the map into an html file

# In[ ]:


for date in cases_df[cases_df['date'] > '2020-03-01'].date.unique():
    maps = plot_map(f"{date}", cases_df)
    maps.save(f"./uscovid19/Maps/{date}.html")


# The following command simply loops through all files saved under `Maps`, opens them, and takes a screenshot.
# 
# `sudo apt-get install cutycapt` installs the software.

# In[ ]:


# for file in os.listdir(f"./Maps/"):
#     command = f"cutycapt --url=file://{os.getcwd()}/Maps/{file} --out=./images/{file.split('.')[0]}.png --delay=1000"
#     subprocess.run(command.split(" "))


# The following collects all screenshots from the previous results and makes a gif image out of it.

# In[ ]:


# pip install pillow==6.2.2

files = sorted(glob.glob('./uscovid19/images/*.png'))
images = list(map(lambda file: Image.open(file), files))
images[0].save('us_covid-19_visualization_cases.gif', save_all=True, append_images=images[1:], duration=120, loop=0)


# <img src="./uscovid19/us_covid-19_visualization_cases.gif">

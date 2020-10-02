#!/usr/bin/env python
# coding: utf-8

# # -----------Visualizing Drought by US County------------
# # ---------------------------------------------------------------

# 
# This data is provided for open use at [this kaggle page.](https://www.kaggle.com/us-drought-monitor/united-states-droughts-by-county/home)
# The purpose of this project is to gain experience with Geospatial analysis and visualization libraries.
# 
# I was first inspired to work on this after seeing kaggle user [@andresionek's](https://www.kaggle.com/andresionek) project analyzing Brazilian E-commerce. Check it out. It's very stylish. I also was able to mimick features from [Holoview's County Mapping Example](https://raw.githubusercontent.com/ioam/holoviews/master/examples/gallery/demos/matplotlib/texas_choropleth_example.ipynb) and Mat Leonard's [Udacity Medium Post](https://medium.com/udacity/creating-map-animations-with-python-97e24040f17b) on moving maps. 

# # Importing Libraries
# First, let's import the plotting library Holoviews, as well as some color schemes, display abilities. We'll also import cv2 for combining our images and we'll import widgets for an interactive section. 

# In[ ]:


#plotting
import holoviews as hv
import holoviews.plotting.mpl
hv.extension('matplotlib')
from colorcet import fire, rainbow
from IPython.display import display_png
import bokeh
#video generation
import cv2
import os
get_ipython().run_line_magic('output', "fig='png'")
#widget creation
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


# Then let's import pandas and do a quick fix to stop matplotlib's deprecation warnings

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# # Importing and Cleaning Data
# We'll take the polygons of US counties from the bokeh library, and we'll import the dataset from the USDM. Let's look at the USDM data. Don't worry about seeing the county data yet because you'll see it a few sections down.

# In[ ]:


import json
with open('../input/bokeh-counties/counties.json') as f:
    counties= json.load(f)
drought=pd.read_csv("../input/united-states-droughts-by-county/us-droughts.csv", encoding='latin1')
drought.head()


# First note that county polygons were imported as a dict where the key is the county code (FIPS code).
# 
# FIPS codes are strings in one dataset but integers (third column) in the other... so we must make the county codes uniform. 

# In[ ]:


counties_int_key={}
for key in counties.keys():
    counties_int_key[int(key)]=counties[key]
counties=counties_int_key


# The reporting on drought level is cumulative so I simply add them together to get a statistic. Description of drought statistics are at [this kaggle page](https://www.kaggle.com/us-drought-monitor/united-states-droughts-by-county/home) for the curious.

# In[ ]:


drought["drought_level"]=drought["D0"]/100+drought["D1"]/100+drought["D2"]/100+drought["D3"]/100+drought["D4"]/100
drought.head()


# We consolidate the entries by month and use the highest week's drought index. Doing so gives us a Pandas Series which we name drought_monthly. Let's take a look at what the head of drought_monthly looks like.

# In[ ]:


drought_monthly=drought[["FIPS","validEnd","drought_level"]]
drought_monthly.validEnd=pd.to_datetime(drought_monthly.validEnd)
drought_monthly.validEnd=drought_monthly.validEnd.dt.to_period("M")
drought_monthly=drought_monthly.groupby(["FIPS","validEnd"]).drought_level.max()
drought_monthly.head()


# We convert counties into a Pandas Dataframe so we can add the drought index of a certain month as a column.

# In[ ]:


counties_drought=pd.DataFrame.from_dict(counties, orient='index')
for year in range(2010,2016):
    for month in range(1,13):
        if month<10:
            month="0"+str(month)
        date=str(year)+"-"+str(month)
        counties_drought[date]=0
        for cid in counties_drought.index:
            if (cid,date) in drought_monthly:
                counties_drought.loc[cid,date]=drought_monthly[(cid,date)]


# In[ ]:


counties_drought.head()


# ## Making a still map
# 1) We are going to make a moving/video map for the continental US later. But let's get a still map first. The dataset has all states and territories, so let's drop some rows.     
# 2) A Holoviews [Polygons Object](http://holoviews.org/reference/elements/matplotlib/Polygons.html) doesn't accept a Pandas Dataframe. We'll input a list with internal dictionaries to comply with the software. Printing a single entry can show well the structure of our list.   
# 3) Let's define our time frame as this decade. Since the data cuts off in 2016 we'll use all of 2010 through 2015, inclusive. We'll store the date strings in a list so we can iterate through them later and access list elements. 

# In[ ]:


#1 
counties_drought_continental=counties_drought[~counties_drought["state"].isin(['hi', 'ha','ak','pr','gu','mp','as','vi'])]
#2
counties_drought_dict=counties_drought_continental.to_dict("index")
counties_drought_list = [dict(county)for cid, county in counties_drought_dict.items()]
#3
dates_list=[]
for year in range(2010,2016):
    for month in range(1,13):
        if month<10:
            month="0"+str(month)
        date=str(year)+"-"+str(month)
        dates_list.append(date)


# Now we plot the thing! If you run this yourself it might take 20 seconds. The rainbow color palette goes from blue to red which makes it a nice visualization tool for drought. We'll plot January, 2010 first.

# In[ ]:


cbar_tick_labels=[(0,'Not in drought'),(1,'Abnormally dry conditions'),(2,'Moderate Drought'),(3,'Severe Drought'),(4,'Extreme Drought'),(5,'Exceptional Drought')]
polygons = hv.Polygons(counties_drought_list, ['lons', 'lats'], vdims=hv.Dimension(dates_list[1], range=(0, 5)), label="Continental US Drought Index")
polygons.options(logz=False, xaxis=None, yaxis=None, cbar_ticks=cbar_tick_labels,
                   show_grid=False, show_frame=False, colorbar=True, color_index=dates_list[1],
                   fig_size=500, edgecolor='black', cmap=list(rainbow))


# # Letting the User Choose a map
# Let the user specify an input for a map to be made. We'll import some widget libraries to help with that.     
# To change the values of the widgets, you must fork your own version of the notebook. 

# In[ ]:


from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


# In[ ]:


month_input=widgets.IntSlider(
    value=7,
    min=1,
    max=12,
    step=1,
    description='Month:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)
year_input=widgets.IntSlider(
    value=2012,
    min=2010,
    max=2015,
    step=1,
    description='Year:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)
state_abbrevs=counties_drought[~counties_drought["state"].isin(['ha','pr','gu','mp','as','vi'])].state.unique()
state_input=widgets.Dropdown(
    options=state_abbrevs,
    value='ca',
    description='State:',
    disabled=False,
    continuous_update=False
)
display(month_input)
display(year_input)
display(state_input)


# Now, running the following code we can quickly generate a snapshot map for any state!

# In[ ]:


if month_input.value<10:
    month_input_str="0"+str(month_input.value)
    date=str(year_input.value)+"-"+str(month_input_str)
else:
    date=str(year_input.value)+"-"+str(month_input.value)
counties_drought_user_input=counties_drought[counties_drought["state"].isin([state_input.value])]
counties_drought_user_input.head()
counties_drought_user_dict=counties_drought_user_input.to_dict("index")
counties_drought_user_list = [dict(county)for cid, county in counties_drought_user_dict.items()]
cbar_tick_labels=[(0,'Not in drought'),(1,'Abnormally dry conditions'),(2,'Moderate Drought'),(3,'Severe Drought'),(4,'Extreme Drought'),(5,'Exceptional Drought')]
polygons = hv.Polygons(counties_drought_user_list, ['lons', 'lats'], vdims=hv.Dimension(date, range=(0, 5)), label="User Specific Map: "+state_input.value+" "+date)
polygons.options(logz=False, xaxis=None, yaxis=None, cbar_ticks=cbar_tick_labels,
                   show_grid=False, show_frame=False, colorbar=True, color_index=date,
                   fig_size=500, edgecolor='black', cmap=list(rainbow))


# # Making a moving map
# We'll loop throught the data, changing the dimension used to color the image (vdims), and rendering the images as pngs. Unfortunately kaggle doesn't support writing to the kaggle environment. I will hopefully edit this soon with a workaround. Tinkering with the code and putting it in a directory in your own computer would work. 

# In[ ]:


plots={}
renderer = hv.plotting.mpl.MPLRenderer.instance(dpi=120)
for date in dates_list:
    print(date)
    polygons.vdims=[hv.Dimension(date, range=(0, 5))]
    plots[date]=polygons.options(logz=False, xaxis=None, yaxis=None, cbar_ticks=cbar_tick_labels,
                   show_grid=False, show_frame=False, colorbar=True, color_index=date,
                   fig_size=500, edgecolor='black', cmap=list(rainbow))
for date in dates_list:
    path=""+date
    renderer.save(plots[date],path ,fmt='png')


# We put them together as a video saved locally below.

# In[ ]:


dir_path = ""
for f in os.listdir(""):
    if f.endswith("png"):
        images.append(f)
image_path = os.path.join("", images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape


# In[ ]:


fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter("vid.mp4", fourcc, 5.0, (width, height))


# In[ ]:


dir_path = ""
for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) 

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): 
        break
out.release()
cv2.destroyAllWindows()


# # We have video! ... (but it can't be played on Kaggle)
# If you want to see the video, download my drought video which is made public in the drought_video folder. It'll work on your computer. 
# 
# To play the video within python, you'll have to fork and copy the code to your local environment. From what I understand so far, kaggle cannot write and the mp4 player I am using has write access. I will update a fix to this soon hopefully. On your own, run the code block below and the video will load. If it loads too large in the window, click the maximize button in the bottom right corner of the video to see a full screen version. 
# 
# The video should also have been saved locally.

# In[ ]:


import base64
import io
from IPython.display import HTML
#video is hardcoded because within Kaggle reading and writing repeatedly is difficult
video = io.open('../input/drought-video/vid.mp4', 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii')))


# In[ ]:





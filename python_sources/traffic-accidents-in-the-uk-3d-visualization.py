#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy
import os
import pandas

import datetime as _dt
import pyproj as _proj

accidents_folder = "../input"


# ### Process accidents

# In[ ]:


accidents_09_11_df = pandas.read_csv(
    os.path.join(accidents_folder, 'accidents_2009_to_2011.csv'))


# In[ ]:


accidents_12_14_df = pandas.read_csv(
    os.path.join(accidents_folder, 'accidents_2012_to_2014.csv'))


# In[ ]:


accidents_df = pandas.concat((accidents_09_11_df, accidents_12_14_df))
accidents_df = accidents_df.reset_index()
accidents_df = accidents_df[["Latitude", "Longitude", "Date", "Time",
              "Day_of_Week", "Accident_Severity",
              "Number_of_Vehicles", "Number_of_Casualties"]].copy()


# In[ ]:


accidents_df.head()


# In[ ]:


# Convert to Web Mercator coordinate system.
src_prj = _proj.Proj("+init=EPSG:4326")
dst_prj = _proj.Proj("+init=EPSG:3857")

x_coords = accidents_df.Longitude.values
y_coords = accidents_df.Latitude.values
x_coords, y_coords = _proj.transform(src_prj, dst_prj, x_coords, y_coords)
accidents_df["x"] = x_coords
accidents_df["y"] = y_coords
accidents_df["z"] = 0


# In[ ]:


accidents_df.head()


# In[ ]:


def convert_datetime(t):
    day = int(t[:2])
    month = int(t[3:5])
    year = int(t[6:10])
    hour = int(t[11:13])
    minute = int(t[14:16])
    return _dt.datetime(year, month, day, hour, minute)

accidents_df.Time = accidents_df.Time.fillna("00:00")
accidents_df["Time"] = (accidents_df.Date + " " + accidents_df.Time).map(convert_datetime)
accidents_df["Time"] = accidents_df["Time"].astype("int64") / 1e9

accidents_df = accidents_df[
    ["x", "y", "z", "Time", "Day_of_Week", "Accident_Severity",
     "Number_of_Vehicles", "Number_of_Casualties"]].copy()


# In[ ]:


accidents_df.head()


# ## Note: The cells below will not run on the Kaggle kernel, as they require the [CityPhi](https://www.inrosoftware.com/cityphi) library, which is not part of the kernel image.
# 
# See [this page](https://imgur.com/a/5nu25) for example visuals produced with this notebook.

# ### Load the data into a CityPhi feature

# In[ ]:


import cityphi.application as _app
import cityphi.attribute as _att
import cityphi.layer as _layer
import cityphi.feature as _feat
import cityphi.widget as _widget


# In[ ]:


accident_feat = _feat.PointFeature(
    accidents_df.index.values,
    accidents_df[["x", "y", "z"]].values)


# In[ ]:


accident_feat.add_attribute(
    "time", "float64", accidents_df.index.values, accidents_df.Time.values)

for name in ["Day_of_Week", "Accident_Severity", "Number_of_Vehicles", "Number_of_Casualties"]:
    accident_feat.add_attribute(
        name, "int32", accidents_df.index.values, accidents_df[name].values)


# ### Launch the app and set up the visualization

# In[ ]:


get_ipython().run_line_magic('gui', 'cityphi')
app = _app.Application()
app.graphics_settings.ambient_occlusion = "SSAO_LOW"


# In[ ]:


accident_heatmap_layer = _layer.HeatmapLayer(accident_feat)

accident_heatmap_layer.color.value_range = 0, 40
accident_heatmap_layer.name = "Accident Heatmap"
accident_heatmap_layer.min_pixel_size = 8
accident_heatmap_layer.radius = 500
accident_heatmap_layer.start_time = _att.FeatureAttribute("time")
accident_heatmap_layer.end_time = _att.FeatureAttribute("time")

app.add_layer(accident_heatmap_layer)


# In[ ]:


app.set_view(app.full_view)


# In[ ]:


accident_layer = _layer.PointLayer(accident_feat.aggregate_grid(14))

accident_layer.name = "Accidents"
accident_layer.start_time = _att.FeatureAttribute("time")
accident_layer.end_time = _att.FeatureAttribute("time")
accident_layer.sort_order = _att.FeatureAttribute("Number_of_Casualties")
accident_layer.stacked = True
accident_layer.height = 200
accident_layer.radius = 400
accident_layer.min_pixel_size = 4

app.add_layer(accident_layer)


# In[ ]:


colors = [
    (228, 26, 28),
    (55, 126, 184),
    (77, 175, 74),
    (152, 78, 163),
    (255, 127, 0),
    (255, 255, 51),
    (166, 86, 40),
    (247, 129, 191)
]
labels = [u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8 and above']
accident_layer.color = _att.DiscreteColorAttribute(
    colors, _att.FeatureAttribute("Number_of_Casualties"), labels,
    values=[[i] for i in range(1, 9)])


# In[ ]:


def datetime_to_float(value):
    return pandas.Timestamp(value).value / 1.0e9

def float_to_datetime(value):
    return pandas.Timestamp(value * 1.0e9).to_pydatetime(warn=False)

def display_time(t):
    time = float_to_datetime(t)
#     return time.strftime("%a, %b %d, %Y %H:%M:%S")
    return time.strftime("%m/%d/%Y")


# In[ ]:


start_time = accident_feat.data().time.min()
end_time = accident_feat.data().time.max()

def change_time(t):
    accident_heatmap_layer.time_window = t - 365 * 24 * 60 * 60, t
#     accident_heatmap_layer.time_window = 0, t
    accident_layer.time_window = accident_heatmap_layer.time_window

time_slider = _widget.TimeSlider(start_time, end_time, display_time, change_time)
app.add_widget(time_slider)


# In[ ]:


time_slider.speed = 6 * 30 * 24 * 60 * 60


# In[ ]:


accident_legend = _widget.HTMLWidget(
    "Number of Casualties<br><br>" + accident_layer.color._repr_html_())
app.add_widget(accident_legend)


#!/usr/bin/env python
# coding: utf-8

# ## NYC  Taxi Fare - Time series

# In[ ]:


#Import Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os #operating system dependent modules of Python
import matplotlib.pyplot as plt #visualization
import seaborn as sns #visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
import warnings
warnings.filterwarnings("ignore")


# ## Data

# In[ ]:


#selecting 5 million rows
nyc_data  = pd.read_csv(r"../input/train.csv",nrows = 5000000)
nyc_data.head()


# ## Data Manipulation

# In[ ]:


#replace 0's in coordinates with null values
coord = ['pickup_longitude','pickup_latitude', 
         'dropoff_longitude', 'dropoff_latitude']

for i in coord :
    nyc_data[i] = nyc_data[i].replace(0,np.nan)
    nyc_data    = nyc_data[nyc_data[i].notnull()]

#Date manipulation
#conver to date format
nyc_data["pickup_datetime"] = nyc_data["pickup_datetime"].str.replace(" UTC","")
nyc_data["pickup_datetime"] = pd.to_datetime(nyc_data["pickup_datetime"],
                                             format="%Y-%m-%d %H:%M:%S")
#extract year
nyc_data["year"]  = pd.DatetimeIndex(nyc_data["pickup_datetime"]).year
#extract month
nyc_data["month"] = pd.DatetimeIndex(nyc_data["pickup_datetime"]).month
nyc_data["month_name"] = nyc_data["month"].map({1:"JAN",2:"FEB",3:"MAR",
                                                4:"APR",5:"MAY",6:"JUN",
                                                7:"JUL",8:"AUG",9:"SEP",
                                                10:"OCT",11:"NOV",12:"DEC"
                                               })
#merge year month
nyc_data["month_year"] = nyc_data["year"].astype(str) + " - " + nyc_data["month_name"]
#extract week day 
nyc_data["week_day"]   = nyc_data["pickup_datetime"].dt.weekday_name
#extract day 
nyc_data["day"]        = nyc_data["pickup_datetime"].dt.day
#extract hour
nyc_data["hour"]        = nyc_data["pickup_datetime"].dt.hour 
nyc_data = nyc_data.sort_values(by = "pickup_datetime",ascending = False)

#Outlier treatment
#drop observations with passengers greater than 6 and equals 0
nyc_data = nyc_data[(nyc_data["passenger_count"] > 0 ) &
                    (nyc_data["passenger_count"] < 7) ]

#drop observations with fareamount  less than 0 and  greater than 99.99% percentile value.
nyc_data = nyc_data[ (nyc_data["fare_amount"] > 0 ) &
                     (nyc_data["fare_amount"]  <  
                      nyc_data["fare_amount"].quantile(.9999))]

#drop outlier observations in data
coords = ['pickup_longitude','pickup_latitude', 
          'dropoff_longitude', 'dropoff_latitude']
for i in coord  : 
    nyc_data = nyc_data[(nyc_data[i]   > nyc_data[i].quantile(.001)) & 
                        (nyc_data[i] < nyc_data[i].quantile(.999))]
    
#create new variable log of fare amount
nyc_data["log_fare_amount"] = np.log(nyc_data["fare_amount"])
    
nyc_data.head()


# ## Finding distances based on Latitude and Longitude
# * The haversine formula determines the great-circle distance between two points on a sphere given their longitudes and latitudes.
# * Formula
# * dlon = lon2 - lon1 
# * dlat  = lat2 - lat1 
# * a = (sin(dlat/2))^2 + cos(lat1) * cos(lat2) * (sin(dlon/2))^2 
# * c = 2 * atan2( sqrt(a), sqrt(1-a) ) 
# * d = R * c (where R is the radius of the Earth)

# In[ ]:


#radius of earth in kilometers
R = 6373.0

pickup_lat  = np.radians(nyc_data["pickup_latitude"])
pickup_lon  = np.radians(nyc_data["pickup_longitude"])
dropoff_lat = np.radians(nyc_data["dropoff_latitude"])
dropoff_lon = np.radians(nyc_data["dropoff_longitude"])

dist_lon = dropoff_lon - pickup_lon
dist_lat = dropoff_lat - pickup_lat

#Formula
a = (np.sin(dist_lat/2))**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * (np.sin(dist_lon/2))**2 
c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) ) 
d = R * c #(where R is the radius of the Earth)

nyc_data["trip_distance_km"] = d

#create new variable log of distance
nyc_data["log_trip_ditance"] = np.log(nyc_data["trip_distance_km"])

nyc_data[coord + ["trip_distance_km"]].head(7)


# ## Variable Summary

# In[ ]:


summary = nyc_data.describe().transpose().reset_index().rename(columns = {"index" : 
                                                                          "variable"})
summary  = np.around(summary,2)

var_lst = [summary["variable"],summary["count"],summary['mean'],summary['std'],
           summary["min"],summary["25%"],summary["50%"],summary["75%"],summary["max"]]

table = go.Table(header = dict(values = summary.columns.tolist(),
                               line = dict(color = ['#506784']),
                               fill = dict(color = ['#119DFF']),
                              ),
                 cells  = dict(values = var_lst,
                               line = dict(color = ['#506784']),
                               fill = dict(color = ["lightgrey",'#F5F8FF']),
                              ),
                 columnwidth = [130,80,80,80,80,80,80,80,80])
                
layout = go.Layout(dict(title = "Variable Summary"))
figure = go.Figure(data=[table],layout=layout)
py.iplot(figure)


# ## Data Loss

# In[ ]:


trace = go.Pie(values = [nyc_data.shape[0],5000000 - nyc_data.shape[0]],
               labels = ["Available data" , "Data loss due to outliers and missing values"],
               marker = dict(colors =  [ 'royalblue' ,'lime'],line = dict(color = "black",
                                                                          width =  1.5)),
               rotation  = 60,
               hoverinfo = "label+percent",
              )

layout = go.Layout(dict(title = "Data Loss due to outliers and missing values",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

fig = go.Figure(data=[trace],layout=layout)
py.iplot(fig)


# ## Distribution plots for numerical features

# In[ ]:


cols = ['fare_amount','passenger_count', 
        'pickup_longitude', 'dropoff_longitude',
        'pickup_latitude', 'dropoff_latitude']

length = len(cols)
cs     = [(0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
          (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
          (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
          (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
          (1.0, 0.4980392156862745, 0.0),"b"]

sns.set_style("darkgrid")
plt.figure(figsize = (13,15))
for i,j,k in itertools.zip_longest(cols,range(length),cs) :
    plt.subplot(length/2,length/3,j+1)
    sns.distplot(nyc_data[i],color = k)
    plt.axvline(nyc_data[i].mean(),linewidth  = 2 ,
                linestyle = "dashed",color = "k" ,
                label = "Mean")
    plt.legend(loc = "best")
    plt.title(i,color = "b")
    plt.xlabel("")
    


# ## Distribution in log of fare amount

# In[ ]:


plt.figure(figsize = (12,7))
sns.distplot(nyc_data["log_fare_amount"],color = "b")
plt.axvline(nyc_data["log_fare_amount"].mean(),color = "k",
            linestyle = "dashed",label = "Avg fare amount")
plt.title("Distribution in log of fare amount")
plt.legend(loc = "best",prop = {"size" : 12})
plt.show()


# ## Distribution of haversine distance in kilometers

# In[ ]:


plt.figure(figsize = (12,7))
sns.distplot(nyc_data["trip_distance_km"],color = "r")
plt.axvline(nyc_data["trip_distance_km"].mean(),color = "k",
            linestyle = "dashed",label = "Avg trip distance (km)")
plt.title("Distribution in trip distance in kilometers")
plt.legend(loc = "best",prop = {"size" : 12})
plt.show()


# ## scatter plot for distance and fare amount

# In[ ]:


plt.figure(figsize = (12,10))

plt.scatter(nyc_data["fare_amount"],
            nyc_data["trip_distance_km"],s = 5,
            linewidths=1, c = "b")
plt.ylabel("Haversine distance in kilometers")
plt.xlabel("Fare amount")
plt.title("scatter plot for distance and fare amount")
plt.show()


# In[ ]:


plt.figure(figsize = (12,10))
plt.scatter(nyc_data["log_fare_amount"],
            nyc_data["log_trip_ditance"],s = 5,
            linewidths=1, c = "b")
plt.ylabel("log of Haversine distance in kilometers")
plt.xlabel("log of Fare amount")
plt.title("scatter plot for distance and fare amount")
plt.show()


# ## Total  trips , passengers and fare amount by year

# In[ ]:


yearly_analysis  = nyc_data.groupby("year").agg({"key":"count",
                                                 "fare_amount":"sum",
                                                 "passenger_count":"sum",
                                                 "trip_distance_km" : "sum"}).reset_index()
#aggregating by year
yearly_analysis = yearly_analysis.rename(columns = {"key" : "trip_count"})

#plotting trips ,passengers and fare amount by year
def plotting(column) : 
    tracer = go.Bar(x= yearly_analysis["year"],y = yearly_analysis[column],
                    marker = dict(line = dict(width = 1)),
                    name = column
                   )
    return tracer

#layout
layout = go.Layout(dict(title = "Total  trips ,passengers,trip_distance and fare amount by year",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',title = "year",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',title = "count",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                       )
                  )
    
data = [plotting("trip_count"),plotting("passenger_count"),
        plotting("trip_distance_km"),plotting("fare_amount")]
fig  = go.Figure(data=data,layout=layout)
py.iplot(fig)


# ## Trend in  trips by months.

# In[ ]:


yrs = [i for i in nyc_data["year"].unique().tolist() if i not in [2015]]

#subset data without year 2015
complete_dat = nyc_data[nyc_data["year"].isin(yrs)]


plt.figure(figsize = (13,15))
for i,j in itertools.zip_longest(yrs,range(len(yrs))) :
    plt.subplot(3,2,j+1)
    trip_counts_mn = complete_dat[complete_dat["year"] == i]["month_name"].value_counts()
    trip_counts_mn = trip_counts_mn.reset_index()
    sns.barplot(trip_counts_mn["index"],trip_counts_mn["month_name"],
                palette = "rainbow",linewidth = 1,
                edgecolor = "k"*complete_dat["month_name"].nunique() 
               )
    plt.title(i,color = "b",fontsize = 12)
    plt.grid(True)
    plt.xlabel("")
    plt.ylabel("trips")


# ## Average fare amount by month

# In[ ]:


fare_mn = complete_dat.groupby("month_name")["fare_amount"].mean().reset_index()

mnth_ord = ['JAN', 'FEB', 'MAR','APR', 'MAY' , 'JUN',
                'JUL',  'AUG', 'SEP','OCT', 'NOV','DEC']

plt.figure(figsize = (12,7))
sns.barplot("month_name","fare_amount",
            data = fare_mn,order = mnth_ord,
            linewidth =1,edgecolor = "k"*len(mnth_ord)
           )
plt.grid(True)
plt.title("Average fare amount by Month")
plt.show()


# ## Trend in trips  by weekdays

# In[ ]:


def plot_day_trend(year) :
    day_count = complete_dat[complete_dat["year"] == year]["week_day"].value_counts().reset_index()
    day_count.columns = ["day","count"]
    day_count["order"]  = day_count["day"].replace({"Sunday" :1,'Monday' : 2, 'Tuesday': 3,
                                                    'Wednesday':4,'Thursday' :5, 'Friday':6,
                                                    'Saturday':7})
    day_count = day_count.sort_values(by = "order",ascending  = True)
    
    tracer = go.Bar(x = day_count["day"],y = day_count["count"],
                    name = year,marker = dict(line = dict(width =1))
                   )
    
    return tracer

#layout
layout = go.Layout(dict(title = "Trend in trips  by weekdays",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     title = "weekday",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     title = "count",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                       )
                  )

t  = plot_day_trend(2009)
t1 = plot_day_trend(2010)
t2 = plot_day_trend(2011)
t3 = plot_day_trend(2012)
t4 = plot_day_trend(2013)
t5 = plot_day_trend(2014)

data = [t,t1,t2,t3,t4,t5]
py.iplot(go.Figure(data = data,layout=layout))


# ## Average fare amount by week day

# In[ ]:


fare_wk = complete_dat.groupby("week_day")["fare_amount"].mean().reset_index()

wk_ord = ["Sunday" ,'Monday' , 'Tuesday','Wednesday',
          'Thursday' ,'Friday', 'Saturday']

plt.figure(figsize = (12,7))
sns.barplot("week_day","fare_amount",
             data = fare_wk,order = wk_ord,palette = "husl",
             linewidth =1,edgecolor = "k"*len(wk_ord)
            )
plt.grid(True)
plt.title("Average fare amount by week day")
plt.show()


# ## Trend in trips  by hour of day

# In[ ]:


trips_hr = nyc_data["hour"].value_counts().reset_index()
trips_hr.columns = ["hour","count"]
trips_hr = trips_hr.sort_values(by = "hour",ascending = True)

trace = go.Scatter(x = trips_hr["hour"],y = trips_hr["count"],
                   mode = "markers+lines",
                  marker = dict(color = "red",size = 9,
                                line = dict(color = "black",width =2)))
#layout
layout = go.Layout(dict(title = "Trend in trips  by hour of day",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',title = "hour",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',title = "count",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                       )
                  )

fig = go.Figure(data = [trace],layout = layout)
py.iplot(fig)


# ## Average fare by hour

# In[ ]:


avg_fare_hr = complete_dat.groupby("hour")["fare_amount"].mean().reset_index()
avg_fare_hr
trace = go.Scatter(x = avg_fare_hr["hour"],y = avg_fare_hr["fare_amount"],
                   mode = "markers+lines",
                  marker = dict(color = "blue",size = 9,
                                line = dict(color = "black",width =2)))

#layout
layout = go.Layout(dict(title = "Average fare by hour",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',title = "hour",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',title = "average_fare",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                       )
                  )

fig = go.Figure(data = [trace],layout = layout)
py.iplot(fig)


# ## Seasonal Trend in trips, passengers and fare amount

# In[ ]:


#aggregate by year-month(trips)
trip_count = nyc_data.groupby(["year","month",
                               "month_name"])["month_year"].value_counts().to_frame()
trip_count.columns = ["count"]
trip_count = trip_count.reset_index()

#aggregate by year-month(passengers)
passenger_count = (nyc_data.groupby(["year","month","month_name",
                                     "month_year"])["passenger_count"].sum().reset_index())

#aggregate by year-month(fare amount)
total_fare = (nyc_data.groupby(["year","month","month_name",
                                     "month_year"])["fare_amount"].sum().reset_index())

#aggregate by year-month(total trip distance)
total_trip_dist = (nyc_data.groupby(["year","month","month_name",
                                     "month_year"])["trip_distance_km"].sum().reset_index())


#plotting
def trend_scatter(data_frame,column) :
    tracer = go.Scatter(x = data_frame["month_year"],y = data_frame[column],
                        mode = "lines+markers",
                        marker = dict(color = data_frame["month"],size = 7,
                                      colorscale = "Picnic",
                                      line = dict(width =1 ,color = "black")
                                     ),
                        line = dict(color = "grey" ),
                   )
    return tracer

def layout_plot(title) :
    layout = go.Layout(dict(title = title,
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                         zerolinewidth=1,ticklen=5,gridwidth=2),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',title = "count",
                                         zerolinewidth=1,ticklen=5,gridwidth=2),
                            margin = dict(b = 100)
                           )
                      )
    return layout

#figure 1
data    = [trend_scatter(trip_count,"count")]
layout  = layout_plot("Trend in trip count from 2009-Jan to 2015-May")
fig = go.Figure(data = data,layout = layout)
py.iplot(fig)

#figure 2
data1    = [trend_scatter(passenger_count,"passenger_count")]
layout1  = layout_plot("Trend in total passenger count from 2009-Jan to 2015-May")
fig1 = go.Figure(data = data1,layout = layout1)
py.iplot(fig1)

#figure 3
data2    = [trend_scatter(total_fare,"fare_amount")]
layout2  = layout_plot("Trend in total fare amount from 2009-Jan to 2015-May")
fig2 = go.Figure(data = data2,layout = layout2)
py.iplot(fig2)

#figure 4
data3    = [trend_scatter(total_trip_dist,"trip_distance_km")]
layout3  = layout_plot("Trend in trip distance from 2009-Jan to 2015-May")
fig3 = go.Figure(data = data3,layout = layout3)
py.iplot(fig3)


# ## Heat map for trips,passengers and fare amount by month year

# In[ ]:


#plot annoted heat map by month - year
def plot_heat_map(column,aggregate_function,title) :
    #pivot table 
    pivot_table = pd.pivot_table(data = nyc_data,columns = "month_name",index = "year",
                                values = column,aggfunc = aggregate_function)
    
    mnth_ord = ['JAN', 'FEB', 'MAR','APR', 'MAY' , 'JUN',
                'JUL',  'AUG', 'SEP','OCT', 'NOV','DEC']
    
    #reverse mnth order list
    def reverse(data_list) :
        return data_list[::-1]
    
    mnth_ord_rev = reverse(mnth_ord)
    
    pivot_table  = pivot_table[mnth_ord_rev].sort_values(by = "year",ascending = True)
    pivot_table  = pivot_table.transpose()
    
    #convert array
    pivot_array  = np.around(np.array(pivot_table))
    
    #color scale
    colorscale=[[0.0, 'rgb(255,255,255)'], [.2, 'rgb(255, 255, 153)'], 
                [.4, 'rgb(153, 255, 204)'], [.6, 'rgb(179, 217, 255)'], 
                [.8, 'rgb(240, 179, 255)'],[1.0, 'rgb(255, 77, 148)']]
    #plot heat map
    figure = ff.create_annotated_heatmap(z = pivot_array,
                                         x = pivot_table.columns.tolist(),
                                         y = pivot_table.index.tolist(),
                                         font_colors = ["black"],
                                         showscale = True,
                                         colorscale = colorscale,
                                         colorbar   = dict(title = "total " + title,
                                                           titleside = "right")
                                        )
    #title
    figure.layout.title = "Total " + title + " by  month - year ."
    figure.layout.plot_bgcolor  = "white"
    figure.layout.paper_bgcolor = "white"
    
    return py.iplot(figure)

#plot1
plot_heat_map("key","count","trip count")
#plot2
plot_heat_map("passenger_count","sum","passengers count")
#plot3
plot_heat_map("fare_amount","sum","fare amount")
#plot4
plot_heat_map("trip_distance_km","sum","trip distance(km)")


# ## Correlation 

# In[ ]:


#Merge passenger_count,trip_count and fare_amount data 
merge1  = trip_count.merge(passenger_count,left_on = "month_year",
                           right_on = "month_year",how= "left")

merge2  = total_fare.merge(total_trip_dist,left_on = "month_year",
                           right_on = "month_year",how= "left")

final_my_dat = merge1.merge(merge2,left_on = "month_year",
                           right_on = "month_year",how= "left")

final_my_dat = final_my_dat[["month_year","count","passenger_count","fare_amount",
                             "trip_distance_km","month_x_x","year_x_x","month_name_x_x"]]


final_my_dat = final_my_dat.rename(columns = {"month_x_x":"month","year_x_x":"year",
                                   "count" : "trips_count","month_name_x_x":"month_name"})


# #correlation
corr  =  np.array(final_my_dat[['trips_count', 'passenger_count',
                                'fare_amount', "trip_distance_km"]].corr())
corr  =  np.around(corr,4)

# #x & y ticks
ticks =  ['trips_count', 'passenger_count', 'fare_amount', "trip_distance_km"]

# #plot heatmap
fig = ff.create_annotated_heatmap(z = corr,x = ticks,y = ticks,showscale=True,
                                   colorscale = "Portland",
                                   colorbar   = dict(title = "correlation coefficient",
                                                     titleside = "right"
                                      ) 
                                  )
fig.layout.title  = "Correlation Matrix"
fig.layout.margin = dict(l = 200,r = 200)
py.iplot(fig)


# ## New york city map

# In[ ]:


#import libraries
import folium
import folium.plugins

#lat and lon center
lat_center = 40.77
lon_center = -73.96

#plot
map_ny = folium.Map(location=[lat_center,lon_center],
                    tiles="openstreetmap",max_zoom=15,zoom_start=11.5)
map_ny


# ## Trips with maximum distance (> 40 km)

# In[ ]:


trips_ln = nyc_data[nyc_data["trip_distance_km"] > 40]

lat_center = 40.78
lon_center = -73.62

map_nyc = folium.Map(location=[lat_center,lon_center],
                    tiles="stamentoner",max_zoom=15,zoom_start=10)


for i in range(0,len(trips_ln)) : 
    
    p1 = [trips_ln["pickup_latitude"].values[i],
          trips_ln["pickup_longitude"].values[i]]
    
    p2 = [trips_ln["dropoff_latitude"].values[i],
          trips_ln["dropoff_longitude"].values[i]]
    
    folium.Marker(location = p1,
                  icon=folium.Icon(color='green',
                                   icon = "home"),
                  popup = "Pick up = " + str(p1),
                 ).add_to(map_nyc)
    
    folium.Marker(location = p2,
                 icon=folium.Icon(color='blue',
                                  icon = "home"),
                  popup = "Drop off = " + str(p2),
                 ).add_to(map_nyc)
    
    folium.PolyLine(locations = [p1,p2] ,
                    color = "red",opacity = .9,
                   ).add_to(map_nyc)
    
   
map_nyc


# ## Pick up locations by passenger count

# In[ ]:


#Import Libraries
from bokeh.models import BoxZoomTool
from bokeh.plotting import figure, output_notebook, show
import datashader as ds
from datashader.bokeh_ext import InteractiveImage
from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select, Greys9, Hot, inferno,Set1
from datashader import transfer_functions as tf
output_notebook()

#plot datapoints by location coordinates
def plot_data_points(longitude,latitude,data_frame,focus_point) :
    #plot dimensions
    x_range, y_range = ((-74.14,-73.73), (40.6,40.9))
    plot_width  = int(750)
    plot_height = int(plot_width//1.2)
    export  = partial(export_image, export_path="export", background="black")
    fig = figure(background_fill_color = "black")    
    #plot data points
    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                    x_range=x_range, y_range=y_range)
    agg = cvs.points(data_frame,longitude,latitude,
                      ds.count(focus_point))
    img = tf.shade(agg, cmap= Hot, how='eq_hist')
    image_xpt  =  tf.dynspread(img, threshold=0.5, max_px=4)
    return export(image_xpt,"NYCT_hot")

plot_data_points('pickup_longitude', 'pickup_latitude',nyc_data,"passenger_count")


# ## Drop off locations by passenger count

# In[ ]:


plot_data_points('dropoff_longitude', 'dropoff_latitude',nyc_data,"passenger_count")


# ## Pick up locations by fare amount

# In[ ]:


plot_data_points('pickup_longitude', 'pickup_latitude',nyc_data,"fare_amount")


# ## Drop off locations by fare amount

# In[ ]:


plot_data_points('dropoff_longitude', 'dropoff_latitude',nyc_data,"fare_amount")


# ## Extracting JFK Airport dropoff's and pickup's
# * calculated haversine distance from pick up and drop off coordinates to jfk and subset dataset with less than 1km distance.

# In[ ]:


jfk_data = nyc_data.copy()

#jfk coordinates
jfk_data["jfk_lat"] = 40.6413
jfk_data["jfk_lon"] = -73.7781

#function to get haversine distance for two set of coordinates
def distance_points(data_frame,x1_lat,x1_lon,x2_lat,x2_lon) :    
    R = 6373.0 #radius of the Earth in kilometers
    
    point1_lat = np.radians(data_frame[x1_lat])
    point1_lon = np.radians(data_frame[x1_lon])
    point2_lat = np.radians(data_frame[x2_lat])
    point2_lon = np.radians(data_frame[x2_lon])
    
    dist_lon = point2_lon - point1_lon
    dist_lat = point2_lat - point1_lat
    
    #Formula
    a = (np.sin(dist_lat/2))**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * (np.sin(dist_lon/2))**2 
    c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) ) 
    d = R * c #(where R is the radius of the Earth)
    
    return d
    
#distance_from jfk airport to pickup coordinates
jfk_data["dist_pickup_jfk"]  = distance_points(jfk_data,"pickup_latitude",
                                               "pickup_longitude",
                                              "jfk_lat","jfk_lon")

#distance_from jfk airport to dropoff coordinates
jfk_data["dist_dropoff_jfk"] = distance_points(jfk_data,"dropoff_latitude",
                                               "dropoff_longitude",
                                              "jfk_lat","jfk_lon")

#pick ups from 1km distance from jfk 
jfk_data_pickups  = jfk_data[(jfk_data["dist_pickup_jfk"] <  2)]
jfk_data_pickups["type"]  = "Pick up"
jfk_data_pickups = jfk_data_pickups.drop(columns = [ 'jfk_lat','jfk_lon'] ,
                                         axis = 1)

#dropoffs from 1km distance from jfk 
jfk_data_dropoffs = jfk_data[(jfk_data["dist_dropoff_jfk"] < 2)]
jfk_data_dropoffs["type"] = "Drop off"
jfk_data_dropoffs = jfk_data_dropoffs.drop(columns = [ 'jfk_lat', 'jfk_lon'] ,
                                           axis = 1)

#concat jfk pickups and dropoff
jfk = pd.concat([jfk_data_pickups,jfk_data_dropoffs],axis = 0)

#subset data which are not in jfk
not_jfk = nyc_data.drop(jfk.index,axis = 0)

#trips to jfk
jfk.head()


# ## Pick up locations from JFK airport`

# In[ ]:


lat_center_jfk = 40.645626
lon_center_jfk = -73.785220

map_jfk_pk = folium.Map(location=[lat_center_jfk,lon_center_jfk],
                         tiles="OpenStreetMap",
                         max_zoom=15,zoom_start=15)

folium.Marker(location = [lat_center_jfk,lon_center_jfk],
              icon = folium.Icon( icon="star",color = "blue")).add_to(map_jfk_pk)

#plotting 2500 dta points
for i in range(0,len(jfk_data_dropoffs[:2500])) :
    p = [jfk_data_pickups["pickup_latitude"][:2500].values[i],
        jfk_data_pickups["pickup_longitude"][:2500].values[i]]
    
    folium.Circle(location = p ,radius = 2,
                  color = "red").add_to(map_jfk_pk)
    

map_jfk_pk


# ## Dropoff locations for JFK Airport

# In[ ]:


map_jfk_dp = folium.Map(location=[lat_center_jfk,lon_center_jfk],
                        tiles="OpenStreetMap",
                        max_zoom=15,zoom_start=15)

folium.Marker(location = [lat_center_jfk,lon_center_jfk],
              icon = folium.Icon( icon="star",color = "blue")).add_to(map_jfk_dp)

#plotting 2500 dta points
for i in range(0,len(jfk_data_dropoffs[:2500])) :
    p = [jfk_data_dropoffs["pickup_latitude"][:2500].values[i],
        jfk_data_dropoffs["pickup_longitude"][:2500].values[i]]
    
    folium.Circle(location = p ,radius = 5,
                  color = "blue").add_to(map_jfk_dp)
    

map_jfk_dp


# ## Fare amount distribution for jfk and non jfk trips

# In[ ]:


plt.figure(figsize = (13,7))
sns.distplot(not_jfk["fare_amount"],color = "b")
sns.distplot(jfk["fare_amount"],color = "r")
plt.axvline(not_jfk["fare_amount"].mean(),
            color = "b",linestyle = "dashed",label = "non_jfk_trip_mean")
plt.axvline(jfk["fare_amount"].mean(),
            color = "r",linestyle = "dashed",label = "jfk_trip_mean")
plt.legend(loc = "best",prop= {"size" : 12})
plt.title("Fare amount distribution for jfk and non jfk trips")
plt.show()


# ## Average fare amount by jfk and non jfk trip  

# In[ ]:


#average fare amounts by jfk pickups and dropoff
jfk_tp_avg   = jfk.groupby("type")["fare_amount"].mean().reset_index()

#average fare amount by jfk and non jfk trip  
jfk_njfk_avg = pd.DataFrame({"type" : ["jfk","non_jfk"], 
                            "fare_amount":[jfk["fare_amount"].mean(),
                                           not_jfk["fare_amount"].mean()]
                            }
                            )

#plot
plt.figure(figsize = (13,5))
plt.subplot(121)
ax = sns.barplot(x = "fare_amount",y = "type",data = jfk_tp_avg,
                 linewidth = 2, edgecolor = "k"*2 ,
                 palette = "husl"
                )
plt.grid(True)
plt.ylabel("trip type")
plt.title("Average fare amount by jfk pickups and drop offs")
for i,j in enumerate(np.around(jfk_tp_avg["fare_amount"].values,2)) :
    ax.text(.9,i,j,fontsize = 15)
    
plt.subplot(122)
ax1 = sns.barplot(x = "fare_amount",y = "type",data = jfk_njfk_avg,
                 linewidth = 2, edgecolor = "k"*2 ,
                 palette = "husl")
plt.ylabel("")
plt.grid(True)
plt.title("Average fare amount by jfk and non jfk trip")
for i,j in enumerate(np.around(jfk_njfk_avg["fare_amount"].values,2)) :
    ax1.text(.9,i,j,fontsize = 15)
plt.show()


# ## Trip distance distribution for jfk and non jfk trips

# In[ ]:


plt.figure(figsize = (13,7))
sns.distplot(not_jfk["trip_distance_km"],color = "b")
sns.distplot(jfk["trip_distance_km"],color = "r")
plt.axvline(not_jfk["trip_distance_km"].mean(),
            color = "b",linestyle = "dashed",label = "non_jfk_trip_mean")
plt.axvline(jfk["trip_distance_km"].mean(),
            color = "r",linestyle = "dashed",label = "jfk_trip_mean")
plt.legend(loc = "best",prop= {"size" : 12})
plt.title("Trip distance distribution for jfk and non jfk trips")
plt.show()


# ## Average trip distance by jfk and non jfk trip  

# In[ ]:


#average distance by jfk pickups and dropoff
jfk_tp_avg   = jfk.groupby("type")["trip_distance_km"].mean().reset_index()

#average distance by jfk and non jfk trip  
jfk_njfk_avg = pd.DataFrame({"type" : ["jfk","non_jfk"], 
                            "trip_distance_km":[jfk["trip_distance_km"].mean(),
                                                not_jfk["trip_distance_km"].mean()]
                            }
                            )

#plot
plt.figure(figsize = (13,5))
plt.subplot(121)
ax = sns.barplot(x = "trip_distance_km",y = "type",data = jfk_tp_avg,
                 linewidth = 2, edgecolor = "k"*2 ,
                 palette = "husl"
                )
plt.grid(True)
plt.ylabel("trip type")
plt.title("Average trip_distance by jfk pickups and drop offs")
for i,j in enumerate(np.around(jfk_tp_avg["trip_distance_km"].values,2)) :
    ax.text(.9,i,j,fontsize = 15)
    
plt.subplot(122)
ax1 = sns.barplot(x = "trip_distance_km",y = "type",data = jfk_njfk_avg,
                 linewidth = 2, edgecolor = "k"*2 ,
                 palette = "husl")
plt.ylabel("")
plt.grid(True)
plt.title("Average trip_distance  by jfk and non jfk trip")
for i,j in enumerate(np.around(jfk_njfk_avg["trip_distance_km"].values,2)) :
    ax1.text(.9,i,j,fontsize = 15)
plt.show()


# ## scatter plot for distance and fare amount

# In[ ]:


plt.figure(figsize = (13,6))
plt.subplot(121)
plt.scatter(jfk_data_pickups["fare_amount"],
            jfk_data_pickups["trip_distance_km"],
            linewidth =1,edgecolor = "k",s = 30,
            color= "r",alpha =.7,label = "pick ups")
plt.legend(loc = "best",prop = {"size" : 15})
plt.xlabel("fare amount")
plt.ylabel("trip distance km")
plt.title("Jfk pick-ups ")

plt.subplot(122)
plt.scatter(jfk_data_dropoffs["fare_amount"],
            jfk_data_dropoffs["trip_distance_km"],
            linewidth =1,edgecolor = "k",s = 30,
            color = "b",alpha =.7,label = "drop offs")
plt.legend(loc = "best",prop = {"size" : 15})
plt.xlabel("fare amount")
plt.ylabel("trip distance km")
plt.title("Jfk drop-offs ")

plt.show()


# ## Average fare amount for jfk airport by year

# In[ ]:


plt.figure(figsize = (12,5))
jfk.groupby(["year"])["fare_amount"].mean().plot(kind = "bar",linewidth = 1,
                                                 figsize = (12,6),
                                                 edgecolor = "k" *jfk["year"].nunique())
plt.xticks(rotation = 0)
plt.ylabel("fare")
plt.title("Average fare amount for jfk airport by year")
plt.show()


# ## Time series forecasting of fare amount

# In[ ]:


import datetime
#Data - total fare amount by month from 2009 to 2015-06
ts_fare = total_fare.copy()
ts_fare["date"] = ts_fare["year"].astype(str) + "-" + ts_fare["month"].astype(str)
#selecting columns
ts_fare = ts_fare[["date","fare_amount"]]
#convert to date format
ts_fare["date"] = pd.to_datetime(ts_fare["date"],format = "%Y-%m")
ts_fare.index   = ts_fare["date"]
ts_fare = ts_fare.drop(columns  = ["date"],axis = 1)
ts_fare.head(10)


# ## Visualizing time series

# In[ ]:


trace = go.Scatter(x = ts_fare.index,y = ts_fare.fare_amount,
                   mode = "lines+markers",
                   marker = dict(color = "royalblue",line = dict(width =1))
                  )
layout = go.Layout(dict(title = "Visualizing time series",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',title = "count",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        margin = dict(b = 100)
                       )
                  )
fig = go.Figure(data = [trace],layout = layout)
py.iplot(fig)


# ## Check stationarity of time series

# In[ ]:


from statsmodels.tsa.stattools import adfuller

def plot_line(x,y,color,name) :
    tracer = go.Scatter(x = x,y = y,mode = "lines",
                        marker = dict(color = color,
                                      line = dict(width =1)),
                       name = name)
    return tracer

def plot_layout(title) :
    layout = go.Layout(dict(title = title,
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                         zerolinewidth=1,ticklen=5,gridwidth=2),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                            zerolinewidth=1,ticklen=5,gridwidth=2),
                        margin = dict(b = 100)
                       )
                  )
    return layout


def stationary_test(timeseries) :
    #rolling mean
    rol_mean = timeseries["fare_amount"].rolling(window = 12,
                                                 center = False).mean()
    #rolling standard deviation
    rol_std  = timeseries["fare_amount"].rolling(window = 12,
                                                 center = False).std()
    
    #plotting
    trace1  = plot_line(timeseries.index,timeseries["fare_amount"],
                        "blue","time_series")
    trace2  = plot_line(rol_mean.index,rol_mean.values,
                        "red","rolling_mean")
    trace3  = plot_line(rol_std.index,rol_std.values,
                        "green", "rolling_std")
    layout  = plot_layout("rolling mean and standard deviation for timeseries")
    figure  = go.Figure(data = [trace1,trace2,trace3],layout = layout)
    
    test_results = adfuller(timeseries["fare_amount"])
    res_list     = ["Test Statistic","p-value",
                    "lags used","no of observations"] 
    res_df = pd.Series(test_results[:4],index = res_list)
    
    for key,value in test_results[4].items() :
        res_df["Critical value (%s)"%key] = value 
        
    print ("Results - Dickey fuller test")
    print (res_df)
    return py.iplot(figure)

stationary_test(ts_fare)


# # Eliminating Trend
# ## Moving Average

# In[ ]:


#log of timeseries
log_ts_fare = np.log(ts_fare)

#rolling average of log timeseries
rol_avg_log_ts = log_ts_fare["fare_amount"].rolling(window = 12,center = False).mean()

#plotting log timeseries and rolling mean
t1 = plot_line(log_ts_fare.index,log_ts_fare.fare_amount,
                "blue","log_time_series")
t2 = plot_line(rol_avg_log_ts.index,rol_avg_log_ts.values,
               "red","moving_average(log)")
lay = plot_layout("log time series and moving average")
fig = go.Figure(data = [t1,t2],layout = lay)
py.iplot(fig)

#difference
log_ts_fare_diff = log_ts_fare - rol_avg_log_ts.to_frame()
log_ts_fare_diff.dropna(inplace = True)
stationary_test(log_ts_fare_diff)


# ## Exponential weighted moving average

# In[ ]:


#exponential moving average of log time series
exp_log_avg = log_ts_fare["fare_amount"].ewm(halflife = 12).mean()

#plotting
t1 = plot_line(log_ts_fare.index,log_ts_fare["fare_amount"],
               "blue","log time series")
t2 = plot_line(exp_log_avg.index,exp_log_avg.values,
               "red","exponential avg")
lay = plot_layout("log time series and exponential moving average")
fig = go.Figure(data = [t1,t2],layout = lay)
py.iplot(fig)

#difference
exp_ts_diff = log_ts_fare - exp_log_avg.to_frame()
stationary_test(exp_ts_diff)


# ## Eliminating Trend and seasonality
# ## Differencing

# In[ ]:


#differencing log series
ts_fare_diff = log_ts_fare - log_ts_fare.shift()
ts_fare_diff.dropna(inplace = True)

#plotting
t1 = plot_line(ts_fare_diff.index,ts_fare_diff["fare_amount"],
              "blue","Differenced log series")
lay = plot_layout("Differenced log series")
fig = go.Figure(data = [t1],layout=lay)
py.iplot(fig)

#stationary test
stationary_test(ts_fare_diff)


# ## Decomposing

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose

#decompose
decompose = seasonal_decompose(log_ts_fare)

#trend
trend       = decompose.trend
#seasonality
seasonality = decompose.seasonal
#residuals
residuals   = decompose.resid

#plotting
t1 = plot_line(ts_fare.index,ts_fare.fare_amount,
               "blue","log_Series")
t2 = plot_line(trend.index,trend.fare_amount,
               "green","Trend")
t3 = plot_line(seasonality.index,seasonality.fare_amount,
               "red","Seasonality")
t4 = plot_line(residuals.index,residuals.fare_amount,
               "black","Residuals")
#subplots
fig = tls.make_subplots(rows = 4,cols = 1,subplot_titles = ("log series",
                                                            "Trend",
                                                            "Seasonality",
                                                            "residuals"))

fig.append_trace(t1,1,1)
fig.append_trace(t2,2,1)
fig.append_trace(t3,3,1)
fig.append_trace(t4,4,1)
#layout
fig["layout"].update(height = 750,
                     plot_bgcolor  = "rgb(243,243,243)",
                     paper_bgcolor = "rgb(243,243,243)",
                     xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                  zerolinewidth=1,ticklen=5,gridwidth=2),
                    yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                 zerolinewidth=1,ticklen=5,gridwidth=2),
                    title = "decomposing"
                    )
py.iplot(fig)

#stationary tert for residuals
residuals.dropna(inplace = True)
stationary_test(residuals)


# # Forecasting
#   ## ACF & PACF plots

# In[ ]:


from statsmodels.tsa.stattools import acf,pacf

#auto correlation function
acf_vals  = acf(ts_fare_diff)
#partial auto correlation function
pacf_vals = pacf(ts_fare_diff)

#plot acf,pacf
def plot_corr_fact(x,y,color,name) :
    tracer = go.Bar(x = x, y= y ,
                    marker = dict(color = color,
                                  line = dict(width =1,
                                              color = "black") 
                                 ),
                    name = name
                   )
    return tracer
#plot confidence intervals
def plot_lines(x,y) :
    trace_line = go.Scatter(x = x, y = y,
                            mode   = "lines",
                            line   = dict(color = "black",
                                          width = 2,
                                          dash = "dash" 
                                         ) ,
                            name = "confidence intervals"
                           )
    return trace_line

#acf values
t_acf  = plot_corr_fact(np.arange(0,len(acf_vals)),
                    acf_vals,"blue","acf")
#confidence intervals for acf
lu_acf = plot_lines(np.arange(0,len(acf_vals)),
                    [1.96/np.sqrt(len(ts_fare_diff))]*len(acf_vals))
ll_acf = plot_lines(np.arange(0,len(acf_vals)),
                    [-1.96/np.sqrt(len(ts_fare_diff))]*len(acf_vals))

#pacf values
t_pacf = plot_corr_fact(np.arange(0,len(pacf_vals)),
                    pacf_vals,"red","pacf")
#confidence intervals for pacf
lu_pacf = plot_lines(np.arange(0,len(pacf_vals)),
                    [1.96/np.sqrt(len(ts_fare_diff))]*len(pacf_vals))
ll_pacf = plot_lines(np.arange(0,len(pacf_vals)),
                    [-1.96/np.sqrt(len(ts_fare_diff))]*len(pacf_vals))

#subplots
fig = tls.make_subplots(rows = 1, cols  = 2,
                        subplot_titles = ("auto correlation function",
                                          "partial auto correlation function"))

fig.append_trace(t_acf,1,1)
fig.append_trace(lu_acf,1,1)
fig.append_trace(ll_acf,1,1)
fig.append_trace(t_pacf,1,2)
fig.append_trace(lu_pacf,1,2)
fig.append_trace(ll_pacf,1,2)

#layout
fig["layout"].update(plot_bgcolor  = "rgb(243,243,243)",
                     showlegend = False,
                     paper_bgcolor = "rgb(243,243,243)",
                     xaxis1 = dict(gridcolor = 'rgb(255, 255, 255)',
                                  zerolinewidth=1,ticklen=5,gridwidth=2),
                     yaxis1 = dict(gridcolor = 'rgb(255, 255, 255)',
                                 zerolinewidth=1,ticklen=5,gridwidth=2),
                     xaxis2 = dict(gridcolor = 'rgb(255, 255, 255)',
                                  zerolinewidth=1,ticklen=5,gridwidth=2),
                     yaxis2 = dict(gridcolor = 'rgb(255, 255, 255)',
                                 zerolinewidth=1,ticklen=5,gridwidth=2))


py.iplot(fig)


# # ARIMA Model
# ## AR model

# In[ ]:


from statsmodels.tsa.arima_model import ARIMA

#ARIMA model
def arima_model(time_series,p,d,q) :
    arima_model   = ARIMA(time_series , order = (p,d,q))
    results_arima = arima_model.fit(disp = -1)
    fitted_values = results_arima.fittedvalues
    
    trace1 = plot_line(fitted_values.index,
                       fitted_values.values,
                       "blue","fitted values")
    
    trace2 = plot_line(ts_fare_diff.index,
                       ts_fare_diff["fare_amount"],
                       "red","log differenced values")

    layout = plot_layout(("ARIMA model p = " + str(p) + 
                          ", d = " + str(d) + ", q = " + str(q)))
    data  = [trace2,trace1]
    fig   = go.Figure(data = data,layout = layout)
    py.iplot(fig)
    print (results_arima.summary())
    
arima_model(log_ts_fare,1,1,0)


# ## MA model

# In[ ]:


arima_model(log_ts_fare,0,1,1)


# ## Combined ARIMA model`

# In[ ]:


arima_model(log_ts_fare,1,1,1)


# In[ ]:


arima_model(log_ts_fare,2,1,2)


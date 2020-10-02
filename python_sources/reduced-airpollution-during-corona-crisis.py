#!/usr/bin/env python
# coding: utf-8

# # Reduced air pollution in India caused by measures covid-19
# ### By Jurjen Verbruggen
# 
# ---
# 
# Many countries have decided to lock down their citizens inside of their homes. People can only leave their houses when they need to be outside, for example to get some groceries.
# I suspect there is a decrease in air pollution during this crisis, but the question is, by how much?
# 
# One of the countries that allows us to measure this accurately is India. Because I could only find accurate pollution data about this country, I will show you all the effects the corona virus had on this country's pollution.

# # Polluters
# There are several polluters you should know about. The ones I will cover in this notebook are the following:
# 
# * NO2 - Nitrogen dioxide
# * CO - Carbon monoxide
# * SO2 - Sulfur dioxide
# * PM10 - Coarse particulate matter
# * PM2.5 - Fine particulate matter
# 
# These make up most of the concerns when looking at airpollution. They are all hazardous for living being when you get them inside your body.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
py.init_notebook_mode(connected=True)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


airquality = pd.read_csv('../input/air-quality-data-in-india/city_day.csv')
cities = pd.read_csv('../input/indian-cities-database/Indian Cities Database.csv')


# In[ ]:


airquality['Date'] = pd.to_datetime(airquality['Date'])
airquality


# Looking for cities I could use with enough datapoints
# ---
# The more data the merrier. I will pick the top few cities to use in this story.

# In[ ]:


#Finding which cities I could use
airquality_count_city = airquality.groupby("City").agg("count").sort_values("Date", ascending=False)
airquality_count_city.head(4)


# In[ ]:


airquality_filtered = airquality[airquality["Date"] > "2020-01-01"][airquality["Date"] < "2020-04-30"]


# # Pollution over time
# These graphs below show the amounts of pollution in cities of different polluters. 
# One thing that stands out with all of these charts is that a significant decrease of air pollution occurs around the end of march. This correlates with the measures that were taken in India, since they entered a lockdown on March 25th 2020.
# 
# There are several spikes in the graphs, one of which is on january 26th, which can be explained by India's "Republic Day", which is a national holiday. I cannot really explain the spikes from mid february until march, but I think it has something to do with the change of season and weather, where people want to go outside.
# 
# Something else that stands out is the fact that the CO emission in Ahmedabad is much higher than in other cities. This can be caused by the rush hours, where traffic is very dense, but also the fact that the city has an international airport in the middle of it, whereas airports at the other cities lie more outside of them. It can also depend on the location of the device that measures these numbers.

# In[ ]:


df = airquality_filtered

def showscatterpolluters(cityname):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df[df["City"] == cityname]["NO2"], name="NO2"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df[df["City"] == cityname]["CO"], name="CO"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df[df["City"] == cityname]["SO2"], name="SO2"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df[df["City"] == cityname]["PM10"], name="PM10"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df[df["City"] == cityname]["PM2.5"], name="PM2.5"))

    fig.update_xaxes(rangeslider_visible=True)
    
    fig.update_layout(title="Polluters in " + cityname)

    fig.show()

showscatterpolluters("Ahmedabad")
showscatterpolluters("Bengaluru")
showscatterpolluters("Mumbai")
showscatterpolluters("Lucknow")


# ![Before and after india pollution](https://smartairfilters.com/wordpress/wp-content/uploads/2020/04/Delhi-India-Gate-1-750x410.jpg)

# In[ ]:


df = airquality_filtered

beforemean = df[df["Date"] <= "2020-03-23"]["NO2"].mean()
aftermean = df[df["Date"] > "2020-03-23"]["NO2"].mean()

labels = ["Before", "After"]
values = [beforemean-aftermean, aftermean]

fig = go.Figure(data=[go.Pie(values=values, labels=labels)])

fig.update_traces(hoverinfo="label+percent", marker=dict(colors=["white", "yellowgreen"], line=dict(color='#000000', width=0.2)))

fig.update_layout(title_text="That's an average decrease of almost 55% in airpolluters!")

fig.show()


# # Air Quality Index (AQI)
# The Air Quality Index can tell us more about the actual quality of the air, and how much it can harm us.
# The AQI usually only considers six kinds of polluters, which are ozone, carbon monoxide, sulfur dioxide, nitrogen dioxide, PM10 and PM2.5 (these are suspended particles).
# The higher the AQI gets, the more harmful it is to us.

# ![AQI info](https://w.ndtvimg.com/sites/3/2019/12/18122812/air_pollution_standards_cpcb.png)

# In[ ]:


df = airquality_filtered

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["Date"], 
    y=df["AQI"], 
    name="Air Quality Index measure", 
    mode='markers', 
    marker=dict(
        color=np.random.randn(100000),
        colorscale='Earth',
        line_width=1,
        opacity=0.4
    )))

avgperdaydf = df[["Date", "AQI", "AQI_Bucket"]]
avgperday = avgperdaydf.groupby('Date').mean()
avgperdaymax = avgperdaydf.groupby('Date').max()

# fig.add_trace(go.Scatter(
#     x=avgperday.index,
#     y=avgperday["AQI"],
#     mode="lines",
#     name="Average per day",
#     marker = dict(
#         color="red"
#     )))

avgper5days = avgperday.rolling(5).mean()
avgper10days = avgperday.rolling(10).mean()

fig.add_trace(go.Scatter(
    x=avgper5days.index,
    y=avgper5days["AQI"],
    mode="lines",
    name="Average over 5 days",
    marker = dict(
        color="red"
    )))


fig.add_trace(go.Scatter(
    x=avgper10days.index,
    y=avgper10days["AQI"],
    mode="lines",
    name="Average over 10 days",
    marker = dict(
        color="purple"
    )))


fig.add_trace(go.Scatter(
    x=avgper5days.index,
    y=avgperdaymax["AQI"],
    mode="lines",
    name="Max that day",
    marker = dict(
        color="black"
    )))

fig.add_shape(
    type="line",
    x0="2020-01-01",
    y0=100,
    x1="2020-05-01",
    y1=100,
    line=dict(
        color="darkgreen",
        width=3,
        dash="dashdot",
    ))

fig.update_yaxes(range=[0, 400])

fig.update_layout(title_text="Air Quality Index over time")

fig.show()


# The graph above shows us all the datapoints we have for looking at the airpollution in all of India. The red and purple lines are the average over a few days, the black line is the maximum measurement on a day, and the green dotted line is the WHO standard of a *Satisfactory* air quality.
# 
# We can also visualize this in another way. By transforming the average into a barchart, and color coding whether the average is within the WHO guidelines, we can quickly see whether the air quality is good.

# In[ ]:


avgperday


# In[ ]:


df = avgperdaymax

#df = df.fillna(0)

bins = [0, 50, 100, 200, 300, 400]
binslabels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

colors = {"Good": "green",
         "Satisfactory": "lightgreen",
         "Moderate": "orange",
         "Poor": "red",
         "Very Poor": "darkred",
         "Severe": "black"}

def getbin(value):
    i = 1
    while i < 6:
        if value < bins[i]:
            return i-1
        else:
            i+=1
    return i-1

def getAQIgraph(df, desc):
    aqilabels = []
    for avg in df["AQI"]:
        binindex = getbin(avg)
        label = binslabels[binindex]
        aqilabels.append(label)

    aqilabels = pd.Series(aqilabels)
    df = df.reset_index()
    aqilabels = pd.concat([df, aqilabels.rename("AQI_label")], axis=1)
    aqilabels

    bars = []
    for label, label_df in aqilabels.groupby('AQI_label'):
        bars.append(go.Bar(x=label_df["Date"],
                           y=label_df["AQI"],
                           name=label,
                           marker={'color': colors[label]}))

    fig = go.FigureWidget(data=bars)
    fig.update_layout(title_text="AQI index, " + desc)
    
    fig.show()
    
getAQIgraph(avgperday, "AVERAGE measurement per day")


# This first graph shows us that the average is not the worst, but it is still worse than what the World Health Organization says is good for you. This is also kinda misleading, because low measurements cancel the very high ones out, so I decided to capture the highest measurements in another chart.

# In[ ]:


getAQIgraph(df, "MAXIMUM measurement per day")


# Those dark red bars are very poor, which can lead to serious health issues. The black bars are just straight up **deadly**.

# This is what those numbers look like
# ---
# 
# 
# ![indiabadpollution](https://www.aljazeera.com/mritems/imagecache/mbdxxlarge/mritems/Images/2018/12/24/003d801203d942a59976fb1fe7e6b974_18.jpg)

# In[ ]:


cities


# In[ ]:


airquality_filteredwlonlat = airquality_filtered.set_index('City').join(cities.set_index('City'))
airquality_filteredwlonlat


# # Before and after measures
# 
# Here you can see very clearly the hotspots of these emissions, before and after corona measures. You can see that emissions are reduced by more than 50%!

# In[ ]:


import plotly.express as px
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2)

aqfll_before = airquality_filteredwlonlat[airquality_filteredwlonlat["Date"] < "2020-03-07"]
aqfll_after = airquality_filteredwlonlat[airquality_filteredwlonlat["Date"] > "2020-04-01"]

# this somehow didnt work they way it should :(
#fig.add_densitymapbox(lat=aqfll_before['Lat'], lon=aqfll_before['Long'], z=aqfll_before['SO2'], radius=30)

fig = px.density_mapbox(aqfll_before, lat='Lat', lon='Long', z='SO2', radius=30, 
                        center=dict(lat=20, lon=80), zoom=3,
                        range_color=[0,150],
                        mapbox_style="stamen-watercolor") #if not good use mapbox_style="carto-positron"
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
    width=500,
)
fig.update_traces(opacity=0.5)
fig.show()

fig = px.density_mapbox(aqfll_after, lat='Lat', lon='Long', z='SO2', radius=30, 
                        center=dict(lat=20, lon=80), zoom=3,
                        range_color=[0,150],
                        mapbox_style="stamen-watercolor") #if not good use mapbox_style="carto-positron"
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
    width=500
)

fig.update_traces(opacity=0.5)
fig.show()


# In[ ]:


so2_before_mean = aqfll_before["SO2"].mean()
so2_after_mean = aqfll_after["SO2"].mean()
pm25_before_mean = aqfll_before["PM2.5"].mean()
pm25_after_mean = aqfll_after["PM2.5"].mean()
pm10_before_mean = aqfll_before["PM10"].mean()
pm10_after_mean = aqfll_after["PM10"].mean()
co_before_mean = aqfll_before["CO"].mean()
co_after_mean = aqfll_after["CO"].mean()
no2_before_mean = aqfll_before["NO2"].mean()
no2_after_mean = aqfll_after["NO2"].mean()

fig = make_subplots(rows=2, cols=3, subplot_titles=("SO2 (ug/m3)","PM2.5 (ug/m3)", "PM10 (ug/m3)", "CO (ug/m3)", "NO2 (ug/m3)"))
fig.add_trace(go.Bar(y=[so2_before_mean], name="Before"), row=1, col=1)
fig.add_trace(go.Bar(y=[so2_after_mean], name="After"), row=1, col=1)
fig.add_trace(go.Bar(y=[pm25_before_mean], name="Before"), row=1, col=2)
fig.add_trace(go.Bar(y=[pm25_after_mean], name="After"), row=1, col=2)
fig.add_trace(go.Bar(y=[pm10_before_mean], name="Before"), row=1, col=3)
fig.add_trace(go.Bar(y=[pm10_after_mean], name="After"), row=1, col=3)

fig.add_trace(go.Bar(y=[co_before_mean], name="Before"), row=2, col=1)
fig.add_trace(go.Bar(y=[co_after_mean], name="After"), row=2, col=1)
fig.add_trace(go.Bar(y=[no2_before_mean], name="Before"), row=2, col=2)
fig.add_trace(go.Bar(y=[no2_after_mean], name="After"), row=2, col=2)

fig.update_layout(showlegend=False, title_text="A detailed look into reduction of several types of polluters")

fig.show()


# In[ ]:


from google.cloud import bigquery

client = bigquery.Client()

dataset_ref = client.dataset("openaq", project="bigquery-public-data")
table_ref = dataset_ref.table("global_air_quality")
table = client.get_table(table_ref)

dfpollution = client.list_rows(table).to_dataframe()
dfpoldelhi = dfpollution[dfpollution["city"] == "Delhi"]
dfpoldelhi = dfpoldelhi[dfpoldelhi["pollutant"] == "co"]

dfpoldelhi.sort_values("timestamp", ascending=False).head(40)
dfpollution


# In[ ]:


# Making sure timestamp is date only, ditch the time 
dfpolfixtimestamp = dfpollution.copy()
dfpolfixtimestamp_coltimestamp = dfpolfixtimestamp["timestamp"].astype(str)
dfpolfixtimestamp_coltimestamp = dfpolfixtimestamp_coltimestamp.str.slice(start=0, stop=10)
dfpolfixtimestamp["timestamp"] = dfpolfixtimestamp_coltimestamp
dfpollution = dfpolfixtimestamp
dfpollution.head()


# In[ ]:


dfpollimit = dfpollution[dfpollution["timestamp"] > "2020-03-15"][dfpollution["timestamp"] < "2020-05-01"]
dfpollimit = dfpollimit[dfpollimit["averaged_over_in_hours"].notna()]
dfpollimitcounts = dfpollimit.groupby("country").agg("count").sort_values("location", ascending=False)
dfpollimitcounts


# In[ ]:


dfpollimitIT = dfpollimit[dfpollimit["pollutant"] == "co"];

dfpollimitIT


# In[ ]:


#dfpollimit = dfpollimit[abs(dfpollimit["value"]) < 2500][dfpollimit["value"] > 0];


fig, ax = plt.subplots();
ax.set_ylabel('');

sns.relplot(data=dfpollimitIT, x='timestamp', y='value', kind="line", aspect=2.5, ax=ax);
mean = dfpollimitIT.mean();
mean.name = "average";
dfpollimitIT.append(mean).plot(kind="line")
#sns.lineplot(data=dfpollimit, x='timestamp', y='value', estimator=None, lw=1)


# In[ ]:


dfvirus = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")


# In[ ]:


dfcountries = pd.read_json("../input/country-data/names.json", orient="records", typ="series")
dfcountries = pd.DataFrame(dfcountries.items(), columns=["code", "name"])


# In[ ]:


dfpollutionwcountries = pd.merge(dfpollution, dfcountries, how='left', left_on='country', right_on='code')


# In[ ]:


#filtering dataframe
dfvirus = dfvirus.loc[dfvirus['country'] == "China"]
dfviruslocations = dfvirus[['longitude', 'latitude']]
dfviruslocations = dfviruslocations.dropna();

#colors
sns.set_palette("husl")

#calculating clusters
kmeans = KMeans(n_clusters=3).fit(dfviruslocations)
centroids = kmeans.cluster_centers_

#give each point a hue according to their cluster
sns.scatterplot(data=dfviruslocations, x='longitude', y='latitude', hue=kmeans.labels_.astype(float));


# In[ ]:


sns.scatterplot(data=dfvirus, x='longitude', y='latitude');


# In[ ]:


dfpollutionchina = dfpollutionwcountries.loc[dfpollutionwcountries['pollutant'] == "co"].loc[dfpollutionwcountries['country'] == "CN"]
dfpollutionchina = dfpollutionchina.sort_values(by=['timestamp'])
#dfpollutionchina['timestamp'] = pd.to_datetime(dfpollutionchina['timestamp'])
dfpollutionchina['timestamp'] = dfpollutionchina['timestamp'].dt.strftime('%Y/%m/%d')

sns.scatterplot(data=dfpollutionchina, x='timestamp', y='value')
sns.regplot(data=dfpollutionchina, x='timestamp', y='value', color="g")


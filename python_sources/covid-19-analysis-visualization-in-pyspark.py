#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


from pyspark.sql.functions import *
from pyspark.sql import SparkSession
spark = SparkSession .builder .appName("Python Spark create RDD example") .config("spark.some.config.option", "some-value") .getOrCreate()


# In[ ]:


df = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("../input/covid-19-datascv/covid_19_data.csv",header=True)


# In[ ]:


df.printSchema()


# In[ ]:


df.show(5)


# In[ ]:


Province = df.select('Province/State').fillna('Unknown')


# In[ ]:


df = df.fillna({'Province/State':'Unknown'})


# In[ ]:


df.show(5)


# ### Replacing "Mainland China" with "China" 

# In[ ]:


df = df.withColumn('Country/Region', regexp_replace('Country/Region', 'Mainland China', 'China'))


# In[ ]:


df.show(5)


# ### Creating New Column in DF "Active_cases"

# In[ ]:


df = df.withColumn("Active_case", df['Confirmed'] - df['Deaths'] - df['Recovered'])


# In[ ]:


df.show(5)


# ### Creating chronological order DataFrame of sum(Recovered), sum(Deaths), sum(Confirmed) and sum(Active_case) 

# In[ ]:


max_date =  df.select(max("ObservationDate")).first()
group = df.groupBy("ObservationDate")
group_data = group.agg({'Confirmed':'sum', 'Deaths':'sum', 'Recovered':'sum', 'Active_case':'sum'}).sort(col("ObservationDate"))
group_data.show()


# ### Line Chart of increase in 'Recovered', 'Deaths', 'Confirmed', 'Active_case'

# In[ ]:


group_data=group_data.toPandas()
import matplotlib.pyplot as plt
ObservationDate = group_data['ObservationDate']
Recovered = group_data['sum(Recovered)']
Deaths = group_data['sum(Deaths)']
Confirmed = group_data['sum(Confirmed)']
Active_case = group_data['sum(Active_case)']
plt.figure(figsize=(20,10))
l1, = plt.plot(ObservationDate, Recovered, color='g')
l2, = plt.plot(ObservationDate, Deaths, color='r')
l3, = plt.plot(ObservationDate, Confirmed, color='b')
l4, = plt.plot(ObservationDate, Active_case, color='orange')
patches = [l1,l2,l3,l4]
labels = ['Recovered', 'Deaths', 'Confirmed', 'Active_case']
plt.legend(patches, labels, loc="best")
plt.xlabel('Date')
plt.ylabel('Number of cases')
plt.title('increace in cases')
plt.xticks(rotation=90)
plt.show()


# ### Showing world wide Latest Data

# In[ ]:


from pyspark.sql import functions as F
mx_date=df.select(F.max("ObservationDate")).collect()[0][0]
Data_world = df.filter(F.col("ObservationDate")==mx_date).groupBy("ObservationDate").agg({'Confirmed':'sum', 'Deaths':'sum', 'Recovered':'sum', 'Active_case':'sum'})
Data_world=Data_world.toPandas()


# In[ ]:


Data_world.head()


# In[ ]:


import plotly.express as px
labels = ["Active cases","Recovered","Deaths"]
values = Data_world.loc[0, ["sum(Active_case)","sum(Recovered)","sum(Deaths)"]]
fig = px.pie(Data_world, values=values, names=labels, color_discrete_sequence=['blue','green','red'])
fig.update_layout(
    title='Total cases : '+str(Data_world["sum(Confirmed)"][0]),
)
fig.show()


# ### sum(Recovered) sum(Deaths) sum(Confirmed) sum(Active_case) in each country

# In[ ]:


Data_world_byCountry = df.filter(F.col("ObservationDate")==mx_date).groupBy("Country/Region").agg({'Confirmed':'sum','Deaths':'sum','Recovered':'sum','Active_case':'sum'}).sort(col("sum(Confirmed)").desc())


# In[ ]:


Data_world_byCountry=Data_world_byCountry.toPandas()
Data_world_byCountry.head()


# In[ ]:


import plotly.graph_objects as go
fig = go.Figure(data=[go.Bar(
            x=Data_world_byCountry['Country/Region'][0:10], y=Data_world_byCountry['sum(Confirmed)'][0:10],
            text=Data_world_byCountry['sum(Confirmed)'][0:10].apply(str),
            textposition='auto')])
fig.update_layout(
    title='Most 10 infected Countries',
    xaxis_title="Countries",
    yaxis_title="Confirmed Cases",
)
fig.show()
fig = go.Figure(data=[go.Scatter(
    x=Data_world_byCountry['Country/Region'][0:10],
    y=Data_world_byCountry['sum(Confirmed)'][0:10],
    mode='markers',
    marker=dict(
        color=[145, 140, 135, 130, 125, 120,115,110,105,100],
        size=[100, 90, 70, 60, 60, 60,50,50,40,35],
        showscale=True
        )
)])
fig.update_layout(
    title='Most 10 infected Countries',
    xaxis_title="Countries",
    yaxis_title="Confirmed Cases",
)
fig.show()


# In[ ]:


import plotly.graph_objects as go
fig = go.Figure(data=[go.Bar(
            x=Data_world_byCountry['Country/Region'][0:10], y=Data_world_byCountry['sum(Deaths)'][0:10],
            text=Data_world_byCountry['sum(Deaths)'][0:10].apply(str),
            textposition='auto')])
fig.update_layout(
    title='Most 10 infected Countries',
    xaxis_title="Countries",
    yaxis_title="Death Cases",
)
fig.show()
fig = go.Figure(data=[go.Scatter(
    x=Data_world_byCountry['Country/Region'][0:10],
    y=Data_world_byCountry['sum(Deaths)'][0:10],
    mode='markers',
    marker=dict(
        color=[145, 140, 135, 130, 125, 120,115,110,105,100],
        size=[100, 90, 70, 60, 60, 60,50,50,40,35],
        showscale=True
        )
)])
fig.update_layout(
    title='Most 10 infected Countries',
    xaxis_title="Countries",
    yaxis_title="Death Cases",
)
fig.show()


# In[ ]:


import plotly.graph_objects as go
fig = go.Figure(data=[go.Bar(
            x=Data_world_byCountry['Country/Region'][0:10], y=Data_world_byCountry['sum(Recovered)'][0:10],
            text=Data_world_byCountry['sum(Recovered)'][0:10].apply(str),
            textposition='auto')])
fig.update_layout(
    title='Most 10 infected Countries',
    xaxis_title="Countries",
    yaxis_title="Recovered Cases",
)
fig.show()
fig = go.Figure(data=[go.Scatter(
    x=Data_world_byCountry['Country/Region'][0:10],
    y=Data_world_byCountry['sum(Recovered)'][0:10],
    mode='markers',
    marker=dict(
        color=[145, 140, 135, 130, 125, 120,115,110,105,100],
        size=[100, 90, 70, 60, 60, 60,50,50,40,35],
        showscale=True
        )
)])
fig.update_layout(
    title='Most 10 infected Countries',
    xaxis_title="Countries",
    yaxis_title="Recovered Cases",
)
fig.show()


# In[ ]:


import plotly.graph_objects as go
fig = go.Figure(data=[go.Bar(
            x=Data_world_byCountry['Country/Region'][0:10], y=Data_world_byCountry['sum(Active_case)'][0:10],
            text=Data_world_byCountry['sum(Active_case)'][0:10].apply(str),
            textposition='auto')])
fig.update_layout(
    title='Most 10 infected Countries',
    xaxis_title="Countries",
    yaxis_title="Active Cases",
)
fig.show()
fig = go.Figure(data=[go.Scatter(
    x=Data_world_byCountry['Country/Region'][0:10],
    y=Data_world_byCountry['sum(Active_case)'][0:10],
    mode='markers',
    marker=dict(
        color=[145, 140, 135, 130, 125, 120,115,110,105,100],
        size=[100, 90, 70, 60, 60, 60,50,50,40,35],
        showscale=True
        )
)])
fig.update_layout(
    title='Most 10 infected Countries',
    xaxis_title="Countries",
    yaxis_title="Active_case Cases",
)
fig.show()


# In[ ]:


Pakistan_Data = df.filter(F.col("Country/Region")=='Pakistan')


# In[ ]:


max_date =  df.select(max("ObservationDate")).first()
groupP = Pakistan_Data.groupBy("ObservationDate")
group_dataP = groupP.agg({'Confirmed':'sum', 'Deaths':'sum', 'Recovered':'sum', 'Active_case':'sum'}).sort(col("ObservationDate"))
group_dataP.show(1000)


# In[ ]:


Pakistan_Data.show()


# In[ ]:


group_dataP=group_dataP.toPandas()
fig = go.Figure()
fig.add_trace(go.Scatter(x=group_dataP['ObservationDate'], y=group_dataP['sum(Confirmed)'],
                    mode='lines',
                    name='Confirmed cases'))

fig.add_trace(go.Scatter(x=group_dataP['ObservationDate'], y=group_dataP['sum(Active_case)'],
                    mode='lines',
                    name='Active cases',line=dict( dash='dot')))
fig.add_trace(go.Scatter(x=group_dataP['ObservationDate'], y=group_dataP['sum(Deaths)'],name='Deaths',
                                   marker_color='black',mode='lines',line=dict( dash='dot') ))
fig.add_trace(go.Scatter(x=group_dataP['ObservationDate'], y=group_dataP['sum(Recovered)'],
                    mode='lines',
                    name='Recovered cases',marker_color='green'))
fig.show()


# In[ ]:


Data_Pakistan_Last = df.filter(F.col("Country/Region")=='Pakistan').filter(F.col("ObservationDate")==mx_date).groupBy("ObservationDate").agg({'Confirmed':'sum', 'Deaths':'sum', 'Recovered':'sum', 'Active_case':'sum'})


# In[ ]:


Data_Pakistan_Last=Data_Pakistan_Last.toPandas()
import plotly.express as px
labels = ["Active cases","Recovered","Deaths"]
values = Data_Pakistan_Last.loc[0, ["sum(Active_case)","sum(Recovered)","sum(Deaths)"]]
fig = px.pie(Data_Pakistan_Last, values=values, names=labels, color_discrete_sequence=['blue','green','red'])
fig.update_layout(
    title='Total cases : '+str(Data_Pakistan_Last["sum(Confirmed)"][0]),
)
fig.show()


# # **Please give an UPVOTE if you like pyspark work notebook**

# In[ ]:





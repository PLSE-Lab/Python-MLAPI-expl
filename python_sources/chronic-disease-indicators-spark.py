#!/usr/bin/env python
# coding: utf-8

# # U.S. Chronic Disease Indicators (CDI)
# 
# *"CDC's Division of Population Health provides cross-cutting set of 124 indicators that were developed by consensus and that allows states and territories and large metropolitan areas to uniformly define, collect, and report chronic disease data that are important to public health practice and available for states, territories and large metropolitan areas. In addition to providing access to state-specific indicator data, the CDI web site serves as a gateway to additional information and data resources."* - DATA.GOV
# 
# **This notebook aims to validate the hypothesis that the probability of a disease to happen is related to where it happened, the location.**
# 
# Data source: https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi
# 
# P.S.: I can't run spark code on Kaggle, so I posted the results and the code is right below it.

# <iframe width="900" height="800" frameborder="0" scrolling="no" src="//plot.ly/~tatianass/7.embed"></iframe>

# #### Results
# 
# As can be seen in the [chart above](https://plot.ly/~tatianass/7/frequency-of-cities-in-top-10-ranking-for-diseases/), some cities, specially Arizona, have come up to 10 times in the top 10 ranking cities where a specific disease appeared. Proving our hypothesis that the probability of a disease to happen is correlated to the location.

# ### Load data and Setup

# In[ ]:


### Spark ###
# To find out where the pyspark
import findspark
findspark.init()

# Creating Spark Context
from pyspark import SparkContext
from pyspark.sql import functions as F
from pyspark.sql import SQLContext
from pyspark.sql import Window
from pyspark.sql.types import StructType, StringType, FloatType

### Working with data ###
import numpy as np
import pandas as pd

### Visualization ###
import altair as alt
import plotly.plotly as py
import plotly.graph_objs as go

### Utils ###
import datetime
from urllib.request import urlopen

# File format #
import csv
import json
import xml.etree.ElementTree as ET


#### Setup plotly
py.sign_in(username='**your username**', api_key='**your key**')

#### Setup Spark
sc = SparkContext("local", "Testing")

spark = SQLContext(sc)

#### Load information
df = spark.read.csv("data/disease_indicators.csv", sep=',', header=True)

# Repartition the data by Topic
df2 = df.repartition('Topic')
df2.cache()
df2.rdd.getNumPartitions()

### Prepare Data
df_location = (df2
               .filter("GeoLocation IS NOT NULL AND Topic IS NOT NULL AND DataValue IS NOT NULL")
               .withColumn('DataValue', F.col('DataValue').cast('Float'))
              ).filter("DataValue IS NOT NULL")

#### Get top 10 cities by disease

### With Hive
df_location.createOrReplaceTempView('df')

top_10_cities = spark.sql("""
SELECT *
FROM (
    SELECT Topic, LocationDesc AS city, DataValue,
    DENSE_RANK() over (PARTITION BY Topic ORDER BY DataValue DESC) as dense_rank
    FROM df
)
WHERE dense_rank <=10
""")

### With PySpark
# top_10_window = Window.partitionBy('Topic').orderBy(F.col('DataValue').desc())
# top_10_cities = (
#     df_location.select('Topic', F.col('LocationDesc').alias('city'), 'DataValue', F.dense_rank().over(top_10_window).alias('rank')) # Using dense rank to get cities with similar positions
#     .filter(F.col('rank') <= 10)
# )

to_n_ranks_cities = (
    top_10_cities
    .groupBy('city')
    .agg(F.countDistinct('Topic').alias('Number of times in top 10'))
    .orderBy('Number of times in top 10') # Since the orientation is horizontal, the sort must be the inverse order of what I want
).toPandas()

### Testing Hypothesis
data = [go.Bar(
            y=to_n_ranks_cities['city'],
            x=to_n_ranks_cities['Number of times in top 10'],
            orientation = 'h',
            text=to_n_ranks_cities['Number of times in top 10']
    )]

layout = go.Layout(
    title='Frequency of Cities in top 10 ranking for diseases',
    titlefont=dict(size=20),
    width=1000,
    height=1400,
    yaxis=go.layout.YAxis(
        ticktext=to_n_ranks_cities['city'],
        tickmode='array',
        automargin=True
    )
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='cities-rank-frequency')


#!/usr/bin/env python
# coding: utf-8

# **Pain Pills in the USA**
# * Data from [The Washington Post](https://www.washingtonpost.com/graphics/2019/investigations/dea-pain-pill-database)
# 

# Step 1: Import Python modules and define helper functions
# * [Dask](https://dask.org/)
# * [Plotly](https://plot.ly/python/)

# In[ ]:


import pandas as pd 
import dask.dataframe as dd
import os
from IPython.core import display as ICD
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')

def valueCounter(dataframe,columnToCount):
    valueCounts = dataframe[columnToCount].value_counts().compute()
    df = pd.DataFrame(data=valueCounts)
    return df

def quantityCounter(dataframe,columnToCount):
    quantityCount = dataframe.groupby(columnToCount)['QUANTITY'].sum()
    quantityCount = quantityCount.compute().sort_values(ascending=False)
    df = pd.DataFrame(data=quantityCount)
    return df

def calculateQuantityPerPerson(dataframe,columnToCount,indexValue,populationValue):
    cityQuantity = dataframe.groupby(columnToCount)['QUANTITY'].sum()
    cityQuantity = cityQuantity.compute().sort_values(ascending=False)[indexValue]
    quantityPerPerson = cityQuantity/populationValue
    return quantityPerPerson

def plotHorizontalBarGraph(dataframe,column,title,yAxisTitle,width,height,begin,end):
    trace1 = go.Bar(
                    x = dataframe[column][begin:end],
                    y = dataframe.index,
                    orientation='h',
                    name = "Kaggle",
                    marker = dict(color = 'rgba(0, 0, 255, 0.8)',
                                 line=dict(color='rgb(0,0,0)',width=1.5)),
                    text = dataframe.index)
    data = [trace1]
    layout = go.Layout(barmode = "group",title=title, 
                       xaxis=dict(title=yAxisTitle),
                       yaxis=dict(autorange="reversed"),
                       width = width,
                       height = height,
                       showlegend=False)
    fig = go.Figure(data = data, layout = layout)
    fig.update_yaxes(automargin=True)
    iplot(fig)

TRAIN_PATH = '../input/arcos-co-statewide-itemized.tsv'
interestingColumns = ['BUYER_NAME','BUYER_ADDRESS1', 'BUYER_ADDRESS2', 'BUYER_CITY', 'BUYER_STATE',
       'BUYER_ZIP', 'BUYER_COUNTY','DRUG_NAME', 'QUANTITY', 'UNIT', 'TRANSACTION_DATE', 'CALC_BASE_WT_IN_GM',
       'DOSAGE_UNIT', 'Product_Name', 'Ingredient_Name','Revised_Company_Name', 'Reporter_family']
df_tmp = pd.read_csv(TRAIN_PATH,sep='\t',usecols=interestingColumns,nrows=5)
df_tmp.head()


# Step 2: Load the data using Dask 
# * 1 x 1.3GB TSV file
# * adapted from [jpmiller/szelee's method](https://www.kaggle.com/szelee/how-to-import-a-csv-file-of-55-million-rows)

# In[ ]:


traintypes = {'BUYER_NAME': 'str',
              'BUYER_ADDRESS1': 'str', 
              'BUYER_ADDRESS2': 'str',
              'BUYER_CITY': 'str',
              'BUYER_STATE': 'str',
              'BUYER_ZIP': 'int64',
              'BUYER_COUNTY': 'str',
              'DRUG_NAME': 'str',
              'QUANTITY': 'int64',
              'UNIT': 'str',
              'TRANSACTION_DATE': 'int64',
              'CALC_BASE_WT_IN_GM': 'float64',
              'DOSAGE_UNIT': 'float64',
              'Product_Name': 'str',
              'Ingredient_Name': 'str',
              'Revised_Company_Name': 'str',
              'Reporter_family': 'str'}
cols = list(traintypes.keys())
df = dd.read_csv(TRAIN_PATH,sep='\t', usecols=interestingColumns,dtype=traintypes)
df.head(5)


# Step 3: Visualize the data

# In[ ]:


drugCounts = quantityCounter(df,'DRUG_NAME')
plotHorizontalBarGraph(drugCounts,'QUANTITY','Most Popular Pain Pills','Quantity of Pain Pills',500,500,0,2)


# In[ ]:


cityCounts = quantityCounter(df,'BUYER_CITY')
plotHorizontalBarGraph(cityCounts,'QUANTITY','Pain Pills per City','Pain Pills per City',750,1500,0,50)


# In[ ]:


# Note population numbers are estimates from Wikipedia for the year 2018
value_list = ['CO']
df = df[df.BUYER_STATE.isin(value_list)]
print('Quantity of Pain Pills per Person: ','\n')
print('Denver: ',calculateQuantityPerPerson(df,'BUYER_CITY',1,716492))
print('Colorado Springs: ',calculateQuantityPerPerson(df,'BUYER_CITY',0,472688))
print('Pueblo: ',calculateQuantityPerPerson(df,'BUYER_CITY',2,111750))


# Step 4: Make conclusions 
# * Oxycodone is more popular than hydrocodone in Colorado.
# * Colorado Springs, Denver, and Pueblo consume the most pain pills in Colorado.
# * Pueblo consumes more than 5x as many pain pills per capita as compared to Denver. 

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
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import warnings
warnings.filterwarnings('ignore')
init_notebook_mode(connected=True)

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
    
TRAIN_PATH = '../input/arcos_all_washpost.tsv/arcos_all_washpost.tsv'
interestingColumns = ['BUYER_NAME','BUYER_ADDRESS1', 'BUYER_ADDRESS2', 'BUYER_CITY', 'BUYER_STATE',
       'BUYER_ZIP', 'BUYER_COUNTY','DRUG_NAME', 'QUANTITY', 'UNIT', 'TRANSACTION_DATE', 'CALC_BASE_WT_IN_GM',
       'DOSAGE_UNIT', 'Product_Name', 'Ingredient_Name','Revised_Company_Name', 'Reporter_family']
df_tmp = pd.read_csv(TRAIN_PATH,sep='\t',usecols=interestingColumns,nrows=5)
df_tmp.head()


# Step 2: Load the data using Dask 
# * 1 x 80GB TSV file
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


# Step 3: Visualize data for the entire USA

# In[ ]:


stateCounts = quantityCounter(df,'BUYER_STATE')
plotHorizontalBarGraph(stateCounts,'QUANTITY','Pain Pills per State','Pain Pills per State',750,1500,0,50)
del stateCounts


# In[ ]:


# Note population numbers are estimates from Wikipedia for the year 2018
print('Quantity of Pain Pills per Person: ','\n')
print('California: ',calculateQuantityPerPerson(df,'BUYER_STATE',0,39557045))
print('Florida: ',calculateQuantityPerPerson(df,'BUYER_STATE',1,21299325))
print('Texas: ',calculateQuantityPerPerson(df,'BUYER_STATE',2,28701845))


# In[ ]:


cityCounts = quantityCounter(df,'BUYER_CITY')
plotHorizontalBarGraph(cityCounts,'QUANTITY','Pain Pills per City','Pain Pills per City',750,1500,0,50)
del cityCounts


# In[ ]:


# Note population numbers are estimates from Wikipedia for the year 2018
print('Quantity of Pain Pills per Person: ','\n')
print('Livermore, California: ',calculateQuantityPerPerson(df,'BUYER_CITY',0,90269))
print('North Charleston, South Carolina: ',calculateQuantityPerPerson(df,'BUYER_CITY',1,113237))
print('Las Vegas, Nevada: ',calculateQuantityPerPerson(df,'BUYER_CITY',2,644644))


# Step 4: Compare top three states to the rest of the USA

# In[ ]:


drugCounts = quantityCounter(df,'DRUG_NAME')
plotHorizontalBarGraph(drugCounts,'QUANTITY','Most Popular Pain Pills in the USA','Quantity of Pain Pills',500,500,0,2)
del drugCounts


# In[ ]:


# Pick a few states to focus on
value_list = ['CA','FL','TX','CO']
df = df[df.BUYER_STATE.isin(value_list)]
drugCounts = quantityCounter(df,'DRUG_NAME')
plotHorizontalBarGraph(drugCounts,'QUANTITY','Most Popular Pain Pills in CA, FL, and TX','Quantity of Pain Pills',500,500,0,2)
del drugCounts


# Step 5: Compare to data from Colorado

# In[ ]:


# Note population numbers are estimates from Wikipedia for the year 2018
value_list = ['CO']
df = df[df.BUYER_STATE.isin(value_list)]


# In[ ]:


print('Quantity of Pain Pills per Person: ','\n')
print('Denver: ',calculateQuantityPerPerson(df,'BUYER_CITY',1,716492))
print('Colorado Springs: ',calculateQuantityPerPerson(df,'BUYER_CITY',0,472688))
print('Pueblo: ',calculateQuantityPerPerson(df,'BUYER_CITY',2,111750))


# Step 6: Make conclusions
# * Hydrocodone and oxycodone have similar sales levels in the USA.
# * California, Florida, and Texas consume more pain pills than any of other state.
# * Hydrocodone is more popular than oxycodone in California, Florida, and Texas.
# * Livermore, California consumes more pain pills than any other city in the USA.

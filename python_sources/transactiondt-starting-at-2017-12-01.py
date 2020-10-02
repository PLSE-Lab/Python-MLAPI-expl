#!/usr/bin/env python
# coding: utf-8

# From the [data description page](https://www.kaggle.com/c/ieee-fraud-detection/data) we know that "The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp)."
# 
# In this kernel we analyse the TransactionDT column, and support our hypothesis that the TransactionDT column starts at the 1st of December.
# 
# We can use the TransactionDT column to calculate features such as weekday and hour.

# In[ ]:


import numpy as np
import pandas as pd
import datetime
import math
import gc

from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, Span
from bokeh.io import show, output_file, output_notebook
from bokeh.layouts import gridplot
from bokeh.models.widgets import Panel, Tabs
from matplotlib.pyplot import plot
output_notebook()

PATH = '../input/'
START_DATE = '2017-12-01'


# In[ ]:


# Load data
train = pd.read_csv(PATH + 'train_transaction.csv')
test = pd.read_csv(PATH + 'test_transaction.csv')

n_train = train.shape[0]
df = pd.concat([train, test], axis = 0, sort = False)
del train, test
gc.collect()


# In[ ]:


# Preprocess date column
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

print(df['TransactionDT'].head())
print(df['TransactionDT'].tail())


# The transaction datetime seems to end at '2018-12-31 23:59:05'.

# In[ ]:


# Helper function for creating a histogram in Bokeh
def create_histogram(df, col_name, resample='D'):        
    counts = df[[col_name]].resample(resample, on=col_name).count()    
    cols = counts.index.astype(str).tolist()
    counts.index.name = 'date'
    counts.reset_index(inplace=True)
    
    # Add color and legend
    max_date_in_train = df.iloc[:n_train][col_name].max().strftime('%Y-%m-%d')
    counts.loc[counts.date <= max_date_in_train,'source'] = 'train'
    counts.loc[counts.date <= max_date_in_train,'color'] = 'mediumslateblue'
    counts.loc[counts.date > max_date_in_train,'source'] = 'test'
    counts.loc[counts.date > max_date_in_train,'color'] = 'mediumaquamarine'
    
    source = ColumnDataSource(counts)
        
    p = figure(title = 'Histogram of column {}'.format(col_name),
               x_axis_label = 'Date',
               x_axis_type = 'datetime',
               y_axis_label = 'Number of records')
    
    p.vbar(x = 'date', 
           top = col_name,
           source = source,
           line_color = 'color',
           fill_color = 'color',
           width = 1,
           legend = 'source')
    
    return p


# In[ ]:


# Generate and show chart
p = create_histogram(df, 'TransactionDT', resample='D')
show(p)


# There are peaks in transactions around christmas 2017 and christmas 2018. For both years, the peaks end at the 26th of December.

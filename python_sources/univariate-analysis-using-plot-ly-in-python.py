#!/usr/bin/env python
# coding: utf-8

# plotting variables from Adult dataset using plot.ly python library. It's really an easy, clean and beautiful way to analyze data.

# In[ ]:



# to show output for all lines in the cell not only the last one :
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# for example check the output of the following cell


# In[ ]:


x = 2
y = 2
x
y


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as gobj
py.init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/adult.csv')


# # first I'll plot variables which values are categorized.

# In[ ]:


categorized_cols = ['race','sex','marital.status',
            'relationship','workclass','occupation',
                    'education.num','education', 'native.country','income']


# In[ ]:


# input col_name :  column name from data(dataframe)
#       table : 0 or 1 : if 1 it will print the data in a table.
#       bar   : 0 0r 1 : if 1 it will draw the data in a bar chart.
# return a dataframe it columns are  : [ col_name, count, percent  ]
def plot_value_counts(col_name,table=False,bar=False):
    
    values_count = pd.DataFrame(data[col_name].value_counts())
    values_count.columns = ['count']
    # convert the index column into a regular column.
    values_count[col_name] = [ str(i) for i in values_count.index ]
    # add a column with the percentage of each data point to the sum of all data points.
    values_count['percent'] = values_count['count'].div(values_count['count'].sum()).multiply(100).round(2)
    # change the order of the columns.
    values_count = values_count.reindex_axis([col_name,'count','percent'],axis=1)
    values_count.reset_index(drop=True,inplace=True)
    
    if bar :
        # add a font size for annotations0 which is relevant to the length of the data points.
        font_size = 20 - (.25 * len(values_count[col_name]))
        
        trace0 = gobj.Bar( x = values_count[col_name], y = values_count['count'] )
        data_ = gobj.Data( [trace0] )
        
        annotations0 = [ dict(x = xi,
                             y = yi, 
                             showarrow=False,
                             font={'size':font_size},
                             text = "{:,}".format(yi),
                             xanchor='center',
                             yanchor='bottom' )
                       for xi,yi,_ in values_count.values ]
        
        annotations1 = [ dict( x = xi,
                              y = yi/2,
                              showarrow = False,
                              text = "{}%".format(pi),
                              xanchor = 'center',
                              yanchor = 'center',
                              font = {'color':'yellow'})
                         for xi,yi,pi in values_count.values if pi > 10 ]
        
        annotations = annotations0 + annotations1                       
        
        layout = gobj.Layout( title = col_name.replace('_',' ').capitalize(),
                             titlefont = {'size': 50},
                             yaxis = {'title':'count'},
                             xaxis = {'type':'category'},
                            annotations = annotations  )
        figure = gobj.Figure( data = data_, layout = layout )
        py.iplot(figure)
    
    if table : 
        values_count['count'] = values_count['count'].apply(lambda d : "{:,}".format(d))
        table = ff.create_table(values_count,index_title="race")
        py.iplot(table)
    
    return values_count


# In[ ]:


for col in categorized_cols:
    _ = plot_value_counts(col,0,1) 


# ----
# 
# # Now I'll plot the other discrete variables.

# In[ ]:


disc_cols = [ 'age', 'fnlwgt', 'capital.gain', 'capital.loss', 'hours.per.week' ]


# In[ ]:


def plot_histogram(col_name):
    series = data[col_name]
    # remove zero values items [ indicates NA values.]
    series = series[ series != 0 ]
    smin,smax = series.min(),series.max()
    # remove outliers for +- three standard deviations.
    series = series[ ~( ( series - series.mean() ).abs() > 3 * series.std() ) ]
    percentiles = [ np.percentile(series,n) for n in (2.5,50,97.5) ]
    
    trace0 = gobj.Histogram( x = series,
                            histfunc = 'avg', 
                            histnorm = 'probability density',
                            opacity=.75,
                           marker = {'color':'#EB89B5'})
    data_ = gobj.Data( [trace0] )
    
    shapes = [{ 'line': { 'color': '#0099FF', 'dash':'solid', 'width':2 },
                'type':'line',
                'x0':percentiles[0], 'x1':percentiles[0], 'xref':'x',
                'y0':-0.1, 'y1':1, 'yref':'paper' },
               
              { 'line': { 'color': '#00999F', 'dash':'solid', 'width':1 },
                'type':'line',
                'x0':percentiles[1], 'x1':percentiles[1], 'xref':'x',
                'y0':-0.1, 'y1':1, 'yref':'paper' },
    
              { 'line': { 'color': '#0099FF', 'dash':'solid', 'width':2 },
                'type':'line',
                'x0':percentiles[2], 'x1':percentiles[2], 'xref':'x',
                'y0':-0.1, 'y1':1, 'yref':'paper' } 
             ]
    
    annotations = [ {'x': percentiles[0], 'xref':'x','xanchor':'right',
                     'y': .3, 'yref':'paper', 
                     'text':'2.5%', 'font':{'size':16},
                     'showarrow':False},
                   
                    {'x': percentiles[1], 'xref':'x','xanchor':'center',
                     'y': .2, 'yref':'paper', 
                     'text':'95%<br>median = {0:,.2f}<br>mean = {1:,.2f}<br>min = {2:,}<br>max = {3:,}'
                         .format(percentiles[1],series.mean(),smin,smax), 
                     'showarrow':False,
                     'font':{'size':20} },
                   
                    {'x': percentiles[2], 'xref':'x','xanchor':'left',
                     'y': .3, 'yref':'paper', 
                     'text':'2.5%','font':{'size':16}, 
                     'showarrow':False},
                   
                    {'x': .5, 'xref':'paper','xanchor':'center',
                     'y': 1.1, 'yref':'paper','yanchor':'center', 
                     'text':'Outliers above or below three standard deviations are excluded from the graph, mean and median calculations.',
                     'font':{'size':15,'color':'rose'}, 
                     'showarrow':False} 
                  ]
    
    layout = gobj.Layout( title = col_name.replace('_',' ').capitalize(),
                        titlefont = {'size':'50'},
                        yaxis = {'title':'Probability/Density'},
                        xaxis = {'title':col_name, 'type':'discrete'},
                        shapes = shapes,
                         annotations = annotations
                         )
    figure = gobj.Figure(data = data_, layout = layout)
    py.iplot(figure)


# In[ ]:


for col in disc_cols :
    plot_histogram( col )


# ----------
# 

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, he
re's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
#from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **VISUALLIZATION-BASIC CHART WITH PLOTLY(BEGINNER)**
# * Amount of Protein, Carbonhydrate and Fat for Each Cereals *#LINE CHART AND FILLED AREA PLOT*
# * Amount of Protein and Fiber for Each Cereal *#SCATTER CHART*
# * Amount of Calori in the Top 3 Cereal According to Rating Score *#BAR PLOT*
# * Amount of Protein and Fiber in the Top 3 Cereal According to Rating Score *#BAR PLOT*
# * Count Cereals According to Amount of Calori *#HISTOGRAM*

# In[ ]:


datacereal=pd.read_csv('../input/cereal.csv')


# **Fields in the dataset**
# 
#     Name: Name of cereal
#     mfr: Manufacturer of cereal
#         A = American Home Food Products;
#         G = General Mills
#         K = Kelloggs
#         N = Nabisco
#         P = Post
#         Q = Quaker Oats
#         R = Ralston Purina 
#     type:
#         cold
#         hot 
#     calories: calories per serving
#     protein: grams of protein
#     fat: grams of fat
#     sodium: milligrams of sodium
#     fiber: grams of dietary fiber
#     carbo: grams of complex carbohydrates
#     sugars: grams of sugars
#     potass: milligrams of potassium
#     vitamins: vitamins and minerals - 0, 25, or 100, indicating the typical percentage of FDA recommended
#     shelf: display shelf (1, 2, or 3, counting from the floor)
#     weight: weight in ounces of one serving
#     cups: number of cups in one serving
#     rating: a rating of the cereals (Possibly from Consumer Reports?)
# 

# In[ ]:


datacereal.head()


# In[ ]:


datacereal.info()


# **Amount of Protein, Carbonhydrate and Fat for Each Cereals**

# In[ ]:


datacereal['new_index']=datacereal.index+1
trace1=go.Scatter(x=datacereal.new_index,y=datacereal.protein,
                 mode='lines',
                 name='Protein',
                 marker=dict(color='rgba(200,20,10,0.8)'),
                 text=datacereal.name)
trace2=go.Scatter(x=datacereal.new_index,y=datacereal.fat,
                 mode='lines',
                 name='Fat',
                 marker=dict(color='rgba(100,10,150,0.8)'),
                 text=datacereal.name)
trace3=go.Scatter(x=datacereal.new_index,y=datacereal.carbo,
                 mode='lines',
                 name='Carbonhydrate',
                 marker=dict(color='rgba(50,220,15,0.8)'),
                 text=datacereal.name)
data=[trace1,trace2,trace3]
layout=dict(title='Amount of Protein, Carbonhydrate and Fat for Each Cereals',
           xaxis=dict(title='Cereals',ticklen=5,zeroline=False),
           yaxis=dict(title='Values',ticklen=5,zeroline=False))
fig=dict(data=data,layout=layout)
iplot(fig)


# In[ ]:


trace1 = go.Scatter(x=datacereal.new_index,y=datacereal.protein,
    name='protein',mode='lines',
    line=dict(width=0.5,color='rgb(111,231, 212)'),
    fill='tonexty',text=datacereal.name)

trace2 = go.Scatter(x=datacereal.new_index,y=datacereal.carbo,
    name='carbonhdyrate',mode='lines',
    line=dict(width=0.5,color='rgb(230,70,111)'),
    fill='tonexty',text=datacereal.name)

trace3 = go.Scatter(x=datacereal.new_index,y=datacereal.fat,
    name='fat',mode='lines',
    line=dict(width=0.5,color='rgb(131,90,231)'),
    fill='tonexty',text=datacereal.name)

data = [trace1,trace2,trace3]
layout = go.Layout(title='Amount of Protein,Carbonhydrate and Fat for Each Cereal',showlegend=True,
    xaxis=dict(title='Number of Cereal',type='category'),
    yaxis=dict(title='Values',type='linear')
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# **Amount of Protein and Fiber for Each Cereal**

# In[ ]:


trace1=go.Scatter(x=datacereal.new_index,y=datacereal.protein,
                 mode='markers',
                 name='Protein',
                 marker=dict(color='rgba(150,20,10,0.8)'),
                 text=datacereal.name)
trace2=go.Scatter(x=datacereal.new_index,y=datacereal.fiber,
                 mode='markers',
                 name='Fiber',
                 marker=dict(color='rgba(10,140,20,0.8)'),
                 text=datacereal.name)
data=[trace1,trace2]
layout=dict(title='Amount of Protein and Fiber for Each Cereals',
           xaxis=dict(title='Cereals',ticklen=5,zeroline=False),
           yaxis=dict(title='Values',ticklen=5,zeroline=False))
fig=dict(data=data,layout=layout)
iplot(fig)
           


# **Amount of Calori in the Top 3 Cereal According to Rating Score**
# 
# 
# mfr: Manufacturer of cereal
# 
#     A = American Home Food Products;
#     G = General Mills
#     K = Kelloggs
#     N = Nabisco
#     P = Post
#     Q = Quaker Oats
#     R = Ralston Purina 

# In[ ]:


new_index1=datacereal.rating.sort_values(ascending=False).index.values
sorted_data=datacereal.reindex(new_index1)

df=sorted_data.iloc[:3,:]
df


# In[ ]:


trace1=go.Bar(x=df.name, y=df.calories,
             marker=dict(color='rgba(220,220,40,0.8)', line=dict(color='rgba(0,0,0)',width=1.5)),
             text=df.mfr)

data=[trace1]
layout=go.Layout(barmode='group',title='Amount of Calori in the Top 3 Cereal According to Rating Score')
fig=go.Figure(data=data,layout=layout)
iplot(fig)


# **Amount of Protein and Fiber in the Top 3 Cereal According to Rating Score**

# In[ ]:


trace1=go.Bar(x=df.name, y=df.protein,name='protein',
             marker=dict(color='rgba(220,220,40,0.8)', line=dict(color='rgba(0,0,0)',width=1.5)),
             )
trace2=go.Bar(x=df.name, y=df.fiber,name='fiber',
             marker=dict(color='rgba(40,230,250,0.8)', line=dict(color='rgba(0,0,0)',width=1.5)),
             )
data=[trace1,trace2]
layout=dict(title='Amount of Protein and Fiber in the Top 3 Cereal According to Rating Score',
           xaxis=dict(title='Cereals'),yaxis=dict(title='Values'))
fig=go.Figure(data=data,layout=layout)
iplot(fig)


# **Count Cereals According to Amount of Calori**

# In[ ]:


datacereal['new_index']=datacereal.index+1
trace1=go.Histogram(x=datacereal.calories,
                 marker=dict(color='rgba(200,20,10,0.8)'))
data=[trace1]
layout=dict(title='Count Cereals According to Amount of Calori',
           xaxis=dict(title='Calories',ticklen=5,zeroline=False),
           yaxis=dict(title='Count',ticklen=5,zeroline=False))
fig=dict(data=data,layout=layout)
iplot(fig)


# In[ ]:





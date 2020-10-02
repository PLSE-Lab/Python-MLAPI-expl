#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from wordcloud import WordCloud #word 
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


team_df=pd.read_csv("../input/nba_2017_team_valuations.csv")
team_df.head()
#team_df.info()


# In[ ]:


attendance_elo_df =pd.read_csv("../input/nba_2017_att_val_elo_with_cluster.csv") 
attendance_elo_df.head()
#attendance_elo_df.info()


# In[ ]:


real_minus_df =pd.read_csv("../input/nba_2017_real_plus_minus.csv")
real_minus_df.head()
#real_minus_df.info()


# In[ ]:


pie_df =pd.read_csv("../input/nba_2017_pie.csv") 
pie_df.head()
#pie_df.info()


# In[ ]:


salary_df=pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv")
salary_df.head()


# In[ ]:


endorsement_df=pd.read_csv("../input/nba_2017_endorsements.csv")
endorsement_df
#endorsement_df.info()


# In[ ]:


#most salary player order
endorsement_df.rename(columns={"NAME":"PLAYER"},inplace=True)
endorsement_df['ENDORSEMENT'] = endorsement_df['ENDORSEMENT'].str.replace(',', '')
endorsement_df['ENDORSEMENT'] = endorsement_df['ENDORSEMENT'].str.replace('$', '')
endorsement_df['ENDORSEMENT'] = endorsement_df['ENDORSEMENT'].astype(float)
endorsement_df


# In[ ]:


endorsement_df['SALARY'] = endorsement_df['SALARY'].str.replace(',', '')
endorsement_df['SALARY'] = endorsement_df['SALARY'].str.replace('$', '')
endorsement_df['SALARY'] = endorsement_df['SALARY'].astype(float)
endorsement_df


# <a id="5"></a> <br>
# # Bar Charts
# <font color='red'>
# First Bar Charts Example:  salry endorsement of top 10
# <font color='black'>
# * Import graph_objs as *go*
# * Creating traces
#     * x = x axis
#     * y = y axis
#     * mode = type of plot like marker, line or line + markers
#     * name = name of the plots
#     * marker = marker is used with dictionary. 
#         * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#         * line = It is dictionary. line between bars
#             * color = line color around bars
#     * text = The hover text (hover is curser)
# * data = is a list that we add traces into it
# * layout = it is dictionary.
#     * barmode = bar mode of bars like grouped
# * fig = it includes data and layout
# * iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


#player of salary and endorsements
# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = endorsement_df.PLAYER,
                y = endorsement_df.SALARY,
                name = "salary",
                marker = dict(color = 'rgba(255, 100, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                )
# create trace2 
trace2 = go.Bar(
                x = endorsement_df.PLAYER,
                y = endorsement_df.ENDORSEMENT,
                name = "endorsement",
                marker = dict(color = 'rgba(10, 255, 178, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                )
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <a id ="3"></a><br>
# # line charts
# Line Charts:
# * import graph_objs as *go*
# * create traces
# * x =x axis
# * y=y axis
# * mode  =type of plot like amrker ,line or line +markers
# * name=name of plots
# * markers=marker is used with dictionary
# * color =color of line .IT takes  RGB and opacity(alpha)
# * text =the hover text(huver is curser)
# * data = is a list that we add traces into it
# * layout =it i dictonary
# * title:tiit of layout
# *  x axis:it i dictionary 
# * ticklen:lenght of x axis ticks
# * zeroline=showing line or no
# * fig =it including data and layout
# * iplot()=plots the figure(fig) that  is created by data and layout

# In[ ]:





# <a id="3"></a> <br>
# # Scatter
# <font color='red'>
# Scatter Example: Minutes Played vs RPM (Real Plus Minus) by Point
# <font color='black'>
# * Import graph_objs as *go*
# * Creating traces
#     * x = x axis
#     * y = y axis
#     * mode = type of plot like marker, line or line + markers
#     * name = name of the plots
#     * marker = marker is used with dictionary. 
#         * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#     * text = The hover text (hover is curser)
# * data = is a list that we add traces into it
# * layout = it is dictionary.
#     * title = title of layout
#     * x axis = it is dictionary
#         * title = label of x axis
#         * ticklen = length of x axis ticks
#         * zeroline = showing zero line or not
#     * y axis = it is dictionary and same with x axis
# * fig = it includes data and layout
# * iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


# import graph objects as "go"
import plotly.graph_objs as go
# creating trace1
trace1 =go.Scatter(
                    x = salary_df.POINTS,
                    y = salary_df.WINS_RPM,
                    mode = "markers",
                    name = "wins_rpm",
                    marker = dict(color = 'rgba(195, 25, 195, 0.8)'),
                    text= salary_df.PLAYER)
# creating trace2
trace2=go.Scatter(
                    x = salary_df.POINTS,
                    y = salary_df.MP,
                    mode = "markers",
                    name = "MP",
                    marker = dict(color = 'rgba(10, 140, 55, 0.6)'),
                    text= salary_df.PLAYER)

data = [trace1,trace2]
layout = dict(title = 'Minutes Played vs RPM (Real Plus Minus) by Point',
              xaxis= dict(title= 'Points',ticklen= 4,zeroline= False),
              
                           )
fig = dict(data = data, layout = layout,)
iplot(fig)


# In[ ]:


#Wordcloud
salary_df=pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv")
text =salary_df.head(20)
wordcloud = WordCloud().generate(text)
     plt.imshow(wordcloud, interpolation="bilinear")
     plt.axis("off")
     plt.savefig("graph.png")
     plt.show()
  
  


# In[ ]:





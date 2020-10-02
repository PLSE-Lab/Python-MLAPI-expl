#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Image
import os
Image("../input/picture/download.jpg")


# In[ ]:


import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import os
from PIL import  Image
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools
import io
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization


# In[ ]:


dataset = pd.read_csv('../input/fifa19/data.csv')


# In[ ]:


dataset = dataset.drop(['Unnamed: 0','Photo','Loaned From','Flag', 'Club Logo'], axis=1)


# In[ ]:


EPL_Columns = ['Liverpool','Manchester City','Tottenham Hotspur', 'Chelsea',
               'Arsenal','Manchester United','Watford','Wolverhampton Wanderers',
               'Leicester City','West Ham United','Everton', 'Bournemouth','Brighton & Hove Albion',
               'Crystal Palace','Southampton','Burnley', 'Newcastle United','Cardiff City','Fulham',
               'Stoke City','Huddersfield Town']


# ## Premier League

# In[ ]:


EPL = dataset["Club"].isin(EPL_Columns) 
dataset = dataset[EPL]


# In[ ]:


dataset.head()


# ## Count Age

# In[ ]:


value_per_age = dataset.groupby('Age')['Age'].count()
data = [go.Bar(x=value_per_age.index,
            y=value_per_age.values)]


layout = go.Layout(
     title='Count Age',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ## Oldest Player

# In[ ]:


Oldest_age = dataset.sort_values(by='Age', ascending=False)[['Name','Club','Overall','Age']].head(10)
table  = ff.create_table(np.round(Oldest_age,4))

py.iplot(table)


# ## Youngest Players

# In[ ]:


youngest_age = dataset.sort_values(by='Age', ascending=True)[['Name','Club','Overall','Age']].head(10)
table  = ff.create_table(np.round(youngest_age,4))

py.iplot(table)


# ## 20 Best Player 

# In[ ]:


best_player = pd.DataFrame.copy(dataset.sort_values(by = 'Overall' , 
                                                   ascending = False ).head(20))
data_best = [go.Bar(x=best_player['Name'].tolist(),
            y=best_player['Overall'].tolist())]


layout = go.Layout(
     title='20 Best Player by Overall',
)

fig = go.Figure(data=data_best, layout=layout)
py.iplot(fig)


# ## Best GK

# In[ ]:


GK = dataset[dataset['Position'] == 'GK']
best_GK = pd.DataFrame.copy(GK.sort_values(by = 'Overall' , 
                                                   ascending = False ).head(20))
data_GK = [go.Bar(x=best_GK['Name'].tolist(),
            y=best_GK['Overall'].tolist())]


layout = go.Layout(
     title='20 Best GK Player by Overall',
)

fig = go.Figure(data=data_GK, layout=layout)
py.iplot(fig)


# ## Best LB

# In[ ]:


LB = dataset[dataset['Position'] == 'LB']
best_LB = pd.DataFrame.copy(LB.sort_values(by = 'Overall' , 
                                                   ascending = False ).head(20))
data_LB = [go.Bar(x=best_LB['Name'].tolist(),
            y=best_LB['Overall'].tolist())]


layout = go.Layout(
     title='20 Best LB Player by Overall',
)

fig = go.Figure(data=data_LB, layout=layout)
py.iplot(fig)


# ## Best CB

# In[ ]:


CB = dataset[dataset['Position'] == 'CB']
best_CB = pd.DataFrame.copy(CB.sort_values(by = 'Overall' , 
                                                   ascending = False ).head(20))
data_CB = [go.Bar(x=best_CB['Name'].tolist(),
            y=best_CB['Overall'].tolist())]


layout = go.Layout(
     title='20 Best CB Player by Overall',
)

fig = go.Figure(data=data_CB, layout=layout)
py.iplot(fig)


# ## Best RB

# In[ ]:


RB = dataset[dataset['Position'] == 'RB']
best_RB = pd.DataFrame.copy(RB.sort_values(by = 'Overall' , 
                                                   ascending = False ).head(20))
data_RB = [go.Bar(x=best_RB['Name'].tolist(),
            y=best_RB['Overall'].tolist())]


layout = go.Layout(
     title='20 Best RB Player by Overall',
)

fig = go.Figure(data=data_RB, layout=layout)
py.iplot(fig)


# ## Best CDM

# In[ ]:


CDM = dataset[dataset['Position'] == 'CDM']
best_CDM = pd.DataFrame.copy(CDM.sort_values(by = 'Overall' , 
                                                   ascending = False ).head(20))
data_CDM = [go.Bar(x=best_CDM['Name'].tolist(),
            y=best_CDM['Overall'].tolist())]


layout = go.Layout(
     title='20 Best CDM Player by Overall',
)

fig = go.Figure(data=data_CDM, layout=layout)
py.iplot(fig)


# ## Best RCM

# In[ ]:


RCM = dataset[dataset['Position'] == 'RCM']
best_RCM = pd.DataFrame.copy(RCM.sort_values(by = 'Overall' , 
                                                   ascending = False ).head(20))
data_RCM = [go.Bar(x=best_RCM['Name'].tolist(),
            y=best_RCM['Overall'].tolist())]


layout = go.Layout(
     title='20 Best RCM Player by Overall',
)

fig = go.Figure(data=data_RCM, layout=layout)
py.iplot(fig)


# ## Best LCM

# In[ ]:


LCM = dataset[dataset['Position'] == 'LCM']
best_LCM = pd.DataFrame.copy(LCM.sort_values(by = 'Overall' , 
                                                   ascending = False ).head(20))
data_LCM = [go.Bar(x=best_LCM['Name'].tolist(),
            y=best_LCM['Overall'].tolist())]


layout = go.Layout(
     title='20 Best LCM Player by Overall',
)

fig = go.Figure(data=data_LCM, layout=layout)
py.iplot(fig)


# ## Best RW

# In[ ]:


RW = dataset[dataset['Position'] == 'RW']
best_RW = pd.DataFrame.copy(RW.sort_values(by = 'Overall' , 
                                                   ascending = False ).head(20))
data_RW = [go.Bar(x=best_RW['Name'].tolist(),
            y=best_RW['Overall'].tolist())]


layout = go.Layout(
     title='20 Best RW Player by Overall',
)

fig = go.Figure(data=data_RW, layout=layout)
py.iplot(fig)


# ## Best LW

# In[ ]:


LW = dataset[dataset['Position'] == 'LW']
best_LW = pd.DataFrame.copy(LW.sort_values(by = 'Overall' , 
                                                   ascending = False ).head(20))
data_LW = [go.Bar(x=best_LW['Name'].tolist(),
            y=best_LW['Overall'].tolist())]


layout = go.Layout(
     title='20 Best LW Player by Overall',
)

fig = go.Figure(data=data_LW, layout=layout)
py.iplot(fig)


# In[ ]:


dataset.Position.unique()


# ## Best ST

# In[ ]:


ST = dataset[dataset['Position'] == 'ST']
best_ST = pd.DataFrame.copy(ST.sort_values(by = 'Overall' , 
                                                   ascending = False ).head(20))
data_ST = [go.Bar(x=best_ST['Name'].tolist(),
            y=best_LW['Overall'].tolist())]


layout = go.Layout(
     title='20 Best ST Player by Overall',
)

fig = go.Figure(data=data_ST, layout=layout)
py.iplot(fig)


# ## Dream Team EPL

# In[ ]:


dream = [best_GK.head(1), best_CB.head(2), best_RB.head(1),best_LB.head(1),best_CDM.head(1),best_RCM.head(1),best_LCM.head(1),
         best_RW.head(1),best_LW.head(1),best_ST.head(1)]

dream_team = pd.concat(dream).reset_index(drop=True)


# In[ ]:


Dream_EPL = dream_team.sort_values(by='Age', ascending=False)[['Name','Position','Age','Club','Overall','Value']]
table  = ff.create_table(np.round(Dream_EPL,6))

py.iplot(table)


# In[ ]:


from IPython.display import Image
import os
Image("../input/picture/download.jpg")


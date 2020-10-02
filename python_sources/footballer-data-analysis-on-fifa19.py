#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings("ignore")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/data.csv")
data.info()


# In[ ]:


data.head(20)


# In[ ]:


data.dtypes


# In[ ]:


data.columns


# In[ ]:


f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


tm = data.groupby('Nationality').count()['ID'].sort_values(ascending = False)
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=False,
    width=5000,
    height=600,
    title = "Total players from a Nation in the whole game"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)


# In[ ]:


melted = pd.melt(frame=data,id_vars = 'Name', value_vars= ['Age','Finishing'])
melted


# In[ ]:


data1 = data['Age'].head()
data2= data['Finishing'].head()
conc_data_col = pd.concat([data1,data2],axis =1)
conc_data_col


# In[ ]:


data1 = data['Age'].tail()
data2= data['Finishing'].tail()
conc_data_col = pd.concat([data1,data2],axis =1)
conc_data_col


# In[ ]:


tm = data['Preferred Foot'].value_counts()
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=False,
    width=500,
    height=500,
    title = "Count of players prefered foot"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)


# In[ ]:


forwards = ['ST','LF','RF','CF','LW','RW']
midfielders = ['CM','LCM','RCM','RM','LM','CDM','LDM','RDM','CAM','LAM','RAM','LCM','RCM']
defenders = ['CB','RB','LB','RCB','LCB','RWB','LWB'] 
goalkeepers = ['GK']
data['Overall_position'] = None
forward_players = data[data['Position'].isin(forwards)]
midfielder_players = data[data['Position'].isin(midfielders)]
defender_players = data[data['Position'].isin(defenders)]
goalkeeper_players = data[data['Position'].isin(goalkeepers)]
data.loc[forward_players.index,'Overall_position'] = 'forward'
data.loc[defender_players.index,'Overall_position'] = 'defender'
data.loc[midfielder_players.index,'Overall_position'] = 'midfielder'
data.loc[goalkeeper_players.index,'Overall_position'] = 'goalkeeper'

tm = data['Overall_position'].value_counts()
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=True,
    width=500,
    height=500,
    title = "Total players playing in the Overall position"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)

plt.figure(figsize = (16, 12))
sns.set(style = 'dark', palette = 'colorblind', color_codes = True)
ax = sns.countplot('Position', data = data, color = 'blue')
ax.set_xlabel(xlabel = 'Different Positions', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Positions and Players', fontsize = 16)
plt.show()


# In[ ]:


plt.figure(figsize = (32, 20))
fig, axes = plt.subplots(nrows=2,ncols=1)
data.plot(kind = "hist",y = "Penalties",bins = 50,range= (0,250),normed = True,ax = axes[0])
data.plot(kind = "hist",y = "Penalties",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt.show()


# In[ ]:


boolean = data.Potential > 93
data[boolean]


# ## *Best 50 Turkish players*

# In[ ]:


data[data["Nationality"] == "Turkey"][['Name' , 'Position' , 'Overall' , 'Age', 'Wage', 'Club']].head(50)


# ## *Footballers of Realmadrid*

# In[ ]:


data[data["Club"] == "Real Madrid"][['Name' , 'Position' , 'Overall' , 'Age', 'Wage', 'Nationality']].head(12)


# ## *The oldest and youngest players*

# In[ ]:


data[['Name', 'Age', 'Wage', 'Value', 'Nationality']].max()


# In[ ]:


data[['Name', 'Age', 'Wage', 'Value', 'Nationality' ]].min()


# ## *Random youngest players*

# In[ ]:


data.sort_values(by = 'Age' , ascending = True)[['Name', 'Age', 'Wage']].set_index('Name').sample(10)


# In[ ]:


data[data["Position"] == "ST"][['Name' , 'Position' , 'Overall' , 'Age', 'Wage', 'Nationality']].head()


# In[ ]:


data.Position.unique()


# ## *Dribbling and finishing by preferred foot*

# In[ ]:


sns.swarmplot(x="Dribbling", y="Finishing",hue="Preferred Foot",data = data, color = 'red')
plt.show()


# ## *Position and finishing by preferred foot***

# In[ ]:


sns.swarmplot(x="Position", y="Finishing",hue="Preferred Foot", data=data)
plt.show()


# In[ ]:


data.describe()


# In[ ]:


sns.countplot(x="Age", data=data)
data.loc[:,'Age'].value_counts()
plt.show()


# ## *** ABOUT MISSING DATA***

# In[ ]:


data['Club'].fillna('No Club', inplace = True)
data['Club'].value_counts(dropna = False)


# In[ ]:


data['Preferred Foot'].fillna('Right', inplace = True)
data['Preferred Foot'].value_counts(dropna = False)


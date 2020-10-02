#!/usr/bin/env python
# coding: utf-8

# # **Understanding Sports of past 120 years **

# ![](https://unmanned-aerial.com/wp-content/uploads/2018/02/intel-2.jpg)

# 

# ## There's some more time for Oylmpic season to get started but the fever is still alive. This is my first notebook and as a fresher in Kaggle, i am open to corrections, recommendations and suggestions. Here i have basically tried to find insights of Oylmpic over the years and to visualise it.

# ## **<code>Lets Start</code>**

# ### First Lets import the packages and import the data.

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
#import the data
athlete = pd.read_csv('../input/athlete_events.csv',index_col = False)
regions = pd.read_csv('../input/noc_regions.csv',index_col = False)
athlete.head()


# ## here we got to know basic info of the dataset and what overall it shows, and boy it has some good information to know about.

# In[ ]:


athlete.info()


# ## Here only Age, Height and Weight have some missing values. We can ignore the whole row, but from an oylmpic standpoint it would be wise right now to not remove these rows as even a single row is important as it repesent an individual in an sport.

# ## We can also see Medal also has missing values but because it says that that person havent got any medal in that event. so we will convert them to 0

# 

# In[ ]:


athlete.Medal = athlete.Medal.fillna(0)


# In[ ]:


athlete.head()


# # we can visualise this dataset in three types 
# 
# * Players who played in summer oylmpics
# * Players who played in winter oylmpics
# * All the Players combined
# 
# ## So now we will represent 
# 
# * **summ** -  for summer oylmpics
# * **wint** -  for winter oylmpics
# * **athlete** -  for overall all players combined

# In[ ]:


summ = athlete[athlete.Season == 'Summer'] 
wint = athlete[athlete.Season == 'Winter']


# In[ ]:


fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(x="Sex", hue="Medal", data=athlete,ax=ax)


#  It was a dumb visualisation from me ... But still it shows How much male and female Athletes had got the medals. In the coming codes i will really create some interesting plots.

# ## Let us see how much Male and Female had got medals w.r.t Summer and Winter oylmpics

# In[ ]:


fig, ax = plt.subplots(figsize=(8,6))
a = athlete[~athlete['Medal'].isnull()].Sex
sns.countplot(x=a, hue="Season", data=athlete,ax=ax)


# ## Now let's group the data according to gold medal, silver medal and bronze medal based on countries.

# If someone has some good optimized way then i will be happy to see that.

# In[ ]:


# Group data according to gold
counts_gold = {}
for key,values in athlete.iterrows():
    if(values['Medal'] == 'Gold'):
        if(counts_gold.get(values.Team) == None):
            counts_gold[values.Team] = 1
        else:
            counts_gold[values.Team] +=1  
gdata = pd.concat({k:pd.Series(v) for k, v in counts_gold.items()}).unstack().astype(float).reset_index()


# In[ ]:


gdata.columns = ['Team', 'gold']
gdata.info()


# # Here there is an important information to note that i have created two dataframe, ie 
#  * **gdata** - for Team and gold
#  
#  * **g_data** - for Team and gold (but it is sorted in descending order)

# In[ ]:


g_data = gdata.sort_values(by='gold',ascending=False)


# ## This Visualisation shows that top countries that have most golds in oylmpics

# In[ ]:


fig, ax = plt.subplots(figsize=(15,6))
ax = sns.barplot(x="Team", y="gold", data=g_data.head(10),ax=ax)


# ## Suprisingly, we haven't saw much change in summer and overall oylmpics, but in winter, looks like canada is on lead

# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go


# # Now same for Silver 

# In[ ]:


counts_silver = {}
for key,values in athlete.iterrows():
    if(values['Medal'] == 'Silver'):
        if(counts_silver.get(values.Team) == None):
            counts_silver[values.Team] = 1
        else:
            counts_silver[values.Team] +=1
            
sdata = pd.concat({k:pd.Series(v) for k, v in counts_silver.items()}).unstack().astype(float).reset_index()
sdata.columns = ['Team', 'silver']
s_data = sdata.sort_values(by='silver',ascending=False)


# In[ ]:


sdata.head()


# # And for Bronze

# In[ ]:


counts_bronze = {}
for key,values in athlete.iterrows():
    if(values['Medal'] == 'Bronze'):
        if(counts_bronze.get(values.Team) == None):
            counts_bronze[values.Team] = 1
        else:
            counts_bronze[values.Team] +=1
            
bdata = pd.concat({k:pd.Series(v) for k, v in counts_bronze.items()}).unstack().astype(float).reset_index()
bdata.columns = ['Team', 'bronze']
b_data = bdata.sort_values(by='bronze',ascending=False)


# ## Now we will create a new dataframe data1 that stores all gold, silver and bronze medal according to their country name.

# In[ ]:


data1 = {}
data1['Team'] = athlete.Team.unique()

data1=pd.DataFrame(data1)

data1 = pd.merge(data1, gdata, on='Team')
data1 =pd.merge(data1, sdata, on='Team')
data1 =pd.merge(data1, bdata, on='Team')

data1 = data1.sort_values(by=['gold','silver','bronze'],ascending=False)


# ## Now here the interesting part comes... We will visualise top countries which has highest medals using Plotly

# In[ ]:


# prepare data frames
dfd = data1.head(20)
# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = dfd.Team,
                y = dfd.gold,
                name = "gold",
                marker = dict(color = 'rgba(255, 223, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
# create trace2 
trace2 = go.Bar(
                x = dfd.Team,
                y = dfd.silver,
                name = "silver",
                marker = dict(color = 'rgba(192, 192, 192, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
# create trace3
trace3 = go.Bar(
                x = dfd.Team,
                y = dfd.bronze,
                name = "bronze",
                marker = dict(color = 'rgba(205, 127, 50, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
data = [trace1, trace2, trace3]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


#  ## Now we will find same for summer and winter oylmpics

# Please note the code to display is same as above. I have just changed the datatypes

# In[ ]:


# Group data according to gold for summer
counts_gold = {}
for key,values in summ.iterrows():
    if(values['Medal'] == 'Gold'):
        if(counts_gold.get(values.Team) == None):
            counts_gold[values.Team] = 1
        else:
            counts_gold[values.Team] +=1  
gdata = pd.concat({k:pd.Series(v) for k, v in counts_gold.items()}).unstack().astype(float).reset_index()
gdata.columns = ['Team', 'gold']
g_data = gdata.sort_values(by='gold',ascending=False)
fig, ax = plt.subplots(figsize=(15,6))
ax = sns.barplot(x="Team", y="gold", data=g_data.head(10),ax=ax).set_title('Top 10 countries in summer oylmpics')
import plotly.plotly as py
import plotly.graph_objs as go

counts_silver = {}
for key,values in summ.iterrows():
    if(values['Medal'] == 'Silver'):
        if(counts_silver.get(values.Team) == None):
            counts_silver[values.Team] = 1
        else:
            counts_silver[values.Team] +=1
            
sdata = pd.concat({k:pd.Series(v) for k, v in counts_silver.items()}).unstack().astype(float).reset_index()
sdata.columns = ['Team', 'silver']
s_data = sdata.sort_values(by='silver',ascending=False)

counts_bronze = {}
for key,values in summ.iterrows():
    if(values['Medal'] == 'Bronze'):
        if(counts_bronze.get(values.Team) == None):
            counts_bronze[values.Team] = 1
        else:
            counts_bronze[values.Team] +=1
            
bdata = pd.concat({k:pd.Series(v) for k, v in counts_bronze.items()}).unstack().astype(float).reset_index()
bdata.columns = ['Team', 'bronze']
b_data = bdata.sort_values(by='bronze',ascending=False)

data1 = {}
data1['Team'] = summ.Team.unique()

data1=pd.DataFrame(data1)

data1 = pd.merge(data1, gdata, on='Team')
data1 =pd.merge(data1, sdata, on='Team')
data1 =pd.merge(data1, bdata, on='Team')

data1 = data1.sort_values(by=['gold','silver','bronze'],ascending=False)


# In[ ]:


# prepare data frames
dfd = data1.head(20)
# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = dfd.Team,
                y = dfd.gold,
                name = "gold",
                marker = dict(color = 'rgba(255, 223, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
# create trace2 
trace2 = go.Bar(
                x = dfd.Team,
                y = dfd.silver,
                name = "silver",
                marker = dict(color = 'rgba(192, 192, 192, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
# create trace3
trace3 = go.Bar(
                x = dfd.Team,
                y = dfd.bronze,
                name = "bronze",
                marker = dict(color = 'rgba(205, 127, 50, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
data = [trace1, trace2, trace3]
layout = go.Layout(title='Top countries in summer oylmpics',barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# ## Now for Winter

# In[ ]:


# Group data according to gold
counts_gold = {}
for key,values in wint.iterrows():
    if(values['Medal'] == 'Gold'):
        if(counts_gold.get(values.Team) == None):
            counts_gold[values.Team] = 1
        else:
            counts_gold[values.Team] +=1  
gdata = pd.concat({k:pd.Series(v) for k, v in counts_gold.items()}).unstack().astype(float).reset_index()
gdata.columns = ['Team', 'gold']
g_data = gdata.sort_values(by='gold',ascending=False)
fig, ax = plt.subplots(figsize=(15,6))
ax = sns.barplot(x="Team", y="gold", data=g_data.head(10),ax=ax).set_title('Top 10 countries in winter oylmpics')
import plotly.plotly as py
import plotly.graph_objs as go

counts_silver = {}
for key,values in wint.iterrows():
    if(values['Medal'] == 'Silver'):
        if(counts_silver.get(values.Team) == None):
            counts_silver[values.Team] = 1
        else:
            counts_silver[values.Team] +=1
            
sdata = pd.concat({k:pd.Series(v) for k, v in counts_silver.items()}).unstack().astype(float).reset_index()
sdata.columns = ['Team', 'silver']
s_data = sdata.sort_values(by='silver',ascending=False)

counts_bronze = {}
for key,values in wint.iterrows():
    if(values['Medal'] == 'Bronze'):
        if(counts_bronze.get(values.Team) == None):
            counts_bronze[values.Team] = 1
        else:
            counts_bronze[values.Team] +=1
            
bdata = pd.concat({k:pd.Series(v) for k, v in counts_bronze.items()}).unstack().astype(float).reset_index()
bdata.columns = ['Team', 'bronze']
b_data = bdata.sort_values(by='bronze',ascending=False)

data1 = {}
data1['Team'] = wint.Team.unique()

data1=pd.DataFrame(data1)

data1 = pd.merge(data1, gdata, on='Team')
data1 =pd.merge(data1, sdata, on='Team')
data1 =pd.merge(data1, bdata, on='Team')

data1 = data1.sort_values(by=['gold','silver','bronze'],ascending=False)


# In[ ]:


# prepare data frames
dfd = data1.head(20)
# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = dfd.Team,
                y = dfd.gold,
                name = "gold",
                marker = dict(color = 'rgba(255, 223, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
# create trace2 
trace2 = go.Bar(
                x = dfd.Team,
                y = dfd.silver,
                name = "silver",
                marker = dict(color = 'rgba(192, 192, 192, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
# create trace3
trace3 = go.Bar(
                x = dfd.Team,
                y = dfd.bronze,
                name = "bronze",
                marker = dict(color = 'rgba(205, 127, 50, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
data = [trace1, trace2, trace3]
layout = go.Layout(title='Top countries in Winter Oylmpics',barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# ## Wow it shows that United States is way ahead in terms of medals and then Soviet Union comes second in comparision of Oylmpics.
# 
# ### Let's just focus only these two countries and see their progression over the years

# In[ ]:


gusa = {}
for key,values in athlete.iterrows():
    if((values['Team'] == 'United States') and (values['Medal'] == 'Gold')):
        if(gusa.get(values.Year) == None):
            gusa[values.Year] = 1
        else:
            gusa[values.Year] +=1
            
gusa = pd.concat({k:pd.Series(v) for k, v in gusa.items()}).unstack().astype(float).reset_index()
gusa.columns = ['Year', 'gold']


#silver for usa
susa = {}
for key,values in athlete.iterrows():
    if((values['Team'] == 'United States') and (values['Medal'] == 'Silver')):
        if(susa.get(values.Year) == None):
            susa[values.Year] = 1
        else:
            susa[values.Year] +=1
            
susa = pd.concat({k:pd.Series(v) for k, v in susa.items()}).unstack().astype(float).reset_index()
susa.columns = ['Year', 'silver']  


# bronze for usa

busa = {}
for key,values in athlete.iterrows():
    if((values['Team'] == 'United States') and (values['Medal'] == 'Bronze')):
        if(busa.get(values.Year) == None):
            busa[values.Year] = 1
        else:
            busa[values.Year] +=1
            
busa = pd.concat({k:pd.Series(v) for k, v in busa.items()}).unstack().astype(float).reset_index()
busa.columns = ['Year', 'bronze']


#### for Soviet Union

# gold for Soviet Union

gsu = {}
for key,values in athlete.iterrows():
    if((values['Team'] == 'Soviet Union') and (values['Medal'] == 'Gold')):
        if(gsu.get(values.Year) == None):
            gsu[values.Year] = 1
        else:
            gsu[values.Year] +=1
            
gsu = pd.concat({k:pd.Series(v) for k, v in gsu.items()}).unstack().astype(float).reset_index()
gsu.columns = ['Year', 'gold']


ssu = {}
for key,values in athlete.iterrows():
    if((values['Team'] == 'Soviet Union') and (values['Medal'] == 'Silver')):
        if(ssu.get(values.Year) == None):
            ssu[values.Year] = 1
        else:
            ssu[values.Year] +=1
            
ssu = pd.concat({k:pd.Series(v) for k, v in ssu.items()}).unstack().astype(float).reset_index()
ssu.columns = ['Year', 'silver']

bsu = {}
for key,values in athlete.iterrows():
    if((values['Team'] == 'Soviet Union') and (values['Medal'] == 'Bronze')):
        if(bsu.get(values.Year) == None):
            bsu[values.Year] = 1
        else:
            bsu[values.Year] +=1
            
bsu = pd.concat({k:pd.Series(v) for k, v in bsu.items()}).unstack().astype(float).reset_index()
bsu.columns = ['Year', 'bronze']


# In[ ]:


# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1 USA Gold
trace1 = go.Scatter(
                    x = gusa.Year,
                    y = gusa.gold,
                    mode = "lines+markers",
                    name = "USA gold",
                    marker = dict(color = 'rgba(240,230,140 0.8)'),
                    )
# Creating trace2 USA silver
trace2 = go.Scatter(
                    x = susa.Year,
                    y = susa.silver,
                    mode = "lines+markers",
                    name = "USA silver",
                    marker = dict(color = 'rgba(211,211,211, 0.8)'),
                    )
# Creating trace3 USA bronze
trace3 = go.Scatter(
                    x = busa.Year,
                    y = busa.bronze,
                    mode = "lines+markers",
                    name = "USA bronze",
                    marker = dict(color = 'rgba(220,165,112)'),
                    )
# Creating trace4 Soviet gold
trace4 = go.Scatter(
                    x = gsu.Year,
                    y = gsu.gold,
                    mode = "lines+markers",
                    name = "Soviet Union gold",
                    marker = dict(color = 'rgba(218,165,32, 0.8)'),
                    )
# Creating trace5 Soviet silver
trace5 = go.Scatter(
                    x = ssu.Year,
                    y = ssu.silver,
                    mode = "lines+markers",
                    name = "Soviet Union Silver",
                    marker = dict(color = 'rgba(128,128,128, 0.8)'),
                    )
# Creating trace6 Soviet bronze
trace6 = go.Scatter(
                    x = bsu.Year,
                    y = bsu.bronze,
                    mode = "lines+markers",
                    name = "Soviet Union bronze",
                    marker = dict(color = 'rgba(144,89,35, 0.8)'),
                    )

data = [trace1, trace2,trace3,trace4,trace5,trace6]
layout = dict(title = 'Comparision between USA and Soviet union in years',
              xaxis= dict(title= 'Years',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# ## Oh It shows some interesting blast in the past. we can see that the Soviet Union started in 1952 and last played in 1988 as after that it was changed.
# 
# <a href="https://www.google.co.in/search?q=soviet+union+1988&rlz=1C1GGRV_enIN801IN801&oq=soviet+union+1988&aqs=chrome..69i57j69i60j0l4.9119j0j9&sourceid=chrome&ie=UTF-8">See here</a>

# ## Now one last as this will get boring if i excedded any further, lastly i will see corelation between Athletes on Height and weight

# In[ ]:


athlete = athlete[pd.notnull(athlete['Height'])]
athlete = athlete[pd.notnull(athlete['Weight'])]


# In[ ]:


# prepare data frames
athg = athlete[athlete.Medal == 'Gold']
aths = athlete[athlete.Medal == 'Silver']
athb = athlete[athlete.Medal == 'Bronze']

# creating trace1
trace1 =go.Scatter(
                    x = athg.Weight,
                    y = athg.Height,
                    mode = "markers",
                    name = "Gold",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= athg.Medal)
# creating trace2
trace2 =go.Scatter(
                    x = aths.Weight,
                    y = aths.Height,
                    mode = "markers",
                    name = "Silver",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= aths.Medal)
# creating trace3
trace3 =go.Scatter(
                    x = athb.Weight,
                    y = athb.Height,
                    mode = "markers",
                    name = "Bronze",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= athb.Medal)
data = [trace1, trace2, trace3]
layout = dict(title = 'Corelation between Height and Weight',
              xaxis= dict(title= 'Weight',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Height',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)




# ## Thats it for now. I know there will be some mistakes and i will improve it in later times.
# ## as this is my first notebook, if it pleases you for atleast 1% , please upvote this notebook. and if you didnt like then please let me know my mistakes. 

# # Happy Coding :)

# In[ ]:





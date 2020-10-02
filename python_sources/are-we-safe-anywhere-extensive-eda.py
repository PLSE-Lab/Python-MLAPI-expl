#!/usr/bin/env python
# coding: utf-8

# **IMPORTING LIBRARIES**

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from wordcloud import WordCloud, STOPWORDS

pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)


# **IMPORTING DATA**

# Let's have a look at our data how it looks and what kind of insights can we explore .

# In[3]:


Guns=pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
Guns.head()


# In[ ]:


Guns.shape[0]


# In[ ]:


Guns.describe()#lots of null values everywhere


# we can see from the description that not all the columns have complete data .
# 
# so our first step will be to figure out which columns will have how much null values and think accordingly how to deal with them.

# In[ ]:


per=[]
for i in Guns.columns:
    num=Guns[i].isnull().sum()
    final=(num/Guns.shape[0])*100
    per.append(final)

d={'Col': Guns.columns,'%null': per}
nulls=pd.DataFrame(data=d)
nulls


# we can see from the results the columns that do not have any null value.
# 
# so it is safe for the starting to look in those for so really cool insights.

# **1- States with most recorded incidences **
# 
# we will first look in which states we have most amount of Gun violence cases registerd.

# In[ ]:


plt.figure(figsize=(18,12))
state=Guns['state'].value_counts()
sns.barplot(state.values,state.index)
plt.xlabel("Number of incidences",fontsize=15)
plt.ylabel("States",fontsize=15)
plt.title("Recoreded incidences in states",fontsize=20)
sns.despine(left=True,right=True)
plt.show()


# **2- Cities with most recorded incidences**
# 
# Now having a look at cities with most Gun Violence

# In[ ]:


plt.figure(figsize=(18,12))
state=Guns['city_or_county'].value_counts()[:20]
sns.barplot(state.values,state.index)
plt.xlabel("Number of incidences",fontsize=15)
plt.ylabel("cities",fontsize=15)
plt.title("Recored incidences in cities",fontsize=20)
sns.despine(left=True,right=True)
plt.show()


# LOOKS LIKE CHICAGO IS FAR BEYOND OTHERS

# **3- Growth rate of Gun Violence **

# Now we are digging a little bit deeper to see how the gun Violence is affected in different states in course of 2013-2018,

# In[ ]:


Guns['date']=pd.to_datetime(Guns['date'],format='%Y-%m-%d')
Guns['date'].head()


# In[ ]:


Guns['year']=Guns['date'].dt.year


# In[ ]:


state_lst=[Guns['state'].unique()]
year_lst=[Guns['date'].dt.year.unique()]

state_lst_new=[]
for i in range(0,51):
    new=state_lst[0][i]
    state_lst_new.append(new)
    
year_lst_new=[]
for i in range(0,6):
    new=year_lst[0][i]
    year_lst_new.append(new)

plt.figure(figsize=(18,9))
for state in state_lst_new:
    yearly_incd=[]
    for year in year_lst_new:
        my= Guns.loc[Guns['state']==state]
        sum=my.loc[my['year']==year]
        sol=sum.shape[0]
        yearly_incd.append(sol)
    plt.plot(yearly_incd,label=state)
plt.xticks(np.arange(6),tuple(year_lst_new),rotation=60)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
plt.show()


# The graph clearly shows that some states are definately using GUNS in their daily life far more than others.

# **4-People killed and injured over time**
# 
# We are having a look at how many people are victims of these Gun violence or the course of time.

# In[ ]:


years_killed=Guns.groupby(Guns['year']).sum()
x=years_killed['n_killed'].index.tolist()
y=years_killed['n_killed'].values.tolist()
z=years_killed['n_injured'].values.tolist()

#create style trace
trace0=go.Scatter(
x = x,
y = y,
name='no. of people killed',
  line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4,dash='dot')
)
trace1=go.Scatter(
x = x,
y = z,
name='no. of people injured',
  line = dict(
        color = ('rgb(10, 205, 26)'),
        width = 4,dash='dot')
)
trace2=go.Scatter(
x = x,
y = [y+z for y,z in zip(y,z)],
name='Total no. of people effected',
  line = dict(
        color = ('rgb(20, 20, 205)'),
        width = 4,dash='dot')
)


data=[trace0,trace1,trace2]

#edit layout

layout=dict(title='people killed or injured every year',
           xaxis=dict(title='Years'),
           yaxis=dict(title='NO. of people killed or injured'))

fig = dict(data=data, layout=layout)
py.iplot(fig , filename='styled-line')


# We can definately see a trend over there .
# 
# obviously more peoples are injured tha  killed.

# **5-People killed or injured in states**
# 
# Now we are digging deep to see population effected by these GUN abuses in different states.

# In[ ]:


state_killed=Guns.groupby(Guns['state']).sum()
sk_x=state_killed['n_killed'].index.tolist()
sk_y=state_killed['n_killed'].values.tolist()
si=state_killed['n_injured'].values.tolist()

trace1=go.Scatter(
x=sk_x,
y=sk_y,
name='people killed')

trace2=go.Scatter(
x=sk_x,
y=si,
name='people injured',
yaxis='y2')
data=[trace1,trace2]

layout = go.Layout(
    title='Incidences in states',
    xaxis=dict(title='states'),
    yaxis=dict(
        title='People killed',
        titlefont=dict(
            color='rgb(140,38,78)')
    ),
    yaxis2=dict(
        title='people injured',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    )
 )
                 
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='multiple-axes-double.html')
            


# **6-Gun use in congressional district**
# 
# In U.S.A. each states are divided into many congressional districts and it ranges from as few as 2 - (40s)
# 

# In[ ]:


cd=Guns[np.isfinite(Guns['congressional_district'])]
my_cd=cd['congressional_district'].value_counts()

plt.figure(figsize=(28,12))
sns.barplot(my_cd.index,my_cd.values)
plt.xlabel("Congressional district",fontsize=12)
plt.ylabel("NO. of incidences",fontsize=12)
plt.title("Gun abuse in different CD ",fontsize=16)
plt.show()


# It makes sense that value of CD-1 is so much more than others because each state has at least 1 CD.
# 
# we can't make out much from this graph as it is more about number of CD each state have

# **7- Types of Guns used**

# In[ ]:


type=Guns.dropna(how='any',axis=0)
my_type=type['gun_type'].values.tolist()
del( my_type[5:11])

my_set=set()
for guns in my_type:
    if len(guns)<=18:
        adds=guns.split("::")[1]
        my_set.add(adds)
    else:
        my_item=[]
        my_items=[]
        lst1=guns.split("||")
        for item in lst1:
            my=item.split("::")
            my_item.append(my)
        for items in my_item:
            adds=items[1]
            my_items.append(adds)
        for adding in my_items:
            my_set.add(adding)
        
        
remove=['45 Auto||1','9mm||1','Handgun||1','Rifle||1']
for rem in remove:
    my_set.remove(rem)
my_set

str_set=[]
for e in my_set:
    string=str(e)
    str_set.append(string)
str_set


# In[ ]:


stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color='black',
                          stopwords=stopwords,
                         ).generate(' '.join(str_set))
print(wordcloud)
fig = plt.figure(figsize=(14,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# **8-Areas of maximum Gun related incidences**
# 
# Now we are going to plot the same graph of Gun usage incidence in different states but now our motive is to see weather geographical condition has to do anything with these incidences.

# In[9]:


item=Guns['state'].value_counts().index.tolist()
item_size=Guns['state'].value_counts().values.tolist()

cities = []
scale = 250


for i in range(len(item)):
    lim = item[i]
    df_sub = Guns.loc[Guns['state']==lim][:1]
    city = dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df_sub['longitude'],
        lat = df_sub['latitude'],
        text = item[i] + '<br>Gun abuse ' + str(item_size[i]),
        marker = dict(
            size = item_size[i]/scale,
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = lim )
    cities.append(city)

layout = dict(
        title = 'Gun abuse around USA states
        showlegend = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

fig = dict( data=cities, layout=layout )
py.iplot( fig, validate=False, filename='d3-bubble-map-populations' )
plt.savefig('abc.png')    


# 

# It is so clear from the plot that the region on the **EAST** is having more incidences of gun usage than any other part.
# 
# In map we can point out ares where it is relatively very low**(safe for living)** 

# **KEEP AN EYE ON THE NOTEBOOK LOT MORE IS COMMING**

# In[ ]:





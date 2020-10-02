#!/usr/bin/env python
# coding: utf-8

# ## Initial Setup

# In[88]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
from plotly import tools


# In[89]:


py.init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# ## Loading the data

# In[47]:


heroes = pd.read_csv('../input/heroes_information.csv')
heroes.head(10)


# In[48]:


heroes.info()


# In[49]:


print("missing value count in Publisher:",heroes['Publisher'].isnull().sum())
print("missing value count in Weight:",heroes['Weight'].isnull().sum())


# In[50]:


# dropping first column 
heroes.drop(['Unnamed: 0'], axis=1, inplace=True)

# replacing '-' and NaN values with 'unknown' in Publisher attribute
heroes.replace(to_replace='-', value='unknown', inplace=True)
heroes['Publisher'].fillna('unknown', inplace=True)


# In[51]:


heroes.info()


# In[52]:


heroes['Weight'].value_counts()


# Umm there's alot of negative weights. Ideally weights can't be negative the super heroes could be light as air but not negative, so let's replace them by NaN.

# In[53]:


heroes[heroes['Weight'].isnull()]


# In[54]:


# replacing negative Heights and Weights with NaN
heroes.replace(-99.0, np.nan, inplace=True)


# In[55]:


heroes.info()


# So it turns out even the Height attribute had a lot of negative values. Now we've a lot of missing values to fill

# In[56]:


ht_wt = heroes[['Height','Weight']]


# In[57]:


# imputing missing heights and weights with median
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")

X = imputer.fit_transform(ht_wt)
# X = imputer.transform(ht_wt)
heroes_h_w = pd.DataFrame(X, columns=ht_wt.columns)


# In[58]:


heroes_h_w.isnull().any()


# In[59]:


heroes_without_h_w = heroes.drop(['Height','Weight'],axis=1)
heroes = pd.concat([heroes_without_h_w, heroes_h_w], axis=1)
heroes.head()


# ## Some Insights
# 
# First lets see the distribution of the number of super heroes from each of the Publishers

# In[60]:


publisher_series = heroes['Publisher'].value_counts()
publishers = list(publisher_series.index)
publications = list((publisher_series/publisher_series.sum())*100)


# In[61]:


trace = go.Pie(labels=publishers, values=publications)
layout = go.Layout(
    title='comic-wise publications distributions',
    height=950,
    width=950
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='comics-wise-distribution')


# __Marvel Comics and DC Comics definitely have a huge market captured__

# Not sure what the Alignment attribute means but the values tell me this is more like a flag whether the character is a 'Hero' or a 'Villian' or 'Neutral' character. Now I'm also curious as to which characters are actually neutral and which ones fall into the unknown alignment.

# In[62]:


heroes.loc[heroes['Alignment']=='unknown']


# Well Venom was definitely a bad guy, that means Anti-Venom must a heroic character. May be someone who stood up against Venom (cutting Spiderman some free-time for Mary Jane) or may be someone born out of Venom idk if you know about this character do let me know...
# But atleast this explains the kind of characters whose alignment is not known.
# 
# Now there's someone as Venompool also....idk how many characters Venom has had the relationship with in the past. But then lets first check the alignment of Deadpool

# In[63]:


heroes.loc[heroes['name']=='Deadpool']


# Hmm now Deadpool's alignment is set to Neutral may be because he doesn't restricts himself only to the bad guys and enjoys kicking butts of some of the X-Men as well  :P 

# In[64]:


heroes['Alignment'].value_counts()


# ## Number of Heroes vs Number of Villians
# Let's see the count of total heroes, total villian and neutral characters in each of the publications

# In[65]:


tot_pub = (heroes.Publisher.value_counts().index)
col_names = ['Publisher', 'total_heroes', 'total_villian', 'total_neutral', 'total_unknown']
df = pd.DataFrame(columns=col_names)

for publisher in tot_pub:
    data=[]
    data.append(publisher)
    data.append(len(list(heroes['name'].loc[(heroes['Alignment']=='good') & (heroes['Publisher']==publisher)])))
    data.append(len(list(heroes['name'].loc[(heroes['Alignment']=='bad') & (heroes['Publisher']==publisher)])))
    data.append(len(list(heroes['name'].loc[(heroes['Alignment']=='neutral') & (heroes['Publisher']==publisher)])))
    data.append(len(list(heroes['name'].loc[(heroes['Alignment']=='unknown') & (heroes['Publisher']==publisher)])))
    df.loc[len(df)] = data

# print(df)


# In[66]:


trace1 = go.Bar(
    x=list(df.Publisher),
    y=list(df.total_heroes),
    name='total_heroes'
)

trace2 = go.Bar(
    x=list(df.Publisher),
    y=list(df.total_villian),
    name='total_villians'
)

trace3 = go.Bar(
    x=list(df.Publisher),
    y=list(df.total_neutral),
    name='total_neutral'
)

trace4 = go.Bar(
    x=list(df.Publisher),
    y=list(df.total_unknown),
    name='total_unknown'
)

data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    title='Publisher-wise number of heroes vs number of villians',
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='heroes-vs-villians per Publisher')


# clearly there are more heroes than villians in any of the comics' publication. This is really disheartening as all the battles i ever read about during my childhood....__it was never an equal fight as the heroes are outnumbered in both Marvel and DC Universe!!__
# 
# 
# except for <i>Image Comics</i> [ zoom-in ] where there are just 2 heroes against 11 villians. Interesting!

# ## Gender Distribution - overall and alignment-wise

# In[67]:


# gender distribution
gender_series = heroes['Gender'].value_counts()
genders = list(gender_series.index)
distribution = list((gender_series/gender_series.sum())*100)

trace = go.Pie(labels=genders, values=distribution)
layout = go.Layout(
    title='overall gender distributions',
    height=500,
    width=500
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='gender-distribution')

# gender distribution by alignment
heroes_gender_series = heroes['Gender'].loc[heroes['Alignment']=='good'].value_counts()
heroes_genders = list(heroes_gender_series.index)
heroes_distribution = list((heroes_gender_series/heroes_gender_series.sum())*100)

villian_gender_series = heroes['Gender'].loc[heroes['Alignment']=='bad'].value_counts()
villian_genders = list(villian_gender_series.index)
villian_distribution = list((villian_gender_series/villian_gender_series.sum())*100)

neutral_gender_series = heroes['Gender'].loc[heroes['Alignment']=='neutral'].value_counts()
neutral_genders = list(neutral_gender_series.index)
neutral_distribution = list((neutral_gender_series/neutral_gender_series.sum())*100)

unknown_gender_series = heroes['Gender'].loc[heroes['Alignment']=='unknown'].value_counts()
unknown_genders = list(unknown_gender_series.index)
unknown_distribution = list((unknown_gender_series/unknown_gender_series.sum())*100)

fig = {
    "data": [
        {
            "labels": heroes_genders, 
            "values": heroes_distribution, 
            "type": "pie", 
            "name": "heroes",
            "domain":{'x': [0, 0.48],
                      'y': [0.51, 1]},
            "textinfo": "label"
        },
        {
            "labels": villian_genders, 
            "values": villian_distribution, 
            "type": "pie", 
            "name": "villians",
            "domain":{'x': [0.52, 1],
                     'y': [0.51, 1]},
            "textinfo": "label"
        },
        {
            "labels": neutral_genders, 
            "values": neutral_distribution, 
            "type": "pie", 
            "name": "neutral characters",
            "domain":{'x': [0, 0.48],
                      'y': [0, 0.49]},
            "textinfo": "label"
        },
        {
            "labels": unknown_genders, 
            "values": unknown_distribution, 
            "type": "pie", 
            "name": "unknown characters",
            "domain":{'x': [0.52, 1],
                      'y': [0, 0.49]},
            "textinfo": "label"
        }
    ],
    "layout": {"title": "Gender distribution among Heroes, Villians and Neutral Characters", 
               "showlegend": False}
}

py.iplot(fig, filename='Gender distribution')


# so less! We need more women to join the Dark side....Harley Quinn, Catwomen, Poison Ivy, etc have been a few of my fav negative characters. __The world needs more sexy negative feminine characters!__

# ## Alignment of superheroes by gender

# In[68]:


male_df = heroes.loc[heroes['Gender']=='Male']
female_df = heroes.loc[heroes['Gender']=='Female']


# In[69]:


trace_m = go.Bar(
    x=male_df['Alignment'].value_counts().index,
    y=male_df['Alignment'].value_counts().values,
    name='male'
)

trace_f = go.Bar(
    x=female_df['Alignment'].value_counts().index,
    y=female_df['Alignment'].value_counts().values,
    name='female'
)

data = [trace_m, trace_f]
layout = go.Layout(
    title='Alignment of super heroes by gender',
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='alignment-by-gender')


# ## Distribution of superheroes by Race

# In[70]:


trace = go.Bar(
    x=heroes['Race'].value_counts().index,
    y=heroes['Race'].value_counts().values,
    name='Races'
)

layout = go.Layout(
    title='Distribution of heroes across different races',
    barmode='bar'
)

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig, filename='distribution-by-race')


# ## Skin color distribution

# __Let's find more heroes like HULK__
# 
# [ basically whose races are Humans or Human-like but skin colores have changed ]
# 
# <i>super-powers at the cost of skin-change!</i> Nope...I'd rather stay human.

# In[71]:


human_heroes_changed_skin_color = heroes.loc[(heroes['Race'].isin(['Human','Human / Radiation','Human / Clone','Human-Kree','Human / Cosmic','Human / Altered','Human-Vuldarian','Human-Vulcan','Human-Spartoi','']) & 
            (~heroes['Skin color'].isin(['unknown','white','black','gray','grey'])))]

# print(human_heroes_changed_skin_color[['name','Skin color']])

green_skin=str(human_heroes_changed_skin_color['name'].loc[heroes['Skin color']=='green'].values)
red_skin=str(human_heroes_changed_skin_color['name'].loc[heroes['Skin color']=='red'].values)
blue_skin=str(human_heroes_changed_skin_color['name'].loc[heroes['Skin color']=='blue'].values)
silver_skin=str(human_heroes_changed_skin_color['name'].loc[heroes['Skin color']=='silver'].values)
gold_skin=str(human_heroes_changed_skin_color['name'].loc[heroes['Skin color']=='gold'].values)
purple_skin=str(human_heroes_changed_skin_color['name'].loc[heroes['Skin color']=='purple'].values)

trace = go.Bar(
    x=list(human_heroes_changed_skin_color['Skin color'].value_counts().index),
    y=list(human_heroes_changed_skin_color['Skin color'].value_counts().values),
    text=[green_skin,red_skin,blue_skin,silver_skin,gold_skin,purple_skin],
    marker=dict(color=[
        'rgba(0, 250, 32, 0.8)', #green
        'rgba(255,0,0,0.9)', #red
        'rgb(0,0,255,0.7)', #blue
        'rgb(192,192,192,0.8)', #silver
        'rgb(255,255,0,0.9)', #gold
        'rgb(128,0,128,0.7)' #purple
    ])
)

layout = go.Layout(
    title='Human Heroes with changed skin color',
    barmode='bar'
)

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig, filename='Human-Heroes-changed-skin-color')


# In[72]:


# overall skin color distribution
skin_series = heroes['Skin color'].value_counts()
skins = list(skin_series.index)
color_distribution = list((skin_series/skin_series.sum())*100)

trace = go.Pie(labels=skins, values=color_distribution)

layout = go.Layout(
    title='skin color distributions',
    height=500,
    width=500
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='skin color distribution')


# In[73]:


# skin color distribution among the rest 10% heroes whose skin color is known

skin_series = heroes.loc[heroes['Skin color']!='unknown']['Skin color'].value_counts()
skins = list(skin_series.index)
color_distribution = list((skin_series/skin_series.sum())*100)

trace = go.Pie(
    labels=skins, 
    values=color_distribution,
    hoverinfo='label+percent', 
    textinfo='value',
    marker=dict(colors=[
        'rgba(0,255,0,0.9)', #green
        'rgba(0,0,255,0.8)', #blue
        'rgba(255,0,0,1)', #red
        'rgba(255,255,255,0.5)', #white
        'rgba(128,128,128,0.8))', #grey 
        'rgba(192,192,192,1)', #silver 
        'rgba(128,0,128,0.8)', #gold 
        'rgba(255,215,0,1)', #purple 
        'rgba(255,255,0,0.7)', #yellow
        'rgba(255,0,0,0.5)', #pink
        'rgba(0,255,255,0.6)', #blue/white
        'rgba(255,165,0,0.7)', #orange
        'rgba(128,0,0,0.9)', #red/black
        'rgba(128,128,128,0.6)', #gray
        'rgba(255,140,0,0.8)', #orange/white
        'rgba(0,0,0,0.8)', #black
    ],
        line=dict(
            color='rgb(8,48,107)',
            width=0.3))
)

layout = go.Layout(
    title='skin color distributions',
    height=700,
    width=700
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='skin color distribution')


# ## Hair color distribution

# __Now let's find more heroes like Professor X__
# 
# [ basically those who're bald ]

# In[74]:


bald_or_not = heroes['Hair color'].where(heroes['Hair color']=="No Hair", other='Hair')

trace = go.Bar(
    x=bald_or_not.value_counts().index,
    y=bald_or_not.value_counts().values,
    name='bald vs not-bald',
    text=['not-bald','bald']
)

layout = go.Layout(
    title='bald vs not-bald',
    barmode='bar'
)

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig, filename='distribution-by-baldness')


# In[75]:


# hair color distribution

hair_df = heroes.loc[~heroes['Hair color'].isin(['No Hair','unknown'])]['Hair color']
# excluding the bald types and unknown hair colors

# some values are same but with diff cases e.g. 'Blond' & 'blond' should be same, need to change all to lower case first
hair_df=hair_df.astype(str).str.lower()

hair_df=hair_df.str.replace('brownn','brown')

hair_series = hair_df.value_counts()
hair_colors = list(hair_series.index)
color_distribution = list((hair_series/hair_series.sum())*100)

trace = go.Pie(
    labels=hair_colors, 
    values=color_distribution,
    hoverinfo='label+percent', 
    textinfo='value',
    marker=dict(colors=[
        'rgba(0,0,0,0.8)', #black
        'rgba(243,243,164,0.8)', #blond
        'rgba(165,104,42,0.7)', #brown
        'rgba(255,0,0,0.9)', #red
        'rgba(255,255,255,1)', #white
        'rgba(165,42,42,0.9)', #auburn
        'rgba(0,255,0,0.7)', #green
        'rgba(165,88,29,0.7)', #strawberry blond
        'rgba(128,128,128,0.9)', #grey
        'rgba(128,0,128,0.8)', #purple
        'rgba(216,210,181,1)', #brown/white 
        'rgba(192,192,192,0.5)', #silver
        'rgba(0,0,255,0.8)', #blue
        'rgba(255,255,0,1)', #yellow 
        'rgba(255,165,0,0.6)', #orange
        'rgba(251,223,214,0.8)', #red/white
        'rgba(75,0,130,1)', #indigo
        'rgba(60,42,8,1)', #brown / black
        'rgba(255,0,255,1)', #magenta
        'rgba(116,68,56,1)', #red/grey
        'rgba(20,28,27,1)', #black / blue
        'rgba(255,215,0,0.9)', #gold
        'rgba(255,64,0,1)', #red/orange
        'rgba(251,194,123,0.7)', #orange/white
        'rgba(255,192,203,1)', #pink
    ],
        line=dict(
            color='rgb(8,48,107)',
            width=0.3))
)

layout = go.Layout(
    title='distributions by Hair color',
    height=750,
    width=750
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='hair color distribution')


# ### Let's look into the super-powers of these heroes now

# In[76]:


powers = pd.read_csv('../input/super_hero_powers.csv')
powers.head()


# wow! everything is in boolean...my life is so much simplified now :))

# In[77]:


# converting all values to 0s and 1s
powers = powers * 1
powers.head(2)


# ## Most powerful superhero

# In[78]:


powers.loc[:,'total_powers'] = powers.iloc[:, 1:].sum(axis=1)
powers.head(2)


# In[79]:


trace = go.Bar(
    x=powers['hero_names'],
    y=powers['total_powers'],
    text=['names','total powers']
)

layout = go.Layout(
    title='most powerfull superhero',
    barmode='bar'
)

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig, filename='most-powerfull-superhero')


# > that's a lot, lets plot for the top 30 superheroes only

# In[80]:


powers = powers.sort_values('total_powers', ascending=False)

trace = go.Bar(
    x=powers['hero_names'].head(30),
    y=powers['total_powers'].head(30),
    text=['names','total powers']
)

layout = go.Layout(
    title='most powerfull superhero',
    barmode='bar'
)

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig, filename='most-powerfull-superhero')


# ### and the Winner is: Spectre
# 
# so there are characters even more powerful than Superman and Goku. 
# 
# #### <i>Atleast the internet will now have the answer for the "Superman vs Goku" thing. Superman is clearly a winner here! </i> 
# Although Goku can evolve into multiple levels of Super-Saiyan but we don't know about the powers of a Super Saiyan yet.

# ## Most common super-powers
# 
# with a lil inspiration from other kernels, i thought it would be interesting to see the most common super powers as well.

# In[81]:


df = powers.drop(['hero_names'], axis=1)

df2 = pd.DataFrame()
for col in list(df.columns):
    df2[col] = df[col].value_counts()
    
df2.drop(['total_powers'], axis=1, inplace=True)
df2 = df2.T
df2.drop([0], axis=1, inplace=True)


# In[82]:


df2.sort_values(1, ascending=False)
df2.rename(columns={1: 'total_heroes'}, inplace=True)


# In[83]:


df2.sort_values('total_heroes', ascending=False, inplace=True)
df2['super_power']=df2.index


# In[84]:


trace = go.Bar(
    x=np.array(df2['super_power'].loc[df2['total_heroes']>100]),
    y=np.array(df2['total_heroes'].loc[df2['total_heroes']>100])
)
layout = go.Layout(
    title='most common super powers',
    barmode='bar'
)

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig, filename='most common super powers')


# ## Most unique super-powers

# In[85]:


list(df2['super_power'].loc[df2['total_heroes']==1])


# ##### This is my first kernel and I'm still working on this, also this is the first time I'm working with plotly, so please excuse if my approaches seem very basic. Also please upvote if you liked the insights projected in this kernel. Any feedback or suggestions you've on this dataset that I should include, please leave in the comments I'll be happy to add them as well. Thank you! :)

# In[ ]:





# In[ ]:





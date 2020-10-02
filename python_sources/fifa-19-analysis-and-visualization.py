#!/usr/bin/env python
# coding: utf-8

# # FIFA-19
# 
# This dataset is as follows:
# Data columns (total 90 columns):
# ````
#  #Column                    Non-Null Count  Dtype  
# ---  ------                    --------------  -----  
#  0   Unnamed: 0                18207 non-null  int64  
#  1   ID                        18207 non-null  int64  
#  2   Name                      18207 non-null  object 
#  3   Age                       18207 non-null  int64  
#  4   Photo                     18207 non-null  object 
#  5   Nationality               18207 non-null  object 
#  6   Flag                      18207 non-null  object 
#  7   Overall                   18207 non-null  object 
#  8   Potential                 18207 non-null  int64  
#  9   Club                      17966 non-null  object 
#  10  Club Logo                 18207 non-null  object 
#  11  Value                     18207 non-null  object 
#  12  Wage                      18207 non-null  object 
#  13  Special                   18207 non-null  int64  
#  14  Preferred Foot            18159 non-null  object 
#  15  International Reputation  18159 non-null  float64
#  16  Weak Foot                 18159 non-null  float64
#  17  Skill Moves               18159 non-null  float64
#  18  Work Rate                 18159 non-null  object 
#  19  Body Type                 18159 non-null  object 
#  20  Real Face                 18159 non-null  object 
#  21  Position                  18147 non-null  object 
#  22  Jersey Number             18147 non-null  float64
#  23  Joined                    16654 non-null  object 
#  24  Loaned From               1264 non-null   object 
#  25  Contract Valid Until      17918 non-null  object 
#  26  Height                    18159 non-null  object 
#  27  Weight                    18159 non-null  object 
#  28  LS                        16122 non-null  object 
#  29  ST                        16122 non-null  object 
#  30  RS                        16122 non-null  object 
#  31  LW                        16122 non-null  object 
#  32  LF                        16122 non-null  object 
#  33  CF                        16122 non-null  object 
#  34  RF                        16122 non-null  object 
#  35  RW                        16122 non-null  object 
#  36  LAM                       16122 non-null  object 
#  37  CAM                       16122 non-null  object 
#  38  RAM                       16122 non-null  object 
#  39  LM                        16122 non-null  object 
#  40  LCM                       16122 non-null  object 
#  41  CM                        16122 non-null  object 
#  42  RCM                       16122 non-null  object 
#  43  RM                        16122 non-null  object 
#  44  LWB                       16122 non-null  object 
#  45  LDM                       16122 non-null  object 
#  46  CDM                       16122 non-null  object 
#  47  RDM                       16122 non-null  object 
#  48  RWB                       16122 non-null  object 
#  49  LB                        16122 non-null  object 
#  50  LCB                       16122 non-null  object 
#  51  CB                        16122 non-null  object 
#  52  RCB                       16122 non-null  object 
#  53  RB                        16122 non-null  object 
#  54  Crossing                  18159 non-null  float64
#  55  Finishing                 18159 non-null  float64
#  56  HeadingAccuracy           18159 non-null  float64
#  57  ShortPassing              18159 non-null  float64
#  58  Volleys                   18159 non-null  float64
#  59  Dribbling                 18159 non-null  float64
#  60  Curve                     18159 non-null  float64
#  61  FKAccuracy                18159 non-null  float64
#  62  LongPassing               18159 non-null  float64
#  63  BallControl               18159 non-null  float64
#  64  Acceleration              18159 non-null  float64
#  65  SprintSpeed               18159 non-null  float64
#  66  Agility                   18159 non-null  float64
#  67  Reactions                 18159 non-null  float64
#  68  Balance                   18159 non-null  float64
#  69  ShotPower                 18159 non-null  float64
#  70  Jumping                   18159 non-null  float64
#  71  Stamina                   18159 non-null  float64
#  72  Strength                  18159 non-null  float64
#  73  LongShots                 18159 non-null  float64
#  74  Aggression                18159 non-null  float64
#  75  Interceptions             18159 non-null  float64
#  76  Positioning               18159 non-null  float64
#  77  Vision                    18159 non-null  float64
#  78  Penalties                 18159 non-null  float64
#  79  Composure                 18159 non-null  float64
#  80  Marking                   18159 non-null  float64
#  81  StandingTackle            18159 non-null  float64
#  82  SlidingTackle             18159 non-null  float64
#  83  GKDiving                  18159 non-null  float64
#  84  GKHandling                18159 non-null  float64
#  85  GKKicking                 18159 non-null  float64
#  86  GKPositioning             18159 non-null  float64
#  87  GKReflexes                18159 non-null  float64
#  88  Release Clause            16643 non-null  float64
#  89  Age_dist                  18207 non-null  object 
# ````

# ## Importing and Analysis of the dataset

# In[ ]:


import numpy as np
import pandas as pd
data=pd.read_csv("/kaggle/input/fifa19/data.csv")
data.head()


# In[ ]:


data.describe()


# # WordClod and Map Visualization
# Here we will see how to create a wordcloud and visualize country data

# In[ ]:


import wordcloud as wc
text=np.array(data['Nationality'])
cloud=wc.WordCloud()
cloud.generate(" ".join(text))
cloud.to_image()


# In[ ]:


import plotly.graph_objects as go
from plotly.offline import init_notebook_mode,iplot
import plotly.express as px
countries=data.Nationality.value_counts()
f= go.Figure(data=go.Choropleth(
    locations=countries.index,
    z =countries, 
    locationmode = 'country names', 
    colorscale =px.colors.sequential.Plasma,
    colorbar_title = "NO. of players",
))

f.update_layout(
    title_text = 'Number of players from each country',
)
iplot(f)


# # Basic Plotting
# 
# Here we will analyse the relation between some variables through plotting

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,10))
sns.scatterplot(data.Composure,data.Potential,color='r')
sns.scatterplot(data.Composure,data.Overall,color='g')


# In[ ]:


fig=plt.figure(figsize=(15,10))
ax=fig.add_subplot(121)
ax.plot(data.StandingTackle,'.',color='r')
bx=fig.add_subplot(122)
bx.plot(data.SlidingTackle,'.',color='g')


# In[ ]:


fig=plt.figure(figsize=(15,10))
ax=fig.add_subplot(321)
ax.plot(data.GKDiving,'.',color='y')
bx=fig.add_subplot(322)
bx.plot(data.GKHandling,'.',color='c')
cx=fig.add_subplot(323)
cx.plot(data.GKKicking,'.',color='r')
dx=fig.add_subplot(324)
dx.plot(data.GKPositioning,'.',color='b')


# In[ ]:


Clubs=data.Club.value_counts()
plt.plot(np.unique(Clubs),'.',color='r')


# # Feature Extraction
# Here I have extracted the money associated with each player and afterwards will perform some basic plotting

# In[ ]:


def Money(x):
    if type(x)==float:
        pass
    else:
        m=x[1:]
        x=m[:-1]
        return round(float(x))
data['Release Clause']=data['Release Clause'].apply(Money)


# In[ ]:


data.head()


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(data['Release Clause'],'.',color='c')


# In[ ]:


sns.scatterplot(x='International Reputation',y='Release Clause',data=data,color='y')


# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,cmap='inferno')


# In[ ]:


sns.scatterplot(data.Age,data['Release Clause'])


# In[ ]:


plt.figure(figsize=(15,10))
sns.distplot(data['Release Clause'],color='g')


# # Categorising numerical Data
# Here I will categorise the age values and the overall potential values for players to check if there is any connection between their overall values and age

# In[ ]:


def age_d(x):
    if x>30:
        return 'Above 30'
    if x<=30 and x>25:
        return 'Between 25-30'
    if x<=25 and x>20:
        return 'Between 20-25'
    else:
        return 'Below 20'
        
data['Age_dist']=data.Age.apply(age_d)


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(data['Age_dist'])


# In[ ]:


def overall_d(x):
    if x>90:
        return 'Above 90'
    if x<=90 and x>80:
        return 'Between 90-80'
    if x<=80 and x>70:
        return 'Between 80-70'
    if x<=70 and x>60:
        return 'Between 70-60'
    if x<=60 and x>50:
        return 'Between 60-50'
    else:
        return 'Below 50'
    
data.Overall=data.Overall.apply(overall_d)


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(data['Overall'])


# # Creating a Crosstab
# 
# A crosstab is used for visualising categorical vs categorical data

# In[ ]:


crosstab=pd.crosstab(data['Age_dist'],data['Overall'])
crosstab.plot.bar(stacked=True,figsize=(15,10))


# # Insights gained from above plotting
# From the above plotting, we can say that players having age>25 have more overall values in comparison to other ages

# # Relation between player jersey numbers and their skills
# 
# Here i will be checking if there is any connection between the player skills and their jersey numbers

# In[ ]:


plt.figure(figsize=(15,10))
sns.lineplot(x='Jersey Number',y='Skill Moves',data=data,hue='Overall')


# # Insights gained from above plotting
# From the above plotting we can say that players having jersey number between 1-20,especially around 10 have high skill values

#  # Relation between potential values and jersey numbers
#  Here i will check the connection between player potential and their jersey number

# In[ ]:


plt.figure(figsize=(15,10))
sns.lineplot(x='Jersey Number',y='Potential',data=data,hue='Overall')


# # Insights gained from above plotting 
# From the above plots we can say that players having high overall and high skill values have jersey numbers between 1-20.I think that's why *messi have a jersey number of 10*.

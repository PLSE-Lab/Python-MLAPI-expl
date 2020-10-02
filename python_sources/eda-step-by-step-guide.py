#!/usr/bin/env python
# coding: utf-8

# ### The notebook contains some EDA on the PubG training dataset.
# 
# #### The initial part contains univariate analysis of some of the important variable in the dataset while the later half of the notebook involves multivariate and bivariate analysis
# 
# #### I have also looked at correlations of the top-10 variables with the target variable at the end
# 
# ### Some of the important findings of the notebook are:
# 
#     * Variable have variation in them, but most of them seem to be scattered in confined boundaries.
#     * Some of the variables have really good correlations with the target variable
#     * The variable have good correlations among themselves as well, hence, it will be crucial to handle this 
#       situation by either using multicollinearity reduction techniques or go with some non-linear model which can
#       handle such multicollinearity issues very well, for e.g. Tree based models or Neural Networks.
#     * Since, it is a team game, hence group level agggregate values (mean, min, max) of these variables can also 
#       play a sinificant role in estimating the value of the target variable
# 
# #### * Please give a thumbs up if you like the notebook !

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Load other essential libraries

# In[ ]:


import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt #Visulization
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import gc

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load Datasets

# In[ ]:


df_train = pd.read_csv('../input/train_V2.csv')
df_train = df_train[df_train['winPlacePerc'].notnull()].reset_index(drop=True)

df_test  = pd.read_csv('../input/test_V2.csv')


# In[ ]:


print("Train : ",df_train.shape)
print("Test : ",df_test.shape)


# In[ ]:


df_train.head()


# ### Visualising some of the important/interesting variables as per competition

# #### Total number of assists in killing an opponent

# In[ ]:


temp = df_train['assists'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='assists'
    ),
    yaxis=dict(
        title='Count of assists'
        )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# #### Number of boost items used

# In[ ]:


temp = df_train['boosts'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='boosts',
    ),
    yaxis=dict(
        title='Count of boosts',
        )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# #### Number of enemy players killed with headshots

# In[ ]:


temp = df_train['headshotKills'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='headshotKills',
    ),
    yaxis=dict(
        title='Count of headshotKills',
        )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### Number of healing items used

# In[ ]:


temp = df_train['heals'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='heals',
    ),
    yaxis=dict(
        title='Count of heals',
        )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### Max number of enemy players killed in a short amount of time

# In[ ]:


temp = df_train['killStreaks'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='killStreaks',
    ),
    yaxis=dict(
        title='Count of killStreaks',
        )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### Number of players killed

# In[ ]:


temp = df_train['kills'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='kills',
    ),
    yaxis=dict(
        title='Count of kills',
        )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### Different match Types

# In[ ]:


temp = df_train['matchType'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='matchType',
    ),
    yaxis=dict(
        title='Count of matchType',
        )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### Number of weapons acquired

# In[ ]:


temp = df_train['weaponsAcquired'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='weaponsAcquired',
    ),
    yaxis=dict(
        title='Count of weaponsAcquired',
        )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### Longest kill distance - this variable does have some interesting looking variance

# In[ ]:


#histogram
f, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['longestKill'])


# ### Distance traveled walking

# In[ ]:


#histogram
f, ax = plt.subplots(figsize=(18, 6))
sns.distplot(df_train['walkDistance'])


# ### Swimming distance

# In[ ]:


#histogram
f, ax = plt.subplots(figsize=(18, 6))
sns.distplot(df_train['swimDistance'])


# ### ride Distance (on vehicle)

# In[ ]:


#histogram
f, ax = plt.subplots(figsize=(18, 6))
sns.distplot(df_train['rideDistance'])


# ## The target variable

# In[ ]:


#histogram
f, ax = plt.subplots(figsize=(18, 6))
sns.distplot(df_train['winPlacePerc'])


# ## From the above plots it's evident that while the explored attributes do have some variance, their values are mostly confined in s small range.
# 
# ## After this univariate analysis, let's move on to multivariate analysis and try to see some correlations of the variables with each other as well as the target variable
# 
# 

# In[ ]:


#winPlacePerc correlation matrix
k = 10 #number of variables for heatmap
corr = df_train.corr() 
cols = corr.nlargest(k, 'winPlacePerc').index # nlargest : Return this many descending sorted values
cm = np.corrcoef(df_train[cols].values.T) # correlation 
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(14, 10))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# ## The above plot shows some correlations which are pretty obvious like - (damageDealt-kills), (kills-killStreaks), etc. while some others are also interesting to see like - (boosts -winPlacePerc) , (walkDistance - winPlacePerc) 
# ### The commonly striking attributes like kills, longestKill, etc. seem to show somewhat less correlation with winPlacePerc
# 
# ## Let's see some of these attributes individually

# In[ ]:


df_train.plot(x="kills",y="winPlacePerc", kind="scatter", figsize = (8,6))


# In[ ]:


df_train.plot(x="damageDealt",y="winPlacePerc", kind="scatter", figsize = (8,6))


# In[ ]:


df_train.plot(x="longestKill",y="winPlacePerc", kind="scatter", figsize = (8,6))


# In[ ]:


df_train.plot(x="weaponsAcquired",y="winPlacePerc", kind="scatter", figsize = (8,6))


# In[ ]:


### Removing some anomalies from walkDistance to give a better idea of it's association with the target variable
df_train.plot(x="walkDistance",y="winPlacePerc", kind="scatter", figsize = (8,6))


# ### Removing some anomalies from walkDistance to give a better idea of it's association with the target variable

# In[ ]:


f = df_train[df_train["walkDistance"] < 15000]
f.plot(x="walkDistance",y="winPlacePerc", kind="scatter", figsize = (8,6))


# In[ ]:


f, ax = plt.subplots(figsize=(18, 6))
fig = sns.boxplot(x='boosts', y="winPlacePerc", data=df_train)
fig.axis(ymin=0, ymax=1);


# ### It's clearly visible from above charts, specially the latest two that as compared to other variables, boosts and walkDistance definitely show a disciminating property as fas as winPlacePerc is concerned

# ### One other attribute which seemed interesting from it's description was matchType, let's see if it has some information on the target variable or not.

# In[ ]:


f, ax = plt.subplots(figsize=(18, 6))
fig = sns.boxplot(x='matchType', y="winPlacePerc", data=df_train)
fig.axis(ymin=0, ymax=1);


# ### Looks like it does not say much about the target variable and median value of winPlacePerc is mostly around 0.5 for all of them.

# ### Since PubG is a team game, hence it is also worth to see how the top correlated attributes behave at match and group level in relation to the target variable

# In[ ]:


k = 10 #number of variables with highest correlation with winPlacePerc
corr = df_train.corr() 
cols = corr.nlargest(k, 'winPlacePerc').index # nlargest : Return this many descending sorted values

agg = df_train.groupby(['matchId','groupId'],as_index=False)[cols].agg('mean')


# In[ ]:


cols


# ### As done previously, let's remove some extreme values from walkDistance and look at it's correlation at roup level values

# In[ ]:


f = agg[agg["walkDistance"] < 15000]
f.plot(x="walkDistance",y="winPlacePerc", kind="scatter", figsize = (8,6))


# In[ ]:


agg.plot(x="boosts",y="winPlacePerc", kind="scatter", figsize = (8,6))


# In[ ]:


agg.plot(x="weaponsAcquired",y="winPlacePerc", kind="scatter", figsize = (8,6))


# In[ ]:


agg.plot(x="kills",y="winPlacePerc", kind="scatter", figsize = (8,6))


# ## Looks like the mean values at group and match level do have some reat deal of importance in relation to the taret variable
# 
# ## Let's also look at how max and min values across these top variables behave in correlation with group level winPlacePerc values

# In[ ]:


agg = df_train.groupby(['matchId','groupId'],as_index=False)[cols].agg('max')
f = agg[agg["walkDistance"] < 15000]
f.plot(x="walkDistance",y="winPlacePerc", kind="scatter", figsize = (8,6))


# In[ ]:


f, ax = plt.subplots(figsize=(18, 6))
fig = sns.boxplot(x='boosts', y="winPlacePerc", data=agg)
fig.axis(ymin=0, ymax=1);


# In[ ]:


agg.plot(x="longestKill",y="winPlacePerc", kind="scatter", figsize = (8,6))


# In[ ]:


agg = df_train.groupby(['matchId','groupId'],as_index=False)[cols].agg('min')

f = agg[agg["walkDistance"] < 15000]
f.plot(x="walkDistance",y="winPlacePerc", kind="scatter", figsize = (8,6))


# In[ ]:


f, ax = plt.subplots(figsize=(18, 6))
fig = sns.boxplot(x='boosts', y="winPlacePerc", data=agg)
fig.axis(ymin=0, ymax=1);


# In[ ]:


agg.plot(x="longestKill",y="winPlacePerc", kind="scatter", figsize = (8,6))


# # From the above analysis it is pretty much clear that along with individual level attribute values, team level aggregated values will also play an important role in estimation of the target variable

# In[ ]:





# In[ ]:





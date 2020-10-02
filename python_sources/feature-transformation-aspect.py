#!/usr/bin/env python
# coding: utf-8

# # Feature transformation: Aspect
# ![](https://www.newbedford-ma.gov/nbcompass/wp-content/uploads/sites/65/Compass2-e1557776614257.jpg)
# 
# In physical geology, aspect is the compass direction that a slope faces. For example, a slope on the eastern edge of the Rockies toward the Great Plains is described as having an easterly aspect. [Aspect (geography)](https://en.wikipedia.org/wiki/Aspect_%28geography%29)
# 
# Let's see how to **Aspect** feature is distributed by cover types.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('/kaggle/input/learn-together/train.csv')
train.shape


# In[ ]:


type_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
type_ids = sorted(train['Cover_Type'].unique())

current_palette = sns.color_palette()
n = 3
fig, ax = plt.subplots(3, n, figsize=(25,15))
for t in type_ids:
    x, y = (t-1)//n, (t-1)%n
    ax[x,y].set_title(str(t) + ': ' + type_names[t-1])
    data = train['Aspect'][train['Cover_Type']==t]
    sns.distplot(data, ax=ax[x, y], color=current_palette[t-1]);
    


# We can notice that some cover types mostly grouped around particuler aspect values.
# 
# **Cottonwood/Willow** is grouped around **120 degrees** on graph 4.
# 
# **Aspen** is grouped around **80 degrees** on graph 5.
# 
# Also we can see that some cover types have **splashes** around the edges: graph 1 and 6.

# Let's split *Aspect* by bins 
# 
# | Aspect_Name | min | max |
# | --- | --- | --- |
# | North | 0 | 22.5 |
# | Northeast | 22.5 | 67.5 |
# | East | 67.5 | 112.5 |
# | Southeast | 112.5 | 157.5 |
# | South | 157.5 | 202.5 |
# | Southwest | 202.5 | 247.5 |
# | West | 247.5 | 292.5 |
# | Northwest | 292.5 | 337.5 |
# | North (!) | 337.5 | 360 |

# Pay attention that **North** means range from 337.5 to 22.5, because 360 degrees it's a circle and 0 degrees == 360 degrees[](http://).

# In[ ]:


aspect_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
degree = np.array([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360])
def get_aspect_name(aspect):
    d = degree - aspect
    indx = np.where(d > 0, d, np.inf).argmin()
    return aspect_names[indx]

train['Aspect_Name'] = train['Aspect'].apply(get_aspect_name).astype('category')
train['Aspect_Name'].cat.set_categories(aspect_names[:-1])


# Now we can plot cover types distributing by compass directions.

# In[ ]:


fig, ax = plt.subplots(3, 3, figsize=(25,15))
for t in type_ids:
    x, y = (t-1)//3, (t-1)%3
    ax[x,y].set_title(str(t) + ': ' + type_names[t-1])
    data = train['Aspect_Name'][train['Cover_Type']==t]
    sns.countplot(x=data, ax=ax[x, y], order=aspect_names[:-1]);


# **Cottonwood/Willow** is more presented on **East** and **Southeast** (see graph 4).
# 
# **Aspen** is more presented on **East** and  **Northeast** (see graph 5).
# 
# The both **Spruce/Fir** and **Douglas-fir** on graph 1 and 6 are grouped around **Nort** direction.
# 
# For greater clarity, let's draw pie charts.
# 

# In[ ]:


fig, ax = plt.subplots(3, 3, figsize=(20,15))
for t in type_ids:
    x, y = (t-1)//3, (t-1)%3
    ax[x,y].set_title(str(t) + ': ' + type_names[t-1])
    val = train['Aspect_Name'][train['Cover_Type']==t].value_counts().to_dict()
    val_sorted = dict(sorted(val.items(), key=lambda x: aspect_names[:-1].index(x[0])))    
    labels = [key + ' (' + str(val) + ')' for key,val in val_sorted.items()]
    ax[x, y].pie(val_sorted.values(), labels=labels)
    hole = plt.Circle((0,0), 0.5, color='white')
    ax[x, y].add_artist(hole)

plt.show();


# It's seems as if new feature **Aspect_Name** can be useful in prediction model :)

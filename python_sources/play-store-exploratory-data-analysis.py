#!/usr/bin/env python
# coding: utf-8

# # **Introduction**

# > I'm kind of new in data science and I've just started learning statistics. My purpose of sharing kernels is to prove what I've learned up to now. When I choose a data for my kernel, I usually check different kernels of another users with that dataset. I do this since I want to improve my statistical and analytical thinking about analysis of a data. Sometimes I copy out some of the codes with little changes on it, but I make sure that I've learned the method, the statistical meaning and benefit of that kind of plotting or coding. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/googleplaystore.csv')


# # **Data Overview**

# As you see below, most of the numerical data in object type and there are also NaN values in them. Let's go deeper in recognizing data, then deal with those issues.

# In[ ]:


df.info()


# In[ ]:


df.sample(10)


# *Finding out the number of  NaN values in each column*

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df.isnull().sum().sort_values(ascending=False)


# *Replacing str values and changing types*

# # **Cleaning Data**

# In[ ]:


df = df[df['Android Ver'] != 'NaN']
df = df[df['Installs'] != 'Free']


# In[ ]:


df.Installs.unique()


# In[ ]:


df.Installs = df.Installs.apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
df.Installs = df.Installs.apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
df.Installs = df.Installs.apply(lambda x: int(x))


# In[ ]:


df.info()


# In[ ]:


df.Size = df.Size.apply(lambda x: x.replace('M', '000') if 'M' in x else x)
df.Size = df.Size.apply(lambda x: x.replace('k','') if 'k' in str(x) else x)
df.Size = df.Size.apply(lambda x: x.replace('Varies with device','0') if 'Varies with device' in str(x) else x)
df.Size = df.Size.apply(lambda x: float(x))


# In[ ]:


df.Price = df.Price.apply(lambda x: x.replace('$','') if '$' in str(x) else x)
df.Price = df.Price.astype(float)


# In[ ]:


df.Reviews = df.Reviews.apply(lambda x: int(x))


# In[ ]:


df.sample(10)


# # **Plots**

# In[ ]:


fig = {
  "data": [
    {
      "values": df.Category.value_counts().values,
      "labels": df.Category.value_counts().index,
      "type": "pie"
    }],
  "layout": {
        "title":"Percentages of Categories in PlayStore"
    }
}
iplot(fig)


# **The distribution of prices**

# In[ ]:


plt.figure(figsize=(10,8))
sns.kdeplot(df.Price, shade=True,color='red')
plt.grid()
plt.show()


# *High kurtosis means a really heavy-tailed relative to a normal distribution and also there are **outliers**. Since most of the prices aggregate around 0 and price cannot be cheaper than free, distribution will have a positive skewness because of the prices being higher than 0.  Let's prove that there are too many outliers and most of the price centered around the zero.*

# In[ ]:


print('Kurtosis:',df.Price.kurt())
print('Skewness:',df.Price.skew())


# In[ ]:


df.head()


# In[ ]:


fig = {
    "data": [{
        "type": 'violin',
        "y": df.Price,
        "box": {
            "visible": True
        },
        "line": {
            "color": 'black'
        },
        "box": {
                "visible": True
        },
        "meanline": {
                "visible": True
        },
        "x0": 'Price'
    }],
    "layout" : {
        "title": "Violin & Box Plots of Price",
        "yaxis": {
            "zeroline": False,
        }
    }
}
iplot(fig)


# *What are the first 10 application categories which are spended on the most?*

# In[ ]:


plt.figure(figsize=(13,9))
df.groupby('Category')['Price'].sum().nlargest(10).sort_values().plot('barh')
plt.xlabel('Total Price', fontsize = 12)
plt.ylabel('Categories',fontsize=12)
plt.show()


# In case you have "NaN" values in your data,  you're like to be given an error like that in an EDA analysis; **"max must be larger than min in range parameter".** So we need to drop them all firstly.
# 
# > * EDA is the process of figuring out what the data can tell us and we use EDA to find patterns, relationships, or anomalies to inform our subsequent analysis.
# * A pairplot is one of the most effective basic tools for EDA allowing us to see both distribution of single variables and relationships between two variables.
# * The distplots on diagonal below (histogram for discrete values), shows us the distribution of features. 

# In[ ]:


rating= df['Rating'].dropna()
size= df['Size'].dropna()
installs= df['Installs'][df.Installs!=0].dropna()
reviews= df['Reviews'][df.Reviews!=0].dropna()
types = df['Type'].dropna()
price = df['Price']

data = pd.concat([rating, size, np.log(installs), np.log10(reviews), types, price], axis=1)

plt.figure(figsize=(15,15))
sns.pairplot(data, hue='Type', palette="husl", markers= ['s','d'])
plt.show()


# ***The intersections and hypothesis of "Reviews" and "Installs"***

# In[ ]:


plt.figure(figsize=(13,13))
sns.jointplot(x=data['Reviews'], y=data['Installs'], kind='reg', color='blue')
plt.show()


# ***Pricing w.r.t. categories***
# * There are some apps even more expensive than 300$, let's see them as well.

# In[ ]:


plt.figure(figsize=(13,10))
sns.stripplot(x= df[df.Price>0].Price, y=df.Category) # a scatterplot where one variable is categorical
plt.grid()
plt.show()

df[['Category','App']][df.Price >= 300]


# ***Statistics of Ratings w.r.t. all categories (Violin & Box)***

# In[ ]:


len(df.Category.value_counts().sort_values(ascending=False).index)


# In[ ]:





# In[ ]:


print(df.Category.nunique())

datalist = []
c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 255, 33)]

for i in range(33):
    violins = {
            "type": 'violin',
            "y": df.Rating[df.Category == df.Category.value_counts().sort_values(ascending=False).index[i]],
            "name": df.Category.value_counts().sort_values(ascending=False).index[i],
            "marker":{
                "color":c[i]},
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        }
    datalist.append(violins)
iplot(datalist)


# **Now let's just see the first 5 significant categories in  box plots**

# In[ ]:


print(df.Category.value_counts().sort_values(ascending=False).nlargest(5))

df_family = df[df.Category=='FAMILY']
df_game = df[df.Category=='GAME']
df_tools = df[df.Category=='TOOLS']
df_business = df[df.Category=='BUSINESS']
df_medical = df[df.Category=='MEDICAL']

box1 = go.Box(
                y= df_family.Rating,
                name= 'Family',
                marker = dict(color = 'rgb(12, 128, 128)'))
box2 = go.Box(
                y= df_game.Rating,
                name= 'Game',
                marker = dict(color = 'rgb(100, 12, 38)'))
box3 = go.Box(
                y= df_tools.Rating,
                name= 'Tools',
                marker = dict(color = 'rgb(12, 128, 128)'))
box4 = go.Box(
                y= df_business.Rating,
                name= 'Business',
                marker = dict(color = 'rgb(50, 40, 100)'))
box5 = go.Box(
                y= df_medical.Rating,
                name= 'Medical',
                marker = dict(color = 'rgb(45, 179, 66)'))

data_boxes = [box1,box2,box3,box4,box5]
iplot(data_boxes)


# **Percentages of Most Downloaded 5 Categories with Their Types**

# In[ ]:


new_df = df.groupby(['Category', 'Type']).agg({'App' : 'count'}).reset_index()

outer_group_names = ['GAME', 'FAMILY', 'MEDICAL', 'TOOLS','BUSINESS']
outer_group_values = [len(df[df.Category == category]) for category in outer_group_names]

a,b,c,d,e =[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]

inner_group_names = ['Paid', 'Free'] * 5
inner_group_values = []

for category in outer_group_names:
    for t in ['Paid', 'Free']:
        x = new_df[new_df.Category == category]
        try:
            inner_group_values.append(int(x.App[x.Type == t].values[0]))
        except:
            inner_group_values.append(0)

explode = (0.025,0.025,0.025,0.025,0.025)

# Outer ring
fig, ax = plt.subplots(figsize=(10,10))
ax.axis('equal')
mypie, texts, _ = ax.pie(outer_group_values, radius=1.5, labels=outer_group_names, autopct='%1.1f%%', pctdistance=1.1,
                                 labeldistance= 0.75,  explode = explode, colors=[a(0.6), b(0.6), c(0.6), d(0.6),e(0.6)], textprops={'fontsize': 16})
plt.setp( mypie, width=1, edgecolor='black')
 
# Inner ring
mypie2, _ = ax.pie(inner_group_values, radius=0.7, labels=inner_group_names, labeldistance= 0.7, 
                   textprops={'fontsize': 12}, colors = [a(0.4), a(0.2), b(0.4), b(0.2), c(0.4), c(0.2), d(0.4), d(0.2),e(0.4)])
plt.setp( mypie2, width=0.7, edgecolor='black')
 
plt.show()


# * Thanks in advance for your upvotes.
# 
# 
# 
# # END

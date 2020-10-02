#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#data visualizationship enables us to understand the data and its internal relationships using our visual sense
#this allows us to engage more effectively with the data as opposed to studying rawnumbers etc 



####purpose of data visualization####
##by understanding the inter-relationships between the features of a dataset, we can make a more
##informed decision about how to treat the data and which ml models to use to get 
##the best possible accuracy or prediction.

##without understanding this through visualization tools we will be treating 
##the data in the darkness. 


# **regression/linear-regression**
# 
# **KDE Kernal Density estimation **
# 
# **bins**
# 
# **confidence interval**
# 
# **outlier**
# 
# **quartile**
# 
# **interquartile range**
# 
# **normal distribution, standard distribution**

# In[ ]:


#importing library 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(16, 6))
figsize=(20,10)
sns.set(style='darkgrid')


# In[ ]:


#use built in dataset 'tips'
df = sns.load_dataset('tips')
df.head()


# In[ ]:


df.dtypes
df.shape


# In[ ]:


#relationship plot how one var relates to another 
sns.relplot(x = 'total_bill',y='tip',data=df)


# In[ ]:


#by adding hue we can differentiate based on categorical feature 
sns.relplot(x = 'total_bill',y='tip',data=df,hue='smoker')


# In[ ]:


#by using style we can differentiate further with another category 
#by adding so many elements to the plot it is possible that the result will be confusing 


sns.relplot(x = 'total_bill',y='tip',data=df,hue='sex',style='smoker')


# **When hue is categorical it will be distict based on categorical, when it is numeric it will be a gradient going from lower to upper limit **

# In[ ]:


#changing the hue of the plots using size feature intead of sex
sns.relplot(x = 'total_bill',y='tip',data=df,hue='size',style='time')


# **#use .fig.set_size_inches(x,y) to set the figure size **

# In[ ]:


#changing the size of the plot mark based on size variable, if size bigger dataplot also bigger 
#influence the size gradient with sizes () input




f = sns.relplot(x = 'total_bill',y='tip',data=df,hue='size',size = 'size',sizes = (45,900),style='time')

f.fig.set_size_inches(9.35,8)


# In[ ]:


#rel plot can also be used for line graphs 


# In[ ]:


#correlation
#find inter-relationship between different features 

df.corr()
#correlation data


# In[ ]:


#use heat map to visualize corelation 
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,7.75))
#above lines required for changing the size of heatmap 

sns.heatmap(df.corr())


# In[ ]:


#using joint joint plot 

sns.jointplot(x='tip',y='total_bill',data=df
              ,kind='reg').fig.set_size_inches(8.25,7.65)


# In[ ]:


#pair plot 
#for multiple bi variable relationships 
sns.pairplot(df)


# In[ ]:


#separating too graphs based on chosen variable, using 'col' input 
#if vertical arrangement is required 'row' can be used 
f = sns.relplot(height=7,x = 'total_bill',y='tip',data=df,hue='size',size = 'size',sizes = (45,1000),col='time')


# In[ ]:


#if chosen variable for col has many categories, that many graphs will be formed 
f = sns.relplot(x = 'total_bill',y='tip',data=df,hue='size',size = 'size',sizes = (45,1000),col='size',col_wrap=3)

#by choosing size as the input for col, all data points with the same size feature have been grouped 
#col_wrap can be used to arrange the plots 


# **arranging multiple charts next to each other, as a group can have greater impact **

# In[ ]:


sns.pairplot(df,hue='smoker').fig.set_size_inches(10,8)


# In[ ]:


#distribution plot 

sns.distplot(df['tip'])


# In[ ]:


#visualizing categorical data
##influence size of graph with height input 
sns.catplot(x='day',y='total_bill',data=df,height=7)


# In[ ]:


#align datapoints by turning off jitter 
sns.catplot(x='day',y='total_bill',data=df,height=7,jitter=False)


# In[ ]:


#swarm kind of categorical plot 
sns.catplot(x='day',y='total_bill',data=df,height=8,kind='swarm',hue='size')


# In[ ]:


sns.catplot(x='smoker',y='tip',data=df,height=5,hue='sex')


# In[ ]:


#manualy setting the order of categorical feature by using order input 
sns.catplot(x='smoker',order=['No','Yes'],y='tip',data=df,height=5,hue='sex')


# In[ ]:


#change orientation by switching x and y
sns.catplot(y='day',x='total_bill',data=df,height=7)


# In[ ]:


#count plot
#graphs the count of categories in a column 
sns.countplot('smoker',data=df)


# In[ ]:


#for categrorical features 


# In[ ]:


sns.countplot('day',data=df)


# In[ ]:


#x axis is default 
sns.countplot(y='time',data=df)


# In[ ]:


#bar plot, similar to count plot but x and y axis both set by us 

sns.barplot(x='total_bill',y='sex',data=df)


# In[ ]:


#box plot

sns.boxplot('smoker','total_bill', data=df)


# In[ ]:


sns.boxplot(x="total_bill", y="day", hue="smoker",data=df)


# In[ ]:


#using catplot function to create a box plot 

sns.catplot(x='day',y='total_bill',kind='box',data=df,hue='smoker',dodge=True).fig.set_size_inches(9.35,8)


# In[ ]:


#by using dogde = false ,hue element ie smoker/nonsmoker can be combined 
sns.catplot(x='day',y='total_bill',kind='box',data=df,hue='sex',dodge=False)


# In[ ]:


#boxen plot similar to box plot 
sns.catplot(x='day',y='total_bill',kind='boxen',data=df,dodge=False).fig.set_size_inches(9.35,8)


# In[ ]:



#violin plot 

#import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,7.75))
#above lines required for changing the size of plot 

sns.violinplot(y="total_bill", x="day", data=df,palette='rainbow')


# In[ ]:


#violin plot with cat plot , set 'kind' to violin 
sns.catplot(y='day',x='total_bill',kind='violin',hue='time',data=df,height=7,dodge=False)


# In[ ]:


#adding inner shading elements ? 
sns.catplot(y='day',x='total_bill',kind='violin',hue='time',data=df,height=7,dodge=False,inner='stick')


# In[ ]:


#using split feature for violin plot 


sns.catplot(y='day',x='total_bill',kind='violin',hue='sex',data=df,height=7,split=True)


# In[ ]:


#creating swarm plot on top of violin plot 
#playing with color palette options to get a good colours that can be seen easily 

f=sns.catplot(y='day',x='total_bill',kind='violin',data=df,height=10,split=True,palette='OrRd')
sns.swarmplot(y='day',x='total_bill',size=12,data=df,ax=f.ax,hue='size',palette='Blues_d')


# In[ ]:


#exploring line plots

#generating a new data set for this 
from numpy.random import randn
 
df = pd.DataFrame(dict(time = np.arange(500), value = randn(500).cumsum()))
df.head()


# In[ ]:



sns.relplot(x = 'time', y = 'value', kind = 'line', data = df,height=10)


# In[ ]:


#creating dataset where values are not ordered 

df = pd.DataFrame(randn(250, 2).cumsum(axis = 0), columns = ['time', 'value'])
df.head()


# In[ ]:


#when the values in the dataset are not sorted the line plot will be a chaotic zigzag
sns.relplot(x = 'time', y = 'value', kind = 'line', data = df, sort = False)


# In[ ]:


#sort = true 
sns.relplot(x = 'time', y = 'value', kind = 'line', data = df, sort = True)


# #using fmri dataset

# In[ ]:


#visualizing data with confidence interval 

#using fmri dataset
df = sns.load_dataset('fmri')
df.head()


# In[ ]:



#without confidence interval 
sns.relplot(x='timepoint',y='signal',estimator=None,kind ='line',data=df)


# In[ ]:


sns.relplot(x='timepoint',y='signal',kind ='line',data=df)


# In[ ]:


#using standard deviation as confidence interval
sns.relplot(x='timepoint',y='signal',kind ='line',data=df,ci='sd')


# In[ ]:


#adding hue element for this plot 
sns.relplot(x='timepoint',y='signal',hue='event',kind ='line',data=df,ci='sd').fig.set_size_inches(9.35,8)


# In[ ]:


#adding style element to this plot with a different hue variable
sns.relplot(x='timepoint',y='signal',hue='region',style='event',kind ='line',data=df,ci='sd')


# In[ ]:


#adding markers 
sns.relplot(x='timepoint',y='signal',hue='event',style='event',kind ='line',data=df,markers=True)


# In[ ]:


#using units feature set to 'subject'
sns.relplot(x='timepoint',y='signal',hue='region',data=df,kind='line',estimator=None ,units='subject')


# In[ ]:


df['event'].value_counts()


# In[ ]:


#previous plot looks very confusing 
#using same method to plot only when event is 'stim' in an entry
sns.relplot(x='timepoint',y='signal',hue='region',data=df.query('event =="stim"'),kind='line',estimator=None ,units='subject')


# In[ ]:


#creating a set of plots, split using row and column, using event and subject feature respectively 



sns.relplot(x='timepoint',y='signal',hue='subject',row='event',col='region',kind='line',estimator=None,data=df).fig.set_size_inches(10,8)


# In[ ]:


#lineplot
#error style 

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,7.75))
#above lines required for changing the size of plot 


sns.lineplot(x='timepoint',y='signal',hue='subject',data=df,err_style='bars')


# In[ ]:


#using units for multiple plots 

#import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,7.75))
#above lines required for changing the size of plot 


sns.lineplot(x = 'timepoint', y = 'signal', hue = 'event', units = 'subject', estimator = None, lw = 1, data = df.query("region == 'frontal'")
            )


# using dots data set

# In[ ]:


df = sns.load_dataset('dots').query("align == 'dots'")
df.head()


# In[ ]:


#simple line plot 
sns.relplot(x='time',y='firing_rate',kind='line',data = df)


# In[ ]:


#differentiating based on coherence using hue 
sns.relplot(x='time',y='firing_rate',kind='line',data = df,hue='coherence').fig.set_size_inches(10,8)


# In[ ]:


#adding style with choice category in data
#
#arranging the code a little different to include figure size statement 

sns.relplot(x='time',y='firing_rate',kind='line',data = df,hue='coherence'
            ,style='choice').fig.set_size_inches(10.25,6.65)


# In[ ]:


#influencing the line width with size and sizes input

sns.relplot(x='time',y='firing_rate',kind='line',data = df,hue='coherence'
            ,size='coherence',sizes=(3,18)).fig.set_size_inches(10.25,6.65)


# *
# 
# 
# 
# 
# 
# *

# #visualizing categorical data 
# 
# using titanice dataset

# In[ ]:


titanic = sns.load_dataset('titanic')
titanic.head()


# In[ ]:



#visualizing categorical plot using bar plot 
sns.catplot(y='survived',x='sex',kind='bar',hue='class',data=titanic,height=7)


# In[ ]:


#count plot 
sns.catplot(x='deck',kind='count',palette='ch:1.9',data=titanic,height=7)


# In[ ]:


#point plot

sns.catplot(x='sex',y='survived',kind='point',hue='class',data=titanic,height=7)


# In[ ]:


#understanding distribution plot 


# In[ ]:


#univariate distribution
#create random variable x, now to chart its distribution 
x=randn(300)

x[:30]


# In[ ]:


#import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,7.75))
#above lines required for changing the size of plot 

sns.distplot(x)


# In[ ]:


#printing distribution without the histogram 

#import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,7.75))
#above lines required for changing the size of plot 

sns.distplot(x,hist=False)


# In[ ]:


#disable the kernel density estimation line 

#adding bins 
sns.distplot(x,kde=False,bins=14
            )


# In[ ]:


#adding rug 

#import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,7.75))
#above lines required for changing the size of plot 


sns.distplot(x,kde=False,bins=30,rug=True)


# In[ ]:


#using kde plot method

#import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,7.75))
#above lines required for changing the size of plot 

#adding shading 
sns.kdeplot(x,shade=True)


# In[ ]:


#adding bandwidth to kde plot 

#import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,7.75))
#above lines required for changing the size of plot 

sns.kdeplot(x,shade=True,bw=0.1)


# In[ ]:


#adding bandwidth to kde plot 

#import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,7.75))
#above lines required for changing the size of plot 

sns.kdeplot(x,shade=True,bw=15)


# In[ ]:


#lower bandwidth produces less smooth plot 

#import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,7.75))
#above lines required for changing the size of plot 

sns.kdeplot(x,shade=True,bw=0.04)


# In[ ]:


#using cut to affect the plot 

#import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,7.75))
#above lines required for changing the size of plot 

#cut = -3
sns.kdeplot(x,shade=True,bw=0.5,cut=-3)


# In[ ]:


#import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,7.75))
#above lines required for changing the size of plot 

#cut = 3.25
sns.kdeplot(x,shade=True,bw=0.5,cut=3.254)


# **bivariate distribution**
# 
# using two variables

# In[ ]:


#use tips dataset again
df = sns.load_dataset('tips')
df.head()


# In[ ]:


#using joint joint plot 

sns.jointplot(x='tip',y='total_bill',data=df
              ,kind='reg').fig.set_size_inches(8.25,7.65)


# In[ ]:


#hex type joint plot
sns.jointplot(x='tip',y='total_bill',data=df
              ,kind='hex').fig.set_size_inches(8.25,7.65)


# In[ ]:


#kde type joint plot
sns.jointplot(x='tip',y='total_bill',data=df
              ,kind='kde').fig.set_size_inches(8.25,7.65)


# In[ ]:


x=df['tip']
y=df['total_bill']


# In[ ]:


#kde plot with cmap palette 

f, ax = plt.subplots(figsize = (6,6))

#x='tip',y='total_bill'

cmap = sns.cubehelix_palette(as_cmap = True, dark = 0, light = 1, reverse= True)

sns.kdeplot(x, y, cmap = cmap, n_levels=60, shade=True)


# **IMPORTANT, Generating graphics with tabular data**

# In[ ]:


g = sns.jointplot(x, y, kind = 'kde', color = 'm')


# In[ ]:


iris = sns.load_dataset('iris')
iris.head()


# **using iris dataset**

# In[ ]:


#pair plot 
#for multiple bi variable relationships 
sns.pairplot(iris)


# In[ ]:


#changing the style for the plotsw 

g=sns.PairGrid(iris)#set up grid

g.map_diag(sns.kdeplot)#create plot for graphs on the diagonal

g.map_offdiag(sns.kdeplot,n_levels=10)#create plot for off diagonal girds 


# > **visualizing linear regression and relationship**

# In[ ]:


#import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7,6))
#above lines required for changing the size of plot 

#regression plot 
sns.regplot(x = 'total_bill', y = 'tip', data = df)


# In[ ]:


#linear model plot 


sns.lmplot(x = 'total_bill', y= 'tip', data = df,height=6)


# In[ ]:


#creating two fit lines based on hue


g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=df,order=3)


# In[ ]:



g = sns.lmplot(x="total_bill", y="tip", hue="smoker",
               data=df,order=1,col='time',row='sex').fig.set_size_inches(10.25,9.65)


# In[ ]:


sns.lmplot(x = 'size', y = 'tip', data = df, x_jitter = 0.05)


# In[ ]:


sns.lmplot(x = 'size', y = 'tip', data = df, x_estimator = np.mean)


# In[ ]:


data = sns.load_dataset('anscombe')
data.head()


# In[ ]:


#plot a liner relationship between two variables 

sns.lmplot(x = 'x', y = 'y', data = data.query("dataset == 'I'"), 
           ci = None, 
           scatter_kws={'s': 80})


# In[ ]:


#in this dataset there is no linear relationship 
sns.lmplot(x = 'x', y = 'y', data = data.query("dataset == 'II'"), 
           ci = None, 
           scatter_kws={'s': 80},
           order=3#change this to get polinomial relationship plot
          )


# In[ ]:


#the outlier in this data will throw off the fit line 
#dealing with outliers , 

sns.lmplot(x = 'x', y = 'y', data = data.query("dataset == 'III'"), 
           ci = None, 
           scatter_kws={'s': 80},
           order=1
           
          )


# outside dot is clearly an outlier 

# In[ ]:


#making the graph robust 
sns.lmplot(x = 'x', y = 'y', data = data.query("dataset == 'III'"), 
           ci = None, 
           scatter_kws={'s': 80},
           order=1,
           robust=True#deals with outliers
          )


# In[ ]:





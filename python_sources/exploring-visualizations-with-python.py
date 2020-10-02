#!/usr/bin/env python
# coding: utf-8

# # Exploring visualizations with python
# The goal here is to explore some python visualizations libraries. After a brief talk about matplotlib, I'm going to focus in seaborn and show an alternative for ggplot2 lovers (like me!)

# In[ ]:


import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from numpy import median
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# # 0 - Knowing the dataset

# In[ ]:


df=pd.read_csv('../input/iris/Iris.csv')
df.info()


# In[ ]:


#checking for duplicates and missinng values
print(df.duplicated().sum())
print(df.isnull().sum().sum())


# # 1- Brief talk about matplotlib x seaborn
# Seaborn is a python library for statistical visualizations, built on top of matplotlib.
# Despite seaborn uses matplotlib behind the scenes, sometimes it's necessary to use matplotlib directly to do some customizations.
# (Source: https://seaborn.pydata.org/introduction.html) 
# 
# Let's do some plots with matplotlib to get:
# * PetalLengthCm distribution (first histogram)
# * PetalLengthCm distribution by species (second histogram)
# * The relationship between PetalLengthCm and PetalWidthCm (scatterplot)
# * PetalLengthCm and PetalWidthCm distributions and the relationship between them (two-dimensional histogram)

# In[ ]:


plt.style.context('bmh')
fig, axes = plt.subplots(2,2, figsize=(8,8),constrained_layout=True)

axes[0,0].hist(df.PetalLengthCm, color='mediumseagreen',edgecolor='black', alpha=0.8)
axes[0,0].set_title('PetalLengthCm Distribution', fontsize=10, fontweight='bold')
axes[0,0].set_xlabel('PetalLengthCm')

axes[0,1].hist(df[df.Species=='Iris-setosa'].PetalLengthCm, label= 'Iris-setosa',edgecolor='black', color='mediumpurple',alpha=0.8)
axes[0,1].hist(df[df.Species=='Iris-versicolor'].PetalLengthCm, label= 'Iris-versicolor',edgecolor='black', color='pink',alpha=0.8)
axes[0,1].hist(df[df.Species=='Iris-virginica'].PetalLengthCm, label= 'Iris-virginica',edgecolor='black', color='aqua',alpha=0.8)
axes[0,1].set_title('PetalLengthCm Distribution by Species', fontsize=10, fontweight='bold')
axes[0,1].set_xlabel('PetalLengthCm')
axes[0,1].legend(loc="upper right")

axes[1,0].hist2d(df.PetalLengthCm, df.PetalWidthCm, cmap='Greens', bins=20)
axes[1,0].set_title('PetalLengthCm and PetalWidthCm Distributions', fontsize=10, fontweight='bold')
axes[1,0].set_xlabel('PetalLengthCm')
axes[1,0].set_ylabel('PetalWidthCm')

axes[1,1].scatter(df.PetalLengthCm, df.PetalWidthCm, color='mediumseagreen', edgecolor='black')
axes[1,1].set_title('PetalLengthCm and PetalWidthCm Relationship', fontsize=10, fontweight='bold')
axes[1,1].set_xlabel('PetalLengthCm')
axes[1,1].set_ylabel('PetalWidthCm');


# # 2- Seaborn
# Let's dive into the deep

# ## 2.1- Plots: how each of them can help us to know the data?
# On this section I will present some plot types to get the following information:
# 
# * Central tendency measures of PetalLengthCm by Species (barplot and pointplot)
# * PetalLengthCm distribution (histograms and KDE plots)
# * PetalLengthCm distribution by species (histograms, stripplot, swarmplot, boxplots and violinplots)
# * The relationship between PetalLengthCm and PetalWidthCm (scatterplots and regression plots)

# In[ ]:


# setting my own palette
mypalette = ['mediumpurple', 'pink','aqua']
sns.set_palette(mypalette)
sns.palplot(sns.color_palette())


# ### Let's use pointplot and barplot to get an estimative of central tendencies of PetalLength by Species: Mean and Median (with uncertantly indication)

# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(10,5), sharey=True)

ax1=sns.barplot(x='Species', y='PetalLengthCm', data= df, edgecolor='black',ax=axes[0])
ax1.set_title('Mean PetalLength by Species', fontsize=12, fontweight='bold')
ax1=sns.despine()

ax2=sns.pointplot(x='Species', y='PetalLengthCm', estimator=np.median, data= df, ax=axes[1], color='mediumseagreen')
ax2.set_title('Median PetalLength by Species', fontsize=12, fontweight='bold')
ax2=sns.despine()


# ### Now, let's check the distribution of PetalLength. Histogram, KDE and CDF plots are good options for it.

# In[ ]:


fig, axes = plt.subplots(1,3, figsize=(15,5))

ax1=sns.distplot(df.PetalLengthCm, bins=30, hist_kws={'edgecolor':'k'}, color='mediumseagreen',ax=axes[0])
ax1.set_title('Histogram + KDE', fontsize=12)
ax1=sns.despine()

ax2=sns.distplot(df.PetalLengthCm, hist=False, rug= True, color='mediumseagreen',ax=axes[1])
ax2.set_title('KDE', fontsize=12)
ax2=sns.despine()

def cdf (column):
    n=len(column)
    x=np.sort(column)
    y=np.arange(1, n+1)/n
    return x,y

xPetalLength, yPetalLength = cdf(df.PetalLengthCm)
ax3=sns.lineplot(xPetalLength, yPetalLength, color='mediumseagreen',ax=axes[2])
ax3.set_title('Cumulative Distribution Function', fontsize=12)
ax3=sns.despine()

fig.suptitle('PetalLength distribution', fontsize=12, fontweight='bold');


# ### Now, how can we compare the distributions of PetalLengthCm by species? Besides using colors to subset the values, we can use other types of visualizations.

# In[ ]:


fig, axes = plt.subplots(1,3, figsize=(15,5))

ax1=sns.distplot(df[df.Species=='Iris-setosa'].PetalLengthCm, hist_kws={"edgecolor":"k"}, ax=axes[0])
ax1=sns.distplot(df[df.Species=='Iris-versicolor'].PetalLengthCm, hist_kws={"edgecolor":"k"},ax=axes[0])
ax1=sns.distplot(df[df.Species=='Iris-virginica'].PetalLengthCm, hist_kws={"edgecolor":"k"},ax=axes[0])
ax1.set_title('Histogram by Species', fontsize=12)
ax1=sns.despine()

ax2=sns.distplot(df[df.Species=='Iris-setosa'].PetalLengthCm, hist=False, kde_kws={"shade": True},ax=axes[1])
ax2=sns.distplot(df[df.Species=='Iris-versicolor'].PetalLengthCm, hist=False,kde_kws={"shade": True}, ax=axes[1])
ax2=sns.distplot(df[df.Species=='Iris-virginica'].PetalLengthCm, hist=False,kde_kws={"shade": True}, ax=axes[1])
ax2.set_title('KDE by Species', fontsize=12)
ax2=sns.despine()

xsePetalLength, ysePetalLength = cdf(df[df.Species=='Iris-setosa'].PetalLengthCm)
xvePetalLength, yvePetalLength = cdf(df[df.Species=='Iris-versicolor'].PetalLengthCm)
xviPetalLength, yviPetalLength = cdf(df[df.Species=='Iris-virginica'].PetalLengthCm)
ax3=sns.lineplot(xsePetalLength, ysePetalLength, ax=axes[2])
ax3=sns.lineplot(xvePetalLength, yvePetalLength, ax=axes[2])
ax3=sns.lineplot(xviPetalLength, yviPetalLength, ax=axes[2])
ax3.set_title('Cumulative Distribution Function by Species', fontsize=12)
ax3=sns.despine()

fig.suptitle('PetalLength distribution by Species', fontsize=12, fontweight='bold');


# In[ ]:


fig, axes = plt.subplots(1,4, figsize=(20,5),sharey=True)  

ax1= sns.stripplot(x='Species', y='PetalLengthCm', data=df, jitter=True,  ax=axes[0])
ax1.set_title('Stripplot', fontsize=14)
ax1=sns.despine()

ax2= sns.swarmplot(x='Species', y='PetalLengthCm', data=df, ax=axes[1])
ax2.set_title('Swarmplot', fontsize=14)
ax2=sns.despine()

ax3= sns.boxplot(x='Species', y='PetalLengthCm', data=df, ax=axes[2])
ax3.set_title('Boxplot', fontsize=14)
ax3=sns.despine()

ax4= sns.violinplot(x='Species', y='PetalLengthCm', data=df, ax=axes[3])
ax4.set_title('Violinplot', fontsize=14)
ax4=sns.despine()


fig.suptitle('PetalLength distribution by Species', fontsize=12, fontweight='bold');


# ### What about the relationship between variables? Regression plots are useful for it.

# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(10,5))  

ax1=sns.regplot(x='PetalLengthCm', y='PetalWidthCm', data=df, color='mediumseagreen',order=1, ax=axes[0])
ax1.set_title('Regression Plot', fontsize=14)
ax1=sns.despine()

ax2=sns.residplot(x='PetalLengthCm', y='PetalWidthCm', data=df, color='grey',order=1, ax=axes[1])
ax2.set_title('Residual Plot', fontsize=14)
ax2=sns.despine()


# ### With lmplot we can subset by species...

# In[ ]:


sns.lmplot(x='PetalLengthCm', y='PetalWidthCm', data=df, hue='Species');


# ### ... and faceting!

# ## 2.2 - Faceting
# Another way to see the diferents distributions/relationships is faceting!
# Here, besides lmplot(), I'm going to use Facetgrid.  
# (we could use factorplot, it is simpler, but less powerfull).

# In[ ]:


sns.lmplot(x='PetalLengthCm', y='PetalWidthCm', data=df, hue='Species', col='Species');


# In[ ]:


g=sns.FacetGrid(df, col='Species', hue='Species')
g.map(sns.distplot,'PetalLengthCm', hist=False, kde_kws={"shade": True},rug= True);


# ## 2.3 - Joint
# Here, I am going to use jointplot and Jointgrid (more powerfull) to compare distributions and check relationships selecting 2 variables.

# In[ ]:


sns.jointplot(data=df, x='PetalLengthCm', y='PetalWidthCm',color='mediumseagreen');


# In[ ]:


j=sns.JointGrid(data=df, x='PetalLengthCm', y='PetalWidthCm')
j=j.plot_joint(sns.regplot, color='mediumseagreen')
j=j.plot_marginals(sns.kdeplot, color='mediumseagreen', shade=True);


# ## 2.4 - Pair
# Here, I am going to use pairplot and Pairgrid (more powerfull) to compare distributions and check relationships selecting all variables.

# In[ ]:


sns.pairplot(df.drop("Id", axis=1),hue='Species');


# In[ ]:


p=sns.PairGrid(df.drop("Id", axis=1),hue='Species')
p=p.map_upper(sns.regplot)
p=p.map_diag(sns.kdeplot, shade=True)
p=p.map_lower(sns.kdeplot)
p=p.add_legend()


# # 3- Plotnine
# Plotnine is a python library based on ggplot2!
# I'm going to show some examples

# In[ ]:


(ggplot(df)+
aes(x='PetalLengthCm')+
geom_histogram(aes(fill='Species'),bins=30, color='black', size=0.3, alpha=0.7)+
scale_fill_manual(values=('mediumpurple', 'pink','aqua'))+
ggtitle('Distribution PetalLengthCm by Species'))


# In[ ]:


(ggplot(df)+
aes(x='Species', y='PetalLengthCm')+
geom_violin(aes(fill='Species'),size=0.7, color='black',alpha=0.5)+
geom_boxplot(aes(fill='Species'),size=0.7, color='black',alpha=0.5, width=0.1)+
scale_fill_manual(values=('mediumpurple', 'pink','aqua'))+
ggtitle('Distribution PetalLengthCm by Species'))


# In[ ]:


(ggplot(df)+
aes(x='PetalLengthCm')+
stat_ecdf(aes(color='Species'), size=1)+
scale_color_manual(values=('mediumpurple', 'pink','aqua'))+
ggtitle('Cumulative Distribution Functions: PetalLengthCm'))


# In[ ]:


(ggplot(df)+
aes(x='PetalLengthCm', y='PetalWidthCm')+
geom_point(aes(fill='Species', size='SepalWidthCm'), color='black',alpha=0.8)+
scale_color_manual(values=('mediumpurple', 'pink','aqua'))+
scale_fill_manual(values=('mediumpurple', 'pink','aqua'))+
ggtitle('PetalLengthCm x PetalWidthCm'))


# In[ ]:


(ggplot(df)+
aes(x='PetalLengthCm', y='PetalWidthCm')+
geom_point(aes(fill='Species'), color= 'black',alpha=0.8)+
geom_smooth(aes(color='Species'), method='lm')+
scale_color_manual(values=('mediumpurple', 'pink','aqua'))+
scale_fill_manual(values=('mediumpurple', 'pink','aqua'))+
facet_wrap('Species')+
ggtitle('PetalLengthCm x PetalWidthCm'))


# # 4- Plotly
# Interactive visualizations
# 
# Source: https://plot.ly/python/

# In[ ]:


fig=make_subplots(rows=2, cols=2, 
                  vertical_spacing=0.3,
                  horizontal_spacing=0.3,
                  specs=[[{'type':'xy'},{'type':'xy'}],
                        [{'type':'xy'},{'type':'xy'}]],
                  subplot_titles=('Histogram by Species', 'Boxplot by Species',
                                  'CDF  by Species', 'ScatterPlot by Species'))

################################################################################

fig.add_trace(go.Histogram(x=df[df.Species=='Iris-setosa'].PetalLengthCm,
                          marker_color='mediumpurple',
                          name='Setosa'),
                          row=1, col=1)


fig.add_trace(go.Histogram(x=df[df.Species=='Iris-versicolor'].PetalLengthCm,
                          marker_color='pink',
                          name='Versicolor'),
                          row=1, col=1)

fig.add_trace(go.Histogram(x=df[df.Species=='Iris-virginica'].PetalLengthCm,
                          marker_color='aqua',
                          name='Virginica'),
                          row=1, col=1)

################################################################################

fig.add_trace(go.Box(y=df[df.Species=='Iris-setosa'].PetalLengthCm,
                          marker_color='mediumpurple',
                          name='Setosa',
                          showlegend=False),
                          row=1, col=2)

fig.add_trace(go.Box(y=df[df.Species=='Iris-versicolor'].PetalLengthCm,
                          marker_color='pink',
                          name='Versicolor',
                          showlegend=False),
                          row=1, col=2)

fig.add_trace(go.Box(y=df[df.Species=='Iris-virginica'].PetalLengthCm,
                          marker_color='aqua',
                          name='Virginica',
                          showlegend=False),
                          row=1, col=2)

################################################################################

fig.add_trace(go.Scatter(x=xsePetalLength, y=ysePetalLength,
                         marker_color='mediumpurple',
                         name='Setosa',
                         showlegend=False),
                         row=2, col=1)

fig.add_trace(go.Scatter(x=xvePetalLength, y=yvePetalLength,
                         marker_color='pink',
                         name='Versicolor',
                         showlegend=False),
                         row=2, col=1)

fig.add_trace(go.Scatter(x=xviPetalLength, y=yviPetalLength,
                         marker_color='aqua',
                         name='Virginica',
                         showlegend=False),
                         row=2, col=1)

################################################################################

fig.add_trace(go.Scatter(x=df[df.Species=='Iris-setosa'].PetalLengthCm,
                         y=df[df.Species=='Iris-setosa'].PetalWidthCm,
                         mode='markers',
                         marker_color='mediumpurple',
                         name='Setosa',
                         showlegend=False),
                         row=2, col=2)

fig.add_trace(go.Scatter(x=df[df.Species=='Iris-versicolor'].PetalLengthCm,
                         y=df[df.Species=='Iris-versicolor'].PetalWidthCm,
                         mode='markers',
                         marker_color='pink',
                         name='Versicolor',
                         showlegend=False),
                         row=2, col=2)

fig.add_trace(go.Scatter(x=df[df.Species=='Iris-virginica'].PetalLengthCm,
                         y=df[df.Species=='Iris-virginica'].PetalWidthCm,
                         mode='markers',
                         marker_color='aqua',
                         name='Virginica',
                         showlegend=False),
                         row=2, col=2)

################################################################################

fig.update_xaxes(title_text='PetalLenth', row=1, col=1)
fig.update_xaxes(title_text='', row=1, col=2)
fig.update_xaxes(title_text='PetalLenth', row=2, col=1)
fig.update_xaxes(title_text='PetalLenth', row=2, col=2)

fig.update_yaxes(title_text='', row=1, col=1)
fig.update_yaxes(title_text='PetalLenth', row=1, col=2)
fig.update_yaxes(title_text='', row=2, col=1)
fig.update_yaxes(title_text='PetalWidth', row=2, col=2)

################################################################################

fig.update_layout(template='plotly_dark',
                 title='Statistical Plots',
                 width=650,
                 height=600,
                 font=dict(size=12),
                 barmode='overlay')

fig.update_traces(opacity=0.9)
fig.show()


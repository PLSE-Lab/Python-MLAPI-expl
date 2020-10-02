#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import os

import itertools
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


os.listdir("../input")


# In[ ]:


df15 = pd.read_csv("../input/2015.csv")
df16 = pd.read_csv("../input/2016.csv")
df17 = pd.read_csv("../input/2017.csv")
df18 = pd.read_csv("../input/2018.csv")
df19 = pd.read_csv("../input/2019.csv")


# In[ ]:


df15.head()


# In[ ]:


df16.head()


# In[ ]:


df17.head()


# In[ ]:


df18.head()


# In[ ]:


df19.head()


# In[ ]:


def cluster_map(d, title=""):
    sns.set(style="white")
    sns.clustermap(d.corr(), 
                   pivot_kws=None, 
    #                method='average', 
    #                metric='euclidean', 
                   z_score=None, 
                   standard_scale=None,
                   figsize=None,
                   cbar_kws=None, 
                   row_cluster=True, 
                   col_cluster=True, 
                   row_linkage=None, 
                   col_linkage=None,
                   row_colors=None, 
                   col_colors=None, 
                   mask=None,
                   center=0,
                   cmap="vlag",
                   linewidths=.75, 
    #                figsize=(13, 13)
                  )
    
    plt.title(title)
    
def corr_plot(d, title=""):
    # Compute the correlation matrix
    corr = d.corr()
    columns = d.columns
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(len(columns), len(columns)))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap = sns.diverging_palette(h_neg=220, h_pos=10, s=75, l=50, sep=10, n=len(columns), center='light', as_cmap=True)

    sns.set(style="white")
    sns.heatmap(corr,
             vmin=None,
             vmax=None,
             cmap=cmap,
             center=None,
             robust=True,
             annot=True, 
    #          fmt='.2g',
             annot_kws=None, 
    #          linewidths=0.5, 
    #          linecolor='white',
             cbar=True,
             cbar_kws={"shrink": .5},
             cbar_ax=None, 
             square=True, 
             xticklabels='auto',
             yticklabels='auto', 
             mask=mask, 
             ax=None)


    plt.yticks(rotation=0)
    plt.xticks(rotation=90)  
    plt.title(title)


# In[ ]:


cluster_map(df15, title="2015")
cluster_map(df16, title="2016")
cluster_map(df17, title="2017")
cluster_map(df18, title="2018")
cluster_map(df19, title="2019")


# In[ ]:


corr_plot(df15, title="2015")
corr_plot(df16, title="2016")
corr_plot(df17, title="2017")
corr_plot(df18, title="2018")
corr_plot(df19, title="2019")


# In[ ]:


df19.head()


# In[ ]:


features = ['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 
            'Generosity','Perceptions of corruption']


# In[ ]:


df = df18

y = "Score"

plt.rcParams["figure.figsize"] = (18,3)

for x in features:
    sns.lineplot(x=x, y=y, data=df18)
    sns.lineplot(x=x, y=y, data=df19)
    plt.title(x.title() + " vs. " + y.title())
    plt.plot()
    plt.show()


# Following can be found from above line/relational, correlation plot
# - 'GDP per capita', 'Social support', 'Healthy life expectancy' was positively correlated with happiness scores
# - 'Generosity', 'Perceptions of corruption' was not correlated with happines scores

# In[ ]:


df19.columns


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import linear_model


features = ['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 
            'Generosity','Perceptions of corruption']

target = "Score"
# target = "Overall rank"


X = df19[features]
y = df19[target]


clf = linear_model.Ridge()


scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_absolute_error')
scores.mean()


# In[ ]:





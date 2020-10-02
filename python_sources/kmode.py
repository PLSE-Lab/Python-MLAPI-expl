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


import re
import json
import shutil
import string
from termcolor import colored
from tqdm import tqdm
from itertools import chain
from collections import defaultdict
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go 
from plotly import tools
import plotly.figure_factory as ff


init_notebook_mode(connected=True)
import types


# In[ ]:


df_data = pd.read_csv("../input/clustering-categorical-peoples-interests/kaggle_Interests_group.csv")
df_data.head()


# In[ ]:


title_mapping = {"P": 0, "C": 1, "R": 2,  "I": 3 }
df_data['group'] = df_data['group'].map(title_mapping)
df_data.head()


# In[ ]:


df_data = df_data.fillna(0)
df_data.head()


# In[ ]:


for c in df_data.columns:
    print(c,'\n',df_data[c].nunique(),df_data[c].unique()[0:15])


# In[ ]:


def oh_enc(df,cat_cols):
    df.fillna(value="NA")
    X = pd.DataFrame()
    for c in cat_cols:
        X = pd.concat([X, pd.get_dummies(df[c])],axis=1)
    return X


# In[ ]:


X_cats = df_data
X_cats.head()

X = X_cats
X.head()


# In[ ]:


all_cols = ['group',
'grand_tot_interests',
'interest1',
'interest2',
'interest3',
'interest4',
'interest5',
'interest6',
'interest7',
'interest8',
'interest9',
'interest10',
'interest11',
'interest12',
'interest13',
'interest14',
'interest15',
'interest16',
'interest17',
'interest18',
'interest19',
'interest20',
'interest21',
'interest22',
'interest23',
'interest24',
'interest25',
'interest26',
'interest27',
'interest28',
'interest29',
'interest30',
'interest31',
'interest32',
'interest33',
'interest34',
'interest35',
'interest36',
'interest37',
'interest38',
'interest39',
'interest40',
'interest41',
'interest42',
'interest43',
'interest44',
'interest45',
'interest46',
'interest47',
'interest48',
'interest49',
'interest50',
'interest51',
'interest52',
'interest53',
'interest54',
'interest55',
'interest56',
'interest57',
'interest58',
'interest59',
'interest60',
'interest61',
'interest62',
'interest63',
'interest64',
'interest65',
'interest66',
'interest67',
'interest68',
'interest69',
'interest70',
'interest71',
'interest72',
'interest73',
'interest74',
'interest75',
'interest76',
'interest77',
'interest78',
'interest79',
'interest80',
'interest81',
'interest82',
'interest83',
'interest84',
'interest85',
'interest86',
'interest87',
'interest88',
'interest89',
'interest90',
'interest91',
'interest92',
'interest93',
'interest94',
'interest95',
'interest96',
'interest97',
'interest98',
'interest99',
'interest100',
'interest101',
'interest102',
'interest103',
'interest104',
'interest105',
'interest106',
'interest107',
'interest108',
'interest109',
'interest110',
'interest111',
'interest112',
'interest113',
'interest114',
'interest115',
'interest116',
'interest117',
'interest118',
'interest119',
'interest120',
'interest121',
'interest122',
'interest123',
'interest124',
'interest125',
'interest126',
'interest127',
'interest128',
'interest129',
'interest130',
'interest131',
'interest132',
'interest133',
'interest134',
'interest135',
'interest136',
'interest137',
'interest138',
'interest139',
'interest140',
'interest141',
'interest142',
'interest143',
'interest144',
'interest145',
'interest146',
'interest147',
'interest148',
'interest149',
'interest150',
'interest151',
'interest152',
'interest153',
'interest154',
'interest155',
'interest156',
'interest157',
'interest158',
'interest159',
'interest160',
'interest161',
'interest162',
'interest163',
'interest164',
'interest165',
'interest166',
'interest167',
'interest168',
'interest169',
'interest170',
'interest171',
'interest172',
'interest173',
'interest174',
'interest175',
'interest176',
'interest177',
'interest178',
'interest179',
'interest180',
'interest181',
'interest182',
'interest183',
'interest184',
'interest185',
'interest186',
'interest187',
'interest188',
'interest189',
'interest190',
'interest191',
'interest192',
'interest193',
'interest194',
'interest195',
'interest196',
'interest197',
'interest198',
'interest199',
'interest200',
'interest201',
'interest202',
'interest203',
'interest204',
'interest205',
'interest206',
'interest207',
'interest208',
'interest209',
'interest210',
'interest211',
'interest212',
'interest213',
'interest214',
'interest215',
'interest216',
'interest217'
           ]


# In[ ]:


def optimal_cluster_num(data, algorithm):
    """
    data - cleaned dataset
    algorithm - kmeans for numeric data; kmodes for categorical data
    
    """
    K = range(1,10)
    if algorithm == "kmeans":
        wcss = [KMeans(n_clusters=i,random_state=0).fit(data).inertia_ for i in K]
    elif algorithm == "kmodes":
        wcss = [KModes(n_clusters=i, init='Huang', n_init=5, verbose=0).fit(data).cost_ for i in K]
        
        
    trace = go.Scatter(x = list(range(1,10)), y = wcss, mode = 'lines', name = algorithm)  
    layout = go.Layout(autosize=False, width=600, height=300, title= 'Optimal Number of Clusters - ' + algorithm)
    
    iplot({'data': [trace],'layout':layout})


# In[ ]:


# Clustering
def clustering(data, n_clusters, algorithm):
    """
    data - cleaned dataset
    n_clusters - number of clusters
    algorithm - kmeans for numeric data; kmodes for categorical data
    
    """
    
    if algorithm == 'kmeans':
        clf = KMeans(n_clusters=n_clusters, max_iter=100, init='k-means++', n_init=10, random_state=0)
             #clf = KMeans(n_clusters=n_clusters, max_iter=100, init='k-means++', n_init=10, random_state=0)
        labels = clf.fit_predict(data)
        centroids = clf.cluster_centers_
    
    elif algorithm == 'kmodes':
        clf = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0, random_state=0)
        labels = clf.fit_predict(data)
        centroids = clf.cluster_centroids_        
    
    return labels, centroids

# Interpreting
def interpret_clusters(df, labels, cols, method):
    df["labels"] = labels
    v_counts = df["labels"].value_counts()
    
    if method == "quantity":
        for i in set(labels):
            print(colored("\n Cluster {} - {}\n".format(i, v_counts[i]),color="magenta", attrs=["bold"]))
            for c in cols:
                ls = df.loc[df['labels'] == i, c].value_counts()
                l = list(ls.index[0:2])
                v = ls.values[0:2]
                print(colored("{}:".format(c),color="blue"), colored(l,color="red", attrs=["bold"]),"-", v)
    if method == 'percentage':
        for i in set(labels):
            print(colored("\n Cluster {} - {}\n".format(i, v_counts[i]),color="magenta", attrs=["bold"]))
            for c in cols:
                ls = df.loc[df['labels'] == i, c].value_counts()
                all_ls = df[c].value_counts()[ls.index]
                ls = (ls/all_ls)*100
                ls.sort_values(ascending=False,inplace=True)
                l = list(ls.index[0:2])                    
                v = np.round(ls, decimals=0).values[0:2]
                print(colored("{}:".format(c),color="blue"), colored(l,color="red", attrs=["bold"]), "-", v, "%")

# Plotting
def plot_clusters(data, labels, centroids, int_method, title=''):
    # First we need to make 2D coordinates from the sparse matrix.
    customPalette = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]
    
    pca = PCA(n_components=2).fit(data)
    coords = pca.transform(data)
    
    pca_data = pd.DataFrame()
    pca_data['PC1'] = coords[:, 0]
    pca_data['PC2'] = coords[:, 1]
    pca_data['label'] = labels
    pca_data['label'] = pca_data['label'].apply(lambda i: 'C' + str(i))

    # Plot the cluster centers
    centroid_coords = pca.transform(centroids)
    groups = {}
    for i in range(0, centroids.shape[0]):
        groups['C' + str(i)] =  centroid_coords[i]


    annots = []

    fig = tools.make_subplots(rows=1, cols=1,print_grid=False)

    for i, label in enumerate(groups.keys()):
        ## Scatter Plot
        trace1 = go.Scatter(x = pca_data.loc[pca_data['label']==label, 'PC1'],
                            y = pca_data.loc[pca_data['label']==label, 'PC2'],
                            mode = 'markers',
                            name = label, marker=dict(size=12, color = customPalette[i]))

        annot = dict(x = groups[label][0], y = groups[label][1], xref='x1', yref='y1', text=label,showarrow=False,
                     font=dict(family='Courier New, monospace', size=16, color='#ffffff'),
                     bordercolor='#c7c7c7', borderwidth=2, borderpad=4, bgcolor=customPalette[i], opacity=1)

        annots.append(annot)
        fig.append_trace(trace1, 1, 1)

        
    fig.layout.update(xaxis = dict(showgrid=False, title='PC1'), yaxis = dict(showgrid=False, title='PC2'),barmode = 'stack', annotations=annots, title= title + ' segmentation interest')
    iplot(fig)
    interpret_clusters(df_data,labels,all_cols, method=int_method)


# In[ ]:


optimal_cluster_num(X,algorithm='kmodes')


# In[ ]:


labels, centroids = clustering(X,n_clusters=2,algorithm='kmodes')


# In[ ]:


plot_clusters(X,labels=labels,centroids=centroids, int_method="quantity")  #percentage   #quantity


# In[ ]:


@plot_clusters(X,labels=labels,centroids=centroids, int_method="percentage")  #percentage   #quantity


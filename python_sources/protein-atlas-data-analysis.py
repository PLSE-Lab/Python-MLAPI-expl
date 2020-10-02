#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import os
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Load dataset info
# **

# In[ ]:


train = pd.read_csv("../input/train.csv")
train.head()
# train_df.Target.value_counts()


# There are 28 different target proteins
# creating label dict

# In[ ]:


label_names={
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles"   ,
5:  "Nuclear bodies"   ,
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus"   ,
8:  "Peroxisomes"   ,
9:  "Endosomes"   ,
10:  "Lysosomes"   ,
11:  "Intermediate filaments",   
12:  "Actin filaments"   ,
13:  "Focal adhesion sites",   
14:  "Microtubules"   ,
15:  "Microtubule ends",   
16:  "Cytokinetic bridge",   
17:  "Mitotic spindle"   ,
18:  "Microtubule organizing center" ,  
19:  "Centrosome"   ,
20:  "Lipid droplets",   
21:  "Plasma membrane",   
22:  "Cell junctions"  , 
23:  "Mitochondria"   ,
24:  "Aggresome"   ,
25:  "Cytosol",
26:  "Cytoplasmic bodies",   
27:  "Rods & rings" 
}

reverse_train_labels = dict((v,k) for k,v in label_names.items())

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = label_names[int(num)]
        row.loc[name] = 1
    return row


# Count of proteins occur in each images 

# In[ ]:


for key in label_names.keys():
    train[label_names[key]] = 0

train = train.apply(fill_targets, axis=1)
train.head()


# In[ ]:


print("Total number of samples in the training data:", train.shape[0])
print("Total number of unique IDs in the training data: ",len(train.Id.unique()))
train["number_of_targets"] = train.drop(["Id", "Target"],axis=1).sum(axis=1)


# Count of Images for number of labes present in images
# 
# All counts:
# 
# **        label count     and       Images count**
#         
#         1                 15126
#         
#         2                 12485
#         
#         3                  3160
#         
#         4                  299
#         
#         5                  2
#         

# In[ ]:


count_perc = np.round(100 * train["number_of_targets"].value_counts() / train.shape[0], 2)
plt.figure(figsize=(20,5))
sns.barplot(x=count_perc.index.values, y=count_perc.values, palette="Oranges")
plt.xlabel("Number of targets per image")
plt.ylabel("% of data")


# **Distribution of training labels**

# In[ ]:


import gc
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

target_array = list(train.Target)
target_array = [label_names[int(n)] for a in target_array for n in a]
fig, ax = plt.subplots(figsize=(15, 5))
pd.Series(target_array).value_counts().plot('bar', fontsize=14 )


# 1. **Single vs Multi label distribution of train data**

# In[ ]:


# train["nb_labels"] = train["Target"].apply(lambda x: len(x.split(" ")))
single_labels_count = train[train['number_of_targets']==1]['number_of_targets'].count()
multi_labels_count = train[train['number_of_targets']>1]['number_of_targets'].count()

import gc
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

data=[go.Bar(x=['Single label', 'Multi-label'], y=[single_labels_count, multi_labels_count],marker=dict(color='rgb(58,200,225)'))]
layout=dict(height=10, width=10, title='Single vs Multi label distribution')
fig=dict(data=data, layout=layout)
py.iplot(data, filename='Label type vs Count')


# **correlations between training labes**

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
sns.heatmap(train[train.number_of_targets>1].drop(
    ["Id", "Target", "number_of_targets"],axis=1
).corr(), cmap="YlGnBu", vmin=-1, vmax=1)


# In[ ]:


# def heatMap(self, df):
df = train[train.number_of_targets>1].drop( ["Id", "Target", "number_of_targets"],axis=1).corr()

mirror = False
# Create Correlation df
corr = df.corr()
# Plot figsize
fig, ax = plt.subplots(figsize=(20, 15))
# Generate Color Map
colormap = sns.diverging_palette(220, 10, as_cmap=True)

if mirror == True:
   #Generate Heat Map, allow annotations and place floats in map
   sns.heatmap(train[train.number_of_targets>1].drop( ["Id", "Target", "number_of_targets"],axis=1).corr(), cmap=colormap, annot=True, fmt=".2f")
   #Apply xticks
   plt.xticks(range(len(corr.columns)), corr.columns);
   #Apply yticks
   plt.yticks(range(len(corr.columns)), corr.columns)
   #show plot

else:
   # Drop self-correlations
   dropSelf = np.zeros_like(corr)
   dropSelf[np.triu_indices_from(dropSelf)] = True# Generate Color Map
   colormap = sns.diverging_palette(220, 10, as_cmap=True)
   # Generate Heat Map, allow annotations and place floats in map
   sns.heatmap(train[train.number_of_targets>1].drop( ["Id", "Target", "number_of_targets"],axis=1).corr(), cmap='YlGnBu', annot=True, fmt=".2f", mask=dropSelf)
   # Apply xticks
   plt.xticks(range(len(corr.columns)), corr.columns);
   # Apply yticks
   plt.yticks(range(len(corr.columns)), corr.columns)
# show plot
plt.show()
    


# **Matrix for coreation values of labels**

# In[ ]:


# coreation values of protein
corr_matrix = train[train.number_of_targets>1].drop( ["Id", "Target", "number_of_targets"],axis=1).corr()
corr_matrix


# **High corelation between labels**

# In[ ]:


# High corelation between proteins

high_corr_var_=np.where(corr_matrix>0.02)
high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y], corr_matrix[corr_matrix.columns[x]][corr_matrix.columns[y]]) for x,y in zip(*high_corr_var_) if x!=y and x<y]

high_corr_var


# **Low corelation between labels**

# In[ ]:



low_corr_var_=np.where(corr_matrix<=-.1)
low_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y], corr_matrix[corr_matrix.columns[x]][corr_matrix.columns[y]]) for x,y in zip(*low_corr_var_) if x!=y and x<y]

low_corr_var


# In[ ]:


sns.pairplot(corr_matrix, diag_kind="kde", palette="husl")


# In[ ]:


import cv2
from PIL import Image
import imageio
from scipy.misc import imread
train_path = "../input/train/"


# In[ ]:





# In[ ]:





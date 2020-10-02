#!/usr/bin/env python
# coding: utf-8

# <center><h1 style="color:red">Mushroom Exploration (EDA with Visualizations)</h1>
# <h2>BY/ Kassem@elcaiseri<h2></center>

# # Content:
# ## 1. Data handel:
#    * Import data  
#    * Work with data
#    * Check for NAN values
#    * Encode data
#    * Show data descriptions
#    
# ## 2. Visualizations:
#    * Set Visualizations function and params
#    * Plot count value of  columns function
#    * Visualize the number of mushrooms for each cap categorize.
#    * Number of mushrooms based on "odor"
#    * Plot pairwise relationships in a mushrooms for each stalk categorize
#    * Between habitat and population

# ## Data Handel
# * **Import main libraries for Visualizations and data handels**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# * **Walk into input file path **

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# * **Import data**

# In[ ]:


data_path = os.path.join(dirname, filename)
data = pd.read_csv(data_path)
data.head()


# * **Show data size**

# In[ ]:


data.shape


# * **Check for any NAN values**

# In[ ]:


data.isnull().sum()


# * **Show text data descriptions**

# In[ ]:


data.describe()


# * **Encode texted data to can describe it**

# In[ ]:


from sklearn.preprocessing import LabelEncoder

data_encoded = data.copy()
le = LabelEncoder()
for col in data_encoded.columns:
    data_encoded[col] = le.fit_transform(data_encoded[col]) 
    
data_encoded.head()


# In[ ]:


data_encoded.describe()


# * **Show data columns**

# In[ ]:


data.columns


# ## Visualizations
# * **Set Visualizations function and params**

# In[ ]:


import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


# * * **Plot count value of  columns function**

# In[ ]:


def plot_col(col, hue=None, color=['red', 'lightgreen'], labels=None):
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.countplot(col, hue=hue, palette=color, saturation=0.6, data=data, dodge=True, ax=ax)
    ax.set(title = f"Mushroom {col.title()} Quantity", xlabel=f"{col.title()}", ylabel="Quantity")
    if labels!=None:
        ax.set_xticklabels(labels)
    if hue!=None:
        ax.legend(('Poisonous', 'Edible'), loc=0)


# In[ ]:


class_dict = ('Poisonous', 'Edible')
plot_col(col='class', labels=class_dict)


# * **Visualize the number of mushrooms for each cap categorize.**

# In[ ]:


shape_dict = {"bell":"b","conical":"c","convex":"x","flat":"f", "knobbed":"k","sunken":"s"}
labels = ('convex', 'bell', 'sunken', 'flat', 'knobbed', 'conical')
plot_col(col='cap-shape', hue='class', labels=labels)


# In[ ]:


color_dict = {"brown":"n","yellow":"y", "blue":"w", "gray":"g", "red":"e","pink":"p",
              "orange":"b", "purple":"u", "black":"c", "green":"r"}
plot_col(col='cap-color', color=color_dict.keys(), labels=color_dict)


# In[ ]:


plot_col(col='cap-color', hue='class', labels=color_dict)


# In[ ]:


surface_dict = {"smooth":"s", "scaly":"y", "fibrous":"f","grooves":"g"}
plot_col(col='cap-surface', hue='class', labels=surface_dict)


# In[ ]:


def get_labels(order, a_dict):    
    labels = []
    for values in order:
        for key, value in a_dict.items():
            if values == value:
                labels.append(key)
    return labels


# * **Number of mushrooms based on "odor"**

# In[ ]:


odor_dict = {"almond":"a","anise":"l","creosote":"c","fishy":"y",
             "foul":"f","musty":"m","none":"n","pungent":"p","spicy":"s"}
order = ['p', 'a', 'l', 'n', 'f', 'c', 'y', 's', 'm']
labels = get_labels(order, odor_dict)      
plot_col(col='odor', color=color_dict.keys(), labels=labels)


# * **<p>Plot pairwise relationships in a mushrooms for each stalk categorize.</p>**

# In[ ]:


stalk_cats = ['class', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
              'stalk-color-above-ring', 'stalk-color-below-ring']
data_cats = data_encoded[stalk_cats]
sns.pairplot(data_cats, hue='class', kind='reg')


# * **Visualize the distribution in a mushrooms for each stalk categorize.**

# In[ ]:


fig, ax = plt.subplots(3, 2, figsize=(20, 15))
for i, axis in enumerate(ax.flat):
    sns.distplot(data_cats.iloc[:, i], ax=axis)


# * **Between habitat and population**

# In[ ]:


pop_dict = {"abundant":"a","clustered":"c","numerous":"n","scattered":"s","several":"v","solitary":"y"}
hab_dict = {"grasses":"g","leaves":"l","meadows":"m","paths":"p","urban":"u","waste":"w","woods":"d"}


# In[ ]:


f, ax = plt.subplots(figsize=(15, 10))
order = list(data['population'].value_counts().index)
pop_labels = get_labels(order, pop_dict)
explode = (0.0,0.01,0.02,0.03,0.04,0.05)
data['population'].value_counts().plot.pie(explode=explode , autopct='%1.1f%%', labels=pop_labels, shadow=True, ax=ax)
ax.set_title('Mushroom Population Type Percentange');


# In[ ]:


f, ax = plt.subplots(figsize=(15, 10))
order = list(data['habitat'].value_counts().index)
hab_labels = get_labels(order, hab_dict)
explode = (0.0,0.01,0.02,0.03,0.04,0.05, 0.06)
data['habitat'].value_counts().plot.pie(explode=explode, autopct='%1.1f%%', labels=hab_labels, shadow=True, ax=ax)
ax.set_title('Mushroom Habitat Type Percentange');


# In[ ]:





# <h3>Thanks for being here, if you find it useful <span style="color:red">UPVOTE</span>  it, feel free in comment</h3>

# In[ ]:





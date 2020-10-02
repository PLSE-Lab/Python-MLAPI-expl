#!/usr/bin/env python
# coding: utf-8

# ##  Objective
# Objective of this kernel is to 
# - Quickly get an idea of the dataset and the images
# - Tricky coding that helps one to consummate all image info in a data frame.
# - Render some insights well ahead of getting into technical stuff
# - Exploration of the dataset and distribution using intuitive plots

# ## Some Interesting Facts
# * How many unique faces are there?  
# **2316**  
# * How many familys are there?  
# **470**
# * What is the total number of images in the train set?  
# **12379**
# * In train data, how many relationship combinations are present?  
# **3598**
# 
# ## Largest Families
# 
# * Which is the largest family?  
# It's **British Royal Family**
# * Which is the second largest family?  
# It's family of **Jeff Bridges** 
# * Who's got maximum number of images?  
# **Prince William**, Duke of Cambridge
# 
# ## More Info on Largest Family
# * Which family got more number of members?  
# **F0601** with **41** Members
# * Which family got least number of members?  
# **F0275** with **1** Members
# * Which family got maximum number of images?  
# **F0601, 776 Images**
# * Which family maximum number of members?  
# **F0601, 41 Members**
# * Which member got maximum number of images?  
# **F0601/MID6**, Image Count: **95**
# * Is there an anomaly in the data?  
# **No** but the largest family data might bias the learning
# 

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


# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
pd.options.display.float_format = '{:20, .2f}'.format


# In[ ]:


train_df = pd.read_csv("../input/train_relationships.csv")
train_df.head()


# In[ ]:


train_df.shape


# ## Create Image Identification DF

# In[ ]:


files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("../input/train")) for f in fn]
train_images_df = pd.DataFrame({
    'files': files,
    'familyId': [file.split('/')[3] for file in files],
    'kinId': [file.split('/')[4] for file in files],
    'uniqueId': [file.split('/')[3] + '/' + file.split('/')[4] for file in files]
})
train_images_df.head()


# ## Count: Family and Kin

# In[ ]:


print("Total number of members in the dataset: {0}".format(train_images_df["uniqueId"].nunique()))
print("Total number of families in the dataset: {0}".format(train_images_df["familyId"].nunique()))


# In[ ]:


family_with_most_pic = train_images_df["familyId"].value_counts()
kin_with_most_pic = train_images_df["uniqueId"].value_counts()
print("Family with maximum number of images: {0}, Image Count: {1}".format(family_with_most_pic.index[0], family_with_most_pic[0]))
print("Member with maximum number of images: {0}, Image Count: {1}".format(kin_with_most_pic.index[0], kin_with_most_pic[0]))


# About 30% of the pics fall under 1 family and this might result bias. Treating this bias is key for prediction accuracy.

# In[ ]:


family_series = family_with_most_pic[:25]
labels = (np.array(family_series.index))
sizes = (np.array((family_series / family_with_most_pic.sum()) * 100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title='Pic Count by Families')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Families')


# In[ ]:


most_pic_members = train_images_df[train_images_df["uniqueId"] == kin_with_most_pic.index[0]].files.values
fig, ax = plt.subplots(4, 6, figsize=(50, 40))
row = 0
col = 0
for index in range(len(most_pic_members[:24])):
    with open(most_pic_members[index], 'rb') as f:
        img = Image.open(f)
        ax[row][col].imshow(img)

        if(col < 5):
            col = col + 1
        else: 
            col = 0
            row = row + 1
fig.show()


# In[ ]:


family_with_most_members = train_images_df.groupby("familyId")["kinId"].nunique().sort_values(ascending=False)
print("Family with maximum number of members: {0}, Member Count: {1}".format(family_with_most_members.index[0], family_with_most_members[0]))
print("Family with least number of members: {0}, Member Count: {1}".format(
    family_with_most_members.index[len(family_with_most_members)-1], 
    family_with_most_members[len(family_with_most_members)-1]))


# In[ ]:





# ## British Royal Family
# Lets look at the family with max members

# In[ ]:


large_family_df = train_images_df[train_images_df["familyId"]  == family_with_most_members.index[0]]
large_family_df.head()


# ## Pic of Every Member from British Royalty
# This snippet of code picks the first picture of every member of the family.

# In[ ]:





def render_bar_chart(data_df, column_name, title, filename):
    series = data_df[column_name].value_counts()
    count = series.shape[0]
    
    trace = go.Bar(x = series.index, y=series.values, marker=dict(
        color=series.values,
        showscale=True
    ))
    layout = go.Layout(title=title)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename=filename)
    
    
render_bar_chart(large_family_df, 'uniqueId', 'Pic Count by Members', 'members')


# In[ ]:


def render_images(large_family_df):
    large_family_pics = [large_family_df.loc[large_family_df.loc[large_family_df["uniqueId"] == aKin].index[0]]["files"] for aKin in large_family_df["uniqueId"].unique()]
    nrows = round(len(large_family_pics) / 6) + 1


    fig, ax = plt.subplots(nrows, 6, figsize=(50, 40))
    row = 0
    col = 0
    for index in range(len(large_family_pics)):
        with open(large_family_pics[index], 'rb') as f:
            img = Image.open(f)
            ax[row][col].imshow(img)

            if(col < 5):
                col = col + 1
            else: 
                col = 0
                row = row + 1
    fig.show()
render_images(large_family_df)


# ## Jeff Bridges and his Family
# Family with second largest collection of pictures.

# In[ ]:


large_family_df = train_images_df[train_images_df["familyId"]  == family_with_most_members.index[1]]
render_images(large_family_df)
render_bar_chart(large_family_df, 'uniqueId', 'Pic Count by Members', 'members')


# ## Third family with largest number of members

# In[ ]:


large_family_df = train_images_df[train_images_df["familyId"]  == family_with_most_members.index[2]]
render_images(large_family_df)
render_bar_chart(large_family_df, 'uniqueId', 'Pic Count by Members', 'members')


# ## Train Set

# In[ ]:


train_df = pd.read_csv("../input/train_relationships.csv")
train_df.head()


# ## Data Augmentation

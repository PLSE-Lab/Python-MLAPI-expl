#!/usr/bin/env python
# coding: utf-8

# We use plotly to perform some basic EDA on the classes 
# 
# Then we check some sample images from the top 3 classes and start to wonder what are the media contributed by ETH (Zurich) ! 
# 
# 
# Inspiration:
# 
# https://www.kaggle.com/seriousran/google-landmark-retrieval-2020-eda/notebook
# 
# https://www.kaggle.com/sudeepshouche/identify-landmark-name-from-landmark-id
# 

# In[ ]:


import os
import glob
import cv2
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

from scipy import stats

# Load plotly related packages
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[ ]:


# Read train csv and human-radable categories
train_df = pd.read_csv('../input/landmark-retrieval-2020/train.csv')
train_df.head()


# In[ ]:


# Download csn and get classes
url = 'https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv'
classes = pd.read_csv(url, index_col = 'landmark_id', encoding='latin', engine='python') #['category'].to_dict()
classes['classes'] = classes['category'].apply(lambda x : x.replace('http://commons.wikimedia.org/wiki/Category:', ''))
classes.head()


# In[ ]:


class_cnt = pd.DataFrame(train_df['landmark_id'].value_counts(False))
class_cnt.rename(columns={'landmark_id':'count'}, inplace=True)
class_cnt.head()


# In[ ]:


classNameCnt = class_cnt.merge(classes, left_index=True, right_index=True)
classNameCnt.head(10)


# In[ ]:


# Select top 20 classes

top20 = classNameCnt.head(20)

trace = go.Bar(
    x=top20['classes'],
    y=top20['count'],
    marker=dict(color = 'rgba(255, 17, 25, 0.8)')
)

data = [trace]
layout = go.Layout(title='Top 20 class names', 
                   yaxis = dict(title = '# of images in train set')
                  )

fig = go.Figure(data=data, layout=layout)
fig['layout']['xaxis'].update(dict(title = 'Classes', 
                                   tickfont = dict(size = 12)))
fig = go.Figure(data = data, layout = layout)
iplot(fig)

#write image to file
#fig.write_image('top20classes.jpeg')


# In[ ]:


train_list = glob.glob('../input/landmark-retrieval-2020/train/*/*/*/*')
test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')
index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')

print("Train images: ", len(train_list) )
print("Test images: ", len(test_list))
print("Index images: ", len(index_list))


# Lets check some sample images from the different classes. We will check for the top 3 classes, viz:
# 
# 1. Media_contributed_by_the_ETH, Bibliothek
# 2. Corktown, Toronto
# 3. Noraduz Cemetery

# In[ ]:


# Function to display 12 images 

def display_sample(sample_df):
    plt.rcParams["axes.grid"] = False
    f, axarr = plt.subplots(4, 3, figsize=(24, 22))

    curr_row = 0
    for i in range(12):
        imageFile = sample.iloc[i]['id']
        path = "../input/landmark-retrieval-2020/train/"+imageFile[0]+"/"+imageFile[1]+"/"+imageFile[2]+"/"+imageFile+'.jpg'
        example = cv2.imread(path)
        example = example[:,:,::-1]

        col = i%4
        axarr[col, curr_row].imshow(example)
        #cv2.imwrite(imageFile + '.jpg', example)
        if col == 3:
            curr_row += 1


# In[ ]:


# Lets check some random images from the largest class

sample = train_df[train_df['landmark_id'] == 138982].sample(12) #.reset_index(inplace=True)
display_sample(sample)


# In[ ]:


# Lets check some random images from the largest class (Corktown Toronto)

sample = train_df[train_df['landmark_id'] == 126637].sample(12) #.reset_index(inplace=True)
display_sample(sample)


# In[ ]:


# Lets check some random images from the largest class (Noraduz Cemetry)

sample = train_df[train_df['landmark_id'] == 20409].sample(12) #.reset_index(inplace=True)
display_sample(sample)


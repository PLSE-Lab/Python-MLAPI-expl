#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.image as mpimg
import zipfile
import cv2
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


df_bounding_boxes = pd.read_csv('../input/train_bounding_boxes.csv', nrows=100, index_col=0)


# In[ ]:


df_bounding_boxes.head(16)


# In[ ]:


for img, bounding_boxes in df_bounding_boxes.groupby('ImageID'):
    fig,ax = plt.subplots(1)

    ax.set_title(img)
    
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,1.1])
    
    # Display the image
    #ax.imshow(im)

    def add_rect (row):
        # Create a Rectangle patch
        w = row.XMax - row.XMin
        h = row.YMax - row.YMin
        rect = patches.Rectangle((row.XMin, row.YMin), w, h, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)        
    
    for _, data in bounding_boxes.iterrows():
        add_rect(data)

plt.show()


# In[ ]:


df_human_labels = pd.read_csv('../input/train_human_labels.csv', nrows=200, index_col=0)


# In[ ]:


df_human_labels.head(140)


# In[ ]:


df_machine_labels = pd.read_csv('../input/train_machine_labels.csv', nrows=100, index_col=0)


# In[ ]:


df_machine_labels.head(16)


# In[ ]:


df_human_labels.groupby('ImageID')['LabelName'].value_counts()


# In[ ]:


df_tuning_labels = pd.read_csv('../input/tuning_labels.csv', nrows=100, index_col=0, header=None, names=['image_id', 'labels'])


# In[ ]:


df_tuning_labels.head(20)


# In[ ]:


df_classes_descriptions = pd.read_csv('../input/class-descriptions.csv', index_col=0)


# In[ ]:


df_classes_descriptions.shape


# In[ ]:


df_classes_descriptions.nunique()


# In[ ]:


df_classes_trainable = pd.read_csv('../input/classes-trainable.csv', index_col=0)


# In[ ]:


df_classes_trainable.shape


# In[ ]:


df_classes_descriptions.loc[df_classes_trainable.index, :].head()


# In[ ]:


df_classes_descriptions.loc[df_classes_trainable.index, :].nunique()


# In[ ]:


# display an image from the stage_1_test collection
def display_image(image_id):
    #archive = zipfile.ZipFile('../input/stage_1_test_images.zip', 'r')
    #imgdata = archive.read('{}.jpg'.format(image_id))

    with open('../input/stage_1_test_images/{}.jpg'.format(image_id), 'rb') as f:
        imgdata = f.read()
    
    imgdata = np.asarray(
            bytearray(imgdata)
            , dtype=np.uint8)

    img = cv2.imdecode(imgdata, 1)
    #img = cv2.imread(imgdata, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
display_image('395076764176457a63324d3d')


# In[ ]:


df_tuning_labels = pd.read_csv('../input/tuning_labels.csv', index_col=0, header=None, names=['image_id', 'labels'])


# In[ ]:


df_tuning_labels.head(3)


# In[ ]:


for row, data in df_tuning_labels.head(3).iterrows():
    display_image(row)
    print(df_classes_descriptions.loc[data['labels'].split(), :])


# In[ ]:





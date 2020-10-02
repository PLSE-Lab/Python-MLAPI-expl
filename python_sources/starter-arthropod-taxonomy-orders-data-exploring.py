#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import json, os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # ArTaxOr Data Exploration
# The Arthropod Taxonomy Orders dataset is a collection of highres images annotated with labels from the taxanomy rank [order](https://basicbiology.net/biology-101/taxonomy). Annotations have been made with [VoTT](https://github.com/microsoft/VoTT). VoTT stores all metadata in json files. In this kernel we will import all the metadata into DataFrames, do some statistical exploration, visualize images and their objects and finally export metadata to a variety of formats for object detection model training.  
# The dataset is distributed under CC BY-NC-SA 4.0

# In[ ]:


# Check the revision log
with open('/kaggle/input/ArTaxOr/revision history.txt', 'r') as f:
    print(f.read())


# ## Metadata import
# There are multiple image directories, each with a VoTT project. The project file contains a list of image files. Each image is associated with a separate json file that contains all the object boundary boxes. 

# In[ ]:


import glob

pfiles=glob.glob('/kaggle/input/ArTaxOr/**/*.vott', recursive=True)
df=pd.DataFrame()
for f in pfiles:
    with open(f) as file:
        pdata=json.load(file)
        df=df.append(pd.DataFrame(list(pdata['assets'].values())), ignore_index=True)
df['path']=df['path'].str.replace('file:F:/','')
df.head()


# Extract the labels for later use:

# In[ ]:


tags=pd.DataFrame(list(pdata['tags']))
pattern=r'[A-Z]'
labels=tags[tags.name.str.match(pattern)]
labels


# ## Image resolution
# Plot the distribution of image size - there is a peak around 3Mpix.

# In[ ]:


import seaborn as sns

ps=np.zeros(len(df))
for i in range(len(df)):
    ps[i]=df['size'][i]['width'] * df['size'][i]['height']/1e6
sns.distplot(ps, bins=21,kde=False).set_title('Image resolution in Mpix (total {})'.format(len(df)));


# ## Object data import
# We will now import all the object data from the json files into a dataframe. In the process, we convert object positions to relative values. This step might take some time.

# In[ ]:


get_ipython().run_cell_magic('time', '', "anno=pd.DataFrame(columns=['label', 'label_idx', 'xres', 'yres', 'height', 'width', 'left', 'top', \n                           'right', 'bottom', 'area', 'xcenter', 'ycenter', 'blurred',\n                           'occluded', 'truncated', 'file', 'id'])\nfor i in range(len(df)):\n    p=df['path'][i].split('/')\n    p='/'.join(p[:2])\n    afile='/kaggle/input/'+p+'/annotations/'+df['id'][i]+'-asset.json'\n    if os.path.isfile(afile):\n        with open(afile) as file:\n            adata=json.load(file)\n        xres,yres=adata['asset']['size']['width'],adata['asset']['size']['height'] \n        for j in range(len(adata['regions'])):\n            h=adata['regions'][j]['boundingBox']['height']/yres\n            w=adata['regions'][j]['boundingBox']['width']/xres\n            tags=adata['regions'][j]['tags']\n            anno=anno.append({'label': tags[0],\n                              'label_idx': labels[labels.name==tags[0]].index[0],\n                              'xres': xres,\n                              'yres': yres,\n                              'height': h,\n                              'width': w,                              \n                              'left': adata['regions'][j]['boundingBox']['left']/xres,\n                              'top': adata['regions'][j]['boundingBox']['top']/yres,\n                              'right': adata['regions'][j]['boundingBox']['left']/xres+w,\n                              'bottom': adata['regions'][j]['boundingBox']['top']/yres+h, \n                              'area': h*w,\n                              'xcenter': adata['regions'][j]['boundingBox']['left']/xres+0.5*w,\n                              'ycenter': adata['regions'][j]['boundingBox']['top']/yres+0.5*h,\n                              'blurred': int(any(ele == '_blurred' for ele in tags)),\n                              'occluded': int(any(ele == '_occluded' for ele in tags)),\n                              'truncated': int(any(ele == '_truncated' for ele in tags)),\n                              'file': adata['asset']['path'].replace('file:F:/',''),\n                              'id': adata['asset']['id'],}, ignore_index=True)")


# In[ ]:


anno.sample(5)


# OK, let's take a look at how the object size (relative image size) distribution compares between labels:

# In[ ]:


sns.relplot(x="width", y="height", hue="label", col="label", data=anno);


# The Seaborn jointplot gives more details:

# In[ ]:


sns.jointplot(x="width", y="height", data=anno.loc[anno['label'] == 'Lepidoptera']);


# How are the centers of each object distributed? Spiders (Araneae) are most centered in the image while ants, bees and wasps (Hymenoptera) are most spread.

# In[ ]:


sns.relplot(x="xcenter", y="ycenter", hue="label", col="label", data=anno);


# The violin plot gives another view on object size distribution:

# In[ ]:


sns.set(rc={'figure.figsize':(12,6)})
sns.violinplot(x=anno['label'],y=anno['area']);


# Next, how many objects are there for each label? There should be at least 2000, and reasonably balanced (less than 2:1 rato between highest and lowest count).

# In[ ]:


graph=sns.countplot(data=anno, x='label')
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# Let's look at how many objects are blurred, occluded or truncated.

# In[ ]:


df2=anno[['label', 'blurred']]
df2=df2.loc[df2['blurred'] == 1]
sns.set(rc={'figure.figsize':(10,6)})
sns.countplot(x='blurred', hue='label', data=df2);


# In[ ]:


df2=anno[['label', 'occluded']]
df2=df2.loc[df2['occluded'] == 1]
sns.countplot(x='occluded', hue='label', data=df2);


# In[ ]:


df2=anno[['label', 'truncated']]
df2=df2.loc[df2['truncated'] == 1]
sns.countplot(x='truncated', hue='label', data=df2);


# ## Image exploration
# To understand the dataset, we will plot some images with object bounding boxes on top.

# In[ ]:


def attribution(fname):
    img = Image.open(fname)
    exif_data = img._getexif()
    img.close()
    if len(exif_data[315]) > 0:
        s='Photo: '+exif_data[315]
    else:
        s=exif_data[37510][8:].decode('ascii')
    return s

def plot_img(axes, idf, highlight=True):
    f='/kaggle/input/'+idf.iloc[0].file
    im = Image.open(f)
    im.thumbnail((300,300),Image.ANTIALIAS)
    draw = ImageDraw.Draw(im)
    xres, yres = im.size[0], im.size[1]
    for i in range(len(idf)):
        if highlight==True:
            color=(255, 0, 0) if i == 0 else (128, 128, 128)          
        else:
            color=labels[labels.name == idf.iloc[i].label].color.iloc[0]
        draw.rectangle([int(idf.iloc[i]['left']*xres),
                        int(idf.iloc[i]['top']*yres),
                        int(idf.iloc[i]['right']*xres),
                        int(idf.iloc[i]['bottom']*yres)], outline=color, width=2)
    plt.setp(axes, xticks=[], yticks=[])
    axes.set_title(idf.iloc[0].label+'\n'+attribution(f))
    plt.imshow(im)


# For each label, view the images that have the largest objects (relative image size).

# In[ ]:


from PIL import Image, ImageDraw

fig = plt.figure(figsize=(16,26))
for i in range(len(labels)):
    ldf=anno[anno.label == labels.name[i]].nlargest(3, 'area')
    for j in range (3):
        axes = fig.add_subplot(len(labels), 3, 1+i*3+j)
        plot_img(axes, anno[anno.id == ldf.iloc[j].id].sort_values(by=['area'], ascending=False), highlight=True)


# For each label, view the images that have the most objects:

# In[ ]:


fig = plt.figure(figsize=(16,26))

for i in range(len(labels)): 
    a=anno[anno.label == labels.name[i]]['id'].value_counts()
    for j in range (3):
        ldf=anno[anno.id == a.index[j]]
        axes = fig.add_subplot(len(labels), 3, 1+i*3+j)
        plot_img(axes, anno[anno.id == ldf.iloc[j].id], highlight=False)


# Clearly, the object detection model must handle both very large and very small objects!  
# Finally, view some random images (re-run cell for a new selection).

# In[ ]:


fig = plt.figure(figsize=(20,18))
for i in range (3):
    ldf=anno.sample(n=3)
    for j in range(3):
        axes = fig.add_subplot(3, 3, 1+i*3+j)
        plot_img(axes, anno[anno.id == ldf.iloc[j].id], highlight=False)


# # Metadata export
# Finally, we can export metadata to various formats for object detection model training. Let's start with a .csv file.

# In[ ]:


header = ['file', 'label', 'height', 'width', 'left', 'top', 'right', 'bottom'] # change as required
anno.to_csv('./ArTaxOr.csv', index=False, columns = header) 


# ## Pascal VOC
# Pascal VOC files are xml format, and there is one xml file per image file, with same name.

# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install pascal_voc_writer')
from pascal_voc_writer import Writer

if not os.path.exists('voc'):
    os.mkdir('voc')

#for i in range(len(df)):
for i in range(10): # use above line for full dataset
    ldf=anno[anno.id == df.id[i]].reset_index()
    p=df.path[i].split('/') 
    width, height = ldf.xres[0], ldf.yres[0]
    writer = Writer(df.path[i], width, height)
    for j in range(len(ldf)):
        writer.addObject(ldf.label[j], 
                         int(ldf.left[j]*width), 
                         int(ldf.top[j]*height), 
                         int(ldf.right[j]*width),
                         int(ldf.bottom[j]*height))
    writer.save('./voc/'+p[2].replace('.jpg','.xml'))
print(os.listdir("./voc"))


# ## Darknet YOLOv3
# Darknet expects one annotation file per image file. Each object is described by:  
# `class x_center y_center width height`  

# In[ ]:


if not os.path.exists('labels'):
    os.mkdir('labels')

#for i in range(len(df)):
for i in range(10): # use above line for full dataset
    ldf=anno[anno.id == df.id[i]].reset_index()
    p=df.path[i].split('/') 
    file=open('./labels/'+p[2].replace('.jpg','.txt'),'w')
    for j in range(len(ldf)):
        l=labels[labels.name == ldf.label[j]].index.to_list()
        file.write('{} {} {} {} {}\n'.format(l[0], ldf.xcenter[j], ldf.ycenter[j], ldf.width[j], ldf.height[j]))
    file.close()
print(os.listdir("./labels"))


# ## TensorFlow TFRecords
# TFRecords are created in a [separate notebook](https://www.kaggle.com/mistag/tensorflow-tfrecords-demystified).

# ## Pickle
# Finally, store labels, file list and object bounding boxes in pickle files for later use.

# In[ ]:


labels.to_pickle('./ArTaxOr_labels.pkl')
df.to_pickle('./ArTaxOr_filelist.pkl')
anno.to_pickle('./ArTaxOr_objects.pkl')


# That's it! 

#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This notebook shows some characteristics of the images and labels used for the two contests. I've taken a small subset of training images and files from the Open Images dataset and put them here: Excerpt from OpenImages 2020 Train.
# 
# ### Some specific objectives:
# 
# * Get a feel for the the images and the objects/segments they contain.
# * Implement some basic object detection.
# * Look at label counts image sizes, and object relationships.
# 
# On the technical side, there are some things you might find useful:
# 
# * Modeling with large datasets in Kaggle notebooks
# * Making interactive plots with hvplot
# * Visualizing graph networks with networkX

# In[ ]:


get_ipython().system(' conda install -y hvplot')


# In[ ]:


import os
import glob
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hvplot.pandas


# # Images and annotations
# Annotations and classes are different for the two competitions. The detection dataset has more classes of objects and more objects per image most of the time. In other words, images common to both challenges will have more boxes than masks.
# 
# Here are a few images containing boxes and segment masks along with labels. I limited it to 6 for display purposes. It's easy enough to browse through hundreds and get a more complete idea.

# In[ ]:


data_dir = Path('../input/excerpt-from-openimages-2020-train')
im_list = sorted(data_dir.glob('train_00_part/*.jpg'))
mask_list = sorted(data_dir.glob('train-masks-f/*.png'))
boxes_df = pd.read_csv(data_dir/'oidv6-train-annotations-bbox.csv')

names_ = ['LabelName', 'Label']
labels =  pd.read_csv(data_dir/'class-descriptions-boxable.csv', names=names_)

im_ids = [im.stem for im in im_list]
cols = ['ImageID', 'LabelName', 'XMin', 'YMin', 'XMax', 'YMax']
boxes_df = boxes_df.loc[boxes_df.ImageID.isin(im_ids), cols]                    .merge(labels, how='left', on='LabelName')
boxes_df


# In[ ]:


# Annotate and plot
cols, rows  = 3, 2
plt.figure(figsize=(20,30))


for i,im_file in enumerate(im_list[9:15], start=1):
    df = boxes_df.query('ImageID == @im_file.stem').copy()
    img = cv2.imread(str(im_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Add boxes
    h0, w0 = img.shape[:2]
    coords = ['XMin', 'YMin', 'XMax', 'YMax']
    df[coords] = (df[coords].to_numpy() * np.tile([w0, h0], 2)).astype(int)

    for tup in df.itertuples():
        cv2.rectangle(img, (tup.XMin, tup.YMin), (tup.XMax, tup.YMax),
                      color=(0,255,0), thickness=2)
        cv2.putText(img, tup.Label, (tup.XMin+2, tup.YMax-2),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, color=(0,255,0), thickness=2)
    
    # Add segmentation masks
    mask_files = [m for m in mask_list if im_file.stem in m.stem]    
    mask_master = np.zeros_like(img)
    np.random.seed(10)
    for m in mask_files:
        mask = cv2.imread(str(m))
        mask = cv2.resize(mask, (w0,h0), interpolation = cv2.INTER_AREA)
        color = np.random.choice([0,255], size=3)
        mask[np.where((mask==[255, 255, 255]).all(axis=2))] = color
        mask_master = cv2.add(mask_master, mask)
    img = cv2.addWeighted(img,1, mask_master,0.5, 0)    
    
    plt.subplot(cols, rows, i)    
    plt.axis('off')
    plt.imshow(img)

plt.show()


# # Object detection demo
# Below is a simple demo that detects objects. I'm using YOLOv3 as implemented in opencv (yes, opencv has a function for object detection). Weights and the network config come from @pjreddie's darknet repo at https://github.com/pjreddie/darknet.
# 
# If you are space constrained, you might try loading images from their URLs. You can download and resize them in a loop before saving to a hard drive. Alternately, you could pull images in batches and feed them to the dataloader of your model. It's probably super-slow, but you could do it all in RAM (I think - never tried it).

# In[ ]:


urls = pd.read_csv(data_dir/"image_ids_and_rotation.csv", 
                   usecols=['ImageID', 'OriginalURL'])


# In[ ]:


classes = np.loadtxt(data_dir/"openimages.names", dtype=np.str, delimiter="\n")
net = cv2.dnn.readNet(str(data_dir/"yolov3-openimages.weights"), str(data_dir/"yolov3-openimages.cfg"))

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Here again is the indoor scene from above. This time the boxes are produced from object detection and not from the box file.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom skimage import io\n\nim_url = urls.loc[urls.ImageID==im_list[11].stem, 'OriginalURL'].squeeze()\nimg = io.imread(im_url)\n\nheight,width,channels = img.shape\n\n# Make a blob array and run it through the network\nblob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)\nnet.setInput(blob)\nouts = net.forward(outputlayers)\n\n# Get confidence scores and objects\nclass_ids=[]\nconfidences=[]\nboxes=[]\nfor out in outs:\n    for detection in out:\n        scores = detection[5:]\n        class_id = np.argmax(scores)\n        confidence = scores[class_id]\n        if confidence > 0.2:   # threshold\n            print(confidence)\n            center_x= int(detection[0]*width)\n            center_y= int(detection[1]*height)\n            w = int(detection[2]*width)\n            h = int(detection[3]*height)\n            x=int(center_x - w/2)\n            y=int(center_y - h/2)\n            boxes.append([x,y,w,h]) #put all rectangle areas\n            confidences.append(float(confidence)) #how confidence was that object detected and show that percentage\n            class_ids.append(class_id) #name of the object tha was detected\n            \n# Non-max suppression\nindexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)\nprint(indexes, boxes, class_ids)")


# In[ ]:


font = cv2.FONT_HERSHEY_DUPLEX
for i in range(len(boxes)):
#     if i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)
        cv2.putText(img, label, (x,y+30), font, 2, (255,255,0), 2)
        
plt.clf()
plt.figure(figsize=(10,15))
plt.imshow(img)


# Pretty cool overall. You can see there are fewer objects detected by this network than appear in the ground truth. YoloV3 has a hard time detecting and delivering objects close together with overlapping boxes. It's a great option for say, flying a drone around and spotting cars, whereas R-CNNs will probably work better for this case.

# # Label counts per image
# 
# Now for some EDA. The distributions are interesting. Some of the images carry dozens or even hundreds of annotations. There's a long tail out past 700 although most images are more like 50 or fewer. Unique annotations are far fewer with many of the highly-labeled pictures being something like a skyscraper wth 102 windows (real example).

# In[ ]:


annotations = boxes_df.groupby('ImageID').agg(
                        box_count=('LabelName', 'size'),
                        box_unique=('LabelName', 'nunique')
                        )

pd.options.display.float_format = '{:,.1f}'.format
annotations.describe()


# In[ ]:


all = annotations.hvplot.hist('box_count', width=600, bins=30)
unique = annotations.hvplot.hist('box_unique', width=600)
(all + unique).cols(1)


# Here's another look at the number of boxes per image with the largest 1% removed.

# In[ ]:


onepct = annotations.box_count.quantile(0.99)
annotations.query('box_count < @onepct').box_count.value_counts(normalize=True)     .sort_index().hvplot.bar(xticks=list(range(0,60,10)), width=600,
                            line_alpha=0, xlabel='objects per image',
                            ylabel='fraction of images')


# Here's the skyscraper.

# In[ ]:


print(boxes_df.loc[boxes_df.ImageID=="fe7c6f7d298893da"]          .groupby(['ImageID', 'Label'])['LabelName'].size()
     )

im_file = "../input/excerpt-from-openimages-2020-train/train_00_part/fe7c6f7d298893da.jpg"
im = cv2.imread(im_file)
plt.imshow(im)


# # Image size
# The data set is huge! Knowing image sizes can give an idea of the impact of size reduction. Here is how the test image sizes are distributed.

# In[ ]:


from PIL import Image
from dask import bag, diagnostics


def faster_get_dims(file):
    dims = Image.open(file).size
    return dims

dfile_list = glob.glob('../input/open-images-object-detection-rvc-2020/test/*.jpg')
print(f"Getting dimensions for {len(dfile_list)} files.")

# parallelize
dfile_bag = bag.from_sequence(dfile_list).map(faster_get_dims)
with diagnostics.ProgressBar():
    dims_list = dfile_bag.compute()


# In[ ]:


sizes = pd.DataFrame(dims_list, columns=['width', 'height'])
counts = sizes.groupby(['width', 'height']).agg(count=('width', 'size'))               .reset_index()


# In[ ]:


plot_opts = dict(xlim=(0,1200), 
                 ylim=(0,1200), 
                 grid=True, 
                 xticks=[250, 682, 768, 1024], 
                 yticks=[250, 682, 768, 1024], 
                 height=500, 
                 width=550
                 )

style_opts = dict(scaling_factor=0.2,
                  line_alpha=1,
                  fill_alpha=0.1
                  )

counts.hvplot.scatter(x='width', y='height', size='count', **plot_opts)              .options(**style_opts)


# # Distribution of object labels
# 
# Here's a chart showing the frequency of the various types of objects. This is for detection, and for the train set, which will be different for instance segmentation and maybe for the test set. Overall though the data will mostly be pictures of "people with faces, wearing clothes, and standing near trees":)

# In[ ]:


train_labels = boxes_df[['ImageID', 'LabelName']].merge(labels, how='left', on='LabelName')
train_labels.Label.value_counts(normalize=True)[:45]             .hvplot.bar(width=650, height=350, rot=60, line_alpha=0,
                        title='Label Frequencies',
                        ylabel='fraction of all objects')


# # Hierarchy of objects
# 
# The Description page on the website has great information on the objects and how they relate. The picture below gives an idea of the relationships. A more complete picture appears on the website.
# 
# 
# ![hier](https://storage.googleapis.com/openimages/web/images/v2-bbox_labels_vis_screenshot.png)
# 
# You can also see relationships in our data with the file called 'oidv6-relationship-triplets.csv'. It look like this.

# In[ ]:


relations = pd.read_csv(data_dir/'oidv6-relationship-triplets.csv')
relations = relations.merge(labels, how='left', left_on='LabelName1', right_on='LabelName')                      .merge(labels, how='left', left_on='LabelName2', right_on='LabelName',
                            suffixes=['1', '2']) \
                     .loc[:, ['Label1', 'RelationshipLabel', 'Label2']] \
                     .dropna() \
                     .sort_values('RelationshipLabel') \
                     .reset_index(drop=True)


#  Mapping the entire network is quite complex. Here's a map for only two entities, boy and girl, and all the things to which they connect in the images.

# In[ ]:


import networkx as nx

kids = relations.query('Label1=="Girl" or Label1=="Boy"')
G = nx.from_pandas_edgelist(kids, 'Label1', 'Label2', 'RelationshipLabel')


graph_opts = dict(arrows=False,
                  node_size=5,
                  width=0.5,
                  alpha=0.8,
                  font_size=10,
                  font_color='darkblue',
                  edge_color='gray'
                
                 )

fig= plt.figure(figsize=(12,10))
nx.draw_spring(G, with_labels=True, **graph_opts)


# ## ------------------Done the Notebook-------------------

#!/usr/bin/env python
# coding: utf-8

# fork of https://www.kaggle.com/mistag/global-wheat-to-tfrecords
# 
# change - make the source_id numeric

# ## Create TFRecords
# So we will convert the Global Wheat Detection dataset to TFRecords, for use in TensorFlow-based models.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os, ast, glob
from PIL import Image, ImageFont, ImageDraw
import hashlib
from io import BytesIO
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read annotation data

# In[ ]:


LABEL='Wheat'
df=pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")
df.bbox = df.bbox.apply(ast.literal_eval)
for i in range(len(df)):
    df.bbox.iloc[i][2]=df.bbox.iloc[i][0]+df.bbox.iloc[i][2]
    df.bbox.iloc[i][3]=df.bbox.iloc[i][1]+df.bbox.iloc[i][3]
df.sample(5)


# ## TFRecords creation
# The function below creates TFRecords with all the bells and whistles.

# In[ ]:


def create_tf_example(imagedf, longest_edge=1024, source_id=0):  
    fname = '/kaggle/input/global-wheat-detection/train/'+imagedf.image_id.iloc[0]+'.jpg'
    filename=fname.split('/')[-1] # exclude path
    img = Image.open(fname, "r")
    # resize image if larger that longest edge while keeping aspect ratio
    if max(img.size) > longest_edge:
        img.thumbnail((longest_edge, longest_edge), Image.ANTIALIAS)
    height = img.size[1] # Image height
    width = img.size[0] # Image width
    buf= BytesIO()
    img.save(buf, format= 'JPEG') # encode to jpeg in memory
    encoded_image_data= buf.getvalue()
    image_format = b'jpeg'
    source_id = str(source_id) # must be unique number
    # A hash of the image is used in some frameworks
    key = hashlib.sha256(encoded_image_data).hexdigest()   
    # object bounding boxes 
    boxes = np.array(imagedf['bbox'].tolist())
    xmins = boxes[:,0]/width # List of normalized left x coordinates in bounding box (1 per box)
    ymins = boxes[:,1]/height # List of normalized top y coordinates in bounding box (1 per box)
    xmaxs = boxes[:,2]/width # List of normalized right x coordinates in bounding box
    ymaxs = boxes[:,3]/height # List of normalized bottom y coordinates in bounding box
    # List of string class name & id of bounding box (1 per box)
    object_cnt = len(imagedf)
    classes_text = []
    classes = []
    cname = LABEL
    for i in range(object_cnt):
        classes_text.append(cname.encode())
        classes.append(1)
    # unused features from Open Image 
    depiction = np.zeros(object_cnt, dtype=int)
    group_of = np.zeros(object_cnt, dtype=int)
    occluded = np.zeros(object_cnt, dtype=int) #also Pascal VOC
    truncated = np.zeros(object_cnt, dtype=int) # also Pascal VOC
    # Pascal VOC
    view_text = []
    for i in range(object_cnt):
        view_text.append('frontal'.encode())
    difficult = np.zeros(object_cnt, dtype=int)

    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[source_id.encode()])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode()])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/depiction': tf.train.Feature(int64_list=tf.train.Int64List(value=depiction)),
        'image/object/group_of': tf.train.Feature(int64_list=tf.train.Int64List(value=group_of)),
        'image/object/occluded': tf.train.Feature(int64_list=tf.train.Int64List(value=occluded)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=view_text))
    }))
    return tf_record


# We also need a labels.pbtxt file with the labels (only one).

# In[ ]:


labels=[LABEL]
pbfile=open('./labels.pbtxt', 'w') 
for i in range (len(labels)): 
    pbfile.write('item {{\n id: {}\n name:\'{}\'\n}}\n\n'.format(i+1, labels[i])) 
pbfile.close()


# ## Helper functions
# A few helper functions are defined below for visualizing images.

# In[ ]:


def bbox(img, xmin, ymin, xmax, ymax, color, width):
    draw = ImageDraw.Draw(img)
    xres, yres = img.size[0], img.size[1]
    box = np.multiply([xmin, ymin, xmax, ymax], [xres, yres, xres, yres]).astype(int).tolist()
    draw.rectangle(box, outline=color, width=width)
           
def plot_img(img, axes, xmin, ymin, xmax, ymax, classes, class_label, by):
    for i in range(len(xmin)):
        #color=hex_to_rgb(colors[class_label[i]-1])
        color='#e81123'
        bbox(img, xmin[i], ymin[i], xmax[i], ymax[i], color, 5)
    plt.setp(axes, xticks=[], yticks=[])
    axes.set_title(by)
    plt.imshow(img)


# ## Create sharded TFRecords
# We create a sharded dataset here, 20 shards will give a granularity of 5% for train/validate split.

# In[ ]:


get_ipython().run_cell_magic('time', '', "import contextlib2\n\nfilelist = glob.glob('/kaggle/input/global-wheat-detection/train/*')\n\ndef open_sharded_tfrecords(exit_stack, base_path, num_shards):\n    tf_record_output_filenames = [\n        '{}-{:04d}-of-{:04d}.tfrecord'.format(base_path, idx, num_shards)\n        for idx in range(num_shards)\n        ]\n    tfrecords = [\n        exit_stack.enter_context(tf.io.TFRecordWriter(file_name))\n        for file_name in tf_record_output_filenames\n    ]\n    return tfrecords\n\nnum_shards=20\noutput_filebase='./Wheat'\n\n# A context2.ExitStack is used to automatically close all the TFRecords created \nwith contextlib2.ExitStack() as tf_record_close_stack:\n    output_tfrecords = open_sharded_tfrecords(tf_record_close_stack, output_filebase, num_shards)\n    for i in tqdm(range(len(filelist))):\n        fid = filelist[i].replace('/kaggle/input/global-wheat-detection/train/','').split('.')[0]\n        ldf=df[df.image_id == fid].reset_index()\n        if len(ldf) > 0:\n            tf_record = create_tf_example(ldf, longest_edge=1024, source_id=i)\n            output_shard_index = i % num_shards\n            output_tfrecords[output_shard_index].write(tf_record.SerializeToString())")


# ## Check the output
# The last step is to check a few records to see that everything went OK:

# In[ ]:


fname='./Wheat-0005-of-0020.tfrecord'
dataset3 = tf.data.TFRecordDataset(fname)
fig = plt.figure(figsize=(12,18))
idx=1
for raw_record in dataset3.take(6):
    axes = fig.add_subplot(3, 2, idx)
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    xmin=example.features.feature['image/object/bbox/xmin'].float_list.value[:]
    xmax=example.features.feature['image/object/bbox/xmax'].float_list.value[:]
    ymin=example.features.feature['image/object/bbox/ymin'].float_list.value[:]
    ymax=example.features.feature['image/object/bbox/ymax'].float_list.value[:]
    classes=example.features.feature['image/object/class/text'].bytes_list.value[:]
    class_label=example.features.feature['image/object/class/label'].int64_list.value[:]
    img_encoded=example.features.feature['image/encoded'].bytes_list.value[0]
    img = Image.open(BytesIO(img_encoded))
    plot_img(img, axes, xmin, ymin, xmax, ymax, classes, class_label, '')
    idx=idx+1


# Yup - everything OK!

# In[ ]:





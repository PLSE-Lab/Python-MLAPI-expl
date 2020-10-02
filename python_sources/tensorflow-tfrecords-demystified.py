#!/usr/bin/env python
# coding: utf-8

# # TensorFlow TFRecords Demystified
# TensorFlow TFRecords have a reputation of being both clunky and very large. But that is just a reputation. In this kernel we will learn how to master writing and reading TFRecords, and control exactly how large or small these files are.  
# In this kernel we will:
#   * Create basic TFRecords (write & read)
#   * Create TFRecords with image data (write & read)
#   * Create TFRecords for a large image dataset using sharding

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Protocol buffers
# TFRecords are [Protocol buffers](https://developers.google.com/protocol-buffers/docs/overview). Protocol buffers are a way of serializing structured data in a compact and efficient way. A Protocol buffer message is defined by a `.proto` file. Example:  
# ```
# syntax = "proto3"; // define protocol version
# 
# // Define "Employee" message with 3 fields
# message Employee {
#  string name = 1; // a unique number is assigned to each field
#  int32 company_id = 2;
#  string address = 3;
# }
# ```  
# A single `.proto`-file can define multiple messages and messages can be nested. The beauty of protocol buffers is that the `.proto`-files can be compiled into a library (language of choice) with access functions using the `protoc` compiler. But for TFRecords the access functions we need are already part of TensorFlow, so no need to worry about compiling `.proto` files.

# ## TFRecord format
# TFRecords are made to support just about any type of data, and that means nesting basic features into a hierarchy of features. Look at the definition of [TFExample protobuffer here.](https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/core/example/feature.proto) TFRecords supports only three different data types:  
#   * bytes
#   * float
#   * int64
# 
# So any data must be converted into one of these three types. `tf.train.Feature` can be used to convert data into lists of these types. Let's create a few lists (or **features**).

# In[ ]:


# 8 int64 values
my_integers=tf.train.Feature(int64_list=tf.train.Int64List(value=np.arange(8)))
my_integers


# In[ ]:


# A single float value
my_floats=tf.train.Feature(float_list=tf.train.FloatList(value=[3.14]))
my_floats


# In[ ]:


# A range of bytes. Note that strings must be .encode()'ed
txt='This string must be encoded'
my_txt=tf.train.Feature(bytes_list=tf.train.BytesList(value=[txt.encode()]))
my_txt


# Next we can combine these individual features into a list of features, using `tf.train.Features`. Each feature is given a name (or key):

# In[ ]:


features=tf.train.Features(feature={
    'integers': my_integers,
    'pi': my_floats,
    'description': my_txt})
features


# The last step is to create an `example` TFRecord. So, `example` is not a very good name, but that is what the TFRecord is called. We simply pass the list of features to `tf.train.Example`.

# In[ ]:


tf_record = tf.train.Example(features=features)
tf_record


# Finally, we can write this record to a file using `tf.io.TFRecordWriter`. 

# In[ ]:


fname='example1.tfrecord'
with tf.io.TFRecordWriter(fname) as writer:
    writer.write(tf_record.SerializeToString())
print("Size of {} is {}bytes".format(fname, os.path.getsize(fname)))


# Next, read the file back in and access the data using `tf.data.TFRecordDataset` and `tf.train.Example.FromString`:

# In[ ]:


dataset = tf.data.TFRecordDataset(fname)
raw_example = next(iter(dataset)) # only one example in this file
parsed = tf.train.Example.FromString(raw_example.numpy())
parsed


# Now we can access each feature using `parsed.features.feature['<feature name>']`:

# In[ ]:


# strings must be decoded
parsed.features.feature['description'].bytes_list.value[0].decode() 


# In[ ]:


parsed.features.feature['integers'].int64_list.value[:] # get all data


# In[ ]:


parsed.features.feature['integers'].int64_list.value[5] # get a single value


# In[ ]:


parsed.features.feature['pi'].float_list.value[0] # get the value


# In[ ]:


parsed.features.feature['pi'].float_list.value[:] # get the value as a list


# Awesome! We just created a TFRecord, wrote it to a file, read it back in and accessed the features. If you are writing you own training code and input function, you can name the features to anything. If you want to create TFRecords as input to someone else's code, make sure all the required keys are present, otherwise there will be plenty of errors. Unfortunately, documentation of the expected TFRecord format is often hard to find.

# ## Using TFRecords with image data
# Right, time to create a more advanced TFRecord, this time with image data. There is kind of an established way to name keys for image data. To make input function efficient, all preprocessing steps should be performed when creating TFRecords, otherwise the input function must repeat preprocessing each time the training passed over the TFRecord. Typical preprocessing steps are:  
#   * Decode image from jpg etc.
#   * Resize image
#   * Convert to float
#   * Normalize to [0,1] range  
#   
# Augumentation is *not* something to do in a TFRecord, since agumentation could be different each time training passes over the data. We will make a TFrecord that is compatible with [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).  
# We are going to get images from the [ArTaxOr dataset](https://www.kaggle.com/mistag/arthropod-taxonomy-orders-object-detection-dataset), and the object data is imported from a pickled dataframe created in the [starter kernel](https://www.kaggle.com/mistag/starter-arthropod-taxonomy-orders-data-exploring).

# In[ ]:


import hashlib
from io import BytesIO
from PIL import Image, ImageFont, ImageDraw
ARTAXOR_PATH = '/kaggle/input/arthropod-taxonomy-orders-object-detection-dataset/'

pickles='/kaggle/input/starter-arthropod-taxonomy-orders-data-exploring/'
objectdf=pd.read_pickle(pickles+'ArTaxOr_objects.pkl')
labels=pd.read_pickle(pickles+'ArTaxOr_labels.pkl')
objectdf.sample(5)


# Define a function that creates a `tf.train.Example` from an image and the objects contained within. Note that TensorFlow Object Detection API expects jpeg-encoded image data, so the only preprocessing to be done is resize (optional) before the image is re-encoded (via a memory buffer using `BytesIO`. The input to this function is a DataFrame that contains one row per object and the columns shown above. This record is generous with features, but that is to make sure that the TF Object Detection API will successfully run the evaluation phase during training.

# In[ ]:


# Fetch attribution string from image EXIF data
def get_attribution(file):
    with Image.open(file) as img:
        exif_data = img._getexif()
    s='Photo: unknown'
    if exif_data is not None:
        if 37510 in exif_data:
            if len(exif_data[37510]) > 0:
                s = exif_data[37510][8:].decode('ascii')
        if 315 in exif_data:
            if len(exif_data[315]) > 0:
                s = 'Photo: ' + exif_data[315]
    return s

# Create example for TensorFlow Object Detection API
def create_tf_example(imagedf, longest_edge=1024):  
    fname = ARTAXOR_PATH+imagedf.file.iloc[0]
    filename=fname.split('/')[-1] # exclude path
    by = get_attribution(fname)
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
    source_id = filename.split('.')[0]
    license = 'CC BY-NC-SA 4.0'
    # A hash of the image is used in some frameworks
    key = hashlib.sha256(encoded_image_data).hexdigest()   
    # object bounding boxes 
    xmins = imagedf.left.values # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = imagedf.right.values # List of normalized right x coordinates in bounding box
    ymins = imagedf.top.values # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = imagedf.bottom.values # List of normalized bottom y coordinates in bounding box
    # List of string class name & id of bounding box (1 per box)
    object_cnt = len(imagedf)
    classes_text = []
    classes = []
    for i in range(object_cnt):
        classes_text.append(imagedf.label.iloc[i].encode())
        classes.append(1+imagedf.label_idx.iloc[i])
    # unused features from Open Image 
    depiction = np.zeros(object_cnt, dtype=int)
    group_of = np.zeros(object_cnt, dtype=int)
    occluded = imagedf.occluded.values #also Pascal VOC
    truncated = imagedf.truncated.values # also Pascal VOC
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
        'image/license': tf.train.Feature(bytes_list=tf.train.BytesList(value=[license.encode()])),
        'image/by': tf.train.Feature(bytes_list=tf.train.BytesList(value=[by.encode()])),
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


# Create an example record and save it to file. The example file happens to be the ArTaxOr dataset logo which contains a butterfly and a bumblebee.

# In[ ]:


sample_file='ArTaxOr/Lepidoptera/002b37ac08e1.jpg'
imagedf=objectdf[objectdf.file == sample_file]
tfr=create_tf_example(imagedf)
fname='./image_ex1.tfrecord'
with tf.io.TFRecordWriter(fname) as writer:
    writer.write(tfr.SerializeToString())
print("Size of {} is {}kbytes".format(fname, os.path.getsize(fname)//1024))


# Now that the `.tfrecord` file is stored, let's load it back in and visualize the contents.

# In[ ]:


# Some helper functions to draw image with object boundary boxes
fontname = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(fontname, 40) if os.path.isfile(fontname) else ImageFont.load_default()

def bbox(img, xmin, ymin, xmax, ymax, color, width, label, score):
    draw = ImageDraw.Draw(img)
    xres, yres = img.size[0], img.size[1]
    box = np.multiply([xmin, ymin, xmax, ymax], [xres, yres, xres, yres]).astype(int).tolist()
    txt = " {}: {}%" if score >= 0. else " {}"
    txt = txt.format(label, round(score, 1))
    ts = draw.textsize(txt, font=font)
    draw.rectangle(box, outline=color, width=width)
    if len(label) > 0:
        if box[1] >= ts[1]+3:
            xsmin, ysmin = box[0], box[1]-ts[1]-3
            xsmax, ysmax = box[0]+ts[0]+2, box[1]
        else:
            xsmin, ysmin = box[0], box[3]
            xsmax, ysmax = box[0]+ts[0]+2, box[3]+ts[1]+1
        draw.rectangle([xsmin, ysmin, xsmax, ysmax], fill=color)
        draw.text((xsmin, ysmin), txt, font=font, fill='white')

def plot_img(img, axes, xmin, ymin, xmax, ymax, classes, class_label, by):
    for i in range(len(xmin)):
        color=labels.color[class_label[i]-1]
        bbox(img, xmin[i], ymin[i], xmax[i], ymax[i], color, 5, classes[i].decode(), -1)
    plt.setp(axes, xticks=[], yticks=[])
    axes.set_title(by)
    plt.imshow(img)


# In[ ]:


# load tfrecord
fname='image_ex1.tfrecord'
dataset = tf.data.TFRecordDataset(fname)
img_example = next(iter(dataset)) 
img_parsed = tf.train.Example.FromString(img_example.numpy())
# only extract features we will actually use
xmin=img_parsed.features.feature['image/object/bbox/xmin'].float_list.value[:]
xmax=img_parsed.features.feature['image/object/bbox/xmax'].float_list.value[:]
ymin=img_parsed.features.feature['image/object/bbox/ymin'].float_list.value[:]
ymax=img_parsed.features.feature['image/object/bbox/ymax'].float_list.value[:]
by=img_parsed.features.feature['image/by'].bytes_list.value[0].decode()
classes=img_parsed.features.feature['image/object/class/text'].bytes_list.value[:]
class_label=img_parsed.features.feature['image/object/class/label'].int64_list.value[:]
img_encoded=img_parsed.features.feature['image/encoded'].bytes_list.value[0]


# In[ ]:


fig = plt.figure(figsize=(10,10))
axes = axes = fig.add_subplot(1, 1, 1)
img = Image.open(BytesIO(img_encoded))
plot_img(img, axes, xmin, ymin, xmax, ymax, classes, class_label, by)


# Yes - it works! But putting jpeg-encoded images in TFRecords does not release the full potential of TFRecords. If you are training on a machine with fast I/O and lots of disk space, it is better to do all the preprocessing before storing the image as normalized float array. Let's define another function for this:

# In[ ]:


def create_tf_example2(imagedf, longest_edge=1024):  
    # Filename of the image (full path is useful when there are multiple image directories)
    fname = ARTAXOR_PATH+imagedf.file.iloc[0]
    filename=fname.split('/')[-1] # exclude path
    by = get_attribution(fname)
    img = Image.open(fname, "r")
    source_id = filename.split('.')[0]
    # resize image if larger that longest edge while keeping aspect ratio
    if max(img.size) > longest_edge:
        img.thumbnail((longest_edge, longest_edge), Image.ANTIALIAS)
    image_data = np.asarray(img)
    # storing shape will make it easy to reconstruct image later
    image_shape = np.array(image_data.shape)
    # convert to float
    image_data = image_data.reshape(image_data.shape[0]*image_data.shape[1]*image_data.shape[2])
    image_data = image_data.astype(float)/255. # normalize to [0,1]
    # object bounding boxes 
    xmins = imagedf.left.values # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = imagedf.right.values # List of normalized right x coordinates in bounding box
    ymins = imagedf.top.values # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = imagedf.bottom.values # List of normalized bottom y coordinates in bounding box
    # List of string class name & id of bounding box (1 per box)
    classes_text = []
    classes = []
    for i in range(len(imagedf)):
        classes_text.append(imagedf.label.iloc[i].encode())
        classes.append(1+imagedf.label_idx.iloc[i])

    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'image/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image_shape)),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[source_id.encode()])),
        'image/by': tf.train.Feature(bytes_list=tf.train.BytesList(value=[by.encode()])),
        'image/data': tf.train.Feature(float_list=tf.train.FloatList(value=image_data)),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes))
    }))
    return tf_record


# In[ ]:


sample_file2='ArTaxOr/Hymenoptera/ab30b4c2f70c.jpg'
imagedf=objectdf[objectdf.file == sample_file2]
tfr2 = create_tf_example2(imagedf)
fname2 = 'image_ex2.tfrecord'
with tf.io.TFRecordWriter(fname2) as writer:
    writer.write(tfr2.SerializeToString())
print("Size of {} is {}kbytes".format(fname2, os.path.getsize(fname2)//1024))


# Right! Storing image data as float gives a filesize of almost 10MB while the jpeg-encoded one was only 104kB. But now all preprocessing steps are complete, and the image can be fed directly to training (or possibly agumentation). Now, let's read it back in and display. 

# In[ ]:


dataset2 = tf.data.TFRecordDataset(fname2)
img_example2 = next(iter(dataset2)) 
img_parsed2 = tf.train.Example.FromString(img_example2.numpy())
# extract features
xmin=img_parsed2.features.feature['image/object/bbox/xmin'].float_list.value[:]
xmax=img_parsed2.features.feature['image/object/bbox/xmax'].float_list.value[:]
ymin=img_parsed2.features.feature['image/object/bbox/ymin'].float_list.value[:]
ymax=img_parsed2.features.feature['image/object/bbox/ymax'].float_list.value[:]
by=img_parsed2.features.feature['image/by'].bytes_list.value[0].decode()
classes=img_parsed2.features.feature['image/object/class/text'].bytes_list.value[:]
class_label=img_parsed2.features.feature['image/object/class/label'].int64_list.value[:]
img_shape=img_parsed2.features.feature['image/shape'].int64_list.value[:]
img_data=img_parsed2.features.feature['image/data'].float_list.value[:]


# In[ ]:


image2=np.array(img_data).reshape(img_shape) # reshape
image2=image2*255. # scale back to [0, 255] and convert to int
image2=image2.astype(int)
img=Image.fromarray(np.uint8(image2))
fig = plt.figure(figsize=(10,10))
axes = axes = fig.add_subplot(1, 1, 1)
plot_img(img, axes, xmin, ymin, xmax, ymax, classes, class_label, by)


# OK, we have seen different ways of how images can be stored in TFRecords. Now, let's consider the case when we have a huge dataset, and need to create multiple TFRecords.

# ## Sharding large datasets
# Sharding is a method of splitting and storing a large dataset into multiple files. The last part of this kernel we will convert the entire ArTaxOr dataset into TFRecords.

# In[ ]:


filelist=pd.read_pickle(pickles+'ArTaxOr_filelist.pkl')
filelist=filelist.sample(frac=1)
filelist.head()


# The `contextlib2` library is used to automatically close all TFRecords files after writing is finished.

# In[ ]:


get_ipython().run_cell_magic('time', '', "import contextlib2\n\ndef open_sharded_tfrecords(exit_stack, base_path, num_shards):\n    tf_record_output_filenames = [\n        '{}-{:05d}-of-{:05d}.tfrecord'.format(base_path, idx, num_shards)\n        for idx in range(num_shards)\n        ]\n    tfrecords = [\n        exit_stack.enter_context(tf.io.TFRecordWriter(file_name))\n        for file_name in tf_record_output_filenames\n    ]\n    return tfrecords\n\nnum_shards=50\noutput_filebase='./ArTaxOr'\n\nwith contextlib2.ExitStack() as tf_record_close_stack:\n    output_tfrecords = open_sharded_tfrecords(tf_record_close_stack, output_filebase, num_shards)\n    for i in range(len(filelist)):\n        ldf=objectdf[objectdf.id == filelist.id.iloc[i]].reset_index()\n        tf_record = create_tf_example(ldf, longest_edge=1280)\n        output_shard_index = i % num_shards\n        output_tfrecords[output_shard_index].write(tf_record.SerializeToString())")


# In[ ]:


get_ipython().system('ls -lh ArTaxOr*.tfrecord')


# If we are going to use TF Object Detection API, a label definition file is also needed:

# In[ ]:


labels=pd.read_pickle(pickles+'ArTaxOr_labels.pkl')
pbfile=open('./ArTaxOr.pbtxt', 'w') 
for i in range (len(labels)): 
    pbfile.write('item {{\n id: {}\n name:\'{}\'\n}}\n\n'.format(i+1, labels.name[i])) 
pbfile.close()


# Again, read a few records from one of the `.tfrecord` files to check that everything is OK.

# In[ ]:


fname='./ArTaxOr-00029-of-00050.tfrecord' 
dataset3 = tf.data.TFRecordDataset(fname)
fig = plt.figure(figsize=(16,18))
idx=1
for raw_record in dataset3.take(6):
    axes = fig.add_subplot(3, 2, idx)
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    xmin=example.features.feature['image/object/bbox/xmin'].float_list.value[:]
    xmax=example.features.feature['image/object/bbox/xmax'].float_list.value[:]
    ymin=example.features.feature['image/object/bbox/ymin'].float_list.value[:]
    ymax=example.features.feature['image/object/bbox/ymax'].float_list.value[:]
    by=example.features.feature['image/by'].bytes_list.value[0].decode()
    classes=example.features.feature['image/object/class/text'].bytes_list.value[:]
    class_label=example.features.feature['image/object/class/label'].int64_list.value[:]
    img_encoded=example.features.feature['image/encoded'].bytes_list.value[0]
    img = Image.open(BytesIO(img_encoded))
    plot_img(img, axes, xmin, ymin, xmax, ymax, classes, class_label, by)
    idx=idx+1


# ## Summary
# We have seen how TFRecords can store just about any type of data in a format that makes it easy to read using key-value pairs. And new keys (or features) can be added without breaking backwards compatibility. When image data is written into TFRecords, the size of the TFRecord is directly dependent on how the image data is stored (encoded jpeg vs. float arrays).

# ## References
# *  [Protocol buffers](https://developers.google.com/protocol-buffers/)  
# *  [TensorFlow TFRecord tutorial](https://www.tensorflow.org/tutorials/load_data/tfrecord)

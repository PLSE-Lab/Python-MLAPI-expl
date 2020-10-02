#!/usr/bin/env python
# coding: utf-8

# Hello Kagglers!! Enjoying life and spending time on Kaggle? Well, since last few days, I am not feeling well at all but my addiction to Kaggle is totally different. Even though I do so much of data science and ML/DL in my daily life but if I don't visit Kaggle or our slack,  [KaggleNoobs](https://kagglenoobs.herokuapp.com/), the work doesn't seem to be complete. There was a discussion in that slack over Tensorflow Object Detection API and I agree that to a newcomer, it might be slightly overwhelming. But once you get it, it's the most straight forward thing to apply to your work and achieve amazing results before you try something that isn't out there.  Yes, you guessed it right, today's kernel is going to be a walkthrough for the TF object detection API and I hope you will like it. 
# 
# ![detection](https://media.giphy.com/media/StRnSltcS0n04/giphy.gif)
# 
# **PS:** There are certain constraints in the kernel, regarding the installation of some ubuntu packages that are required to be installed for using the API but be assured that even though there is no straightforward way to use it in kernels, you will find this tutorial very very helpful to speed up your modeling. So, let's dive in.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import glob
import cv2
import numba as nb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from time import time
from numba import jit
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from skimage.io import imread
from PIL import Image
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
np.random.seed(111)
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


# Defining some paths as usual
input_dir = Path('../input/')
data_dir = input_dir / 'data_300x300/data_300x300'


# Quoting from the data description: `This dataset contains images and labels of feline reticulocytes (an immature red blood cell without a nucleus, having a granular or reticulated appearance when suitably stained). The dataset was created using equipment that is easily accessible to veterinarians: a standard laboratory microscope and two types of cameras: a basic microscope camera and a smartphone camera`
# 
# Let's look how the dataset is arranged

# In[3]:


os.listdir(data_dir)


# The `images` directory contains the images for training,  the `labels` directory contains the corresponding annotations for the training images and `TEST` contains the test images. How are the annotations done? When you annotate any object with a bounding box, there are certain things that you need to take care of in the annotations. The annotation corresponding to an image should contain the coordinates of the bounding boxes, the height of the image, the width of the image and the label corresponding to that box.  Here is an example of the annotation in our dataset:
# 
# ```
# <annotation>
# 	<folder>images</folder>
# 	<filename>000045.jpg</filename>
# 	<path>/home/vini/Desktop/data_300x300/images/000045.jpg</path>
# 	<source>
# 		<database>Unknown</database>
# 	</source>
# 	<size>
# 		<width>300</width>
# 		<height>300</height>
# 		<depth>3</depth>
# 	</size>
# 	<segmented>0</segmented>
# 	<object>
# 		<name>aggregate reticulocyte</name>
# 		<pose>Unspecified</pose>
# 		<truncated>0</truncated>
# 		<difficult>0</difficult>
# 		<bndbox>
# 			<xmin>140</xmin>
# 			<ymin>115</ymin>
# 			<xmax>169</xmax>
# 			<ymax>143</ymax>
# 		</bndbox>
# 	</object>
# 	<object>
# 		<name>punctate reticulocyte</name>
# 		<pose>Unspecified</pose>
# 		<truncated>0</truncated>
# 		<difficult>0</difficult>
# 		<bndbox>
# 			<xmin>72</xmin>
# 			<ymin>155</ymin>
# 			<xmax>103</xmax>
# 			<ymax>187</ymax>
# 		</bndbox>
# 	</object>
# 	<object>
# 		<name>erythrocyte</name>
# 		<pose>Unspecified</pose>
# 		<truncated>0</truncated>
# 		<difficult>0</difficult>
# 		<bndbox>
# 			<xmin>184</xmin>
# 			<ymin>195</ymin>
# 			<xmax>213</xmax>
# 			<ymax>228</ymax>
# 		</bndbox>
# 	</object>
# </annotation>
# ```
# 
# You can see the height and width of the image, the bounding box `bndbox`, the coordinates of the bounding box `xmin, ymin, xmax, ymax`, the label corresponding to that bounding box  given by the node `name` . There are a lot of opensource tools that you can use for annotating datasets but amongst all of them, the simplest and the best one is [labelImg](https://github.com/tzutalin/labelImg). 

# ## Preprocessing
# 
# The annotations are given as `xmls`. The Tensorflow Object detection API accepts data in `TFRecords` format. So, we need to process our annotations

# In[4]:


# A function to parse the xmls
def parse_xmls(xml_files):
    data = []
    # Iterate over each file
    for sample in xml_files:
        # Get the xml tree
        tree = ET.parse(sample)

        # Get the root
        root = tree.getroot()

        # Get the members and extract the values
        for member in root.findall('object'):
            # Name of the image file
            filename = root.find('filename').text
            
            # Height and width of the image
            width =  int((root.find('size')).find('width').text)
            height = int((root.find('size')).find('height').text)
            
            # Bounding box coordinates
            bndbox = member.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            xmax = float(bndbox.find('xmax').text)
            ymin = float(bndbox.find('ymin').text)
            ymax = float(bndbox.find('ymax').text)
            
            # label to the corresponding bounding box
            label =  member.find('name').text

            data.append((filename, width, height, label, xmin, ymin, xmax, ymax))
    
    # Create a pandas dataframe
    columns_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(data=data, columns=columns_name)

    return df


# In[5]:


images = sorted(glob.glob('../input/data_300x300/data_300x300/images/*.jpg'))
xmls = sorted(glob.glob('../input/data_300x300/data_300x300/labels/*.xml'))
print("Total number of images: ", len(images))
print("Total number of xmls: ", len(xmls))


# In[6]:


# Parse the xmls and get the data in a dataframe
df = parse_xmls(xmls)
df.head()


# In[7]:


# How many classes do we have for object detection?
label_counts = df['class'].value_counts()
print(label_counts)

plt.figure(figsize=(20,8))
sns.barplot(x=label_counts.index, y= label_counts.values, color=color[2])
plt.title('Labels in our dataset', fontsize=14)
plt.xlabel('Label', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(len(label_counts.index)), ['erythrocyte', 'punctate reticulocyte', 'aggregate reticulocyte'])
plt.show()


# In[8]:


train, valid = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=111)

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
print("Number of training samples: ", len(train))
print("Number of validation samples: ", len(valid))


# # The TensorFlow Object Detection API setup
# 
# As I said there are certains things that needs to be installed on the host computer for using the API which is not possible in kernels, so I will demonstrate the steps in markdown. 
# 
# 1.  Install tensorflow-gpu. Make sure you have installed the right version of Cuda and cuDNN. For more information, click [here](https://www.tensorflow.org/install/)
# 2. Make a directory where you want to store all of the work and just cd into it
# 3. Clone the tensorflow models repo `git clone https://github.com/tensorflow/models.git`
# 4.  Install protobuf compiler `sudo apt-get install protobuf-compiler`
# 5. Install other dependencies:
#     * pip install Cython
#     * pip install pillow
#     * pip install lxml
#     * pip install jupyter
#     * pip install matplotlib
#  
# 6.  `cd models/research/`
# 
# 7.   `protoc object_detection/protos/*.proto --python_out=.`
# 
# 8.   ```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim```
# 
# 9.   Test your installation: `python object_detection/builders/model_builder_test.py`
# 
# If the last step ran successfully then, you are done with the set up. Yay!!!

# # LabelMap
# 
# Before converting you dataset to TFRecords, you need to make sure that you have a labelmap corresponding to all the labels that are in your dataset. For our dataset, the labelmap looks like this:
# 
# ```
# item {
#   id: 1
#   name: 'erythrocyte'
# }
# 
# item {
#   id: 2
#   name: 'punctate reticulocyte'
# }
# 
# item {
#   id: 3
#   name: 'aggregate reticulocyte'
# }
# 
# ```
# 
# **Note**: Numbering starts from 1 as 0 is treated as background. I have named this labelmap as `bloodmap.pbtxt` and at this point my work directory looks like this:
# 
# ```
# /home
#      /ubuntu
#            /Nain
#                  /models
#                      /research
#                             /blood_train
#                                    /felina
#                                         /data_300x300
#                              bloodmap.pbtxt
# ```
# 
# `blood_train` is the directory that I created for this project.

# # Converting your data to TFRecords format
# 
# This is the part where most of the the beginners get stuck. They have no clue how to do this. But this is quite simple. You have all your data information stored in the dataframe. The most importnat thing to remember is that a single image can contain multiple labels(bounding boxes) in the annotations, so you have to do a `groupby` on your dataframe. Let's see how we can do that.
# 
# ```python
# # Import the packages required
# import sys
# sys.path.append("..")
# import io
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.image as mimg
# from PIL import Image
# from collections import namedtuple, OrderedDict
# from models.research.object_detection.utils import dataset_util
# from models.research.object_detection.utils import label_map_util
# 
# # Function to group data and return the same
# # Group by imagefile name
# def make_groups(df, field=None):
#     if field==None:
#         field = 'filename'
#         
#     data = namedtuple('object', ['filename', 'info'])
#     grouped = df.groupby(field)
#     
#     grouped_data = []
#     for filename, x in zip(grouped.groups.keys(), grouped.groups):
#         grouped_data.append(data(filename, grouped.get_group(x)))
#         
#     return grouped_data
#     
#     
#   # Creating a tf record sample
#   def create_tf_example(group, img_path, label_map_dict)
#       # Read the imagefile. This will be used in features later 
#       with tf.gfile.GFile(os.path.join(img_path, '{}'.format(group.filename)), 'rb') as f:
#           img_file = f.read()
#     
#       # Encode to bytes and read using PIL. Could be done directly too
#       encoded_img = io.BytesIO(img_file)
#       # Read the image using PIL
#       img = Image.open(encoded_img)
#       width, height = img.size
#     
#       # Encode the name of the img file
#       filename = group.filename.encode('utf8')
#       
#       # Define the format of the image file
#       img_format = b'jpg'   # The name will be in bytes
#     
#     
#       # Define the variables that you need as features
#       xmins = []
#       xmaxs = []
#       ymins = []
#       ymaxs = []
#       classes_text = []
#       classes = []
# 
#       # Iterate over the namedtuple object
#       for index, row in group.info.iterrows():
#           xmins.append(row['xmin'] / width)   # store normalized values for bbox
#           xmaxs.append(row['xmax'] / width)
#           ymins.append(row['ymin'] / height)
#           ymaxs.append(row['ymax'] / height)
#           classes_text.append(row['class'].encode('utf8'))
#           classes.append(label_map_dict[row['class']])
# 
#       tf_example = tf.train.Example(features=tf.train.Features(feature={
#           'image/height': dataset_util.int64_feature(height),
#           'image/width': dataset_util.int64_feature(width),
#           'image/filename': dataset_util.bytes_feature(filename),
#           'image/source_id': dataset_util.bytes_feature(filename),
#           'image/encoded': dataset_util.bytes_feature(img_file),
#           'image/format': dataset_util.bytes_feature(img_format),
#           'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
#           'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
#           'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
#           'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
#           'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
#           'image/object/class/label': dataset_util.int64_list_feature(classes),}))
#     
#       return tf_example
# ```
# 
# 

# Great!! We have defined all the functions required for preprocessing and all. While creating TFRecords, all we need to do is to open a TFRecords writer instance and create `train.record` and `valid.record` from our `train` and `valid` dataframes. Let's do that.
# 
# ```python
# # Path where all the images are present
# img_path = './felina/data_300x300/images/'
# # Label map
# label_map_dict = label_map_util.get_label_map_dict('./bloodmap.pbtxt')
# 
# writer = tf.python_io.TFRecordWriter('./train.record')
# # create groups in the df. One image may contain several instances of an object hence the grouping thing
# img_groups = make_groups(train, field='filename')
# # Iterate over the samples in each group create a TFRecord
# for group in img_groups:
#     tf_example = create_tf_example(group, img_path, label_map_dict)
#     writer.write(tf_example.SerializeToString())
# # close the writer
# writer.close()
# print("TFRecords for training data  created successfully")
# 
# 
# writer = tf.python_io.TFRecordWriter('./valid.record')
# # create groups 
# img_groups = make_groups(valid, field='filename')
# # Iterate over the samples in each group create a TFRecord
# for group in img_groups:
#     tf_example = create_tf_example(group, img_path, label_map_dict)
#     writer.write(tf_example.SerializeToString())
# # close the writer
# writer.close()
# print("TFRecords for validation data created successfully")
# ```

# # Adding the model you want to use 
# 
# Now we have almost everything ready.  We need to do two more steps:
# * Choosing the model config file that you want to use for training
# * Downloading the weights of the same model from [TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
# 
# Let's do this.
# 1.  I chose the ssd_inception_v2 model for my training but you can chose whichever you like. `cp models/research/object_detection/samples/configs/ssd_inception_v2_coco.config`
# 
# 2. `wget download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz`
# 
# 3.  `unzip ssd_inception_v2_coco_2017_11_17.tar.gz`
# 
# 4. `mv ssd_inception_v2_coco_2017_11_17 ssd_inceptionv2`
# 
# Now your work directory should be like this:
# 
# ```
# /home
#      /ubuntu
#            /Nain
#                  /models
#                      /research
#                             /blood_train
#                                 /felina
#                                     /data_300x300
#                                  /ssd_inceptionv2      
#                                  bloodmap.pbtxt
#                                  train.record
#                                  valid.record
#                                  ssd_inception_v2_coco.config
# ```

# # Configuring your model config file
# 
# In the model config that you chose to use, you need to make some changes. You need to give the path of the `tfrecords` and the `labelmap` files as well as the checkpoint of that model for fine tuning. Open your config file and edit the following lines:
# 
# 
# ```
# num_classes: 3
# 
# 
# fine_tune_checkpoint:"/home/ubuntu/Nain/models/research/blood_train/ssd_inceptionv2/model.ckpt"
# 
# train_input_reader: {
#   tf_record_input_reader {
#     input_path: "/home/ubuntu/Nain/models/research/blood_train/train.record"
#   }
#   label_map_path: "/home/ubuntu/Nain/models/research/blood_train/bloodmap.pbtxt"
# }
# 
# eval_input_reader: {
#   tf_record_input_reader {
#     input_path: "/home/ubuntu/Nain/models/research/blood_train/valid.record"
#   }
#   label_map_path:"/home/ubuntu/Nain/models/research/blood_train/bloodmap.pbtxt"
#   shuffle: false
#   num_readers: 1
# }
# 
# ```

# # Training
# 
# We are almost done!! I know it's too much in one go but once you do it, you will become very comfortable in using it. To start the training, we need to do two things:
# * Create a directory for storing training checkpoints. I named it `checkpoints`
# * Copy the `train.py`,  `eval.py` and  `export_inference_graph.py` from the `object_detection` directory to our current directory
# 
# This is how your things should be arranged by now:
# 
# ```
# /home
#      /ubuntu
#            /Nain
#                  /models
#                      /research
#                             /blood_train
#                                 /felina
#                                     /data_300x300
#                                  /ssd_inceptionv2
#                                  /checkpoints
#                                  bloodmap.pbtxt
#                                  train.record
#                                  valid.record
#                                  ssd_inception_v2_coco.config
#                                  train.py
#                                  eval.py
#                                  export_inference_graph.py
# ```
# 
# And the final command to run the training!!!!
# 
# ```
# python train.py --logtostderr --train_dir=/home/ubuntu/Nain/models/research/blood_train/checkpoints/ --pipeline_config_path=/home/ubuntu/Nain/models/research/blood_train/ssd_inception_v2_coco.config
# 
# ```
# 
# 

# # Freezing the graph
# 
# Once you are done with the training, you need to freeze the graph for doing inference.  The checkpoint depends on the number of iterations you completed for training.  I completed 25K iterations for this, but you should do more as it's not enough.  Freeze the graph:
# 
# ```
# python export_inference_graph.py --input_type image_tensor --pipeline_config_path /home/ubuntu/Nain/models/research/blood_train/ssd_inception_v2_coco.config --trained_checkpoint_prefix ./checkpoints/model.ckpt-25823 --output_directory ./fine_tuned_model
# ```

# These are the results that I got on some sample images
# 
# ![](https://i.imgur.com/TpRvjvp.png)
# 
# ![](https://i.imgur.com/E4DwYEG.png)

# And that's it folks!! I have tried my best to give you an overview of the TF object detection API in a most simplified way.  The motivation behind this kernel actually came from the discussion  regarding the TF API in our [KaggleNoobs](https://kagglenoobs.herokuapp.com/) slack. **Please upvote if this you liked this kernel**. 

# In[ ]:





# In[ ]:





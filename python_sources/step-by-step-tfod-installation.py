#!/usr/bin/env python
# coding: utf-8

# 1. # TFOD Installation Steps : 
# 
# <p><font size='3' color='green'> If you like my work,please consider giving an upvote !</font></p>

# Official repository for TFOD:
# 
# 
# https://github.com/tensorflow/models/tree/master/research/object_detection

# ## 1. Downloading the files

# **1. Download the model repository**
# 
# 
# 
# Google it : Tensorflow models 
# 
# 
# click on below link :
# 
# 
# https://github.com/tensorflow/models
# 
# 
# 
# From branch select the tag v1.13.0
# 
# 
# 
# ------------------or---------------------
# 
# 
# 
# Download repository directly from below link
# 
# 
# 
# https://github.com/tensorflow/models/archive/v1.13.0.zip
# 
# 
# 
# **2. Download the pre-train model**
# 
# 
# 
# Google it : Tensorflow models zoo
# 
# 
# 
# click on below link :
# 
# 
# 
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# 
# 
# 
# ------------------or---------------------
# 
# 
# 
# Download faster_rcnn_inception_v2_coco directly from below link
# 
# 
# 
# http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
# 
# 
# 
# coco : This is the datasets which are having 91 categories.
# 
# 
# 
# **3. Download the image data**
# 
# 
# 
# https://drive.google.com/file/d/12F5oGAuQg7qBM_267TCMt_rlorV-M7gf/view?usp=sharing
# 
# 
# 
# **Downlaod Annotation tool named as Lablel IMG**
# 
# 
# 
# https://tzutalin.github.io/labelImg/
# 
# 
# 
# For Windows : Downlaod the last option for windows
# 
# 
# 
# For Linux : git clone -> https://github.com/tzutalin/labelImg

# ## 2. Installation

# **1. On Desktop create TFOD folder and save all the downloaded files in that folder**
# 
# 
# 
# **2. Creating virtual env using conda**
# 
# 
# 
# conda create -n TFOD python=3.6
# 
# 
# 
# **2.1 Activate the env**
# 
# 
# 
# conda activate TFOD
# 
# 
# 
# **2.2 Downlaod the required libraries using below command**
# 
# 
# 
# **GPU**
# 
# 
# 
# pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow-gpu==1.14.0
# 
# 
# 
# **CPU**
# 
# 
# 
# pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow ==1.14.0
# 
# 
# 
# **3. Unzip all the files inside TFOD folder.**
# 
# 
# 
# **4. To convert protos file into py file, we need to install protobuf using below conda command.**
# 
# 
# 
# conda install -c anaconda protobuf
# 
# 
# 
# **5. Convert protos file into py files**
# 
# 
# 
# Go to research folder:
# 
# 
# 
# protoc object_detection/protos/*.proto --python_out=.
# 
# 
# 
# **6. Inside research folder run below command in CMD**
# 
# 
# 
# python setup.py install

# ## 3. Verification

# **1. Research folder -> object detection -> open Jupyter notebook and run all the cells** 
# 
# 
# 
# **2. For the testing in jupyter notebook add below lines at the end of Jupyter notebook**
# 
# 
# 
# %matplotlib inline
# 
# plt.figure(figsize=(200,200))
# 
# plt.imshow(image_np)

# ### 4. Annotation/Labelling

# 1. open labelImg.exe
# 
# 
# 
# 2. For TFOD option should be Pascal/VOC

# ### 5. Custom training proess

# **1. From utils folder copy all the files and paste inside the research folder.**
# 
# 
# 
# **2. Conversion from XML to CSV**(inside research folder)
# 
# 
# 
# **note** : for TFOD, imgaes should be color images.
# 
# 
# 
# python xml_to_csv.py
# 
# 
# 
# **3. According to classes of datasets, generate_tfrecord.py file needs to be changed**
# 
# 
# 
# **4. Conversion form csv to tfrecords**
# 
# 
# 
# **For Train:**
# 
# 
# 
# python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=object_detection/train.record
# 
# 
# 
# **For Test:**
# 
# 
# 
# python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=object_detection/test.record

# ## 6. Trainig phase

# **1. Copy train.py file from object_detection/legacy folder to research folder**
# 
# 
# 
# **2. Copy model (faster_rcnn_inception_v2_coco_2018_01_28) folder inside research folder.**
# 
# 
# 
# **3. For all the models which are present in zoo repo for that corresponding config files are availbale inside object_detection/samples folder**
# 
# 
# 
# **4. Copy (faster_rcnn_inception_v2_coco.config)file from object_detection/samples folder and paste inside research\training folder.**
# 
# 
# 
# **5. Inside research\training folder there should be labelmap.pbtxt file which has to be updated with all the classes required for training.**
# 
# 
# 
# **6. In config file (faster_rcnn_inception_v2_coco.config) needs to do 7 changes**
# 
# 
# 
# * a. num_classes: 6
# 
# * b. fine_tune_checkpoint: "faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
# 
# * c. num_steps: 200
# 
# * d. input_path: "object_detection/train.record"
# 
# * e. label_map_path: "training/labelmap.pbtxt"
# 
# * f. input_path: "object_detection/test.record"
# 
# * g. label_map_path: "training/labelmap.pbtxt"
# 
# 
# **7.Copy two folders (deployment and nets) form research\slim folder to research folder**
# 
# 
# 
# **8. Start the training and for that open a terminal inside the research folder**
# 
# 
# 
# python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config
# 
# 
# 
# **9. For re-training use the above command**
# 
# 
# 
# **10. To convert ckpt into pb file, first copy file (export_inference_graph.py) from research/object_detection folder to research flder.**
# 
# 
# 
# **11. Replace the XXXX with the last generated ckpt file inside the training folder and run below command inside research folder to convert ckpt into pb file.**
# 
# 
# 
# python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-200 --output_directory inference_graph

# In[ ]:





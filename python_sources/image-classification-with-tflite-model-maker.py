#!/usr/bin/env python
# coding: utf-8

# # **Flower classification with TensorFlow Lite Model Maker with TensorFlow 2.0**
# 

# Model Maker library simplifies the process of adapting and converting a TensorFlow neural-network model to particular input data when deploying this model for on-device ML applications.
# 
# This notebook shows an end-to-end example that utilizes this Model Maker library to illustrate the adaption and conversion of a commonly-used image classification model to classify flowers on a mobile device.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Prerequisites
# 
# To run this example, we first need to install serveral required packages, including Model Maker package that in github [repo](https://github.com/tensorflow/examples/tree/master/tensorflow_examples/lite/model_maker).

# In[ ]:


get_ipython().system('pip install git+git://github.com/tensorflow/examples.git#egg=tensorflow-examples[model_maker]')


# Import the required packages.

# In[ ]:


import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_maker.core.task import image_classifier
from tensorflow_examples.lite.model_maker.core.task.model_spec import mobilenet_v2_spec
from tensorflow_examples.lite.model_maker.core.task.model_spec import ImageModelSpec

import matplotlib.pyplot as plt


# Simple End-to-End Example
# Get the data path
# 
# Let's get some images to play with this simple end-to-end example. Hundreds of images is a good start for Model Maker while more data could achieve better accuracy.

# In[ ]:


#load flower dataset
image_path = tf.keras.utils.get_file(
      'flower_photos',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      untar=True)

#below code is for unpacking archives/zip/rar
#!pip install pyunpack
#!pip install patool
#from pyunpack import Archive
#Archive('/content/stage 01.rar').extractall('/content/new/')

#load your own dataset
#image_path= '/content/new/stage 01/'


# You could replace image_path with your own image folders. As for uploading data to colab, you could find the upload button in the left sidebar shown in the image below with the red rectangle. Just have a try to upload a zip file and unzip it. The root file path is the current path.
# 
# Upload File
# 
# If you prefer not to upload your images to the cloud, you could try to run the library locally following the guide in github.
# Run the example
# 
# The example just consists of 4 lines of code as shown below, each of which representing one step of the overall process.
# 
#     1. Load input data specific to an on-device ML app. Split it to training data and testing data.
# 
# 

# In[ ]:


data = ImageClassifierDataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)


# 2. Customize the TensorFlow model.

# # **Head over [this](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification) page for training on specific models, cutomization of models.**
# 
# **Change the model**
# 
# Change to the model that's supported in this library.
# 
# This library supports EfficientNet-Lite models, MobileNetV2, ResNet50 by now. EfficientNet-Lite are a family of image classification models that could achieve state-of-art accuracy and suitable for Edge devices. The default model is EfficientNet-Lite0.
# 
# We could switch model to MobileNetV2 by just setting parameter model_spec to mobilenet_v2_spec in create method.
# 
# *model = image_classifier.create(train_data, model_spec=mobilenet_v2_spec, validation_data=validation_data)*
# 

# In[ ]:


model = image_classifier.create(train_data,model_spec=mobilenet_v2_spec)


# # Change to the model in TensorFlow Hub
# 
# Moreover, we could also switch to other new models that inputs an image and outputs a feature vector with TensorFlow Hub format.
# 
# As Inception V3 model as an example, we could define inception_v3_spec which is an object of ImageModelSpec and contains the specification of the Inception V3 model.
# 
# We need to specify the model name name, the url of the TensorFlow Hub model uri. Meanwhile, the default value of input_image_shape is [224, 224]. We need to change it to [299, 299] for Inception V3 model.
# 
# *inception_v3_spec = ImageModelSpec(
#     uri='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
# inception_v3_spec.input_image_shape = [299, 299]*
# 
# Then, by setting parameter model_spec to inception_v3_spec in create method, we could retrain the Inception V3 model.
# 
# The remaining steps are exactly same and we could get a customized InceptionV3 TensorFlow Lite model in the end.
# Change your own custom model
# 
# If we'd like to use the custom model that's not in TensorFlow Hub, we should create and export ModelSpec in TensorFlow Hub.
# 
# Then start to define ImageModelSpec object like the process above.
# Change the training hyperparameters
# 
# We could also change the training hyperparameters like epochs, dropout_rate and batch_size that could affect the model accuracy. For instance,
# 
#     epochs: more epochs could achieve better accuracy until it converges but training for too many epochs may lead to overfitting.
#     dropout_rate: avoid overfitting.
#     batch_size: number of samples to use in one training step.
#     validation_data: number of samples to use in one training step.
# 
# For example, we could train with more epochs.
# 
# *model = image_classifier.create(train_data, validation_data=validation_data, epochs=10)
# *

# 3. Evaluate the model.

# In[ ]:


loss, accuracy = model.evaluate(test_data)


# **4. Export to TensorFlow Lite model. You could download it in the left sidebar same as the uploading part for your own use.**

# In[ ]:


model.export(export_dir='.', with_metadata=True)
from IPython.display import FileLink, FileLinks
FileLinks('.') #generates links of all files
#FileLinks('model.tflite')
#FileLinks('labels.txt')


# After this simple 4 steps, we can now download the model and label files, and continue to the next step in the [codelab](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/#5).
# 
# For a more comprehensive guide to TFLite Model Maker, please refer to this [notebook](https://colab.research.google.com/github/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/demo/image_classification.ipynb) and its documentation.
# 

# **The source of this kernel is [here](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification)**

# # Connect me at "shubham.divakar@gmail.com" for any queries.
# # Soon I will be publishing android apps using the tflite models from here.

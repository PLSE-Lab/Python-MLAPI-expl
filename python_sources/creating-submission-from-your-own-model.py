#!/usr/bin/env python
# coding: utf-8

# There has been a lot of confusion on how exactly we are supposed to submit our model. As the ***Data*** section of the competition states:
# > Your model must be named submission.zip and be compatible with TensorFlow 2.2. The submission.zip should contain all files and directories created by the tf.saved_model_save function using Tensorflow's SavedModel format.
# 
# Now question is what exactly in the [SavedModel](https://www.tensorflow.org/tutorials/keras/save_and_load#savedmodel_format) format do we need to submit.
# 
# Also, majority of us don't want to use tensorflow to train our models. And we don't know how to preprocess. So we'll tackle two things mainly.
# 
# 1. Use our own keras model in submission.
# 2. How to preprocess.
# 
# Let's get started.

# Let's reverse engineer the model that organisers gave us as baseline. We'll use saved_model_cli to visualize it's structure. You may want to check out this [discussion thread](https://www.kaggle.com/c/landmark-retrieval-2020/discussion/163589).

# In[ ]:


get_ipython().system('saved_model_cli show --dir "../input/baseline-landmark-retrieval-model/baseline_landmark_retrieval_model" --all')


# Important things to notice are:
# 
#     inputs['input_image'] tensor_info:
#     dtype: DT_UINT8
#     shape: (-1, -1, 3)
#         
#     outputs['global_descriptor'] tensor_info:
#     dtype: DT_FLOAT
#     shape: (2048)

# Armed with this information, let's create our own model.

# In[ ]:


import numpy as np

import os
import cv2
import glob

import tensorflow as tf
import keras
from keras.models import load_model, save_model
from keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D
import keras.backend as K
from keras.models import Model, load_model
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input


# There are varying shapes of images as you can see below, meaning we'll need to resize images inside the model.

# In[ ]:


files = glob.glob("../input/landmark-retrieval-2020/train/a/b/c/*.jpg")
for i in range(10):
    im = cv2.imread(files[i])
    print(im.shape)


# Now let's load our model. In this case the vanilla VGG16 pretrained model of Keras for demonstration purposes. Since this is not trained on any retrieval dataset, the score will most probably be zero.

# In[ ]:


vgg = VGG16(input_shape=(224,224,3), weights=None, include_top=False)
vgg.load_weights("../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

input_image = Input((224,224,3))
x = vgg(input_image)
output = GlobalMaxPooling2D()(x)

model = Model(inputs=[input_image], outputs=[output])
model.summary()


# Now the main part! The *input_image* will be in it's own variable shape and hence we need to resize it within the model.

# In[ ]:


import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = model
    
    @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image')
    ])
    def call(self, input_image):
        output_tensors = {}
        
        # resizing
        im = tf.image.resize(input_image, (224,224))
        
        # preprocessing
        im = preprocess_input(im)
        
        extracted_features = self.model(tf.convert_to_tensor([im], dtype=tf.uint8))[0]
        output_tensors['global_descriptor'] = tf.identity(extracted_features, name='global_descriptor')
        return output_tensors


# Now we create and save our model instance.

# In[ ]:


m = MyModel() #creating our model instance

served_function = m.call
tf.saved_model.save(
      m, export_dir="./my_model", signatures={'serving_default': served_function})


# In[ ]:


get_ipython().system('ls ./my_model/variables')


# In[ ]:


from zipfile import ZipFile

with ZipFile('submission.zip','w') as zip:           
    zip.write('./my_model/saved_model.pb', arcname='saved_model.pb') 
    zip.write('./my_model/variables/variables.data-00000-of-00002', arcname='variables/variables.data-00000-of-00002')
    zip.write('./my_model/variables/variables.data-00001-of-00002', arcname='variables/variables.data-00001-of-00002') 
    zip.write('./my_model/variables/variables.index', arcname='variables/variables.index') 


# Last but not the least, let's visualize our model to see if the structure is as per the requirements.

# In[ ]:


get_ipython().system('saved_model_cli show --dir ./my_model/ --all')


# Please upvote and let me know if this helps!
